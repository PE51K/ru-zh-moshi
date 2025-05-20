#!/usr/bin/env python3
import argparse
import json
import logging
import os
import sys
import random
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import stats
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
    datefmt="%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Duration limits
MAX_USER_DURATION = 20  # seconds
MAX_ASSISTANT_DURATION = 30  # seconds
RANDOM_PADDING_FLOOR = 2  # seconds
RANDOM_PADDING_CEIL = 5  # seconds

# Audio augmentation parameters
GAIN_AUGMENT_PROB = 0.5
NOISE_AUGMENT_PROB = 0.3
SILENCE_PROB = 0.5
MIN_GAIN_DB = -24
MAX_GAIN_DB = 15
MIN_NOISE_DB = -30
MAX_NOISE_DB = 6
MAX_SILENCE_DURATION = 30  # seconds

def db_to_linear(db):
    """Convert dB value to linear scale."""
    return 10 ** (db / 20)

def generate_noise(duration, sample_rate):
    """Generate synthetic noise with given duration."""
    samples = int(duration * sample_rate)
    amplitude = 11  # As per example
    noise = stats.truncnorm(-1, 1, scale=min(2**16, 2**amplitude)).rvs(samples)
    return noise.astype(np.float32) / 32768.0  # Normalize to [-1, 1]

def generate_silence(duration, sample_rate):
    """Generate silence with given duration."""
    samples = int(duration * sample_rate)
    return np.zeros(samples, dtype=np.float32)

def apply_audio_augmentation(audio, sample_rate):
    """Apply audio augmentations to user's audio stream."""
    # Random gain adjustment (50% probability)
    if random.random() < GAIN_AUGMENT_PROB:
        gain_db = random.uniform(MIN_GAIN_DB, MAX_GAIN_DB)
        gain_linear = db_to_linear(gain_db)
        audio = audio * gain_linear
        
    # Noise or silence addition (30% probability)
    if random.random() < NOISE_AUGMENT_PROB:
        duration = len(audio) / sample_rate
        
        # Generate noise or silence (50% probability for silence)
        if random.random() < SILENCE_PROB:
            silence_duration = random.uniform(0, min(MAX_SILENCE_DURATION, duration))
            noise = generate_silence(silence_duration, sample_rate)
            # Pad noise to match audio length if needed
            if len(noise) < len(audio):
                noise = np.pad(noise, (0, len(audio) - len(noise)), mode='constant')
        else:
            noise = generate_noise(duration, sample_rate)
            
        # Adjust noise level relative to source
        target_db = random.uniform(MIN_NOISE_DB, MAX_NOISE_DB)
        noise_linear = db_to_linear(target_db)
        noise = noise * noise_linear
        
        # Add noise to audio
        audio = np.clip(audio + noise[:len(audio)], -1, 1)
    
    return audio

def load_segment_data(segment_path: Path, metadata: dict, role: str) -> tuple:
    """Load audio segment and its annotation, adjusting timestamps.
    
    Args:
        segment_path: Path to the audio segment file
        metadata: Segment metadata
        role: Speaker role ('assistant' or 'user')
        
    Returns:
        tuple: (audio array, sample rate, annotation dict)
    """
    # Load audio
    audio, sr = sf.read(segment_path)
    if len(audio.shape) > 1:
        audio = audio[:, 0]  # Take first channel if stereo

    # Verify sample rate matches metadata
    if sr != metadata.get("sample_rate", sr):
        logger.warning(f"Audio sample rate {sr} does not match metadata {metadata.get('sample_rate')}")

    # Only load annotations for assistant segments
    annotation = {"alignments": []}  # Default empty annotation
    if role == 'assistant':
        annotation_path = segment_path.with_suffix('.json')
        with open(annotation_path, 'r', encoding='utf-8') as f:
            annotation = json.load(f)

    return audio, sr, annotation

def limit_turn_duration(segments: list, metadata: dict, max_duration: float, sr: int) -> tuple:
    """Limit turn duration by selecting segments up to max_duration + random padding.
    
    Args:
        segments: List of audio segments
        metadata: Turn metadata
        max_duration: Maximum allowed duration in seconds
        sr: Sample rate
        
    Returns:
        tuple: (limited_segments, limited_metadata)
    """
    random_padding = random.uniform(RANDOM_PADDING_FLOOR, RANDOM_PADDING_CEIL)
    total_duration = 0
    limited_segments = []
    limited_metadata_segments = []

    for segment, meta in zip(segments, metadata["segments"]):
        duration = len(segment) / sr
        if total_duration + duration > max_duration + random_padding:
            break
            
        limited_segments.append(segment)
        limited_metadata_segments.append(meta)
        total_duration += duration

    limited_metadata = metadata.copy()
    limited_metadata["segments"] = limited_metadata_segments
    return limited_segments, limited_metadata

def group_qa_pairs(turn_dirs: list) -> list:
    """Group turn directories into user-assistant pairs."""
    pairs = []
    for i in range(0, len(turn_dirs) - 1, 2):
        # Skip if not a proper user-assistant pair
        user_turn = turn_dirs[i]
        assistant_turn = turn_dirs[i + 1] if i + 1 < len(turn_dirs) else None
        
        if assistant_turn and 'user' in user_turn.name and 'assistant' in assistant_turn.name:
            pairs.append((user_turn, assistant_turn))
    return pairs

def process_turn(turn_dir: Path, sample_rate: int, max_duration: float, role: str) -> tuple:
    """Process a single turn with duration limiting."""
    # Load metadata
    with open(turn_dir / "metadata.json", "r", encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Process segments
    turn_audio = []
    turn_alignments = []
    current_time = 0.0
    
    for segment in metadata["segments"]:
        segment_path = turn_dir / segment["file"]
        audio, sr, annotation = load_segment_data(segment_path, segment, role)
        
        if sr != sample_rate:
            raise ValueError(f"Inconsistent sample rates: {sr} vs {sample_rate}")
        
        # Apply augmentation for user audio
        if role == 'user':
            audio = apply_audio_augmentation(audio, sr)
            
        turn_audio.append(audio)
        
        if role == 'assistant':
            # Update timestamps for alignments
            for align in annotation["alignments"]:
                align["timestamp"] = [t + current_time for t in align["timestamp"]]
                turn_alignments.append([
                    align["text"],
                    align["timestamp"],
                    "SPEAKER_MAIN"
                ])
        
        current_time += len(audio) / sr

    # Apply duration limit
    if turn_audio:
        turn_audio, metadata = limit_turn_duration(turn_audio, metadata, max_duration, sample_rate)
        
    return turn_audio, turn_alignments

def merge_qa_pair(qa_pair: tuple, output_dir: Path, conv_name: str, pair_idx: int, sample_rate: int = 24000):
    """Merge a single QA pair into a stereo file with annotations."""
    try:
        user_dir, assistant_dir = qa_pair
        output_name = f"{conv_name}_qa{pair_idx}"
        audio_path = output_dir / f"{output_name}.wav"
        annotation_path = output_dir / f"{output_name}.json"
        
        if audio_path.exists() and annotation_path.exists():
            logger.info(f"Skipping {output_name} - files already exist")
            return
        
        # Process user turn
        user_audio, _ = process_turn(user_dir, sample_rate, MAX_USER_DURATION, "user")
        if not user_audio:
            return
            
        # Process assistant turn
        assistant_audio, alignments = process_turn(assistant_dir, sample_rate, MAX_ASSISTANT_DURATION, "assistant")
        if not assistant_audio:
            return
            
        # Concatenate segments for each turn
        user_audio = np.concatenate(user_audio)
        assistant_audio = np.concatenate(assistant_audio)
        
        # Create stereo segments
        user_stereo = np.zeros((len(user_audio), 2))
        user_stereo[:, 1] = user_audio
        
        assistant_stereo = np.zeros((len(assistant_audio), 2))
        assistant_stereo[:, 0] = assistant_audio
        
        # Combine into final stereo output
        stereo = np.concatenate([user_stereo, assistant_stereo], axis=0)
        
        # Save outputs
        output_dir.mkdir(parents=True, exist_ok=True)
        sf.write(audio_path, stereo, sample_rate)
        
        # Save alignments
        with open(annotation_path, 'w', encoding='utf-8') as f:
            json.dump({"alignments": alignments}, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        logger.error(f"Error processing QA pair {output_name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Merge conversation segments into Moshi format")
    parser.add_argument("audio_dir", type=Path, help="Path to audio dataset directory")
    parser.add_argument("output_dir", type=Path, help="Path to output directory")
    parser.add_argument("--start", type=int, default=0, help="Starting conversation index")
    parser.add_argument("--end", type=int, help="Ending conversation index (exclusive)")
    parser.add_argument("--sample-rate", type=int, default=24000, help="Target sample rate for output audio")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Setup output structure
    stereo_dir = args.output_dir / "data_stereo"
    stereo_dir.mkdir(parents=True, exist_ok=True)
    
    # Get conversation directories
    conv_dirs = sorted([d for d in args.audio_dir.iterdir() if d.is_dir()])
    if args.end:
        conv_dirs = conv_dirs[args.start:args.end]
    else:
        conv_dirs = conv_dirs[args.start:]
    
    # Process conversations
    for conv_dir in tqdm(conv_dirs, desc="Processing conversations"):
        # Get all turn directories and sort them
        turn_dirs = sorted([d for d in conv_dir.iterdir() if d.is_dir()])
        
        # Group turns into QA pairs
        qa_pairs = group_qa_pairs(turn_dirs)
        
        # Process each QA pair
        for pair_idx, qa_pair in enumerate(qa_pairs, 1):
            merge_qa_pair(qa_pair, stereo_dir, conv_dir.name, pair_idx, args.sample_rate)
    
    logger.info("Merge complete")

if __name__ == "__main__":
    main()
