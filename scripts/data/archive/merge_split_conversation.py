import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import stats
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Duration limits
MAX_USER_DURATION = 20  # seconds
MAX_ASSISTANT_DURATION = 30  # seconds

# Padding parameters
RANDOM_PADDING_FLOOR = 0.5  # seconds
RANDOM_PADDING_CEIL = 1.5  # seconds

# Audio augmentation parameters
GAIN_AUGMENT_PROB = 0.5
NOISE_AUGMENT_PROB = 0.3
MIN_GAIN_DB = -5
MAX_GAIN_DB = 5
MIN_NOISE_DB = -5
MAX_NOISE_DB = 5

# Chinese content threshold (90%)
CHINESE_CONTENT_THRESHOLD = 0.9


def db_to_linear(db):
    """Convert dB value to linear scale."""
    return 10 ** (db / 20)


def generate_noise(num_samples):
    """Generate synthetic noise with given number of samples."""
    amplitude = 11  # As per example
    noise = stats.truncnorm(-1, 1, scale=min(2**16, 2**amplitude)).rvs(num_samples)
    return noise.astype(np.float32) / 32768.0  # Normalize to [-1, 1]


def load_segment_data(segment_path: Path, metadata: dict, is_assistant: bool) -> tuple:
    """Load audio segment and its annotation, adjusting timestamps."""
    # Load audio
    audio, sr = sf.read(segment_path)
    if len(audio.shape) > 1:
        audio = audio[:, 0]  # Take first channel if stereo

    # Only load annotations for assistant segments
    annotation = {"alignments": []}  # Default empty annotation
    if is_assistant:
        annotation_path = segment_path.with_suffix('.json')
        with open(annotation_path, 'r', encoding='utf-8') as f:
            annotation = json.load(f)

    return audio, sr, annotation


def process_turn(turn_dir: Path, sample_rate: int) -> tuple:
    """Load and process a single turn's audio and annotations."""
    # Load metadata
    with open(turn_dir / "metadata.json", "r", encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Process segments
    all_audio = []
    all_alignments = []
    current_time = 0.0
    
    for segment in metadata["segments"]:
        segment_path = turn_dir / segment["file"]
        audio, sr, annotation = load_segment_data(segment_path, metadata, turn_dir.name.endswith('assistant'))
        
        if sr != sample_rate:
            raise ValueError(f"Inconsistent sample rates: {sr} vs {sample_rate}")
            
        all_audio.append(audio)
        
        if turn_dir.name.endswith('assistant'):
            # Just collect alignments with current timing
            for align in annotation["alignments"]:
                timestamps = [t + current_time for t in align["timestamp"]]
                all_alignments.append([
                    align["text"],
                    timestamps,
                    "SPEAKER_MAIN"
                ])
        
        current_time += len(audio) / sr
    
    # Concatenate all segments
    full_audio = np.concatenate(all_audio) if all_audio else np.array([], dtype=np.float32)
    
    return full_audio, all_alignments, sr


def merge_qa_pair(user_turn_dir: Path, assistant_turn_dir: Path, output_dir: Path, conv_name: str, assistant_turn_number: int, sample_rate: int = 24000):
    """Merge a user-assistant pair into a stereo file with annotations following specific steps."""
    pair_idx = (assistant_turn_number // 2) + 1
    output_name = f"{conv_name}_qa{pair_idx}"
    audio_path = output_dir / f"{output_name}.wav"
    annotation_path = output_dir / f"{output_name}.json"
    
    if audio_path.exists() and annotation_path.exists():
        logger.info(f"Skipping {output_name} - files already exist")
        return
    
    # Check Chinese content ratio in assistant turn
    assistant_metadata_path = assistant_turn_dir / "metadata.json"
    with open(assistant_metadata_path, 'r', encoding='utf-8') as f:
        assistant_metadata = json.load(f)
    
    # Calculate Chinese content ratio based on segment durations
    total_duration = 0
    chinese_duration = 0
    for segment in assistant_metadata["segments"]:
        duration = segment.get("duration", 0)
        total_duration += duration
        if segment.get("lang") == "zh-cn":
            chinese_duration += duration
    
    if total_duration > 0:
        chinese_ratio = chinese_duration / total_duration
        if chinese_ratio > CHINESE_CONTENT_THRESHOLD:
            logger.info(f"Skipping {output_name} - too much Chinese content ({chinese_ratio:.1%})")
            return
    
    # 1. Get audio and check durations
    user_audio, _, sr = process_turn(user_turn_dir, sample_rate)
    assistant_audio, alignments, sr = process_turn(assistant_turn_dir, sample_rate)
    
    # 2. Check durations and skip if too long
    user_duration = len(user_audio) / sr
    assistant_duration = len(assistant_audio) / sr
    
    if user_duration > MAX_USER_DURATION or assistant_duration > MAX_ASSISTANT_DURATION:
        logger.info(f"Skipping {output_name} - durations exceed limits: user={user_duration:.2f}s, assistant={assistant_duration:.2f}s")
        return
    
    # 3. Apply augmentations to user audio
    if random.random() < NOISE_AUGMENT_PROB:
        noise_db = random.uniform(MIN_NOISE_DB, MAX_NOISE_DB)
        noise = generate_noise(len(user_audio))
        noise_linear = db_to_linear(noise_db)
        noise = noise * noise_linear
        user_audio = np.clip(user_audio + noise, -1, 1)
    
    if random.random() < GAIN_AUGMENT_PROB:
        gain_db = random.uniform(MIN_GAIN_DB, MAX_GAIN_DB)
        gain_linear = db_to_linear(gain_db)
        user_audio = np.clip(user_audio * gain_linear, -1, 1)
    
    # 4. Add random silence
    silence_duration = random.uniform(RANDOM_PADDING_FLOOR, RANDOM_PADDING_CEIL)
    silence_samples = int(silence_duration * sample_rate)
    silence = np.zeros(silence_samples, dtype=np.float32)
    
    # 5 & 6. Create final audio and update annotations
    total_len = len(user_audio) + len(silence) + len(assistant_audio)
    stereo = np.zeros((total_len, 2), dtype=np.float32)
    
    # User audio in right channel
    stereo[:len(user_audio), 1] = user_audio
    
    # Assistant audio in left channel after silence
    start_idx = len(user_audio) + len(silence)
    stereo[start_idx:start_idx + len(assistant_audio), 0] = assistant_audio
    
    # Update alignments considering user speech and silence
    user_duration = len(user_audio) / sr
    silence_duration = len(silence) / sr
    updated_alignments = []
    
    for align in alignments:
        timestamps = [t + user_duration + silence_duration for t in align[1]]
        updated_alignments.append([
            align[0],
            timestamps,
            "SPEAKER_MAIN"
        ])
    
    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    sf.write(audio_path, stereo, sample_rate)
    
    with open(annotation_path, 'w', encoding='utf-8') as f:
        json.dump({"alignments": updated_alignments}, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Merge conversation segments from separate user and assistant directories")
    parser.add_argument("user_dir", type=Path, help="Path to user audio dataset directory")
    parser.add_argument("assistant_dir", type=Path, help="Path to assistant audio dataset directory")
    parser.add_argument("output_dir", type=Path, help="Path to output directory")
    parser.add_argument("--start", type=int, default=0, help="Starting conversation index")
    parser.add_argument("--end", type=int, help="Ending conversation index (exclusive)")
    parser.add_argument("--sample-rate", type=int, default=24000, help="Target sample rate for output audio")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Set up logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
        
    # Validate input directories
    if not args.user_dir.exists():
        raise FileNotFoundError(f"User directory not found: {args.user_dir}")
    if not args.assistant_dir.exists():
        raise FileNotFoundError(f"Assistant directory not found: {args.assistant_dir}")
    
    # Setup output structure
    stereo_dir = args.output_dir / "data_stereo"
    stereo_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate end index
    if args.end is None:
        max_conv = max(
            int(d.name.split("_")[1]) for d in args.assistant_dir.iterdir() 
            if d.is_dir() and d.name.startswith("conversation_")
        )
        args.end = max_conv + 1
    
    # Process conversations
    for conv_idx in tqdm(range(args.start, args.end), desc="Processing conversations"):
        conv_name = f"conversation_{conv_idx}"
        
        user_conv_dir = args.user_dir / conv_name
        assistant_conv_dir = args.assistant_dir / conv_name
        
        if not user_conv_dir.exists():
            logger.warning(f"Skipping {conv_name} - user directory not found")
            continue
        if not assistant_conv_dir.exists():
            logger.warning(f"Skipping {conv_name} - assistant directory not found")
            continue
        
        # Get all directories from assistant conversation
        assistant_turns = sorted(
            [d for d in assistant_conv_dir.iterdir() if d.is_dir()],
            key=lambda x: int(x.name.split("_")[0])
        )
        
        for assistant_turn_dir in assistant_turns:
            try:
                # Get assistant turn number
                assistant_turn_number = int(assistant_turn_dir.name.split("_")[0])
                # Get user turn dir
                user_turn_dir = user_conv_dir / f"{assistant_turn_number - 1}_user"
                
                if not user_turn_dir.exists():
                    logger.warning(f"Skipping turn {assistant_turn_number} - user turn not found")
                    continue
                    
                merge_qa_pair(user_turn_dir, assistant_turn_dir, stereo_dir,
                            conv_name, assistant_turn_number, args.sample_rate)
                            
            except Exception as e:
                logger.error(f"Error processing turn {assistant_turn_dir}: {str(e)}")
                continue


if __name__ == "__main__":
    main()
