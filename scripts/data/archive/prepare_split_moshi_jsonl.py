#!/usr/bin/env python3
import argparse
import json
import logging
import sys
import random
from pathlib import Path
from typing import List, Dict, Any

import soundfile as sf
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
    datefmt="%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def validate_audio_file(audio_path: Path) -> tuple[bool, float | None]:
    """Validate audio file and return its duration if valid."""
    try:
        # Check if file exists
        if not audio_path.exists():
            logger.error(f"Audio file not found: {audio_path}")
            return False, None

        # Check if JSON exists
        json_path = audio_path.with_suffix('.json')
        if not json_path.exists():
            logger.error(f"JSON file not found: {json_path}")
            return False, None

        # Validate audio format
        info = sf.info(audio_path)
        
        # Check if stereo
        if info.channels != 2:
            logger.error(f"Audio must be stereo, got {info.channels} channels: {audio_path}")
            return False, None
        
        # Check sample rate
        if info.samplerate != 24000:
            logger.error(f"Audio must be 24kHz, got {info.samplerate}Hz: {audio_path}")
            return False, None

        return True, info.duration

    except Exception as e:
        logger.error(f"Error validating {audio_path}: {e}")
        return False, None

def filter_entries(entries: List[Dict[str, Any]], conversation_jsonl: Path) -> List[Dict[str, Any]]:
    """Filter entries based on conversation JSONL file.
    Currently a mock function that passes through all entries.
    """
    # TODO: Implement actual filtering logic
    logger.info("Using mock filter function - no entries filtered")
    return entries

def split_train_val(entries: List[Dict[str, Any]], val_ratio: float = 0.1) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split entries into training and validation sets."""
    # Ensure reproducible shuffling
    random.seed(42)
    
    # Shuffle entries
    shuffled_entries = entries.copy()
    random.shuffle(shuffled_entries)
    
    # Calculate split point
    val_size = int(len(shuffled_entries) * val_ratio)
    
    # Split entries
    val_entries = shuffled_entries[:val_size]
    train_entries = shuffled_entries[val_size:]
    
    return train_entries, val_entries

def write_jsonl(entries: List[Dict[str, Any]], output_path: Path):
    """Write entries to a JSONL file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in tqdm(entries, desc=f"Writing entries to {output_path.name}"):
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')

def main():
    parser = argparse.ArgumentParser(description="Prepare split JSONL files for Moshi fine-tuning")
    parser.add_argument("dataset_dir", type=Path, help="Path to dataset directory containing data_stereo/")
    parser.add_argument("--output-dir", type=Path, help="Output directory (default: dataset_dir)")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation set ratio (default: 0.1)")
    parser.add_argument("--conversation-jsonl", type=Path, help="Path to conversation JSONL for filtering (optional)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling (default: 42)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Set random seed
    random.seed(args.seed)

    # Verify data_stereo directory exists
    stereo_dir = args.dataset_dir / "data_stereo"
    if not stereo_dir.exists():
        logger.error(f"data_stereo directory not found in {args.dataset_dir}")
        sys.exit(1)

    # Set output directory
    output_dir = args.output_dir or args.dataset_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process all wav files
    wav_files = sorted(stereo_dir.glob("*.wav"))
    logger.info(f"Found {len(wav_files)} WAV files in {stereo_dir}")

    entries = []
    valid_count = 0
    
    for wav_path in tqdm(wav_files, desc="Processing audio files"):
        # Validate file and get duration
        is_valid, duration = validate_audio_file(wav_path)
        
        if is_valid and duration is not None:
            # Create relative path from dataset root
            rel_path = wav_path.relative_to(args.dataset_dir)
            
            # Create entry
            entry = {
                "path": str(rel_path),
                "duration": duration
            }
            entries.append(entry)
            valid_count += 1

    # Filter entries if conversation JSONL provided
    if args.conversation_jsonl:
        entries = filter_entries(entries, args.conversation_jsonl)
        logger.info(f"Filtered entries: {len(entries)}")

    # Split into train and validation sets
    train_entries, val_entries = split_train_val(entries, args.val_ratio)
    logger.info(f"Split into {len(train_entries)} train and {len(val_entries)} validation entries")

    # Write output files
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    
    write_jsonl(train_entries, train_path)
    write_jsonl(val_entries, val_path)

    logger.info(f"Successfully processed {valid_count} files")
    logger.info(f"Invalid files: {len(wav_files) - valid_count}")
    logger.info(f"Output written to:")
    logger.info(f"  Train: {train_path}")
    logger.info(f"  Validation: {val_path}")

if __name__ == "__main__":
    main()