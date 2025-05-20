#!/usr/bin/env python3
import argparse
import json
import logging
import sys
from pathlib import Path

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

def main():
    parser = argparse.ArgumentParser(description="Prepare JSONL file for Moshi fine-tuning")
    parser.add_argument("dataset_dir", type=Path, help="Path to dataset directory containing data_stereo/")
    parser.add_argument("-o", "--output", type=Path, help="Output JSONL file (default: dataset_dir/data.jsonl)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Verify data_stereo directory exists
    stereo_dir = args.dataset_dir / "data_stereo"
    if not stereo_dir.exists():
        logger.error(f"data_stereo directory not found in {args.dataset_dir}")
        sys.exit(1)

    # Set output path
    output_path = args.output or (args.dataset_dir / "data.jsonl")

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

    # Write output file
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in tqdm(entries, desc="Writing entries to JSONL"):
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')

    logger.info(f"Successfully processed {valid_count} files")
    logger.info(f"Invalid files: {len(wav_files) - valid_count}")
    logger.info(f"Output written to {output_path}")

if __name__ == "__main__":
    main()
