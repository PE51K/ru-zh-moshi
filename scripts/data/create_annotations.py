#!/usr/bin/env python3
import argparse
import gc
import json
import logging
import os
import sys
from pathlib import Path

from tqdm import tqdm
import sphn
import torch
import torchaudio.functional as F
import whisper_timestamped as whisper

# Configure logging
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
    datefmt="%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000

def normalize_language_code(lang: str) -> str:
    """Convert language codes like 'zh-cn' to 'zh'."""
    return lang.split('-')[0]

def process_segment(audio_path: Path, language: str, w_model, source_sample_rate: int) -> dict:
    """Process a single audio segment with whisper.
    
    Args:
        audio_path: Path to audio file
        language: Language code (e.g. 'zh-cn', 'en-us')
        w_model: Loaded whisper model
        source_sample_rate: Original audio sample rate
    """
    gc.collect()
    torch.cuda.empty_cache()

    # Load audio
    x, sr = sphn.read(audio_path)
    if sr != source_sample_rate:
        logger.warning(f"Audio sample rate {sr} does not match metadata {source_sample_rate}")
    x = torch.from_numpy(x).cuda()
    if len(x.shape) > 1:
        x = x[0]  # Take first channel if stereo
    x = x[None]  # Add batch dimension for resampling
    if sr != SAMPLE_RATE:
        x = F.resample(x, sr, SAMPLE_RATE)
    x = x.cpu().numpy()[0]

    # Normalize language code for whisper
    normalized_lang = normalize_language_code(language)

    result = whisper.transcribe(
        w_model,
        x,
        language=normalized_lang,
        vad=None,  # No VAD needed for segments
        best_of=5,
        beam_size=5,
        temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        verbose=None,
    )

    # Extract word-level alignments
    alignments = []
    for segment in result["segments"]:
        if "words" not in segment:
            continue
        for word in segment["words"]:
            try:
                alignments.append({
                    "text": word["text"],
                    "timestamp": [word["start"], word["end"]]
                })
            except KeyError as e:
                logger.error(f"Missing key in word data: {e}")
                continue

    return {"alignments": alignments}

def process_turn(turn_dir: Path, w_model):
    """Process all segments in a conversation turn."""
    # Skip non-assistant turns
    if not turn_dir.name.startswith("1_assistant") and not turn_dir.name.endswith("_assistant"):
        return True

    # Load metadata
    try:
        with open(turn_dir / "metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load metadata for {turn_dir}: {e}")
        return False

    # Get sample rate from metadata or use default TTS rate
    source_sample_rate = metadata.get("sample_rate", 24000)

    # Process each segment without progress bar
    for segment in metadata["segments"]:
        segment_file = turn_dir / segment["file"]
        annotation_file = segment_file.with_suffix(".json")

        # Skip if annotation exists
        if annotation_file.exists():
            continue

        try:
            # Process segment with correct sample rate
            result = process_segment(segment_file, segment["lang"], w_model, source_sample_rate)

            # Save annotation
            with open(annotation_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logger.debug(f"Saved annotation to {annotation_file}")

        except Exception as e:
            logger.error(f"Failed to process {segment_file}: {e}")
            continue

    return True

def main():
    parser = argparse.ArgumentParser(description="Create annotations for audio segments")
    parser.add_argument("audio_dir", type=Path, help="Path to audio dataset directory")
    parser.add_argument("--start", type=int, default=0, help="Starting conversation index")
    parser.add_argument("--end", type=int, help="Ending conversation index (exclusive)")
    parser.add_argument("--whisper-model", default="medium", help="Whisper model to use")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()

    # Setup logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize Whisper
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    w_model = whisper.load_model(args.whisper_model, device=device)
    logger.info(f"Loaded Whisper model '{args.whisper_model}' on {device}")

    # Get conversation directories and sort them numerically
    def get_conv_number(path):
        try:
            # Extract number from directory name (assumes format like "conv_123" or just "123")
            return int(''.join(filter(str.isdigit, path.name)))
        except ValueError:
            return float('inf')  # Put non-numeric names at the end

    conv_dirs = sorted(
        [d for d in args.audio_dir.iterdir() if d.is_dir()],
        key=get_conv_number
    )
    if args.end:
        conv_dirs = conv_dirs[args.start:args.end]
    else:
        conv_dirs = conv_dirs[args.start:]

    # Process each conversation
    for conv_dir in tqdm(conv_dirs, desc="Processing conversations"):
        logger.info(f"Processing conversation: {conv_dir.name}")
        # Get all turn directories and sort them
        turn_dirs = sorted([d for d in conv_dir.iterdir() if d.is_dir()])
        
        # Process each turn without progress bar
        for turn_dir in turn_dirs:
            process_turn(turn_dir, w_model)

    logger.info("Annotation generation complete")

if __name__ == "__main__":
    main()
