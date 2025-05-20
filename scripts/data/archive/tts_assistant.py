#!/usr/bin/env python3
import torch
from TTS.api import TTS
import json
import os
import soundfile as sf
import sys
import argparse
import logging
from tqdm import tqdm


def count_words(text, language):
    """Count words in a text string with attention to language."""
    if language == "ru":
        # For Russian, we can use a simple split by whitespace
        return len(text.split())
    elif language == "zh-cn":
        # For Chinese, we should count characters as words
        return len(text)


def truncate_text(text, max_words):
    """
    Truncate text to a maximum number of words with attention to punctuation. It means, that we returning last sentence on which max_word limit are exceeded.
    """
    sentences = text.split('.')
    words_count = 0
    truncated_text = []
    for sentence in sentences:
        words_in_sentence = sentence.split()
        truncated_text.append(sentence.strip())
        words_count += len(words_in_sentence)
        if words_count >= max_words:
            break
    return '. '.join(truncated_text).strip() + ('.' if truncated_text else '')


def main():
    # Configure argument parser
    parser = argparse.ArgumentParser(description='Process assistant replies with TTS')
    parser.add_argument('--start', type=int, required=True, help='Starting conversation index (inclusive)')
    parser.add_argument('--end', type=int, required=True, help='Ending conversation index (exclusive)')
    parser.add_argument('--jsonl', type=str, default="dataset/text_cleaned/data.jsonl", help='Path to JSONL file')
    parser.add_argument('--voice', type=str, required=True, help='Voice file for assistant')
    parser.add_argument('--output', type=str, default="dataset/audio", help='Output directory for audio files')
    parser.add_argument('--max-words', type=int, default=20, help='Maximum number of words per reply')
    parser.add_argument('--max-overflow', type=int, default=10, 
                       help='Maximum words to keep in overflow segment before truncating')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                       help='Device to run TTS on (cuda/cpu)')
    args = parser.parse_args()

    # Override the torch.load function to use weights_only=False by default
    original_torch_load = torch.load
    torch.load = lambda f, map_location=None, pickle_module=torch.serialization.pickle, **kwargs: original_torch_load(
        f, map_location=map_location, pickle_module=pickle_module, weights_only=False, **kwargs
    )

    # Configure minimal logging
    logging.basicConfig(level=logging.ERROR)
    logging.getLogger('TTS').setLevel(logging.ERROR)
    logging.getLogger('TTS.utils.synthesizer').setLevel(logging.ERROR)

    # Initialize TTS model
    print(f"Initializing TTS model on {args.device}...")
    tts = TTS("omogr/xtts-ru-ipa").to(args.device)
    print("Model loaded successfully")

    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)

    # Read all conversations from the JSONL file
    print(f"Reading conversations from {args.jsonl}...")
    conversations = []
    with open(args.jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            conversations.append(json.loads(line))

    # Validate slice indices
    if args.start < 0 or args.end > len(conversations) or args.start >= args.end:
        print(f"Invalid slice range: {args.start} to {args.end}. Valid range is 0 to {len(conversations)}")
        sys.exit(1)

    print(f"Processing conversations from index {args.start} to {args.end-1} (total: {args.end-args.start})")

    # Process the specified slice of conversations
    slice_conversations = conversations[args.start:args.end]
    for conv_idx, conversation in enumerate(tqdm(slice_conversations, desc="Processing conversations")):
        actual_idx = args.start + conv_idx
        try:
            # Create a directory for this conversation
            conversation_dir = os.path.join(args.output, f"conversation_{actual_idx}")
            os.makedirs(conversation_dir, exist_ok=True)

            # Process each turn in the conversation
            for turn_idx, message in enumerate(conversation):
                if message.get("role") != "assistant":
                    continue

                # Define output filename (1-indexed turn number)
                output_file = os.path.join(conversation_dir, f"{turn_idx+1}_assistant.wav")

                # Skip if file already exists
                if os.path.exists(output_file):
                    continue

                # Create a directory for this turn's segments
                turn_dir = os.path.join(conversation_dir, f"{turn_idx+1}_assistant")
                os.makedirs(turn_dir, exist_ok=True)

                # Initialize metadata for segments
                segments_metadata = {
                    "sample_rate": 24000,  # TTS output sample rate
                    "segments": []
                }

                # Process segments until max words is reached
                cumulative_words = 0
                
                for seg_idx, segment in enumerate(message["language_segments"]):
                    segment_text = segment["text"].strip()
                    segment_lang = segment["lang"]
                    
                    if not segment_text:
                        continue

                    # Count words in current segment
                    cumulative_words += count_words(segment_text, segment_lang)
                    
                    # Process this segment if we haven't exceeded max words or if this is the segment that exceeds it
                    if not cumulative_words > args.max_words:
                        # Define segment output file
                        segment_file = os.path.join(turn_dir, f"segment_{seg_idx}.wav")
                        
                        if os.path.exists(segment_file):
                            continue

                        try:
                            segment_audio = tts.tts(
                                text=segment_text,
                                speaker_wav=args.voice,
                                language=segment_lang
                            )
                            
                            sf.write(segment_file, segment_audio, tts.synthesizer.output_sample_rate)

                            segments_metadata["segments"].append({
                                "file": f"segment_{seg_idx}.wav",
                                "lang": segment_lang,
                                "text": segment_text,
                                "duration": len(segment_audio) / tts.synthesizer.output_sample_rate
                            })
                            
                        except Exception as e:
                            print(f"Error processing segment {seg_idx} in conversation {actual_idx}, turn {turn_idx+1}: {str(e)}")
                        
                    else:
                        # If we exceed max words, truncate the text
                        truncated_text = truncate_text(segment_text, args.max_overflow)
                        
                        # Define segment output file
                        segment_file = os.path.join(turn_dir, f"segment_{seg_idx}.wav")
                        
                        if os.path.exists(segment_file):
                            continue

                        try:
                            segment_audio = tts.tts(
                                text=truncated_text,
                                speaker_wav=args.voice,
                                language=segment_lang
                            )
                            
                            sf.write(segment_file, segment_audio, tts.synthesizer.output_sample_rate)

                            segments_metadata["segments"].append({
                                "file": f"segment_{seg_idx}.wav",
                                "lang": segment_lang,
                                "text": truncated_text,
                                "duration": len(segment_audio) / tts.synthesizer.output_sample_rate
                            })
                            
                        except Exception as e:
                            print(f"Error processing segment {seg_idx} in conversation {actual_idx}, turn {turn_idx+1}: {str(e)}")

                        finally:
                            # Break the loop after processing the overflow segment
                            break

                # Save metadata if we processed any segments
                if segments_metadata["segments"]:
                    metadata_file = os.path.join(turn_dir, "metadata.json")
                    with open(metadata_file, 'w', encoding='utf-8') as f:
                        json.dump(segments_metadata, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"Error processing conversation {actual_idx}: {str(e)}")
            continue


if __name__ == "__main__":
    main()
