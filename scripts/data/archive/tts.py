#!/usr/bin/env python3
import torch
from TTS.api import TTS
import json
import os
import numpy as np
from tqdm import tqdm
import logging
import soundfile as sf
import sys
import random
import argparse

def main():
    # Configure argument parser
    parser = argparse.ArgumentParser(description='Process a slice of conversations with TTS')
    parser.add_argument('--start', type=int, required=True, help='Starting conversation index (inclusive)')
    parser.add_argument('--end', type=int, required=True, help='Ending conversation index (exclusive)')
    parser.add_argument('--jsonl', type=str, default="dataset/text_cleaned/data.jsonl", help='Path to JSONL file')
    parser.add_argument('--voices', type=str, default="dataset/voices", help='Directory containing voice samples')
    parser.add_argument('--assistant-voice', type=str, help='Specific voice file for assistant (relative to voices directory)')
    parser.add_argument('--output', type=str, default="dataset/audio", help='Output directory for audio files')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help='Device to run TTS on (cuda/cpu)')
    args = parser.parse_args()

    # Override the torch.load function to use weights_only=False by default
    # Only do this if you trust the source of the checkpoint
    original_torch_load = torch.load
    torch.load = lambda f, map_location=None, pickle_module=torch.serialization.pickle, **kwargs: original_torch_load(
        f, map_location=map_location, pickle_module=pickle_module, weights_only=False, **kwargs
    )

    # Configure minimal logging - aggressive suppression
    logging.basicConfig(level=logging.ERROR)
    logging.getLogger('TTS').setLevel(logging.ERROR)
    logging.getLogger('TTS.utils.synthesizer').setLevel(logging.ERROR)

    # Initialize TTS model
    print(f"Initializing TTS model on {args.device}...")
    tts = TTS("omogr/xtts-ru-ipa").to(args.device)
    print("Model loaded successfully")

    # Suppress stdout temporarily during TTS operations
    class NullWriter:
        def write(self, s):
            pass
        def flush(self):
            pass

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

    # Get list of all available voice files
    available_voices = os.listdir(args.voices)
    if not available_voices:
        print(f"No voice samples found in {args.voices}")
        sys.exit(1)
    
    print(f"Found {len(available_voices)} voice samples")

    # Select assistant voice
    if args.assistant_voice:
        if args.assistant_voice not in available_voices:
            print(f"Specified assistant voice '{args.assistant_voice}' not found in {args.voices}")
            sys.exit(1)
        assistant_voice = args.assistant_voice
        available_user_voices = [v for v in available_voices if v != assistant_voice]
    else:
        # If no assistant voice specified, pick one randomly and use it consistently
        assistant_voice = random.choice(available_voices)
        available_user_voices = [v for v in available_voices if v != assistant_voice]
        print(f"Selected '{assistant_voice}' as the consistent assistant voice")

    if not available_user_voices:
        print("No voices available for user roles after selecting assistant voice")
        sys.exit(1)

    # Process the specified slice of conversations
    slice_conversations = conversations[args.start:args.end]
    for conv_idx, conversation in enumerate(tqdm(slice_conversations, desc="Processing conversations")):
        actual_idx = args.start + conv_idx
        try:
            # Create a directory for this conversation
            conversation_dir = os.path.join(args.output, f"conversation_{actual_idx}")
            os.makedirs(conversation_dir, exist_ok=True)
            
            # Select a random user voice for this conversation
            user_voice = random.choice(available_user_voices)
            
            # Map roles to voice files
            role_to_voice = {
                'user': os.path.join(args.voices, user_voice),
                'assistant': os.path.join(args.voices, assistant_voice)
            }
            
            # Process each turn in the conversation
            for turn_idx, message in enumerate(conversation):
                role = message.get("role", "")
                
                # Skip if role is not defined or not in our mapping
                if not role or role not in role_to_voice:
                    continue
                    
                # Get the appropriate voice for this role
                voice_path = role_to_voice[role]
                
                # Define output filename (1-indexed turn number)
                output_file = os.path.join(conversation_dir, f"{turn_idx+1}_{role}.wav")
                
                # Skip if file already exists
                if os.path.exists(output_file):
                    continue
                    
                # Process with language segments if available
                if "language_segments" in message and message["language_segments"]:
                    audio_segments = []
                    sample_rate = None
                    
                    # Create a directory for this turn's segments
                    turn_dir = os.path.join(conversation_dir, f"{turn_idx+1}_{role}")
                    os.makedirs(turn_dir, exist_ok=True)

                    # Initialize metadata for segments
                    segments_metadata = {
                        "sample_rate": 24000,  # TTS output sample rate
                        "segments": []
                    }

                    # Process each language segment
                    for seg_idx, segment in enumerate(message["language_segments"]):
                        segment_text = segment["text"].strip()
                        segment_lang = segment["lang"]
                        
                        # Skip empty segments
                        if not segment_text:
                            continue
                        
                        # Define segment output file
                        segment_file = os.path.join(turn_dir, f"segment_{seg_idx}.wav")
                        
                        # Skip if segment file already exists
                        if os.path.exists(segment_file):
                            continue

                        # Capture and suppress stdout during TTS generation
                        original_stdout = sys.stdout
                        sys.stdout = NullWriter()
                        
                        try:
                            # Generate speech directly as numpy array
                            segment_audio = tts.tts(
                                text=segment_text,
                                speaker_wav=voice_path,
                                language=segment_lang
                            )
                            
                            # Store the sample rate from the first segment
                            if sample_rate is None:
                                sample_rate = tts.synthesizer.output_sample_rate

                            # Save individual segment
                            sf.write(segment_file, segment_audio, sample_rate)

                            # Add segment metadata
                            segments_metadata["segments"].append({
                                "file": f"segment_{seg_idx}.wav",
                                "lang": segment_lang,
                                "text": segment_text,
                                "duration": len(segment_audio) / sample_rate
                            })
                        except Exception as e:
                            print(f"Error processing segment {seg_idx} in conversation {actual_idx}, turn {turn_idx+1}: {str(e)}")
                        finally:
                            # Restore stdout
                            sys.stdout = original_stdout

                    # Save metadata
                    if segments_metadata["segments"]:
                        metadata_file = os.path.join(turn_dir, "metadata.json")
                        with open(metadata_file, 'w', encoding='utf-8') as f:
                            json.dump(segments_metadata, f, ensure_ascii=False, indent=2)
                
                else:
                    # Create a directory for this turn's segments
                    turn_dir = os.path.join(conversation_dir, f"{turn_idx+1}_{role}")
                    os.makedirs(turn_dir, exist_ok=True)

                    # Process the whole message as a single language (default to Russian)
                    text = message.get("content", "").strip()
                    
                    if not text:
                        continue

                    # Define segment output file
                    segment_file = os.path.join(turn_dir, "segment_0.wav")
                    
                    # Skip if segment file already exists
                    if os.path.exists(segment_file):
                        continue

                    # Capture and suppress stdout during TTS generation
                    original_stdout = sys.stdout
                    sys.stdout = NullWriter()
                    
                    try:
                        # Generate the audio directly
                        audio = tts.tts(
                            text=text,
                            speaker_wav=voice_path,
                            language="ru"  # Default to Russian
                        )
                        
                        # Save individual segment
                        sf.write(segment_file, audio, tts.synthesizer.output_sample_rate)

                        # Save metadata
                        segments_metadata = {
                            "segments": [{
                                "file": "segment_0.wav",
                                "lang": "ru",
                                "text": text,
                                "duration": len(audio) / tts.synthesizer.output_sample_rate
                            }]
                        }
                        metadata_file = os.path.join(turn_dir, "metadata.json")
                        with open(metadata_file, 'w', encoding='utf-8') as f:
                            json.dump(segments_metadata, f, ensure_ascii=False, indent=2)
                    except Exception:
                        # Silent exception handling
                        pass
                    finally:
                        # Restore stdout
                        sys.stdout = original_stdout
                        
        except Exception as e:
            # Print exception but continue to next conversation
            print(f"Error processing conversation {actual_idx}: {str(e)}")
            continue

    print(f"Audio generation complete for conversations {args.start} to {args.end-1}")

if __name__ == "__main__":
    main()
