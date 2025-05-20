import os
import json
import glob
import random
import re
from tqdm import tqdm
from collections import Counter


# ============ CONFIGURATION = ============


INPUT_DIR = "dataset/new/text_generation_results"  # Relative to project root
OUTPUT_FILE = "dataset/new/converted_generated_data.jsonl" # Relative to project root
ROLE_MAPPING = {
    "student": "user",
    "teacher": "assistant"
}
# --- ---


# ============ LANGUAGE DETECTION ============


# Counter for Chinese fragments
chinese_fragment_counter = Counter()


def build_chinese_detector():
    """Build regex for Chinese character detection"""

    # Chinese character ranges
    LHan = [[0x2E80, 0x2E99],    # Han # So  [26] CJK RADICAL REPEAT, CJK RADICAL RAP
            [0x2E9B, 0x2EF3],    # Han # So  [89] CJK RADICAL CHOKE, CJK RADICAL C-SIMPLIFIED TURTLE
            [0x2F00, 0x2FD5],    # Han # So [214] KANGXI RADICAL ONE, KANGXI RADICAL FLUTE
            0x3005,              # Han # Lm       IDEOGRAPHIC ITERATION MARK
            0x3007,              # Han # Nl       IDEOGRAPHIC NUMBER ZERO
            [0x3021, 0x3029],    # Han # Nl   [9] HANGZHOU NUMERAL ONE, HANGZHOU NUMERAL NINE
            [0x3038, 0x303A],    # Han # Nl   [3] HANGZHOU NUMERAL TEN, HANGZHOU NUMERAL THIRTY
            0x303B,              # Han # Lm       VERTICAL IDEOGRAPHIC ITERATION MARK
            [0x3400, 0x4DB5],    # Han # Lo [6582] CJK UNIFIED IDEOGRAPH-3400, CJK UNIFIED IDEOGRAPH-4DB5
            [0x4E00, 0x9FC3],    # Han # Lo [20932] CJK UNIFIED IDEOGRAPH-4E00, CJK UNIFIED IDEOGRAPH-9FC3
            [0xF900, 0xFA2D],    # Han # Lo [302] CJK COMPATIBILITY IDEOGRAPH-F900, CJK COMPATIBILITY IDEOGRAPH-FA2D
            [0xFA30, 0xFA6A],    # Han # Lo  [59] CJK COMPATIBILITY IDEOGRAPH-FA30, CJK COMPATIBILITY IDEOGRAPH-FA6A
            [0xFA70, 0xFAD9],    # Han # Lo [106] CJK COMPATIBILITY IDEOGRAPH-FA70, CJK COMPATIBILITY IDEOGRAPH-FAD9
            [0x20000, 0x2A6D6],  # Han # Lo [42711] CJK UNIFIED IDEOGRAPH-20000, CJK UNIFIED IDEOGRAPH-2A6D6
            [0x2F800, 0x2FA1D]]  # Han # Lo [542] CJK COMPATIBILITY IDEOGRAPH-2F800, CJK COMPATIBILITY IDEOGRAPH-2FA1D

    # Chinese punctuation characters
    chinese_punctuation = "，。！？；：""''「」【】《》〈〉（）［］｛｝…～"
    
    # Create regex pattern for Chinese characters
    L = []
    for i in LHan:
        if isinstance(i, list):
            f, t = i
            f = chr(f)
            t = chr(t)
            L.append(f'{f}-{t}')
        else:
            L.append(chr(i))
    
    # Add Chinese punctuation
    for char in chinese_punctuation:
        L.append(re.escape(char))  # Properly escape each character
    
    # Combine all ranges and characters into a regex pattern
    RE = '[%s]' % ''.join(L)

    # Compile the regex pattern
    return re.compile(RE, re.UNICODE)


# Initialize Chinese character detector
chinese_detector = build_chinese_detector()


def contains_chinese(text):
    """Check if text contains any Chinese characters"""

    # Check if text is empty or None
    if not text:
        raise ValueError("Input text is empty or None")
    
    # For each character in the text, check if it matches the Chinese character regex
    for char in text:
        if chinese_detector.match(char):
            return True
    
    # If no Chinese characters are found, return False
    return False


def has_chinese_in_russian_segments(transformed_messages):
    """
    Check if any Russian segments contain Chinese characters.
    """
    # If no messages are provided, raise an error
    if not transformed_messages:
        raise ValueError("No transformed messages provided")

    # Iterate through each message in the transformed messages
    for message in transformed_messages:

        # Get the language segments of the message
        language_segments = message.get("language_segments")

        # If no language segments are found, raise an error
        if not language_segments:
            raise ValueError("No language segments found in the message")
        
        # Iterate through each language segment
        for segment in language_segments:
            # Check if the segment is in Russian and contains Chinese characters
            if segment.get("lang") == "ru" and contains_chinese(segment.get("text", "")):
                return True
            
    # If no Russian segments contain Chinese characters, return False
    return False


def contains_russian(text):
    """Check if text contains any Russian (Cyrillic) characters."""
    
    # Check if text is empty or None
    if not text:
        raise ValueError("Input text is empty or None")
    
    # Compile regex pattern for Cyrillic characters 
    # and search for any match in the text
    cyrillic_match = re.search(r'[\u0400-\u04FF]', text)

    # If a match is found, return True
    return bool(cyrillic_match)


def has_russian_in_chinese_segments(transformed_messages):
    """
    Check if any Chinese segments contain Russian characters.
    """
    # If no messages are provided, raise an error
    if not transformed_messages:
        raise ValueError("No transformed messages provided")
    
    # Iterate through each message in the transformed messages
    for message in transformed_messages:

        # Get the language segments of the message
        language_segments = message.get("language_segments")

        # If no language segments are found, raise an error
        if not language_segments:
            raise ValueError("No language segments found in the message")
        
        # Iterate through each language segment
        for segment in language_segments:
            # Check if the segment is in Chinese and contains Russian characters
            if segment.get("lang") == "zh" and contains_russian(segment.get("text", "")):
                return True
    
    # If no Chinese segments contain Russian characters, return False
    return False


# ============ PROCESSING FUNCTIONS ============


def extract_chinese_fragments(transformed_messages):
    """
    Extract all Chinese fragments from a conversation.
    """
    # If no messages are provided, raise an error
    if not transformed_messages:
        raise ValueError("No transformed messages provided")
    
    # Initialize a list to store Chinese fragments
    chinese_fragments = []

    # Iterate through each message in the transformed messages  
    for message in transformed_messages:

        # Get the language segments of the message
        language_segments = message.get("language_segments")

        # If no language segments are found, raise an error
        if not language_segments:
            raise ValueError("No language segments found in the message")
        
        # Iterate through each language segment
        for segment in language_segments:
            if segment.get("lang") == "zh":
                chinese_fragment = segment.get("text")
                
                # Check if fragment is not empty
                if chinese_fragment:
                    # Append the Chinese fragment to the list
                    chinese_fragments.append(chinese_fragment)
                else:
                    # Raise an error if an empty Chinese fragment is found
                    raise ValueError("Empty Chinese fragment found")
                
    # Return the list of Chinese fragments
    return chinese_fragments


def should_accept_conversation(chinese_fragments):
    """
    Decide whether to accept the conversation based on the frequency of all Chinese fragments.
    Uses a combined probability approach with geometric mean.
    """
    if not chinese_fragments:
        return True  # Always accept dialogues without Chinese content
    
    # Calculate acceptance probability based on all fragments using geometric mean
    unique_fragments = set(fragment for fragment in chinese_fragments if fragment.strip())
    
    if not unique_fragments:
        return True
    
    # Calculate product of individual probabilities
    total_probability = 1.0
    for fragment in unique_fragments:
        count = chinese_fragment_counter.get(fragment, 0)
        # Individual fragment probability: 1.5^(-count)
        fragment_probability = 1.5 ** (-count)
        total_probability *= fragment_probability
    
    # Calculate geometric mean
    total_probability = total_probability ** (1 / len(unique_fragments))
    
    # Random decision based on probability
    return random.random() < total_probability


def update_counter(chinese_fragments):
    """
    Update the counter with the Chinese fragments from the accepted conversation.
    """
    for fragment in chinese_fragments:
        chinese_fragment_counter[fragment] += 1


def is_punctuation_only(text):
    """Check if text contains only punctuation."""
    if not text:
        return False
    # Check if text consists only of punctuation marks
    return all(not char.isalnum() and not char.isspace() for char in text)


def starts_with_punctuation(text):
    """Check if text starts with punctuation."""
    if not text:
        return False
    return not text[0].isalnum() and not text[0].isspace()


def escape_punctuation(text):
    """
    Экранирует все знаки пунктуации, кроме кавычек «», 
    и заменяет «» на обычные ".
    """
    # Сначала заменяем ёлочки на обычные кавычки
    text = text.replace('«', '"').replace('»', '"')
    
    # Теперь экранируем все остальные знаки пунктуации
    result = ""
    for char in text:
        if is_punctuation_only(char) and char not in '"':
            result += '\\' + char
        else:
            result += char
    return result


def transform_conversation(nested_conversation_data):
    """
    Преобразует «сырую» структуру разговора к формату
        [{ "role": "...", "language_segments": [...] }, … ]

    Порядок обработки **обновлён**:

        1.  Считываем исходные phrase_fragments  ➜  raw_fragments
        2.  Переносим ведущую пунктуацию каждого сегмента к предыдущему
            (если сегмент опустел — удаляем).
        3.  Экранируем пунктуацию (`escape_punctuation`).
        4.  Объединяем соседние фрагменты одного языка, добавляя пробел,
            если новый фрагмент не начинается с пунктуации.
    """
    transformed_messages = []

    # --- валидация входных данных ---------------------------------
    if not (isinstance(nested_conversation_data, dict) and "conversation" in nested_conversation_data):
        print("Warning: Invalid nested conversation data format. Skipping.")
        return None

    conversation_list = nested_conversation_data["conversation"]
    if not isinstance(conversation_list, list):
        print("Warning: 'conversation' key does not contain a list. Skipping.")
        return None

    # --- обработка реплик -----------------------------------------
    for message in conversation_list:
        if not (
            isinstance(message, dict)
            and isinstance(message.get("phrase_fragments"), list)
            and "role" in message
        ):
            print(f"Warning: Skipping invalid message format: {message}")
            continue

        original_role = message["role"]
        phrase_fragments = message["phrase_fragments"]

        # 1. читаем «сырые» фрагменты --------------------------------
        raw_fragments = [
            {"text": frag.get("text", "").strip(), "lang": frag.get("lang", "")}
            for frag in phrase_fragments
            if isinstance(frag, dict) and frag.get("text", "").strip()
        ]

        # 2. перенос ведущей пунктуации ------------------------------
        fixed_fragments = []
        for frag in raw_fragments:
            txt = frag["text"]
            if fixed_fragments and starts_with_punctuation(txt):
                # непрерывный префикс пунктуации
                m = re.match(r'^[^\w\s]+', txt)
                leading = m.group(0) if m else ''
                if leading:
                    prev = fixed_fragments[-1]
                    prev["text"] = prev["text"].rstrip() + leading
                    txt = txt[len(leading):].lstrip()
            if txt:                                  # сегмент не пустой?
                fixed_fragments.append({"text": txt, "lang": frag["lang"]})

        # если после переноса не осталось сегментов — пропускаем реплику
        if not fixed_fragments:
            continue

        # 3. экранирование пунктуации --------------------------------
        escaped_fragments = [
            {"text": escape_punctuation(seg["text"]), "lang": seg["lang"]}
            for seg in fixed_fragments
        ]

        # 4. merge соседних фрагментов одного языка -----------------
        merged_fragments = []
        current_lang, current_text = None, ""

        for frag in escaped_fragments:
            txt = frag["text"]
            if current_lang is None:
                current_lang, current_text = frag["lang"], txt
            elif frag["lang"] == current_lang:
                current_text += txt if starts_with_punctuation(txt) else " " + txt
            else:
                merged_fragments.append({"text": current_text, "lang": current_lang})
                current_lang, current_text = frag["lang"], txt

        merged_fragments.append({"text": current_text, "lang": current_lang})

        # --- запись результата -------------------------------------
        transformed_messages.append(
            {
                "role": ROLE_MAPPING.get(original_role, original_role),
                "language_segments": merged_fragments,
            }
        )

    return transformed_messages


def main():
    # Set random seed for reproducibility
    random.seed(0)
    
    # Ensure output directory exists
    abs_output_file = os.path.abspath(OUTPUT_FILE)
    os.makedirs(os.path.dirname(abs_output_file), exist_ok=True)

    search_pattern = os.path.join(INPUT_DIR, "*.jsonl")
    input_files = glob.glob(search_pattern)
    print(f"Found {len(input_files)} files to process in {INPUT_DIR}")

    total_lines_processed = 0
    total_conversations = 0
    total_chinese_in_russian = 0
    total_russian_in_chinese = 0
    
    # First, collect all conversations
    all_conversations = []
    print("Loading and transforming all conversations...")
    for filepath in tqdm(input_files, desc="Loading files"):
        try:
            with open(filepath, "r", encoding="utf-8") as infile:
                for line_num, line in enumerate(infile, 1):
                    try:
                        main_data = json.loads(line.strip())
                        
                        # Navigate to the nested JSON string
                        nested_json_str = main_data.get("response", {}).get("body", {}).get("output", [{}])[0].get("content", [{}])[0].get("text")

                        if not nested_json_str or not isinstance(nested_json_str, str):
                            continue

                        # Parse the nested JSON string
                        nested_data = json.loads(nested_json_str)
                        
                        # Transform the conversation
                        transformed_conv = transform_conversation(nested_data)
                        if transformed_conv:
                            # Filter out dialogues where Russian segments contain Chinese characters
                            if has_chinese_in_russian_segments(transformed_conv):
                                total_chinese_in_russian += 1
                                continue

                            if has_russian_in_chinese_segments(transformed_conv):
                                total_russian_in_chinese += 1
                                continue
                                
                            all_conversations.append(transformed_conv)
                            total_conversations += 1
                            
                    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
                        pass
                    finally:
                        total_lines_processed += 1
        except Exception as e:
            print(f"Error processing file {filepath}: {e}")
    
    # Shuffle conversations
    print(f"Shuffling {len(all_conversations)} conversations...")
    random.shuffle(all_conversations)
    
    # Now apply filtering and write to output
    total_conversations_written = 0
    total_conversations_filtered = 0
    
    print("Applying filtering and writing output...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        for conversation in tqdm(all_conversations, desc="Processing conversations"):
            chinese_fragments = extract_chinese_fragments(conversation)
            
            if should_accept_conversation(chinese_fragments):
                # Write the conversation to the output file
                outfile.write(json.dumps(conversation, ensure_ascii=False) + "\n")
                total_conversations_written += 1
                
                # Update the counter for accepted conversations
                update_counter(chinese_fragments)
            else:
                total_conversations_filtered += 1

    print(f"\nProcessing complete.")
    print(f"Total lines processed: {total_lines_processed}")
    print(f"Total conversations found: {total_conversations}")
    print(f"Total conversations with Chinese in Russian segments (rejected): {total_chinese_in_russian}")
    print(f"Total conversations with Russian in Chinese segments (rejected): {total_russian_in_chinese}")
    print(f"Total conversations filtered out due to frequency: {total_conversations_filtered}")
    print(f"Total conversations written to {OUTPUT_FILE}: {total_conversations_written}")
    print(f"Unique Chinese fragments tracked: {len(chinese_fragment_counter)}")
    
    # Print top 10 most frequent Chinese fragments
    if chinese_fragment_counter:
        print("\nTop 10 most frequent Chinese fragments:")
        for fragment, count in chinese_fragment_counter.most_common(10):
            print(f"- '{fragment}': {count} occurrences")

if __name__ == "__main__":
    main()