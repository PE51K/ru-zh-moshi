#!/usr/bin/env python3
# align_all_segments.py  –  word‑level FA (ru+zh), skip *_user, punct→prev‑word  ✅ fixed "scores"

import os
import re
import json
import unicodedata
from typing import List, Tuple

import torch
from tqdm import tqdm

# ------------------------- Chinese tokenisation ----------------------------
USE_JIEBA = True
if USE_JIEBA:
    try:
        import jieba           # pip install jieba
    except ImportError:
        USE_JIEBA = False

def tok_zh(text: str) -> List[str]:
    if USE_JIEBA:
        return [t for t in jieba.lcut(text) if t.strip()]
    # char‑level fallback
    return [c for c in text if re.match(r"[\u4e00-\u9fff]", c)]

RUS_RE = re.compile(r"\w+|[^\w\s]", flags=re.U)

def is_punct(tok: str) -> bool:
    return all(unicodedata.category(ch).startswith("P") for ch in tok)

# ------------------------- CTC‑forced‑aligner ------------------------------
from ctc_forced_aligner import (
    load_audio,
    load_alignment_model,
    generate_emissions,
    preprocess_text,
    get_alignments,
    get_spans,
    postprocess_results,
)

ROOT = "dataset/new/audio"
device = "cuda" if torch.cuda.is_available() else "cpu"
model, tokenizer = load_alignment_model(device=device)

# ------------------------- helper: align one segment -----------------------
def align_segment(wav: str, spaced_text: str, lang_code: str,
                  clean_tokens: List[str]) -> List[Tuple[str, List[float]]]:
    wav_t = load_audio(wav, model.dtype, model.device)
    em, stride = generate_emissions(model, wav_t)

    t_star, txt_star = preprocess_text(spaced_text,
                                       language=lang_code,
                                       romanize=True)
    idxs, scores, blank = get_alignments(em, t_star, tokenizer)
    spans = get_spans(t_star, idxs, blank)
    words = postprocess_results(txt_star, spans, stride, scores)  # ✅ scores

    # map back to original clean_tokens
    out = []
    for i, w in enumerate(words):
        if i >= len(clean_tokens):
            break
        out.append((clean_tokens[i],
                    [round(w["start"], 3), round(w["end"], 3)]))
    return out

# ------------------------- collect metadata.json (skip *_user) -------------
meta_paths = [
    os.path.join(root, f)
    for root, _, files in os.walk(ROOT)
    if not os.path.basename(root).endswith("_user")   #  skip user replies
    for f in files if f == "metadata.json"
]

print(f"Processing {len(meta_paths)} metadata files (user folders skipped)")

# ------------------------- main loop ---------------------------------------
for mpath in tqdm(meta_paths, desc="forced‑align"):
    with open(mpath, encoding="utf-8") as f:
        meta = json.load(f)

    changed = False
    for seg in meta.get("segments", []):
        if "alignments" in seg:
            continue

        wav_file = os.path.join(os.path.dirname(mpath), seg["file"])
        if not os.path.exists(wav_file):
            continue

        raw_text = seg["text"].strip()
        if not raw_text:
            continue

        lang = seg.get("lang", "ru").lower()
        lang_code = "rus" if lang.startswith("ru") else "zho" if lang.startswith("zh") else "eng"

        # tokenise full text
        tokens_all = tok_zh(raw_text) if lang_code == "zho" else RUS_RE.findall(raw_text)
        clean_tokens = [t for t in tokens_all if not is_punct(t)]
        if not clean_tokens:
            continue

        spaced = " ".join(clean_tokens)

        try:
            aligned_pairs = align_segment(wav_file, spaced, lang_code, clean_tokens)

            # ---- merge punctuation to previous word ------------------------
            alignments, idx_clean = [], 0
            for tok in tokens_all:
                if is_punct(tok):
                    if alignments:            # attach to previous
                        alignments[-1][0] += tok
                    continue
                # normal word
                if idx_clean >= len(aligned_pairs):
                    break
                word, times = aligned_pairs[idx_clean]
                alignments.append([word, times, "SPEAKER_MAIN"])
                idx_clean += 1

            seg["alignments"] = alignments
            changed = True

        except Exception as e:
            print(f"[!] {wav_file}: {e}")

    if changed:
        with open(mpath, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
