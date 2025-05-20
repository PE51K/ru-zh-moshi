import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import List, Dict

import soundfile as sf
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] — %(message)s",
    datefmt="%m-%d %H:%M:%S",
)
LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------- helpers ---
def wav_duration(wav_path: Path) -> float:
    info = sf.info(str(wav_path))
    return info.frames / info.samplerate


def split_train_val(items: List[Dict], val_ratio: float, seed: int):
    random.seed(seed)
    random.shuffle(items)
    pivot = int(len(items) * val_ratio)
    return items[pivot:], items[:pivot]


def write_jsonl(lines: List[Dict], path: Path):
    with path.open("w", encoding="utf-8") as f:
        for entry in lines:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------ main ---
def main():
    p = argparse.ArgumentParser(
        "Create train/val JSONL from conversation_*.wav produced by merge_conversations_stereo.py"
    )
    p.add_argument(
        "root_out", type=Path,
        help="Каталог, куда merge_conversations_stereo писал data_stereo/"
    )
    p.add_argument(
        "--val-ratio", type=float, default=0.1,
        help="Доля валидации (default 0.1)"
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    if args.verbose:
        LOG.setLevel(logging.DEBUG)

    data_stereo = args.root_out / "data_stereo"
    if not data_stereo.exists():
        LOG.error(f"{data_stereo} not found — запустите merge_conversations_stereo сначала")
        sys.exit(1)

    wav_files = sorted(data_stereo.glob("conversation_*.wav"))
    LOG.info(f"найдено {len(wav_files)} wav‑файлов")

    entries: List[Dict] = []
    for wav in tqdm(wav_files, desc="Collect durations"):
        try:
            dur = wav_duration(wav)
            rel = wav.relative_to(args.root_out)   # путь относительно root_out
            entries.append({"path": str(rel), "duration": dur})
        except Exception as e:
            LOG.warning(f"Пропуск {wav}: {e}")

    if not entries:
        LOG.error("нет валидных файлов — нечего писать")
        sys.exit(1)

    train, val = split_train_val(entries, args.val_ratio, args.seed)

    write_jsonl(train, args.root_out / "train.jsonl")
    write_jsonl(val,   args.root_out / "val.jsonl")

    LOG.info(f"train.jsonl: {len(train)} строк")
    LOG.info(f"val.jsonl  : {len(val)} строк")
    LOG.info("Готово.")

if __name__ == "__main__":
    main()
