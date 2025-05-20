import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
from scipy import stats
from tqdm import tqdm

# ---------- параметры -------------------------------------------------------
RANDOM_PADDING_FLOOR = 0.5
RANDOM_PADDING_CEIL  = 1.5
GAIN_AUGMENT_PROB  = 0.5
NOISE_AUGMENT_PROB = 0.3
MIN_GAIN_DB,  MAX_GAIN_DB  = -5, 5
MIN_NOISE_DB, MAX_NOISE_DB = -40, -20
# ---------------------------------------------------------------------------

log = logging.getLogger("merge_conv")

# -------------------- вспомогательные функции ------------------------------
def db2lin(db: float) -> float:
    return 10 ** (db / 20)


def gen_noise(n: int) -> np.ndarray:
    return np.random.randn(n).astype(np.float32) / 32768.0


def load_wav(path: Path, target_sr: int) -> np.ndarray:
    wav, sr = sf.read(path)
    if sr != target_sr:
        raise ValueError(f"{path}: sample‑rate {sr} ≠ {target_sr}")
    if wav.ndim > 1:
        wav = wav[:, 0]
    return wav.astype(np.float32)


# -------------------- обработка одной реплики ------------------------------
def process_turn(turn_dir: Path, sr: int) -> Tuple[np.ndarray, List[List]]:
    """Возвращает (audio, alignments_for_assistant_only)."""
    meta = json.loads((turn_dir / "metadata.json").read_text(encoding="utf-8"))
    is_assistant = turn_dir.name.endswith("_assistant")

    chunks: List[np.ndarray] = []
    aligns: List[List] = []
    t = 0.0

    for seg in meta["segments"]:
        wav = load_wav(turn_dir / seg["file"], sr)
        chunks.append(wav)

        if is_assistant and "alignments" in seg:
            for token, ts, _ in seg["alignments"]:
                aligns.append([token, [ts[0] + t, ts[1] + t], "SPEAKER_MAIN"])
        t += len(wav) / sr

    audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    return audio, aligns


# -------------------- склейка всего разговора ------------------------------
def merge_conversation(conv_dir: Path, out_dir: Path, sr: int) -> None:
    conv_name = conv_dir.name
    wav_out = out_dir / f"{conv_name}.wav"
    json_out = out_dir / f"{conv_name}.json"

    if wav_out.exists() and json_out.exists():
        log.info(f"{conv_name} — уже создано, пропускаю")
        return

    # Все реплики в естественном порядке
    turns = sorted(
        [d for d in conv_dir.iterdir() if d.is_dir()],
        key=lambda p: int(p.name.split("_")[0])
    )

    left_parts, right_parts = [], []
    alignments: List[List] = []
    cur_time = 0.0

    for turn in turns:
        audio, aligns = process_turn(turn, sr)

        if turn.name.endswith("_user"):
            # → R‑канал, с аугментациями
            if random.random() < NOISE_AUGMENT_PROB:
                noise_db = random.uniform(MIN_NOISE_DB, MAX_NOISE_DB)
                noise_gain = db2lin(noise_db)

                # Нормализуем шум по среднеквадратичному значению
                noise = gen_noise(len(audio))
                noise_rms = np.sqrt(np.mean(noise**2))
                target_rms = np.sqrt(np.mean(audio**2)) * noise_gain

                if noise_rms > 1e-8:  # защита от деления на ноль
                    noise *= (target_rms / noise_rms)

                audio = np.clip(audio + noise, -1, 1)

            if random.random() < GAIN_AUGMENT_PROB:
                audio = np.clip(audio*db2lin(
                    random.uniform(MIN_GAIN_DB, MAX_GAIN_DB)), -1, 1)

            right_parts.append(audio)
            left_parts.append(np.zeros_like(audio))

            cur_time += len(audio) / sr

            # пауза после пользователя
            silence_len = int(random.uniform(RANDOM_PADDING_FLOOR,
                                             RANDOM_PADDING_CEIL) * sr)
            if silence_len:
                silence = np.zeros(silence_len, dtype=np.float32)
                left_parts.append(silence)
                right_parts.append(silence)
                cur_time += silence_len / sr

        else:  # assistant
            left_parts.append(audio)
            right_parts.append(np.zeros_like(audio))

            # сдвигаем выравнивания
            for tok, ts, role in aligns:
                alignments.append([tok, [ts[0] + cur_time, ts[1] + cur_time], role])

            cur_time += len(audio) / sr

    # ---------- финальное склеивание ---------------------------------------
    if not left_parts and not right_parts:
        log.warning(f"{conv_name}: empty, пропуск")
        return

    left = np.concatenate(left_parts)
    right = np.concatenate(right_parts)
    stereo = np.stack([left, right], axis=1)

    out_dir.mkdir(parents=True, exist_ok=True)
    sf.write(wav_out, stereo, sr)
    json_out.write_text(json.dumps({"alignments": alignments},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"{conv_name}: готово  —  {len(stereo)/sr:.1f}s")


# -------------------- CLI ---------------------------------------------------
def main():
    ap = argparse.ArgumentParser("Merge whole conversation into one stereo file")
    ap.add_argument("audio_root", type=Path, help="dataset/new/audio")
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int)
    ap.add_argument("--sample-rate", type=int, default=24000)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    stereo_root = args.output_dir / "data_stereo"
    stereo_root.mkdir(parents=True, exist_ok=True)

    # диапазон разговоров
    if args.end is None:
        args.end = 1 + max(int(d.name.split("_")[1])
                           for d in args.audio_root.iterdir()
                           if d.is_dir() and d.name.startswith("conversation_"))

    for idx in tqdm(range(args.start, args.end), desc="conversations"):
        cdir = args.audio_root / f"conversation_{idx}"
        if not cdir.exists():
            log.warning(f"conversation_{idx} отсутствует")
            continue
        try:
            merge_conversation(cdir, stereo_root, args.sample_rate)
        except Exception as e:
            log.error(f"{cdir}: {e}")

if __name__ == "__main__":
    main()
