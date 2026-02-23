import multiprocessing
import pathlib

from violingen import Orchestrator
from violingen.post_processor import PostProcessor

SOURCE_DIR = "datasets"
OUTPUT_DIR = "output/separated"

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    out_dir = pathlib.Path(OUTPUT_DIR)

    all_files = sorted(
        p for p in pathlib.Path(SOURCE_DIR).iterdir()
        if p.suffix.lower() in AUDIO_EXTENSIONS
    )

    if not all_files:
        print(f"No audio files found in {SOURCE_DIR!r}")
        raise SystemExit(1)

    # Stage 1 — stem splitting: skip files whose separated WAV already exists
    pending_split = [p for p in all_files if not (out_dir / f"{p.stem}.wav").exists()]
    n_skip_split = len(all_files) - len(pending_split)
    if n_skip_split:
        print(f"Stage 1: skipping {n_skip_split} already-separated file(s), {len(pending_split)} remaining.")

    if pending_split:
        orch = Orchestrator(out_dir=OUTPUT_DIR)
        orch.process([str(p) for p in pending_split])

    # Stage 2 — post-processing: collect ALL separated WAVs so any that were
    # split in a previous run but not yet post-processed are also picked up.
    # PostProcessor skips files whose _processed.wav already exists internally.
    all_separated = sorted(str(p) for p in out_dir.glob("*.wav")) if out_dir.exists() else []
    if not all_separated:
        print("No separated files found for post-processing.")
        raise SystemExit(0)

    pp = PostProcessor(out_dir=str(out_dir / "processed"))
    pp.process(all_separated)
