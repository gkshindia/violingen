import argparse
import multiprocessing
import pathlib

from violingen import Orchestrator
from violingen.stem_cleaner import StemCleaner

SOURCE_DIR = pathlib.Path("datasets")
OUTPUT_DIR = pathlib.Path("output/separated")
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}
PROCESSED_SUFFIX = "_processed.wav"


def find_audio_files():
    if not SOURCE_DIR.exists():
        return []
    return sorted(p for p in SOURCE_DIR.iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS)


def pending_splits(files):
    return [p for p in files if not (OUTPUT_DIR / f"{p.stem}.wav").exists()]


def pending_postprocess():
    if not OUTPUT_DIR.exists():
        return []
    processed_dir = OUTPUT_DIR / "processed"
    return [
        str(p)
        for p in sorted(OUTPUT_DIR.glob("*.wav"))
        if not (processed_dir / f"{p.stem}{PROCESSED_SUFFIX}").exists()
    ]


def run_split():
    audio_files = find_audio_files()
    if not audio_files:
        print(f"No audio files found in {str(SOURCE_DIR)!r}")
        return

    todo = pending_splits(audio_files)
    skipped = len(audio_files) - len(todo)
    if skipped:
        print(f"Stage 1: skipping {skipped} already-separated file(s), {len(todo)} remaining.")

    if todo:
        orch = Orchestrator(out_dir=str(OUTPUT_DIR))
        orch.process([str(p) for p in todo])


def run_postprocess():
    to_process = pending_postprocess()
    if not to_process:
        print("No separated files pending post-processing.")
        return
    pp = StemCleaner(out_dir=str(OUTPUT_DIR / "processed"))
    pp.process(to_process)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run violin stem splitting and post-processing.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--split-only", action="store_true", help="Only run stem splitting.")
    group.add_argument("--postprocess-only", action="store_true", help="Only run post-processing.")
    args = parser.parse_args()

    multiprocessing.set_start_method("spawn")

    if not args.postprocess_only:
        run_split()
