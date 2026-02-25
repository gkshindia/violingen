import argparse
import multiprocessing
import pathlib

from violingen import Orchestrator

SOURCE_DIR = pathlib.Path("datasets")
OUTPUT_DIR = pathlib.Path("output/separated")
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}


def find_audio_files():
    if not SOURCE_DIR.exists():
        return []
    return sorted(p for p in SOURCE_DIR.iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS)


def pending_splits(files):
    return [p for p in files if not (OUTPUT_DIR / f"{p.stem}.wav").exists()]


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run violin stem splitting.")
    args = parser.parse_args()

    multiprocessing.set_start_method("spawn")

    run_split()
