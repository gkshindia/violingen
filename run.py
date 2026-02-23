import pathlib

from violingen import Orchestrator

SOURCE_DIR = "datasets"
OUTPUT_DIR = "output/separated"

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}

if __name__ == "__main__":
    file_paths = sorted(
        str(p) for p in pathlib.Path(SOURCE_DIR).iterdir()
        if p.suffix.lower() in AUDIO_EXTENSIONS
    )

    if not file_paths:
        print(f"No audio files found in {SOURCE_DIR!r}")
        raise SystemExit(1)

    orch = Orchestrator(out_dir=OUTPUT_DIR)
    results = orch.process(file_paths)
    orch.post_process(results)
