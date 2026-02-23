import pathlib
import tempfile

import numpy as np


class StemSplitter:
    def __init__(
        self,
        model="htdemucs_6s",
        device="cpu",
        shifts=0,
        overlap=0.25,
        clip_mode="rescale",
        stem="other",
        jobs=1,
    ):
        self.model = model
        self.device = device
        self.shifts = shifts
        self.overlap = overlap
        self.clip_mode = clip_mode
        self.stem = stem
        self.jobs = jobs

    def split(self, in_file_path, out_file_path):
        import soundfile as sf

        in_path  = pathlib.Path(in_file_path).expanduser().resolve()
        out_path = pathlib.Path(out_file_path).expanduser().resolve()

        if not in_path.exists():
            raise FileNotFoundError(f"Input file not found: {in_path}")

        device = self._detect_device()
        _models = ("htdemucs", "htdemucs_ft")

        with tempfile.TemporaryDirectory(prefix="demucs_") as tmp_dir:
            for m in _models:
                self._run_demucs(str(in_path), tmp_dir, model=m, device=device)

            path_a = self._locate_stem(tmp_dir, in_path.stem, model=_models[0])
            path_b = self._locate_stem(tmp_dir, in_path.stem, model=_models[1])
            y_combined, sr = self._ensemble_max(path_a, path_b)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_path, np.clip(y_combined, -1, 1).T, sr, subtype="PCM_16")
        print(f"ensemble(htdemucs, htdemucs_ft) saved → {out_path}")
        return str(out_path)

    def _run_demucs(self, in_file, out_dir, model=None, device=None):
        import subprocess
        import sys
        # Run demucs in a subprocess so that CUDA-cleanup segfaults during
        # teardown don't kill the main process.  The stems are written to
        # out_dir before any crash occurs; _locate_stem() validates them.
        subprocess.run(
            [
                sys.executable, "-m", "demucs",
                "-n", model or self.model,
                "-d", device or self.device,
                "--shifts", str(self.shifts),
                "--overlap", str(self.overlap),
                "--clip-mode", self.clip_mode,
                "-j", str(self.jobs),
                "--out", out_dir,
                in_file,
            ],
            check=False,
        )

    def _locate_stem(self, out_dir, audio_stem, model=None):
        m = model or self.model
        stem_wav = (
            pathlib.Path(out_dir) / m / audio_stem / f"{self.stem}.wav"
        )
        if not stem_wav.exists():
            raise RuntimeError(
                f"Expected stem file not found at: {stem_wav}\n"
                f"Available files: {list(stem_wav.parent.iterdir()) if stem_wav.parent.exists() else '(dir missing)'}"
            )
        return stem_wav

    def _ensemble_max(self, path_a, path_b):
        import torch
        import torchaudio

        wav_a, sr = torchaudio.load(str(path_a))
        wav_b, _  = torchaudio.load(str(path_b))

        n_fft  = 2048
        hop    = 512
        window = torch.hann_window(n_fft)

        out_channels = []
        for c in range(wav_a.shape[0]):
            S_a = torch.stft(wav_a[c], n_fft, hop, window=window, return_complex=True)
            S_b = torch.stft(wav_b[c], n_fft, hop, window=window, return_complex=True)
            mask = S_a.abs() >= S_b.abs()
            S_max = torch.where(mask, S_a, S_b)
            ch = torch.istft(S_max, n_fft, hop, window=window, length=wav_a.shape[-1])
            out_channels.append(ch)

        return torch.stack(out_channels).numpy(), sr

    @staticmethod
    def _detect_device():
        import torch
        if torch.cuda.is_available():
            return "cuda"
        try:
            if torch.backends.mps.is_available():
                return "mps"
        except AttributeError:
            pass
        return "cpu"
