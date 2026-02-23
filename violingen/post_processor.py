import csv
import multiprocessing
import pathlib
import traceback

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import ruptures
import soundfile as sf
import torch

from violingen.logging import get_logger

N_FFT, HOP, PELT_PEN = 2048, 512, 3.0
FREQ_LO, FREQ_HI     = 200, 4000
HER_THRESH           = 0.4
HPSS_MARGIN          = 2


def _trim_pelt(y, sr):
    rms  = librosa.feature.rms(y=y, hop_length=HOP)[0]
    bkps = ruptures.Pelt(model="rbf").fit(rms.reshape(-1, 1).astype(np.float64)).predict(pen=PELT_PEN)

    onset_frame  = bkps[0]  if len(bkps) >= 2 else 0
    offset_frame = bkps[-2] if len(bkps) >= 3 else len(rms)

    onset_samp  = int(np.clip(librosa.frames_to_samples(onset_frame,  hop_length=HOP), 0, len(y)))
    offset_samp = int(np.clip(librosa.frames_to_samples(offset_frame, hop_length=HOP), 0, len(y)))

    if onset_samp >= offset_samp:
        onset_samp  = 0
        offset_samp = len(y)

    return y[onset_samp:offset_samp], onset_samp, offset_samp, rms, bkps


def _save_rms_plot(rms, bkps, sr, stem_name, plots_dir):
    pathlib.Path(plots_dir).mkdir(parents=True, exist_ok=True)
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=HOP)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(times, rms, linewidth=0.8)
    for bp in bkps[:-1]:
        ax.axvline(librosa.frames_to_time(bp, sr=sr, hop_length=HOP), color="r", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RMS")
    ax.set_title(stem_name)
    fig.tight_layout()
    fig.savefig(pathlib.Path(plots_dir) / f"{stem_name}_rms.png", dpi=100)
    plt.close(fig)


def _remove_bleed(y, sr):
    freqs     = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    freq_mask = ((freqs >= FREQ_LO) & (freqs <= FREQ_HI)).reshape(-1, 1)

    try:
        device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        y_tensor = torch.tensor(y, dtype=torch.float32, device=device)
        window   = torch.hann_window(N_FFT, device=device)
        S_gpu    = torch.stft(y_tensor, N_FFT, HOP, window=window, return_complex=True)
        S_mag    = S_gpu.abs().cpu().numpy()

        H_mag, _ = librosa.decompose.hpss(S_mag, margin=HPSS_MARGIN)
        mask_np  = (H_mag / (S_mag + 1e-8)) * freq_mask
        mask_gpu = torch.tensor(mask_np, dtype=torch.float32, device=device)

        S_clean = S_gpu * mask_gpu
        y_clean = torch.istft(S_clean, N_FFT, HOP, window=window, length=len(y))
        return y_clean.cpu().numpy().astype(np.float32)

    except Exception:
        S        = librosa.stft(y, n_fft=N_FFT, hop_length=HOP)
        S_mag    = np.abs(S)
        H_mag, _ = librosa.decompose.hpss(S_mag, margin=HPSS_MARGIN)
        mask     = (H_mag / (S_mag + 1e-8)) * freq_mask
        S_clean  = S * mask
        return librosa.istft(S_clean, hop_length=HOP, length=len(y)).astype(np.float32)


def _score(y_orig, y_trimmed, onset_samp, offset_samp, sr):
    S        = librosa.stft(y_trimmed, n_fft=N_FFT, hop_length=HOP)
    H, _     = librosa.decompose.hpss(np.abs(S), margin=HPSS_MARGIN)
    harmonic_ratio = float(np.mean(H ** 2) / (np.mean(np.abs(S) ** 2) + 1e-8))

    rms_active    = float(np.sqrt(np.mean(y_trimmed ** 2)))
    silence_parts = np.concatenate([y_orig[:onset_samp], y_orig[offset_samp:]])
    if len(silence_parts) > 0:
        rms_silence    = float(np.sqrt(np.mean(silence_parts ** 2)))
        contrast_ratio = rms_active / (rms_silence + 1e-8)
    else:
        contrast_ratio = float("inf")

    duration    = len(y_trimmed) / sr
    low_quality = harmonic_ratio < HER_THRESH

    return harmonic_ratio, contrast_ratio, duration, low_quality


# Module-level worker — must be defined here (not inside a class) to be picklable.
def _post_process_worker(args):
    torch.set_num_threads(1)
    in_path   = pathlib.Path(args["in_path"])
    out_dir   = pathlib.Path(args["out_dir"])
    plots_dir = pathlib.Path(args["plots_dir"])
    error_log = pathlib.Path(args["error_log"])
    stem_name = in_path.stem
    out_path  = out_dir / (stem_name + "_processed.wav")

    try:
        out_dir.mkdir(parents=True, exist_ok=True)

        y, sr = librosa.load(in_path, sr=None, mono=True)
        y_trimmed, onset, offset, rms, bkps = _trim_pelt(y, sr)
        _save_rms_plot(rms, bkps, sr, stem_name, plots_dir)
        y_clean = _remove_bleed(y_trimmed, sr)
        harmonic_ratio, contrast_ratio, duration, low_quality = _score(y, y_trimmed, onset, offset, sr)
        sf.write(out_path, y_clean, sr)

        return {
            "filename":       str(in_path),
            "harmonic_ratio": harmonic_ratio,
            "contrast_ratio": contrast_ratio,
            "duration":       duration,
            "low_quality":    low_quality,
        }

    except Exception:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(error_log, "a") as fh:
            fh.write(f"--- {in_path} ---\n{traceback.format_exc()}\n")
        return None


class PostProcessor:
    """
    Post-process a batch of audio stems: trim silence, remove harmonic bleed,
    score quality, and write a CSV report.

    Parameters
    ----------
    out_dir : str
        Root directory for processed WAVs, RMS plots, CSV report, and error log.
    max_workers : int or None
        Worker processes for the multiprocessing pool.
        None defaults to ``os.cpu_count()``.
    """

    def __init__(self, out_dir="output/processed", max_workers=None):
        self.out_dir     = pathlib.Path(out_dir)
        self.plots_dir   = self.out_dir / "plots"
        self.report_csv  = self.out_dir / "quality_report.csv"
        self.error_log   = self.out_dir / "errors.txt"
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self._logger     = get_logger("violingen.post_processor")

    def process(self, file_paths):
        """
        Trim, clean, and score each file in *file_paths*.

        Parameters
        ----------
        file_paths : list[str]
            Paths to audio files (WAV, MP3, FLAC, OGG, M4A).

        Returns
        -------
        list[dict]
            One result dict per successfully processed file with keys:
            ``filename``, ``harmonic_ratio``, ``contrast_ratio``,
            ``duration``, ``low_quality``.
        """
        if not file_paths:
            self._logger.warning("PostProcessor.process() called with empty file list.")
            return []

        self.out_dir.mkdir(parents=True, exist_ok=True)

        worker_args = [
            {
                "in_path":   str(p),
                "out_dir":   str(self.out_dir),
                "plots_dir": str(self.plots_dir),
                "error_log": str(self.error_log),
            }
            for p in file_paths
        ]

        self._logger.info(
            f"PostProcessor starting: {len(file_paths)} file(s), {self.max_workers} worker(s)"
        )

        with multiprocessing.Pool(self.max_workers) as pool:
            results = pool.map(_post_process_worker, worker_args)

        rows = [r for r in results if r is not None]
        self._write_report(rows)

        n_err = len(results) - len(rows)
        n_lq  = sum(1 for r in rows if r["low_quality"])
        self._logger.info(
            f"PostProcessor done: {len(rows)} ok, {n_err} failed, {n_lq} low-quality  "
            f"report={self.report_csv}"
        )
        return rows

    def _write_report(self, rows):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with open(self.report_csv, "w", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["filename", "harmonic_ratio", "contrast_ratio", "duration", "low_quality"],
            )
            writer.writeheader()
            writer.writerows(rows)
