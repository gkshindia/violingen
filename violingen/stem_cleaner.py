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
FREQ_LO, FREQ_HI     = 100, 4400
HER_THRESH           = 0.4
HPSS_MARGIN          = 4.0


def _trim_pelt(y, sr):
    rms  = librosa.feature.rms(y=y, hop_length=HOP)[0]
    bkps = ruptures.Pelt(model="l2").fit(rms.reshape(-1, 1).astype(np.float64)).predict(pen=PELT_PEN)

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
    enable_plots = bool(args.get("enable_plots", False))

    try:
        out_dir.mkdir(parents=True, exist_ok=True)

        y, sr = librosa.load(in_path, sr=None, mono=True)
        y_trimmed, onset, offset, rms, bkps = _trim_pelt(y, sr)
        if enable_plots:
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


class StemCleaner:
    """
    Post-process a batch of audio stems: trim silence, remove harmonic bleed,
    score quality, and write a CSV report.

    Parameters
    ----------
    out_dir : str
        Root directory for processed WAVs, optional RMS plots, CSV report, and error log.
    max_workers : int or None
        Worker processes for the multiprocessing pool.
        None defaults to ``min(2, os.cpu_count())``.
    enable_plots : bool
        When True, write RMS plots to ``{out_dir}/plots``.
    """

    def __init__(self, out_dir="output/processed", max_workers=None, enable_plots=False):
        self.out_dir     = pathlib.Path(out_dir)
        self.plots_dir   = self.out_dir / "plots"
        self.report_csv  = self.out_dir / "quality_report.csv"
        self.error_log   = self.out_dir / "errors.txt"
        default_workers = min(2, multiprocessing.cpu_count())
        self.max_workers = max_workers or default_workers
        self.enable_plots = bool(enable_plots)
        self._logger     = get_logger("violingen.stem_cleaner")

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
            self._logger.warning("StemCleaner.process() called with empty file list.")
            return []

        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Skip files whose processed WAV already exists (resume after crash/interrupt)
        pending = [
            p for p in file_paths
            if not (self.out_dir / (pathlib.Path(p).stem + "_processed.wav")).exists()
        ]
        n_skip = len(file_paths) - len(pending)
        if n_skip:
            self._logger.info(f"Skipping {n_skip} already-processed file(s), {len(pending)} remaining.")
        if not pending:
            self._logger.info("All files already post-processed.")
            return []
        file_paths = pending

        worker_args = [
            {
                "in_path":       str(p),
                "out_dir":       str(self.out_dir),
                "plots_dir":     str(self.plots_dir),
                "error_log":     str(self.error_log),
                "enable_plots":  self.enable_plots,
            }
            for p in file_paths
        ]

        self._logger.info(
            f"StemCleaner starting: {len(file_paths)} file(s), {self.max_workers} worker(s)"
        )

        with multiprocessing.Pool(self.max_workers) as pool:
            async_results = [
                (args, pool.apply_async(_post_process_worker, (args,)))
                for args in worker_args
            ]

            results = []
            for args, ar in async_results:
                try:
                    results.append(ar.get())
                except Exception as e:
                    in_path = args["in_path"]
                    self._logger.error(f"Worker crashed (segfault?) for {in_path}: {e}")
                    self.out_dir.mkdir(parents=True, exist_ok=True)
                    with open(self.error_log, "a") as fh:
                        fh.write(
                            f"--- {in_path} ---\n"
                            f"Worker crash: {e}\n"
                            f"{traceback.format_exc()}\n"
                        )
                    results.append(None)

        rows = [r for r in results if r is not None]
        self._write_report(rows)

        n_err = len(results) - len(rows)
        n_lq  = sum(1 for r in rows if r["low_quality"])
        self._logger.info(
            f"StemCleaner done: {len(rows)} ok, {n_err} failed, {n_lq} low-quality  "
            f"report={self.report_csv}"
        )
        return rows

    def _write_report(self, rows):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        write_header = not self.report_csv.exists()
        with open(self.report_csv, "a", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["filename", "harmonic_ratio", "contrast_ratio", "duration", "low_quality"],
            )
            if write_header:
                writer.writeheader()
            writer.writerows(rows)
