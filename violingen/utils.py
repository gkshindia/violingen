"""
utils.py
~~~~~~~~
Audio utility functions and Plotly visualisation helpers.

Audio I/O uses ``torchaudio`` (bundled with demucs) so mp3, wav, flac
and most common formats are supported without extra dependencies.
"""

from __future__ import annotations

import pathlib

import numpy as np
import soundfile as sf
import torch
import torchaudio
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Audio I/O
# ---------------------------------------------------------------------------

def load_audio(
    file_path: str,
    sr: int | None = 22050,
    mono: bool = True,
    duration: float | None = None,
) -> tuple[np.ndarray, int]:
    """
    Load an audio file as a float32 numpy array.

    Parameters
    ----------
    file_path : str
        Path to any audio file (mp3, wav, flac, …).
    sr : int or None
        Target sample rate.  If ``None`` the file's native rate is kept.
    mono : bool
        Mix down to mono when ``True``.
    duration : float or None
        Seconds to load from the start.  ``None`` loads the full file.

    Returns
    -------
    (y, sr_out) : (np.ndarray, int)
        Float32 waveform array and the actual sample rate used.
    """
    waveform, native_sr = torchaudio.load(file_path)

    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr is not None and sr != native_sr:
        waveform = torchaudio.functional.resample(waveform, native_sr, sr)
        out_sr = sr
    else:
        out_sr = native_sr

    if duration is not None:
        max_samples = int(duration * out_sr)
        waveform = waveform[:, :max_samples]

    y = waveform.squeeze().numpy().astype(np.float32)
    return y, out_sr


def save_audio(y: np.ndarray, file_path: str, sr: int) -> None:
    """
    Save a float32 waveform as a 16-bit PCM WAV file.

    Parameters
    ----------
    y         : waveform array (mono float32, values in [-1, 1])
    file_path : destination path (parent dirs created automatically)
    sr        : sample rate in Hz
    """
    out = pathlib.Path(file_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), np.clip(y, -1.0, 1.0), sr, subtype="PCM_16")
    print(f"[save_audio] saved → {out}")


# ---------------------------------------------------------------------------
# Plotly charts
# ---------------------------------------------------------------------------

def plot_waveform(
    file_path: str,
    sr: int | None = 22050,
    mono: bool = True,
    duration: float | None = None,
    title: str | None = None,
    colour: str = "#4a9eda",
) -> go.Figure:
    """
    Interactive Plotly waveform chart with zoom and pan.

    Parameters
    ----------
    file_path : str   — path to the audio file
    sr        : int   — sample rate to load at (None = native)
    mono      : bool  — mix to mono before plotting
    duration  : float — seconds to display (None = full file)
    title     : str   — chart title (defaults to the filename)
    colour    : str   — line colour hex

    Returns
    -------
    plotly.graph_objects.Figure
    """
    y, out_sr = load_audio(file_path, sr=sr, mono=mono, duration=duration)
    times = np.linspace(0.0, len(y) / out_sr, num=len(y))
    label = pathlib.Path(file_path).name

    fig = go.Figure(
        go.Scatter(
            x=times,
            y=y,
            mode="lines",
            name=label,
            line=dict(width=0.8, color=colour),
        )
    )
    fig.update_layout(
        title=dict(text=title or label, font=dict(size=13)),
        xaxis=dict(title="Time (s)"),
        yaxis=dict(title="Amplitude", range=[-1.05, 1.05]),
        template="plotly_dark",
        dragmode="zoom",
        height=300,
    )
    return fig


def plot_spectrogram(
    file_path: str,
    sr: int | None = 22050,
    mono: bool = True,
    duration: float | None = None,
    n_fft: int = 2048,
    hop_length: int = 512,
    fmax: int | None = 8000,
    title: str | None = None,
) -> go.Figure:
    """
    Interactive Plotly dB-magnitude spectrogram heatmap with zoom and pan.

    Parameters
    ----------
    file_path  : str   — path to the audio file
    sr         : int   — sample rate to load at (None = native)
    mono       : bool  — mix to mono before computing
    duration   : float — seconds to display (None = full file)
    n_fft      : int   — FFT window size
    hop_length : int   — hop between frames
    fmax       : int   — highest frequency bin to display in Hz
    title      : str   — chart title (defaults to the filename)

    Returns
    -------
    plotly.graph_objects.Figure
    """
    y, out_sr = load_audio(file_path, sr=sr, mono=mono, duration=duration)
    label = pathlib.Path(file_path).name

    # Compute STFT magnitude in dB
    window = torch.hann_window(n_fft)
    stft = torch.stft(
        torch.from_numpy(y),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        return_complex=True,
    )
    S = stft.abs().numpy()          # (freq_bins, time_frames)
    S_db = 20.0 * np.log10(np.maximum(S / (S.max() + 1e-9), 1e-6))

    # Frequency and time axes
    freq_bins = S_db.shape[0]
    freqs = np.linspace(0, out_sr / 2, freq_bins)
    n_frames = S_db.shape[1]
    times = np.linspace(0, len(y) / out_sr, n_frames)

    # Trim to fmax
    if fmax is not None:
        freq_mask = freqs <= fmax
        freqs = freqs[freq_mask]
        S_db = S_db[freq_mask, :]

    fig = go.Figure(
        go.Heatmap(
            x=times,
            y=freqs,
            z=S_db,
            colorscale="Magma",
            zmin=-80,
            zmax=0,
            colorbar=dict(title="dB"),
        )
    )
    fig.update_layout(
        title=dict(text=title or label, font=dict(size=13)),
        xaxis=dict(title="Time (s)"),
        yaxis=dict(title="Frequency (Hz)"),
        template="plotly_dark",
        dragmode="zoom",
        height=400,
    )
    return fig


# ---------------------------------------------------------------------------
# Progress bar helpers
# ---------------------------------------------------------------------------

def make_progress_bar(
    total: int,
    desc: str = "Processing",
    unit: str = "file",
):
    """
    Create and return a configured ``tqdm`` progress bar.

    Parameters
    ----------
    total : int
        Total number of items to track.
    desc : str
        Label displayed to the left of the bar.
    unit : str
        Singular unit name shown in the rate counter (e.g. ``"file"``).

    Returns
    -------
    tqdm.tqdm
        A ready-to-use progress bar.  Callers are responsible for closing it
        (``bar.close()``) or using it as a context manager.
    """
    from tqdm import tqdm

    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        unit_scale=False,
        dynamic_ncols=True,
        bar_format=(
            "{desc}: {percentage:3.0f}%|{bar}| "
            "{n_fmt}/{total_fmt} {unit}s "
            "[{elapsed}<{remaining}, {rate_fmt}]"
        ),
        colour="cyan",
    )


def format_elapsed(seconds: float) -> str:
    """
    Convert a raw second count into a human-readable elapsed-time string.

    Examples
    --------
    >>> format_elapsed(3.7)
    '3.70s'
    >>> format_elapsed(75.2)
    '1m 15.20s'

    Parameters
    ----------
    seconds : float
        Duration in seconds.

    Returns
    -------
    str
        Human-readable string, e.g. ``"42.13s"`` or ``"2m 05.40s"``.
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = int(seconds) // 60
    secs = seconds - minutes * 60
    return f"{minutes}m {secs:05.2f}s"
