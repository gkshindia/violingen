"""
stem_splitter.py
~~~~~~~~~~~~~~~~

Ref: https://github.com/facebookresearch/demucs

Isolate the **violin / strings** stem from a single audio file using
Demucs `htdemucs_6s` (6-source model).

In Demucs htdemucs_6s the six stems are:
    drums  · bass  · other  · vocals  · guitar  · piano

The ``other`` stem captures everything that is not one of the five named
sources — in practice this is where violin, strings, and non-guitar
melody instruments land.
"""

from __future__ import annotations

import pathlib
import shutil
import tempfile


class StemSplitter:
    """
    Separate the ``other`` stem (violin / strings) from a single audio file
    using Demucs ``htdemucs_6s``.

    Parameters
    ----------
    model : str
        Demucs model name.  ``htdemucs_6s`` is the 6-source model that
        gives a dedicated ``other`` channel (violin / strings).
    device : str
        Inference device — ``"cpu"``, ``"cuda"``, or ``"mps"``.
        Use ``"mps"`` on Apple Silicon for ~3–4× CPU speed.
    shifts : int
        Number of random time-shift passes for equivariant stabilisation.
        0 = off (fastest).
    overlap : float
        Overlap ratio between consecutive audio chunks (0 < overlap < 1).
    clip_mode : str
        ``"rescale"`` or ``"clamp"`` — how to prevent output clipping.
    stem : str
        Demucs stem file to extract.  ``"other"`` = violin / strings.
    """

    def __init__(
        self,
        model: str = "htdemucs_6s",
        device: str = "cpu",
        shifts: int = 0,
        overlap: float = 0.25,
        clip_mode: str = "rescale",
        stem: str = "other",
    ) -> None:
        self.model = model
        self.device = device
        self.shifts = shifts
        self.overlap = overlap
        self.clip_mode = clip_mode
        self.stem = stem

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split(self, in_file_path: str, out_file_path: str) -> str:
        """
        Run Demucs on *in_file_path* and save the extracted stem to
        *out_file_path*.

        Parameters
        ----------
        in_file_path : str
            Path to the source audio file (mp3, wav, flac, …).
        out_file_path : str
            Destination path for the extracted stem WAV.
            Parent directories are created automatically.

        Returns
        -------
        str
            Absolute path to the saved stem file.
        """
        import demucs.separate

        in_path  = pathlib.Path(in_file_path).expanduser().resolve()
        out_path = pathlib.Path(out_file_path).expanduser().resolve()

        if not in_path.exists():
            raise FileNotFoundError(f"Input file not found: {in_path}")

        with tempfile.TemporaryDirectory(prefix="demucs_") as tmp_dir:
            self._run_demucs(str(in_path), tmp_dir, demucs.separate)
            stem_wav = self._locate_stem(tmp_dir, in_path.stem)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(stem_wav, out_path)

        print(f"[StemSplitter] stem='{self.stem}'  saved → {out_path}")
        return str(out_path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_demucs(self, in_file: str, out_dir: str, demucs_separate) -> None:
        """Build the CLI argument list and invoke ``demucs.separate.main``."""
        demucs_separate.main([
            "-n", self.model,
            "-d", self.device,
            "--shifts", str(self.shifts),
            "--overlap", str(self.overlap),
            "--clip-mode", self.clip_mode,
            "--out", out_dir,
            in_file,
        ])

    def _locate_stem(self, out_dir: str, audio_stem: str) -> pathlib.Path:
        """
        Find the extracted stem WAV inside the Demucs output tree.

        Demucs writes:
            ``{out_dir}/{model}/{audio_stem}/{stem_name}.wav``
        """
        stem_wav = (
            pathlib.Path(out_dir) / self.model / audio_stem / f"{self.stem}.wav"
        )
        if not stem_wav.exists():
            raise RuntimeError(
                f"Expected stem file not found at: {stem_wav}\n"
                f"Available files: {list(stem_wav.parent.iterdir()) if stem_wav.parent.exists() else '(dir missing)'}"
            )
        return stem_wav