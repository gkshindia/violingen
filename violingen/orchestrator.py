"""
orchestrator.py
~~~~~~~~~~~~~~~

Batch-process a list of audio files through :class:`StemSplitter` with
hardware-aware parallelism and structured progress reporting.

Device strategy
---------------
* **CPU** — ``ProcessPoolExecutor`` with one worker per logical core (or a
  user-supplied *max_workers*).  Each worker calls
  ``torch.set_num_threads(1)`` so N workers × N cores of BLAS threads are
  not spawned simultaneously (a common PyTorch CPU bottleneck).
* **CUDA / MPS** — Serial execution on the single GPU context.  Spawning
  processes would reinitialise the model in each process, wasting VRAM and
  risking context conflicts.

Logging and progress
--------------------
All structured logging is delegated to :mod:`violingen.logging`.
The ``tqdm`` progress bar is constructed via :func:`violingen.utils.make_progress_bar`.
Every file is individually timed with ``time.perf_counter()``; total batch
wall time is also recorded and included in the summary log.

Example
-------
    from violingen import Orchestrator

    orch = Orchestrator(out_dir="output/strings")
    results = orch.process([
        "datasets/2.mp3",
        "datasets/3.mp3",
        "datasets/7.mp3",
    ])
    # results: {"datasets/2.mp3": "output/strings/2/other.wav", ...}
"""

from __future__ import annotations

import os
import pathlib
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import torch

from violingen.logging import (
    get_logger,
    log_batch_start,
    log_batch_summary,
    log_file_error,
    log_file_result,
)
from violingen.stem_splitter import StemSplitter
from violingen.utils import format_elapsed, make_progress_bar


# ---------------------------------------------------------------------------
# Module-level worker (must be picklable → defined at module scope)
# ---------------------------------------------------------------------------

def _split_worker(args: dict[str, Any]) -> tuple[str, str | Exception, float]:
    """
    Worker function executed in a subprocess (CPU path).

    Receives a plain dict so it is trivially picklable.  Constructs a fresh
    :class:`StemSplitter` from the config, calls ``split()``, and returns
    a ``(in_path, out_path, elapsed_s)`` tuple on success or
    ``(in_path, exception, elapsed_s)`` on failure.

    ``torch.set_num_threads(1)`` is set before any PyTorch work to prevent
    each worker from spawning a full BLAS thread pool and saturating all
    cores.
    """
    torch.set_num_threads(1)

    in_path:  str = args["in_path"]
    out_path: str = args["out_path"]
    config:   dict = args["config"]

    t0 = time.perf_counter()
    try:
        splitter = StemSplitter(**config)
        splitter.split(in_path, out_path)
        return in_path, out_path, time.perf_counter() - t0
    except Exception as exc:  # noqa: BLE001
        return in_path, exc, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    """
    Batch-process a list of audio files through :class:`StemSplitter`.

    Parameters
    ----------
    out_dir : str
        Base directory for output stems.  Each input file is written to
        ``{out_dir}/{input_filestem}/{stem}.wav``.
    model : str
        Demucs model name forwarded to :class:`StemSplitter`.
    stem : str
        Demucs stem to extract (``"other"`` = violin / strings).
    device : str or None
        Inference device.  ``None`` triggers auto-detection:
        ``"cuda"`` → ``"mps"`` → ``"cpu"`` in priority order.
    shifts : int
        Random time-shift passes for equivariant stabilisation.
    overlap : float
        Overlap ratio between consecutive audio chunks.
    clip_mode : str
        ``"rescale"`` or ``"clamp"``.
    max_workers : int or None
        Worker processes on the CPU path.  ``None`` defaults to
        ``os.cpu_count()``.  Forced to ``1`` on CUDA / MPS.
    """

    def __init__(
        self,
        out_dir: str = "output",
        model: str = "htdemucs_6s",
        stem: str = "other",
        device: str | None = None,
        shifts: int = 0,
        overlap: float = 0.25,
        clip_mode: str = "rescale",
        max_workers: int | None = None,
    ) -> None:
        self.out_dir   = pathlib.Path(out_dir)
        self.model     = model
        self.stem      = stem
        self.device    = device or self._detect_device()
        self.shifts    = shifts
        self.overlap   = overlap
        self.clip_mode = clip_mode

        # GPU contexts are not safe to share across spawned processes
        _is_gpu = self.device in ("cuda", "mps")
        self.max_workers = 1 if _is_gpu else (max_workers or os.cpu_count() or 1)

        # Reused on the GPU serial path (avoid re-loading model per file)
        self._splitter = StemSplitter(
            model=self.model,
            device=self.device,
            shifts=self.shifts,
            overlap=self.overlap,
            clip_mode=self.clip_mode,
            stem=self.stem,
        )

        self._logger = get_logger("violingen.orchestrator")
        self._logger.debug(
            f"Orchestrator ready  device={self.device}  "
            f"max_workers={self.max_workers}  model={self.model}  stem={self.stem}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, file_paths: list[str]) -> dict[str, str | Exception]:
        """
        Run stem separation on every file in *file_paths*.

        Parameters
        ----------
        file_paths : list[str]
            Absolute or relative paths to input audio files.

        Returns
        -------
        dict[str, str | Exception]
            Maps each input path to either the output WAV path (success) or
            the exception that was caught (failure).
        """
        if not file_paths:
            self._logger.warning("process() called with an empty file list — nothing to do.")
            return {}

        pairs = self._build_pairs(file_paths)

        log_batch_start(
            self._logger,
            n_files=len(pairs),
            device=self.device,
            max_workers=self.max_workers,
            model=self.model,
            stem=self.stem,
        )

        batch_t0 = time.perf_counter()
        results: dict[str, str | Exception] = {}

        if self.device == "cpu":
            results = self._process_cpu(pairs)
        else:
            results = self._process_gpu(pairs)

        total_elapsed = time.perf_counter() - batch_t0
        n_ok   = sum(1 for v in results.values() if not isinstance(v, Exception))
        n_fail = len(results) - n_ok
        log_batch_summary(self._logger, n_ok, n_fail, total_elapsed)

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_pairs(self, file_paths: list[str]) -> list[tuple[str, str]]:
        """Build (in_path, out_path) pairs: {out_dir}/{original_name}.wav"""
        pairs: list[tuple[str, str]] = []
        for fp in file_paths:
            in_path  = str(pathlib.Path(fp).expanduser().resolve())
            name     = pathlib.Path(fp).stem          # e.g. "1"
            out_path = str(self.out_dir / f"{name}.wav")
            pairs.append((in_path, out_path))
        return pairs

    def _process_cpu(
        self, pairs: list[tuple[str, str]]
    ) -> dict[str, str | Exception]:
        """Parallel CPU processing via ProcessPoolExecutor."""
        config = dict(
            model=self.model,
            device=self.device,
            shifts=self.shifts,
            overlap=self.overlap,
            clip_mode=self.clip_mode,
            stem=self.stem,
        )
        worker_args = [
            {"in_path": in_p, "out_path": out_p, "config": config}
            for in_p, out_p in pairs
        ]

        results: dict[str, str | Exception] = {}
        bar = make_progress_bar(total=len(pairs), desc="Splitting (CPU)")

        with ProcessPoolExecutor(max_workers=self.max_workers) as pool:
            future_map = {
                pool.submit(_split_worker, args): args["in_path"]
                for args in worker_args
            }
            for future in as_completed(future_map):
                in_path, outcome, elapsed_s = future.result()
                if isinstance(outcome, Exception):
                    log_file_error(self._logger, in_path, outcome)
                    results[in_path] = outcome
                else:
                    log_file_result(self._logger, in_path, outcome, elapsed_s)
                    results[in_path] = outcome
                bar.set_postfix({"last": pathlib.Path(in_path).name, "t": format_elapsed(elapsed_s)})
                bar.update(1)

        bar.close()
        return results

    def _process_gpu(
        self, pairs: list[tuple[str, str]]
    ) -> dict[str, str | Exception]:
        """Serial GPU / MPS processing — one CUDA/MPS context, no subprocess."""
        results: dict[str, str | Exception] = {}
        bar = make_progress_bar(
            total=len(pairs),
            desc=f"Splitting ({self.device.upper()})",
        )

        for in_path, out_path in pairs:
            t0 = time.perf_counter()
            try:
                self._splitter.split(in_path, out_path)
                elapsed_s = time.perf_counter() - t0
                log_file_result(self._logger, in_path, out_path, elapsed_s)
                results[in_path] = out_path
            except Exception as exc:  # noqa: BLE001
                elapsed_s = time.perf_counter() - t0
                log_file_error(self._logger, in_path, exc)
                results[in_path] = exc
            bar.set_postfix({"last": pathlib.Path(in_path).name, "t": format_elapsed(elapsed_s)})
            bar.update(1)

        bar.close()
        return results

    @staticmethod
    def _detect_device() -> str:
        """
        Auto-detect the best available inference device.

        Priority: ``"cuda"`` → ``"mps"`` → ``"cpu"``.
        """
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
