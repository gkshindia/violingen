"""
violingen
~~~~~~~~~

Violin / strings stem extraction and note generation toolkit.
"""

from violingen.orchestrator import Orchestrator
from violingen.post_processor import PostProcessor
from violingen.stem_splitter import StemSplitter
from violingen.logging import get_logger

__all__ = [
    "Orchestrator",
    "PostProcessor",
    "StemSplitter",
    "get_logger",
]

