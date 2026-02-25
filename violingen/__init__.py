"""
violingen
~~~~~~~~~

Violin / strings stem extraction and note generation toolkit.
"""

from violingen.orchestrator import Orchestrator
from violingen.stem_cleaner import StemCleaner
from violingen.stem_splitter import StemSplitter
from violingen.logging import get_logger

__all__ = [
    "Orchestrator",
    "StemCleaner",
    "StemSplitter",
    "get_logger",
]

