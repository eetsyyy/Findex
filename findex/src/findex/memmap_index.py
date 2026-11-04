"""Persist NumPy posting arrays to .npy and reopen them with memory mapping.

A manifest keeps term strings out of path names. Files are hashed to avoid
unsafe characters and possible collisions between distinct terms.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from findex.models import Index
from findex.numpy_ranking import NumPyRankedSearcher


def save_mmap(index: Index, directory: Path) -> None:
    """Persist int32 arrays for each term and one doc-lengths array."""
    directory.mkdir(parents=True, exist_ok=True)
    ranker = NumPyRankedSearcher(index)
    np.save(directory / "doc_lengths.npy", ranker.doc_lengths, allow_pickle=False)
    manifest: dict[str, str] = {}
    for term, (ids, tfs) in ranker.postings.items():
        stem = hashlib.sha256(term.encode("utf-8")).hexdigest()
        np.save(directory / f"{stem}.ids.npy", ids, allow_pickle=False)
        np.save(directory / f"{stem}.tfs.npy", tfs, allow_pickle=False)
        manifest[term] = stem
    (directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_mmap(directory: Path) -> tuple[np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Read numeric arrays with mmap_mode='r', without loading their full payload."""
    manifest: dict[str, str] = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    lengths = np.load(directory / "doc_lengths.npy", mmap_mode="r", allow_pickle=False)
    postings = {
        term: (
            np.load(directory / f"{stem}.ids.npy", mmap_mode="r", allow_pickle=False),
            np.load(directory / f"{stem}.tfs.npy", mmap_mode="r", allow_pickle=False),
        )
        for term, stem in manifest.items()
    }
    return lengths, postings
