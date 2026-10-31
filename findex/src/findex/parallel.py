"""Deterministic, spawn-safe parallel index construction (Lab 05)."""
from __future__ import annotations

import time
from array import array
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Literal

from findex.models import DocMeta, Index, PlainPosting, Posting
from findex.tokenize import tokenize

ExecutorKind = Literal["serial", "threads", "processes"]
Representation = Literal["plain", "slots", "array"]
DocumentTask = tuple[int, Path, str]


@dataclass
class PartialIndex:
    postings: dict[str, list[tuple[int, int]]]
    doc_lengths: dict[int, int]
    doc_meta: dict[int, DocMeta]
    positions: dict[str, dict[int, tuple[int, ...]]] | None
    cpu_seconds: float = 0.0


def build_partial(
    tasks: list[DocumentTask], with_positions: bool = False
) -> PartialIndex:
    """Top-level picklable worker; receives paths, never pre-read document bodies."""
    cpu_start = time.process_time()
    postings: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)
    lengths: dict[int, int] = {}
    meta: dict[int, DocMeta] = {}
    positions: defaultdict[str, dict[int, tuple[int, ...]]] | None = (
        defaultdict(dict) if with_positions else None
    )
    for doc_id, path, relative_name in tasks:
        # Propagate read errors so failures surface through future.result().
        text = path.read_text(encoding="utf-8", errors="replace")
        tokens = list(tokenize(text)) if with_positions else tokenize(text)
        counts: Counter[str] = Counter(tokens)
        lengths[doc_id] = counts.total()
        meta[doc_id] = DocMeta(path=relative_name, title=path.stem)
        for term, tf in counts.items():
            postings[term].append((doc_id, tf))
        if positions is not None:
            offsets: defaultdict[str, list[int]] = defaultdict(list)
            for position, term in enumerate(tokens):
                offsets[term].append(position)
            for term, term_offsets in offsets.items():
                positions[term][doc_id] = tuple(term_offsets)
    return PartialIndex(dict(postings), lengths, meta, dict(positions) if positions is not None else None, time.process_time() - cpu_start)


def merge(
    partials: list[PartialIndex], representation: Representation = "slots"
) -> Index:
    """Combine disjoint ID ranges and restore globally sorted posting lists."""
    if representation not in ("plain", "slots", "array"):
        raise ValueError(f"Invalid representation: {representation}")
    grouped: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)
    lengths: dict[int, int] = {}
    meta: dict[int, DocMeta] = {}
    positions: dict[str, dict[int, tuple[int, ...]]] | None = (
        {} if any(p.positions is not None for p in partials) else None
    )
    for partial in partials:
        if lengths.keys() & partial.doc_lengths.keys():
            raise ValueError("Duplicate doc_id across partial indexes")
        lengths.update(partial.doc_lengths)
        meta.update(partial.doc_meta)
        for term, entries in partial.postings.items():
            grouped[term].extend(entries)
        if positions is not None and partial.positions is not None:
            for term, docs in partial.positions.items():
                positions.setdefault(term, {}).update(docs)
    final: dict[str, list[Posting] | list[PlainPosting] | tuple[array[int], array[int]]] = {}
    for term, entries in grouped.items():
        entries.sort(key=lambda item: item[0])
        if representation == "plain":
            final[term] = [PlainPosting(doc, tf) for doc, tf in entries]
        elif representation == "slots":
            final[term] = [Posting(doc, tf) for doc, tf in entries]
        else:
            final[term] = (array("I", (doc for doc, _ in entries)), array("I", (tf for _, tf in entries)))
    return Index(final, lengths, meta, representation, positions)


def partition_documents(root: Path, workers: int) -> list[list[DocumentTask]]:
    """Contiguous, non-overlapping global ID ranges; stable filename ordering."""
    if workers < 1:
        raise ValueError("workers must be >= 1")
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(str(root))
    paths = [p for p in sorted(root.rglob("*.txt")) if p.is_file()]
    tasks = [(doc_id, path, path.relative_to(root).as_posix()) for doc_id, path in enumerate(paths)]
    if not tasks:
        return [[]]
    chunk_size = (len(tasks) + workers - 1) // workers
    return [tasks[i:i + chunk_size] for i in range(0, len(tasks), chunk_size)]


@dataclass(frozen=True)
class BuildMetrics:
    wall_seconds: float
    merge_seconds: float
    executor: ExecutorKind
    workers: int
    worker_cpu_seconds: float


def build_parallel(
    root: Path,
    *,
    workers: int = 1,
    executor: ExecutorKind = "serial",
    representation: Representation = "slots",
    with_positions: bool = False,
) -> tuple[Index, BuildMetrics]:
    """Build with identical worker/merge logic across all executor modes."""
    if executor not in ("serial", "threads", "processes"):
        raise ValueError(f"Invalid executor: {executor}")
    chunks = partition_documents(root, workers)
    started = time.perf_counter()
    if executor == "serial":
        partials = [build_partial(chunk, with_positions) for chunk in chunks]
    else:
        if executor == "threads":
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(build_partial, chunk, with_positions) for chunk in chunks]
                partials = [future.result() for future in futures]
        else:
            with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
                futures = [pool.submit(build_partial, chunk, with_positions) for chunk in chunks]
                partials = [future.result() for future in futures]
    merging = time.perf_counter()
    index = merge(partials, representation)
    merge_seconds = time.perf_counter() - merging
    return index, BuildMetrics(time.perf_counter() - started, merge_seconds, executor, workers, sum(part.cpu_seconds for part in partials))


def gil_status() -> str:
    """Report CPython 3.13+ GIL state, including builds without the API."""
    import sys

    probe = getattr(sys, "_is_gil_enabled", None)
    return str(probe()) if callable(probe) else "unavailable on this interpreter"
