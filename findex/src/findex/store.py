"""Index serialization. Never unpickle indexes from untrusted sources."""

import json
import pickle
import time
from array import array
from pathlib import Path
from typing import Any, cast

from findex.index import posting_pairs
from findex.models import DocMeta, Index, PlainPosting, Posting
from findex.timing import timed


def save(index: Index, path: Path, fmt: str = "pickle") -> tuple[float, int]:
    path = Path(path)
    start = time.perf_counter()
    if fmt == "pickle":
        with path.open("wb") as f:
            pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
    elif fmt == "json":
        payload = {
            "representation": index.representation,
            "positions": index.positions,
            "postings": {term: posting_pairs(index, term) for term in index.postings},
            "doc_lengths": index.doc_lengths,
            "doc_meta": {str(k): {"path": v.path, "title": v.title}
                         for k, v in index.doc_meta.items()},
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    else:
        raise ValueError(fmt)
    return time.perf_counter() - start, path.stat().st_size


@timed
def load(path: Path, fmt: str | None = None) -> tuple[Index, float]:
    path = Path(path)
    if fmt is None:
        fmt = "json" if path.suffix.lower() == ".json" else "pickle"
    start = time.perf_counter()
    if fmt == "pickle":
        # SECURITY: pickle.load can execute arbitrary code. ONLY load self-created trusted files!
        with path.open("rb") as f:
            result = pickle.load(f)
        if not isinstance(result, Index):
            raise ValueError("Not a findex index")
    elif fmt == "json":
        payload = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
        rep = payload["representation"]
        postings: dict[str, list[Posting] | list[PlainPosting] | tuple[array[int], array[int]]] = {}
        for term, pairs in cast(dict[str, list[tuple[int, int]]], payload["postings"]).items():
            if rep == "array":
                postings[term] = (array("I", (p[0] for p in pairs)),
                                  array("I", (p[1] for p in pairs)))
            elif rep == "plain":
                postings[term] = [PlainPosting(*p) for p in pairs]
            else:
                postings[term] = [Posting(*p) for p in pairs]
        result = Index(
            postings,
            {int(k): int(v) for k, v in cast(dict[str, int], payload["doc_lengths"]).items()},
            {int(k): DocMeta(**v) for k, v in cast(dict[str, dict[str, str]], payload["doc_meta"]).items()},
            rep,
            {term: {int(k): tuple(v) for k, v in docs.items()}
             for term, docs in cast(dict[str, dict[str, list[int]]], payload["positions"]).items()}
            if payload.get("positions") is not None else None,
        )
    else:
        raise ValueError(fmt)
    return result, time.perf_counter() - start
