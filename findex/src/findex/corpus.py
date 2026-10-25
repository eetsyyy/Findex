"""Lazy, deterministic traversal of a directory of UTF-8 text documents."""

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Document:
    """A single document, not the complete corpus."""

    doc_id: str
    path: Path
    text: str


def iter_documents(root: Path) -> Iterator[Document]:
    """Yield one .txt document at a time, traversing folders recursively.

    Bad UTF-8 is replaced with U+FFFD, with a warning; unreadable files are skipped.
    Per-document memory is proportional to the current file's size.
    """
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"Corpus directory does not exist: {root}")
    # Sorting the paths makes --limit reproducible; this materializes file paths,
    # never the documents' contents.
    for path in sorted(root.rglob("*.txt")):
        if not path.is_file():
            continue
        try:
            with path.open("r", encoding="utf-8", errors="replace") as file:
                text = file.read()
        except OSError as exc:
            logger.warning("Skipping unreadable file %s: %s", path, exc)
            continue
        if "\ufffd" in text:
            logger.warning("Invalid UTF-8 bytes replaced in %s", path)
        yield Document(doc_id=path.relative_to(root).as_posix(), path=path, text=text)
