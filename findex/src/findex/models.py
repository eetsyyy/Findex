"""Stable import location for pickled index data models."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Posting:
    doc_id: int
    tf: int


@dataclass(frozen=True, slots=True)
class DocMeta:
    path: str
    title: str


@dataclass
class PlainPosting:
    doc_id: int
    tf: int


@dataclass
class Index:
    postings: dict[str, Any]
    doc_lengths: dict[int, int]
    doc_meta: dict[int, DocMeta]
    representation: str = "slots"


