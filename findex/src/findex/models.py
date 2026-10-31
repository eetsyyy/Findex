"""Stable import location for pickled index data models."""

from array import array
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from functools import cached_property


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
class Index(Mapping[str, list[Posting] | list[PlainPosting] | tuple[array[int], array[int]]]):
    postings: dict[str, list[Posting] | list[PlainPosting] | tuple[array[int], array[int]]]
    doc_lengths: dict[int, int]
    doc_meta: dict[int, DocMeta]
    representation: str = "slots"
    positions: dict[str, dict[int, tuple[int, ...]]] | None = None

    def __len__(self) -> int:
        return len(self.postings)

    def __iter__(self) -> Iterator[str]:
        return iter(self.postings)

    def __getitem__(self, term: str) -> list[Posting] | list[PlainPosting] | tuple[array[int], array[int]]:
        return self.postings[term]

    def __contains__(self, term: object) -> bool:
        return term in self.postings

    def __repr__(self) -> str:
        return f"Index(terms={len(self):_}, docs={self.num_docs:_})"

    @property
    def num_docs(self) -> int:
        return len(self.doc_lengths)

    @cached_property
    def avg_doc_length(self) -> float:
        return sum(self.doc_lengths.values()) / self.num_docs if self.num_docs else 0.0

    def doc_length(self, doc_id: int) -> int:
        return self.doc_lengths[doc_id]

    def df(self, term: str) -> int:
        from findex.index import posting_pairs
        return len(posting_pairs(self, term))


