"""Recursive-descent Boolean and phrase query parser."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from findex.index import posting_pairs
from findex.tokenize import tokenize


class Query(Protocol):
    def evaluate(self, index) -> set[int]: ...
    def terms(self) -> tuple[str, ...]: ...


class Operators:
    def __and__(self, other):
        return And(self, other)

    def __or__(self, other):
        return Or(self, other)

    def __invert__(self):
        return Not(self)


@dataclass(frozen=True)
class Term(Operators):
    value: str

    def evaluate(self, index) -> set[int]:
        return {d for d, _ in posting_pairs(index, self.value)}

    def terms(self) -> tuple[str, ...]:
        return (self.value,)


@dataclass(frozen=True)
class Phrase(Operators):
    words: tuple[str, ...]

    def evaluate(self, index) -> set[int]:
        if not self.words:
            return set()
        positions = getattr(index, "positions", None)
        if positions is None:
            raise ValueError("Phrase search requires an index built with --positions")
        docs = set(positions.get(self.words[0], {}))
        for word in self.words[1:]:
            docs &= set(positions.get(word, {}))
        result = set()
        for doc_id in docs:
            starts = set(positions[self.words[0]][doc_id])
            for offset, word in enumerate(self.words[1:], 1):
                starts &= {p - offset for p in positions[word][doc_id]}
            if starts:
                result.add(doc_id)
        return result

    def terms(self) -> tuple[str, ...]:
        return self.words


@dataclass(frozen=True)
class And(Operators):
    left: Query
    right: Query

    def evaluate(self, index) -> set[int]:
        return self.left.evaluate(index) & self.right.evaluate(index)

    def terms(self) -> tuple[str, ...]:
        return self.left.terms() + self.right.terms()


@dataclass(frozen=True)
class Or(Operators):
    left: Query
    right: Query

    def evaluate(self, index) -> set[int]:
        return self.left.evaluate(index) | self.right.evaluate(index)

    def terms(self) -> tuple[str, ...]:
        return self.left.terms() + self.right.terms()


@dataclass(frozen=True)
class Not(Operators):
    node: Query

    def evaluate(self, index) -> set[int]:
        return set(index.doc_meta) - self.node.evaluate(index)

    def terms(self) -> tuple[str, ...]:
        return ()  # Negative terms must not contribute to ranking.


LEX = re.compile(r'"[^"\n]*"|\(|\)|[^\s()"]+')


def parse(query: str) -> Query:
    """NOT > AND (explicit or implicit) > OR. Adjacent terms mean AND."""
    raw = LEX.findall(query)
    tokens = []
    for item in raw:
        if item.startswith('"'):
            tokens.append(Phrase(tuple(tokenize(item[1:-1]))))
        elif item in ('(', ')') or item.upper() in ('AND', 'OR', 'NOT'):
            tokens.append(item.upper())
        else:
            tokens.extend(Term(t) for t in tokenize(item))
    if not tokens:
        raise ValueError("Empty query")
    pos = 0

    def atom():
        nonlocal pos
        if pos >= len(tokens):
            raise ValueError("Missing operand")
        tok = tokens[pos]
        pos += 1
        if tok == '(':
            node = or_expr()
            if pos >= len(tokens) or tokens[pos] != ')':
                raise ValueError("Unclosed parenthesis")
            pos += 1
            return node
        if tok == 'NOT':
            return Not(atom())
        if tok in ('AND', 'OR', ')'):
            raise ValueError(f"Unexpected token: {tok}")
        return tok

    def and_expr():
        nonlocal pos
        node = atom()
        while pos < len(tokens) and tokens[pos] not in ('OR', ')'):
            if tokens[pos] == 'AND':
                pos += 1
            node = And(node, atom())
        return node

    def or_expr():
        nonlocal pos
        node = and_expr()
        while pos < len(tokens) and tokens[pos] == 'OR':
            pos += 1
            node = Or(node, and_expr())
        return node

    tree = or_expr()
    if pos != len(tokens):
        raise ValueError("Unexpected trailing tokens")
    return tree
