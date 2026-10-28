"""Lab 03: object-model, parser, scores, caching, persistence."""

import pytest

from findex.index import build_index
from findex.query import And, Not, Or, Phrase, Term, parse
from findex.ranked_search import open_index
from findex.ranking import BM25, RankedSearcher, TfIdf
from findex.store import save


def fixture_index(tmp_path):
    (tmp_path / "a.txt").write_text("python async event loop python", encoding="utf-8")
    (tmp_path / "b.txt").write_text("python await event loop", encoding="utf-8")
    (tmp_path / "c.txt").write_text("java async event loop", encoding="utf-8")
    return build_index(tmp_path, with_positions=True)


def test_mapping(tmp_path):
    ix = fixture_index(tmp_path)
    assert len(ix) >= 6 and 'python' in ix
    assert ix['python'] and 'missing' not in ix
    assert list(ix) == list(ix.postings)
    assert ix.num_docs == 3 and ix.df('python') == 2
    assert ix.avg_doc_length > 0
    assert repr(ix).startswith('Index(terms=')
    with pytest.raises(KeyError):
        _ = ix['nonexistent']


def test_parser_tree():
    assert parse('a OR b') == Or(Term('a'), Term('b'))
    assert parse('a b') == And(Term('a'), Term('b'))
    assert parse('NOT x') == Not(Term('x'))
    assert parse('a AND (b OR c)') == And(Term('a'), Or(Term('b'), Term('c')))
    assert parse('"event loop"') == Phrase(('event', 'loop'))
    assert (Term('a') & Term('b')) == And(Term('a'), Term('b'))
    assert (Term('a') | ~Term('b')) == Or(Term('a'), Not(Term('b')))


def test_complex_query(tmp_path):
    ix = fixture_index(tmp_path)
    tree = parse('python AND (async OR await) NOT java "event loop"')
    assert tree.evaluate(ix) == {0, 1}


def test_phrase_requires_positions(tmp_path):
    (tmp_path / 'x.txt').write_text('event loop', encoding='utf-8')
    ix = build_index(tmp_path)
    with pytest.raises(ValueError, match='positions'):
        parse('"event loop"').evaluate(ix)


def test_scores():
    bm = BM25()
    assert bm(1, 1, 100, 100, 100) > bm(1, 70, 100, 100, 100)
    assert bm(1, 10, 100, 50, 100) > bm(1, 10, 100, 500, 100)
    assert bm(20, 10, 100, 100, 100) - bm(19, 10, 100, 100, 100) < bm(1, 10, 100, 100, 100) / 20
    assert TfIdf()(2, 1, 10, 100, 100) > 0


def test_rank_cache_and_results(tmp_path):
    ix = fixture_index(tmp_path)
    s = RankedSearcher(ix)
    first = s.search('python', k=5, corpus=tmp_path)
    assert first and first[0].score > 0
    assert '**python**' in first[0].snippet.lower()
    s.search('python')
    assert s.matching_ids.cache_info().hits >= 1


def test_pickle_and_json_positions(tmp_path):
    corpus = tmp_path / 'corpus'
    corpus.mkdir()
    ix = fixture_index(corpus)
    for fmt, ext in [('pickle','.bin'), ('json','.json')]:
        path = tmp_path / ('idx' + ext)
        save(ix, path, fmt)
        with open_index(path) as reopened:
            assert parse('"event loop"').evaluate(reopened) == {0, 1, 2}


def test_invalid_query():
    for expression in ['', '(python', 'python OR', 'OR python', 'python )']:
        with pytest.raises(ValueError):
            parse(expression)
