"""Additional integration tests for document ingestion and measurements."""

from inspect import isgeneratorfunction

from findex.corpus import iter_documents
from findex.stats import eager_stats, lazy_stats


def test_iter_documents_and_stats(tmp_path):
    (tmp_path / "a.txt").write_text("Київ Київ Python", encoding="utf-8")
    folder = tmp_path / "nested"
    folder.mkdir()
    (folder / "b.txt").write_text("Python!", encoding="utf-8")
    assert isgeneratorfunction(iter_documents)
    iterator = iter_documents(tmp_path)
    assert next(iterator).doc_id == "a.txt"
    lazy = lazy_stats(tmp_path)
    assert lazy.documents == 2 and lazy.tokens == 4 and lazy.vocabulary == 2
    assert lazy == eager_stats(tmp_path)
    assert lazy_stats(tmp_path, limit=1).documents == 1


def test_invalid_utf8_replaced(tmp_path):
    (tmp_path / "broken.txt").write_bytes(b"valid \xff word")
    document = next(iter_documents(tmp_path))
    assert "\ufffd" in document.text
