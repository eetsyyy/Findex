"""Tokenizer acceptance checks for Lab 01."""

from inspect import isgenerator, isgeneratorfunction

from findex.tokenize import tokenize


def test_casefold():
    assert list(tokenize("PyThOn STRASSE Straße")) == ["python", "strasse", "strasse"]


def test_cyrillic():
    assert list(tokenize("УкраЇнська МОВА Київ")) == ["українська", "мова", "київ"]


def test_nfc_combining_mark():
    assert list(tokenize("café cafe\u0301")) == ["café", "café"]


def test_punctuation():
    assert list(tokenize("Hi, world! (Good.)")) == ["hi", "world", "good"]


def test_empty():
    assert list(tokenize("")) == []


def test_apostrophe_forms():
    assert list(tokenize("п'ять п’ять don't")) == ["п'ять", "п'ять", "don't"]


def test_hyphens_split():
    assert list(tokenize("пів-Європи state-of-the-art")) == ["пів", "європи", "state", "of", "the", "art"]


def test_digits_and_underscores():
    assert list(tokenize("2026 python3 hello_world")) == ["2026", "python3", "hello", "world"]


def test_generator_protocol():
    assert isgeneratorfunction(tokenize)
    iterator = tokenize("one two")
    assert isgenerator(iterator)
    assert next(iterator) == "one"
    assert list(iterator) == ["two"]
