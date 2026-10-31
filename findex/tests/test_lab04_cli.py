"""Findex 0.4 CLI integration checks."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from findex.cli import app

runner = CliRunner()


@pytest.fixture
def small_corpus(tmp_path: Path) -> Path:
    corpus = tmp_path / "data"
    corpus.mkdir()
    (corpus / "a.txt").write_text("Python пошукова система Python", encoding="utf-8")
    (corpus / "b.txt").write_text("Київ Python", encoding="utf-8")
    return corpus


def test_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "search" in result.stdout


def test_index_and_search(tmp_path: Path, small_corpus: Path) -> None:
    target = tmp_path / "index.bin"
    built = runner.invoke(app, ["index", str(small_corpus), "--out", str(target)])
    assert built.exit_code == 0, built.output
    assert target.is_file()
    found = runner.invoke(app, ["search", str(target), "python", "--corpus", str(small_corpus)])
    assert found.exit_code == 0, found.output
    assert "a" in found.output


def test_json_output(tmp_path: Path, small_corpus: Path) -> None:
    target = tmp_path / "index.bin"
    assert runner.invoke(app, ["index", str(small_corpus), "--out", str(target)]).exit_code == 0
    result = runner.invoke(app, ["search", str(target), "python", "--json", "--corpus", str(small_corpus)])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert isinstance(payload, list)
    assert payload[0]["title"] in {"a", "b"}


def test_missing_index() -> None:
    result = runner.invoke(app, ["search", "missing-index.bin", "python"])
    assert result.exit_code != 0
    assert "Traceback" not in result.output


def test_stats(small_corpus: Path) -> None:
    result = runner.invoke(app, ["stats", str(small_corpus)])
    assert result.exit_code == 0
    assert "2" in result.stdout
