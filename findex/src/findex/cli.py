"""Installable Typer command-line interface for Findex."""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Annotated, Literal, NoReturn

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from findex.index import build_index
from findex.models import Index
from findex.ranking import BM25, RankedSearcher, SearchResult, TfIdf
from findex.store import load, save

app = typer.Typer(help="Findex — пошукова система для текстових документів.", no_args_is_help=True)
err_console = Console(stderr=True)
console = Console()
log = logging.getLogger("findex")


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING if verbosity == 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr, force=True)


@app.callback()
def root(
    verbose: Annotated[int, typer.Option("-v", count=True, help="Деталі: -v або -vv")] = 0,
) -> None:
    _configure_logging(verbose)


@app.command("index")
def build(
    corpus: Annotated[Path, typer.Argument(help="Папка з .txt документами")],
    out: Annotated[Path, typer.Option("--out", help="Файл індексу")] = Path("index.bin"),
    positions: Annotated[bool, typer.Option("--positions", help="Зберігати позиції для точних фраз")] = True,
    fmt: Annotated[Literal["pickle", "json"], typer.Option("--format")] = "pickle",
) -> None:
    if not corpus.is_dir():
        _fail(f"Папка корпусу не існує: {corpus}")
    started = time.perf_counter()
    idx: Index
    duration: float
    size: int
    try:
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=err_console, transient=True) as progress:
            progress.add_task("Побудова індексу...", total=None)
            idx = build_index(corpus, with_positions=positions)
            duration, size = save(idx, out, fmt)
    except (OSError, ValueError, TypeError) as exc:
        _fail(str(exc))
    console.print(f"Індекс збережено: {out} | документів: {idx.num_docs} | термінів: {len(idx)}")
    log.info("build %.3f s; save %.3f s; bytes %s", time.perf_counter()-started, duration, size)


@app.command("search")
def search(
    path: Annotated[Path, typer.Argument(help="Файл індексу")],
    query: Annotated[str, typer.Argument(help="Пошуковий запит")],
    corpus: Annotated[Path, typer.Option("--corpus", help="Папка з текстами для сніпетів")] = Path("data"),
    scorer: Annotated[Literal["bm25", "tfidf"], typer.Option("--scorer")] = "bm25",
    top: Annotated[int, typer.Option("--top", min=0)] = 5,
    as_json: Annotated[bool, typer.Option("--json", help="JSON лише в stdout")] = False,
) -> None:
    if not path.is_file():
        _fail(f"Файл індексу не існує: {path}")
    matches: list[SearchResult]
    try:
        idx, _ = load(path)
        matches = RankedSearcher(idx).search(query, BM25() if scorer == "bm25" else TfIdf(), top, corpus)
    except (ValueError, OSError, KeyError, EOFError, TypeError) as exc:
        _fail(f"Помилка пошуку: {exc}")
    results = [{"doc_id": r.doc_id, "title": r.title, "score": r.score, "path": r.path, "snippet": r.snippet} for r in matches]
    if as_json:
        sys.stdout.write(json.dumps(results, ensure_ascii=False) + "\n")
        return
    table = Table(title=f"Результати пошуку: {query}")
    for col in ("ID", "Назва", "Оцінка", "Фрагмент"):
        table.add_column(col)
    for r in results:
        table.add_row(str(r["doc_id"]), str(r["title"]), f'{r["score"]:.4f}', str(r["snippet"]))
    console.print(table)


@app.command("stats")
def stats(
    corpus: Annotated[Path, typer.Argument(help="Папка з текстами")],
) -> None:
    if not corpus.is_dir():
        _fail(f"Папка корпусу не існує: {corpus}")
    from collections import Counter

    from findex.corpus import iter_documents
    from findex.tokenize import tokenize
    counter: Counter[str] = Counter()
    doc_count = 0
    for doc in iter_documents(corpus):
        doc_count += 1
        counter.update(tokenize(doc.text))
    console.print(f"Документів: {doc_count}; токенів: {counter.total()}; словник: {len(counter)}")
    for term, count in counter.most_common(10):
        console.print(f"{term}: {count}")


def _fail(message: str) -> NoReturn:
    err_console.print(f"Помилка: {message}")
    raise typer.Exit(code=1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
