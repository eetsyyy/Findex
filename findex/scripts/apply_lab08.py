"""Idempotently integrate Lab 08 into the existing Findex 0.7 project."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    if old not in text:
        raise RuntimeError(f"Cannot update {label}: expected snippet missing. No file written.")
    return text.replace(old, new, 1)


def modify(path: Path, fn) -> None:
    original = path.read_text(encoding="utf-8")
    updated = fn(original)
    if original != updated:
        path.write_text(updated, encoding="utf-8")
        print("Updated", path.relative_to(ROOT))


def pyproject(text: str) -> str:
    if 'version = "0.7.0"' in text:
        text = text.replace('version = "0.7.0"', 'version = "1.0.0"', 1)
    elif 'version = "1.0.0"' not in text:
        raise RuntimeError("Expected findex 0.7.0")
    if '"numpy>=2.0"' not in text:
        text = re.sub(r'(?m)^dependencies = \[', 'dependencies = ["numpy>=2.0", ', text, count=1)
    if '"pytest-benchmark>=5"' not in text:
        text = re.sub(r'(?m)^dev = \[', 'dev = ["pytest-benchmark>=5", ', text, count=1)
    return text


def cli(text: str) -> str:
    text = replace_once(text, 'from findex.ranking import BM25,', 'from findex.numpy_ranking import cached_numpy_searcher\nfrom findex.ranking import BM25,', 'CLI import')
    text = replace_once(text, '    top: Annotated[int, typer.Option("--top", min=0)] = 5,',
                        '    engine: Annotated[Literal["legacy", "numpy"], typer.Option("--engine")] = "numpy",\n    top: Annotated[int, typer.Option("--top", min=0)] = 5,', 'CLI argument')
    text = replace_once(text, '        matches = RankedSearcher(idx).search(query, BM25() if scorer == "bm25" else TfIdf(), top, corpus)',
                        '        matches = (\n            cached_numpy_searcher(idx).search(query, scorer=scorer, k=top, corpus=corpus)\n            if engine == "numpy"\n            else RankedSearcher(idx).search(query, BM25() if scorer == "bm25" else TfIdf(), top, corpus)\n        )', 'CLI search')
    return text


def api(text: str) -> str:
    text = replace_once(text, 'from findex.models import Index', 'from findex.models import Index\nfrom findex.numpy_ranking import cached_numpy_searcher', 'API import')
    text = replace_once(text, '    scorer: Literal["bm25", "tfidf"]\n    total:',
                        '    scorer: Literal["bm25", "tfidf"]\n    engine: Literal["legacy", "numpy"] = "numpy"\n    total:', 'SearchResponse field')
    # Backwards-compatible optional keyword at end; called by all three routes.
    text = replace_once(text, '    corpus: Path | None,\n) -> SearchResponse:',
                        '    corpus: Path | None,\n    engine: Literal["legacy", "numpy"] = "numpy",\n) -> SearchResponse:', 'API calculation signature')
    text = replace_once(text, '        found: list[RankedHit] = searcher.search(q, scorer=score_fn, k=limit, corpus=corpus)',
                        '        found: list[RankedHit] = (\n            cached_numpy_searcher(index).search(q, scorer=scorer, k=limit, corpus=corpus)\n            if engine == "numpy" else searcher.search(q, scorer=score_fn, k=limit, corpus=corpus)\n        )', 'API score call')
    text = replace_once(text, 'return SearchResponse(q=q, k=k, page=page, scorer=scorer, total=total,',
                        'return SearchResponse(q=q, k=k, page=page, scorer=scorer, engine=engine, total=total,', 'API response')
    text = text.replace('version="0.7.0"', 'version="1.0.0"')
    # The first route is /search, the second /search-async, both use same param.
    argument = '        scorer: Literal["bm25", "tfidf"] = "bm25",\n        page:'
    addition = '        scorer: Literal["bm25", "tfidf"] = "bm25",\n        engine: Literal["legacy", "numpy"] = "numpy",\n        page:'
    if text.count(argument) != 2 and text.count(addition) != 2:
        raise RuntimeError("Expected two search routes; stop instead of partial API update")
    text = text.replace(argument, addition)
    text = text.replace('return _calculate(index, q, k, scorer, page, corpus)',
                        'return _calculate(index, q, k, scorer, page, corpus, engine)')
    return text


def ui(text: str) -> str:
    anchor = '<button type="submit">Знайти</button>'
    if 'name="engine"' not in text:
        text = replace_once(text, anchor,
            '<select name="engine" aria-label="Рушій пошуку"><option value="numpy" {% if engine == "numpy" %}selected{% endif %}>NumPy</option><option value="legacy" {% if engine == "legacy" %}selected{% endif %}>Python</option></select>' + anchor,
            'HTML select')
    text = text.replace('href="/?q={{ q|urlencode }}&page=', 'href="/?engine={{ engine }}&q={{ q|urlencode }}&page=')
    return text.replace('Python Lab 07', 'Python Lab 08')


def home_api(text: str) -> str:
    text = replace_once(text, 'def home(request: Request, q: str = "", k: int = 10, page: int = 1):',
                        'def home(request: Request, q: str = "", k: int = 10, page: int = 1, engine: Literal["legacy", "numpy"] = "numpy"):', 'home signature')
    text = replace_once(text, 'data = _calculate(index, q, k, "bm25", page, get_corpus(request))',
                        'data = _calculate(index, q, k, "bm25", page, get_corpus(request), engine)', 'home call')
    text = replace_once(text, 'context={"q": q, "data": data, "error": error,',
                        'context={"q": q, "engine": engine, "data": data, "error": error,', 'template data')
    return text


def main() -> None:
    paths = [ROOT / "pyproject.toml", ROOT / "src/findex/cli.py", ROOT / "src/findex/web/app.py", ROOT / "src/findex/web/templates/index.html"]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise SystemExit("Missing files: " + ", ".join(missing))
    modify(paths[0], pyproject)
    modify(paths[1], cli)
    modify(paths[2], lambda text: home_api(api(text)))
    modify(paths[3], ui)
    print("Lab 08 integration done: version 1.0.0, NumPy engine in CLI and API/UI.")


if __name__ == "__main__":
    main()
