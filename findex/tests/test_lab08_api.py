"""API can switch between the legacy and NumPy scorers without a live index file."""
from __future__ import annotations

from fastapi.testclient import TestClient

from findex.models import DocMeta, Index, Posting
from findex.web.app import Settings, create_app, get_index


def test_legacy_numpy_api_agree() -> None:
    index = Index(
        postings={"alpha": [Posting(0, 2), Posting(1, 1)]},
        doc_lengths={0: 3, 1: 2},
        doc_meta={0: DocMeta("0.txt", "First"), 1: DocMeta("1.txt", "Second")},
    )
    app = create_app(Settings(index_path="missing.bin"))
    app.dependency_overrides[get_index] = lambda: index
    with TestClient(app) as client:
        a = client.get("/search", params={"q": "alpha", "engine": "legacy"})
        b = client.get("/search", params={"q": "alpha", "engine": "numpy"})
        bad = client.get("/search", params={"q": "alpha", "engine": "invalid"})
    assert a.status_code == b.status_code == 200
    assert bad.status_code == 422
    assert [x["doc_id"] for x in a.json()["results"]] == [x["doc_id"] for x in b.json()["results"]]
