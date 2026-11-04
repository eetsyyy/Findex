"""HTTP API tests: dependency overrides, no real index file/network."""
from __future__ import annotations

from fastapi.testclient import TestClient

from findex.models import DocMeta, Index, Posting
from findex.web.app import Settings, create_app, get_index


def _index() -> Index:
    return Index(
        postings={"python": [Posting(doc_id=1, tf=2)]},
        doc_lengths={1: 2},
        doc_meta={1: DocMeta(path="1.txt", title="Python tutorial")},
    )


def test_search_success() -> None:
    app = create_app(Settings(index_path="no-such-file.bin"))
    app.dependency_overrides[get_index] = _index
    with TestClient(app) as client:
        response = client.get("/search", params={"q": "python", "k": 10, "page": 1})
    assert response.status_code == 200
    result = response.json()
    assert result["total"] == 1
    assert result["results"][0]["title"] == "Python tutorial"


def test_invalid_query_422() -> None:
    app = create_app(Settings(index_path="no-such-file.bin"))
    app.dependency_overrides[get_index] = _index
    with TestClient(app) as client:
        assert client.get("/search", params={"q": "", "k": 0}).status_code == 422
        assert client.get("/search", params={"q": "python", "scorer": "bad"}).status_code == 422
        assert client.get("/search", params={"q": "python", "page": -1}).status_code == 422


def test_unknown_document_404() -> None:
    app = create_app(Settings(index_path="no-such-file.bin"))
    app.dependency_overrides[get_index] = _index
    with TestClient(app) as client:
        assert client.get("/docs/999").status_code == 404


def test_unloaded_health_503() -> None:
    app = create_app(Settings(index_path="no-such-file.bin"))
    with TestClient(app) as client:
        assert client.get("/health").status_code == 503


def test_ui_renders() -> None:
    app = create_app(Settings(index_path="no-such-file.bin"))
    app.dependency_overrides[get_index] = _index
    with TestClient(app) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert "find" in response.text


def test_stats() -> None:
    app = create_app(Settings(index_path="no-such-file.bin"))
    app.dependency_overrides[get_index] = _index
    with TestClient(app) as client:
        assert client.get("/stats").json()["documents"] == 1
