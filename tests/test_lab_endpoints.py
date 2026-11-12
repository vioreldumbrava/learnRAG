"""The lab over HTTP and on the command line: chunk preview and embedding map."""

from __future__ import annotations

import csv
import io

import yaml
from typer.testing import CliRunner

from rag_app import cli as cli_module
from rag_app.models import DocumentChunk

from tests.conftest import FakeEmbeddingProvider, FakeVectorStore
from tests.test_server_endpoints import _fill, api  # noqa: F401  (pytest fixture)


_NOTES = "\n\n".join(
    f"## {n}. Section {n}\n\nParagraph {n} explains register R{n} and why its reset "
    f"value matters when the peripheral is enabled for the first time."
    for n in range(1, 7)
)


def _write_notes(tmp_path):
    documents = tmp_path / "documents"
    documents.mkdir(parents=True, exist_ok=True)
    path = documents / "notes.md"
    path.write_text(_NOTES, encoding="utf-8")
    return path


# ----- chunk preview ---------------------------------------------------------


def test_chunk_preview_by_path_stores_nothing(api, tmp_path):  # noqa: F811
    client, store, _, _ = api
    path = _write_notes(tmp_path)

    response = client.post(
        "/api/chunk-preview",
        json={"path": str(path), "settings": {"chunk_size": 200, "chunk_overlap": 0, "strategy": "heading"}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["file"] == "notes.md" and body["stats"]["count"] == 6
    assert body["settings"] == {"chunk_size": 200, "chunk_overlap": 0, "strategy": "heading"}
    assert body["chunks"][0]["text"].startswith("## 1. Section 1")
    assert store.stats()["count"] == 0
    assert client.get("/api/documents").json()["documents"] == []


def test_chunk_preview_of_an_indexed_document_leaves_the_index_alone(api, tmp_path):  # noqa: F811
    client, store, _, _ = api
    _write_notes(tmp_path)
    assert client.post("/api/ingest", json={}).json()["failed_files"] == []
    indexed = store.stats()["count"]
    document_id = client.get("/api/documents").json()["documents"][0]["document_id"]

    as_configured = client.post("/api/chunk-preview", json={"document_id": document_id}).json()
    smaller = client.post(
        "/api/chunk-preview",
        json={"document_id": document_id, "settings": {"chunk_size": 150, "chunk_overlap": 30}},
    ).json()

    # With the configured settings the preview equals what was indexed.
    assert as_configured["stats"]["count"] == indexed
    assert smaller["stats"]["count"] > indexed
    assert smaller["stats"]["repeated_chars"] > 0
    assert store.stats()["count"] == indexed


def test_chunk_preview_rejects_bad_requests(api, tmp_path):  # noqa: F811
    client, *_ = api
    path = _write_notes(tmp_path)
    outside = tmp_path / "elsewhere.md"
    outside.write_text("Not under the documents folder.", encoding="utf-8")

    assert client.post("/api/chunk-preview", json={}).status_code == 422
    assert client.post("/api/chunk-preview", json={"document_id": "unknown"}).status_code == 404
    assert client.post("/api/chunk-preview", json={"path": str(outside)}).status_code == 403
    bad = client.post(
        "/api/chunk-preview", json={"path": str(path), "settings": {"chunk_size": 50, "chunk_overlap": 50}}
    )
    assert bad.status_code == 422 and "smaller than chunk_size" in bad.json()["detail"]
    unknown = client.post("/api/chunk-preview", json={"path": str(path), "settings": {"top_k": 3}})
    assert unknown.status_code == 422


# ----- embedding map -----------------------------------------------------------


def test_embedding_map_endpoint_places_chunks_and_a_query(api):  # noqa: F811
    client, store, embedder, chat = api
    _fill(store, embedder, n=6)

    plain = client.post("/api/embedding-map", json={}).json()
    assert plain["shown"] == plain["total"] == 6 and plain["dimension"] == embedder.dim
    assert plain["query"] is None and len(plain["points"]) == 6
    # These chunks have no module, so they form one group.
    assert plain["groups"] == [{"name": "(none)", "count": 6, "series": 0}]

    with_query = client.post(
        "/api/embedding-map",
        json={"query": "chunk text number 3", "top_k": 2, "group_by": "source_file"},
    ).json()
    ranked = sorted((p for p in with_query["points"] if "rank" in p), key=lambda p: p["rank"])
    assert [p["rank"] for p in ranked] == [1, 2]
    assert ranked[0]["preview"] == "chunk text number 3"
    assert with_query["groups"] == [{"name": "a.txt", "count": 6, "series": 0}]
    assert chat.received == []  # the map never calls the chat model

    assert client.post("/api/embedding-map", json={"group_by": "colour"}).status_code == 422
    assert client.post("/api/embedding-map", json={"limit": 0}).status_code == 422


# ----- the same two tools on the command line ------------------------------------


def _config(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "chat": {"model": "m", "base_url": "http://x"},
                "embeddings": {"model": "e", "base_url": "http://x"},
                "paths": {"index_file": str(tmp_path / "storage" / "index.json")},
            }
        ),
        encoding="utf-8",
    )
    return config


def test_chunks_command_previews_without_any_model(tmp_path, monkeypatch):
    monkeypatch.setenv("COLUMNS", "200")
    path = _write_notes(tmp_path)
    config = _config(tmp_path)

    def run(*args):
        return CliRunner().invoke(
            cli_module.app, ["chunks", "--path", str(path), *args, "--config", str(config)]
        )

    result = run("--size", "200", "--overlap", "0", "--strategy", "heading", "--show", "1")
    assert result.exit_code == 0, result.output
    assert "chunk_size=200, chunk_overlap=0, strategy=heading" in result.output
    assert "Chunks:    6" in result.output and "no text is stored twice" in result.output
    assert "## 2. Section 2" in result.output  # the table and the --show panel

    overlapped = run("--size", "200", "--overlap", "60")
    assert "characters are stored twice because of overlap" in overlapped.output

    assert run("--size", "50", "--overlap", "50").exit_code == 2
    assert run("--show", "99").exit_code == 2
    assert not (tmp_path / "storage").exists()  # nothing was written


def test_embedding_map_command_describes_the_projection_and_exports_it(tmp_path, monkeypatch):
    monkeypatch.setenv("COLUMNS", "200")
    embedder, store = FakeEmbeddingProvider(), FakeVectorStore()
    texts = {
        "CAN": ["CAN FD bit timing uses the NBRP prescaler", "Error ERR080082 means a phase error"],
        "SPI": ["SPI DMA driver underrun handling", "SPI slave transmit buffer is empty"],
    }
    chunks = [
        DocumentChunk(
            id=f"{module}:r:{index}",
            text=text,
            metadata={"source_file": f"{module.lower()}.md", "module": module, "section": "", "location": ""},
        )
        for module, items in texts.items()
        for index, text in enumerate(items)
    ]
    store.upsert_chunks(chunks, embedder.embed_texts([c.text for c in chunks]))
    monkeypatch.setattr(cli_module, "build_embedding_provider", lambda _: embedder)
    monkeypatch.setattr(cli_module, "_make_vector_store", lambda _: store)
    out = tmp_path / "points.csv"

    result = CliRunner().invoke(
        cli_module.app,
        [
            "embedding-map", "--query", "Error ERR080082 means a phase error",
            "--top-k", "2", "--out", str(out), "--config", str(_config(tmp_path)),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Chunks on the map: 4 of 4" in result.output
    assert "of the variation" in result.output
    assert "Where each module sits" in result.output and "CAN" in result.output
    assert "true rank" in result.output and "rank on map" in result.output
    rows = list(csv.DictReader(io.StringIO(out.read_text(encoding="utf-8"))))
    assert [r["kind"] for r in rows] == ["chunk"] * 4 + ["query"]
    best = next(r for r in rows if r["rank"] == "1")
    assert best["preview"] == "Error ERR080082 means a phase error"

    empty = CliRunner().invoke(cli_module.app, ["embedding-map", "--config", str(_config(tmp_path))])
    assert empty.exit_code == 0
    monkeypatch.setattr(cli_module, "_make_vector_store", lambda _: FakeVectorStore())
    empty = CliRunner().invoke(cli_module.app, ["embedding-map", "--config", str(_config(tmp_path))])
    assert "The index is empty" in empty.output
