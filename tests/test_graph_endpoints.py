"""The entity graph over HTTP and on the command line."""

from __future__ import annotations

import yaml
from typer.testing import CliRunner

from rag_app import cli as cli_module
from rag_app import server
from rag_app.models import DocumentChunk

from tests.conftest import FakeChatProvider, FakeEmbeddingProvider, FakeVectorStore
from tests.test_server_endpoints import _fill, api  # noqa: F401  (pytest fixture)


_EXTRACTION = (
    "ENTITY: NBRP | register\nENTITY: DBRP | register\nENTITY: CAN FD | protocol\n"
    "RELATION: NBRP | must match | DBRP\nRELATION: NBRP | configures | CAN FD"
)


def _ingest_with_entities(client, chat, monkeypatch, tmp_path):
    """Ingest one file with entity extraction on, the way a user would."""

    documents = tmp_path / "documents"
    documents.mkdir(parents=True, exist_ok=True)
    (documents / "can.md").write_text(
        "NBRP must match DBRP so that CAN FD nodes keep synchronization.", encoding="utf-8"
    )
    monkeypatch.setattr(server._state.cfg.chunking, "entity_graph", True)
    monkeypatch.setattr(chat, "generate", lambda *args, **kwargs: _EXTRACTION)
    result = client.post("/api/ingest", json={})
    assert result.status_code == 200 and result.json()["failed_files"] == [], result.text


def test_graph_is_empty_until_entities_are_extracted(api):  # noqa: F811
    client, store, embedder, _ = api
    _fill(store, embedder, n=3)

    overview = client.get("/api/graph").json()

    assert overview["entities"] == [] and overview["entities_in_index"] == 0
    assert overview["enabled"] is False
    assert client.get("/api/graph", params={"entity": "NBRP"}).status_code == 404


def test_ingest_extracts_entities_and_the_graph_shows_them(api, monkeypatch, tmp_path):  # noqa: F811
    client, store, _, chat = api
    _ingest_with_entities(client, chat, monkeypatch, tmp_path)

    stored = store.list_chunks()[0].metadata
    assert stored["entities"] == "NBRP|DBRP|CAN FD" and "NBRP>must match>DBRP" in stored["relations"]

    overview = client.get("/api/graph").json()
    assert overview["enabled"] is True and overview["entities_in_index"] == 3 and overview["relations"] == 2
    assert overview["entities"][0] == {"name": "NBRP", "type": "register", "chunks": 1, "related": 2}
    assert [e["name"] for e in client.get("/api/graph", params={"contains": "brp"}).json()["entities"]] == ["NBRP", "DBRP"]
    assert client.get("/api/graph", params={"limit": 0}).status_code == 422

    detail = client.get("/api/graph", params={"entity": "nbrp"}).json()
    assert detail["name"] == "NBRP"
    assert [(r["name"], r["relations"]) for r in detail["related"]] == [
        ("CAN FD", ["configures"]), ("DBRP", ["must match"]),
    ]
    assert detail["chunks"][0]["file"] == "can.md" and detail["chunks"][0]["document_id"]


def test_graph_retrieval_is_a_per_question_setting_with_its_own_stage(api, monkeypatch, tmp_path):  # noqa: F811
    client, _, _, chat = api
    _ingest_with_entities(client, chat, monkeypatch, tmp_path)

    controls = {c["field"]: c for c in client.get("/api/retrieval-controls").json()["controls"]}
    assert controls["graph"]["kind"] == "bool" and not controls["graph"]["needs_chat"]

    body = client.post(
        "/api/retrieve", json={"question": "What must NBRP match?", "retrieval": {"graph": True}}
    ).json()
    stages = {stage["name"]: stage for stage in body["trace"]}
    assert stages["entity_graph"]["detail"]["matched"] == ["NBRP"]
    assert stages["entity_graph"]["detail"]["related"] == ["CAN FD", "DBRP"]
    assert stages["entity_graph"]["table"]["headers"][3] == "graph"  # the score column is labelled

    plain = client.post("/api/retrieve", json={"question": "What must NBRP match?"}).json()
    assert "entity_graph" not in [stage["name"] for stage in plain["trace"]]

    presets = {p["name"]: p for p in client.get("/api/experiment-presets").json()["presets"]}
    assert presets["graph"]["settings"] == {"graph": True} and not presets["graph"]["needs_chat"]


# ----- command line ---------------------------------------------------------------------


def _cli(tmp_path, monkeypatch, chunks):
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
    embedder, store = FakeEmbeddingProvider(), FakeVectorStore()
    if chunks:
        store.upsert_chunks(chunks, embedder.embed_texts([c.text for c in chunks]))
    monkeypatch.setattr(cli_module, "build_embedding_provider", lambda _: embedder)
    monkeypatch.setattr(cli_module, "build_chat_provider", lambda _: FakeChatProvider())
    monkeypatch.setattr(cli_module, "_make_vector_store", lambda _: store)
    monkeypatch.setenv("COLUMNS", "200")
    return lambda *args: CliRunner().invoke(cli_module.app, [*args, "--config", str(config)])


def test_graph_command_lists_entities_and_describes_one(tmp_path, monkeypatch):
    chunks = [
        DocumentChunk(
            id="d:r:0",
            text="NBRP must match DBRP.",
            metadata={
                "source_file": "can.md", "chunk_index": 0, "location": "p. 4",
                "entities": "NBRP|DBRP", "entity_types": "register|register", "relations": "NBRP>must match>DBRP",
            },
        ),
        DocumentChunk(
            id="d:r:1",
            text="NBRP sets the nominal bit rate.",
            metadata={"source_file": "can.md", "chunk_index": 1, "entities": "NBRP", "entity_types": "register"},
        ),
    ]
    run = _cli(tmp_path, monkeypatch, chunks)

    listing = run("graph")
    assert listing.exit_code == 0, listing.output
    assert "Entity graph: 2 entities, 1 relations, from 2 chunks" in listing.output
    assert listing.output.index("NBRP") < listing.output.index("DBRP")  # most mentioned first
    filtered = run("graph", "--contains", "dbr").output
    assert "DBRP" in filtered and "NBRP" not in filtered.split("Most-mentioned entities")[1].split("Look at one")[0]

    one = run("graph", "nbrp")
    assert one.exit_code == 0, one.output
    assert "NBRP  (register)" in one.output and "must match" in one.output
    assert "p. 4" in one.output and "NBRP sets the nominal bit rate." in one.output

    missing = run("graph", "watchdog")
    assert missing.exit_code == 2 and "No entity named 'watchdog'" in missing.output

    stats = run("stats")
    assert "Entity extraction (ingest)" in stats.output and "Entity-graph retrieval" in stats.output


def test_graph_command_explains_an_index_without_entities(tmp_path, monkeypatch):
    chunk = DocumentChunk(id="d:r:0", text="Plain text.", metadata={"source_file": "a.md"})
    run = _cli(tmp_path, monkeypatch, [chunk])

    result = run("graph")

    assert result.exit_code == 0 and "The index has no entities." in result.output
    assert "chunking.entity_graph: true" in result.output
