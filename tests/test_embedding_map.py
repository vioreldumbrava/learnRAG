"""The embedding map: a 2D projection that still tells the truth about neighbours."""

from __future__ import annotations

import csv
import io
import math

import pytest

from rag_app.analysis.embedding_map import embedding_map, map_csv
from rag_app.models import DocumentChunk
from rag_app.providers.base import EmbeddingProvider

from tests.conftest import FakeVectorStore


class AxisEmbedder(EmbeddingProvider):
    """Maps a text to a hand-made vector, so the geometry is known in advance."""

    provider_name = "axis"
    model_name = "axis"

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.vectors[text] for text in texts]


def _store() -> tuple[FakeVectorStore, AxisEmbedder]:
    """Two tight clusters far apart on the first axis, spread a little on the second.

    The third and fourth dimensions carry almost nothing, so two components
    should explain nearly all of the variation.
    """

    vectors = {
        "can 1": [1.0, 0.10, 0.01, 0.0],
        "can 2": [1.0, 0.20, 0.00, 0.01],
        "can 3": [1.0, 0.30, 0.01, 0.0],
        "spi 1": [-1.0, 0.10, 0.00, 0.01],
        "spi 2": [-1.0, 0.20, 0.01, 0.0],
        "spi 3": [-1.0, 0.30, 0.00, 0.0],
        "about can": [1.0, 0.22, 0.0, 0.0],
    }
    embedder = AxisEmbedder(vectors)
    store = FakeVectorStore()
    chunks = [
        DocumentChunk(
            id=f"d:r:{index}",
            text=text,
            metadata={
                "source_file": f"{text.split()[0]}.md",
                "module": text.split()[0].upper(),
                "section": f"# {text}",
                "location": "",
            },
        )
        for index, text in enumerate(t for t in vectors if t != "about can")
    ]
    store.upsert_chunks(chunks, embedder.embed_texts([c.text for c in chunks]))
    return store, embedder


def test_similar_chunks_land_together_and_the_axes_explain_the_spread():
    store, _ = _store()
    result = embedding_map(store)

    assert result["shown"] == result["total"] == 6 and result["dimension"] == 4
    assert result["groups"] == [
        {"name": "CAN", "count": 3, "series": 0},
        {"name": "SPI", "count": 3, "series": 1},
    ]
    by_group = {
        name: [p["x"] for p in result["points"] if p["group"] == name] for name in ("CAN", "SPI")
    }
    # The first axis separates the two modules completely.
    assert max(by_group["CAN"]) < min(by_group["SPI"]) or min(by_group["CAN"]) > max(by_group["SPI"])
    first, second = result["explained_variance"]
    assert first > 0.9 and first + second > 0.99
    assert first >= second >= 0
    assert all({"id", "x", "y", "file", "section", "preview"} <= set(p) for p in result["points"])
    assert result["query"] is None and all("rank" not in p for p in result["points"])


def test_the_query_is_placed_on_the_map_and_its_true_neighbours_are_ranked():
    store, embedder = _store()
    result = embedding_map(store, embedder, query="about can", top_k=3)

    ranked = sorted((p for p in result["points"] if "rank" in p), key=lambda p: p["rank"])
    assert [p["preview"] for p in ranked] == ["can 2", "can 3", "can 1"]
    assert ranked[0]["similarity"] > ranked[1]["similarity"] >= ranked[2]["similarity"] > 0.9
    # Each true neighbour also says where it ranks by distance on the 2D map.
    # Here the map is faithful (two components explain almost everything).
    assert sorted(p["map_rank"] for p in ranked) == [1, 2, 3]
    # On the map the query sits inside the CAN cluster, far from SPI.
    query = result["query"]
    assert query["text"] == "about can"

    def distance(point):
        return math.hypot(point["x"] - query["x"], point["y"] - query["y"])

    can = [distance(p) for p in result["points"] if p["group"] == "CAN"]
    spi = [distance(p) for p in result["points"] if p["group"] == "SPI"]
    assert max(can) < min(spi)


def test_the_map_does_not_flip_when_the_chunks_arrive_in_another_order():
    """A mirrored axis is equally valid maths, but a map that flips between
    two draws of the same corpus looks like the corpus changed."""

    store, embedder = _store()
    reordered = FakeVectorStore()
    chunks = list(reversed(store.list_chunks()))
    reordered.upsert_chunks(chunks, embedder.embed_texts([c.text for c in chunks]))

    def placed(result):
        return {p["preview"]: (p["x"], p["y"]) for p in result["points"]}

    first, second = placed(embedding_map(store)), placed(embedding_map(reordered))
    assert first.keys() == second.keys()
    for name, (x, y) in first.items():
        assert second[name] == pytest.approx((x, y), abs=1e-3)


def test_group_by_file_limit_and_filter():
    store, _ = _store()

    by_file = embedding_map(store, group_by="source_file")
    assert {g["name"] for g in by_file["groups"]} == {"can.md", "spi.md"}

    sampled = embedding_map(store, limit=3)
    assert sampled["shown"] == 3 and sampled["total"] == 6
    # An evenly spaced sample covers both modules, not just the first documents.
    assert {p["group"] for p in sampled["points"]} == {"CAN", "SPI"}

    only_spi = embedding_map(store, where={"module": "SPI"})
    assert only_spi["total"] == 3 and {p["group"] for p in only_spi["points"]} == {"SPI"}


def test_only_the_largest_groups_get_a_colour_slot_and_keep_it():
    """Three hues is what stays tellable apart for every pair on a scatter."""

    sizes = {"ZED": 4, "ADC": 3, "MID": 2, "TINY": 1, "ALSO": 1}
    texts = [f"{module} {index}" for module, count in sizes.items() for index in range(count)]
    embedder = AxisEmbedder({text: [float(i), 1.0, 0.5] for i, text in enumerate(texts)})
    store = FakeVectorStore()
    chunks = [
        DocumentChunk(id=f"d:r:{i}", text=text, metadata={"source_file": "a.md", "module": text.split()[0]})
        for i, text in enumerate(texts)
    ]
    store.upsert_chunks(chunks, embedder.embed_texts(texts))

    groups = embedding_map(store)["groups"]

    # Listed largest first; the slots are handed out in name order, so which
    # group happens to be the largest does not decide its colour.
    assert [(g["name"], g["count"], g["series"]) for g in groups] == [
        ("ZED", 4, 2), ("ADC", 3, 0), ("MID", 2, 1), ("ALSO", 1, None), ("TINY", 1, None),
    ]


def test_degenerate_indexes_do_not_fail():
    empty = embedding_map(FakeVectorStore())
    assert empty["points"] == [] and empty["shown"] == 0 and empty["explained_variance"] == [0.0, 0.0]

    store = FakeVectorStore()
    store.upsert_chunks(
        [DocumentChunk(id="only", text="one chunk", metadata={"source_file": "a.md"})],
        [[0.3, 0.4, 0.0]],
    )
    single = embedding_map(store)
    assert [(p["x"], p["y"], p["group"]) for p in single["points"]] == [(0.0, 0.0, "(none)")]


def test_bad_arguments_and_a_mismatched_query_model_are_reported():
    store, _ = _store()
    with pytest.raises(ValueError, match="group_by"):
        embedding_map(store, group_by="colour")
    with pytest.raises(ValueError, match="limit"):
        embedding_map(store, limit=0)

    other_model = AxisEmbedder({"about can": [1.0, 0.0]})
    with pytest.raises(ValueError, match="different embedding model"):
        embedding_map(store, other_model, query="about can")


def test_csv_export_has_every_point_and_the_query():
    store, embedder = _store()
    rows = list(csv.DictReader(io.StringIO(map_csv(embedding_map(store, embedder, query="about can", top_k=1)))))

    assert len(rows) == 7 and rows[-1]["kind"] == "query" and rows[-1]["preview"] == "about can"
    assert [r["rank"] for r in rows if r["rank"]] == ["1"]
    assert {r["group"] for r in rows[:-1]} == {"CAN", "SPI"}
