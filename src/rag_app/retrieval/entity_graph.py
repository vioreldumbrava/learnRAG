"""Entity-graph retrieval: look things up by name, and follow what they touch.

With `chunking.entity_graph` on, ingestion stored for every chunk the named
things it mentions and the relationships it states between them (see
`ingestion/entity_extractor.py`). Put together across the corpus, that is a
graph:

    nodes  = entities ("NBRP", "ERR080082", "Adx12_Init")
    edges  = relations between them ("NBRP" -must match-> "DBRP")
    and every node knows which chunks mention it.

At query time (`retrieval.graph`) the question is scanned for entity names it
contains — a plain text match, no model call. Those are the *seeds*. Then:

1. every chunk that mentions a seed is a candidate, weighted by how rare the
   entity is (an entity in three chunks says more than one in three hundred);
2. entities one edge away from a seed bring their chunks too, at half weight.
   That second step is what similarity search cannot do: a chunk about DBRP
   can be the right evidence for a question that only says "NBRP".

The result is one more ranked list. It is merged with the vector (and BM25)
lists by reciprocal rank fusion, the same way hybrid search merges its two.

What this is not: full GraphRAG also clusters the graph into communities and
has the model write a summary of each, which is what lets it answer "what are
the main themes of this corpus?". That part is not implemented; this graph
answers questions about named things.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from threading import RLock

from rag_app.ingestion.entity_extractor import read_entity_metadata
from rag_app.models import RetrievedChunk
from rag_app.utils.naming import display_file_name
from rag_app.validation import matches_filter


logger = logging.getLogger(__name__)

# A related entity (one edge from a seed) counts for this much of a seed.
RELATED_WEIGHT = 0.5
_PREVIEW_CHARS = 160


def _key(name: str) -> str:
    return " ".join(name.lower().split())


@dataclass
class Entity:
    name: str
    type: str
    chunk_ids: list[str] = field(default_factory=list)


class EntityGraph:
    """Entities, their relations and the chunks that mention them."""

    def __init__(self) -> None:
        self.entities: dict[str, Entity] = {}
        # entity key -> {other entity key -> verb phrases}, both directions
        self.edges: dict[str, dict[str, set[str]]] = {}
        self.chunks: dict[str, RetrievedChunk] = {}
        self._patterns: dict[str, re.Pattern] = {}

    # ----- building --------------------------------------------------------------

    def build(self, chunks: list[RetrievedChunk]) -> None:
        self.entities.clear()
        self.edges.clear()
        self.chunks.clear()
        self._patterns.clear()
        for chunk in chunks:
            found = read_entity_metadata(chunk.metadata)
            if not found["entities"]:
                continue
            self.chunks[chunk.id] = chunk
            for name, kind in found["entities"]:
                entity = self.entities.setdefault(_key(name), Entity(name=name, type=kind))
                entity.chunk_ids.append(chunk.id)
            for source, verb, target in found["relations"]:
                a, b = _key(source), _key(target)
                if a in self.entities and b in self.entities and a != b:
                    self.edges.setdefault(a, {}).setdefault(b, set()).add(verb)
                    self.edges.setdefault(b, {}).setdefault(a, set()).add(verb)
        for key, entity in self.entities.items():
            # The name as a whole word or phrase: "CAN" must not match "scan".
            self._patterns[key] = re.compile(
                r"(?<![A-Za-z0-9_])" + re.escape(entity.name) + r"(?![A-Za-z0-9_])", re.IGNORECASE
            )

    def __len__(self) -> int:
        return len(self.entities)

    # ----- querying --------------------------------------------------------------

    def match(self, question: str) -> list[str]:
        """Keys of the entities named in `question`, longest name first.

        A name inside a longer matched name is dropped: "CAN FD" in the
        question matches the entity "CAN FD", not also the entity "CAN".
        """

        spans: list[tuple[int, int, str]] = []
        for key, pattern in self._patterns.items():
            for found in pattern.finditer(question):
                spans.append((found.start(), found.end(), key))
        spans.sort(key=lambda span: (-(span[1] - span[0]), span[0]))
        taken: list[tuple[int, int]] = []
        matched: list[str] = []
        for start, end, key in spans:
            if any(start >= a and end <= b for a, b in taken):
                continue
            taken.append((start, end))
            if key not in matched:
                matched.append(key)
        return matched

    def _rarity(self, key: str) -> float:
        return math.log(1.0 + len(self.chunks) / len(set(self.entities[key].chunk_ids)))

    def search(self, question: str, top_k: int = 20, where: dict | None = None) -> tuple[list[RetrievedChunk], dict]:
        """Chunks that mention the question's entities or their neighbours.

        Returns `(ranked chunks, detail)`; `detail` names the matched and the
        related entities, for the pipeline view.
        """

        seeds = self.match(question)
        related: dict[str, float] = {}
        for seed in seeds:
            for other in self.edges.get(seed, {}):
                if other not in seeds:
                    related[other] = max(related.get(other, 0.0), RELATED_WEIGHT)
        scores: dict[str, float] = {}
        for key, weight in [*((seed, 1.0) for seed in seeds), *related.items()]:
            for chunk_id in set(self.entities[key].chunk_ids):
                scores[chunk_id] = scores.get(chunk_id, 0.0) + weight * self._rarity(key)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        results = []
        for chunk_id, score in ranked:
            chunk = self.chunks[chunk_id]
            if not matches_filter(chunk.metadata, where):
                continue
            results.append(
                RetrievedChunk(
                    id=chunk.id, text=chunk.text, metadata=chunk.metadata,
                    score=round(score, 4), score_type="entity_score",
                )
            )
            if len(results) >= top_k:
                break
        detail = {
            "entities_in_index": len(self.entities),
            "matched": [self.entities[key].name for key in seeds],
            "related": [self.entities[key].name for key in sorted(related)],
        }
        return results, detail

    # ----- looking at the graph ---------------------------------------------------

    def overview(self, limit: int = 50, contains: str = "") -> dict:
        """The most-mentioned entities, for a browser of the graph."""

        wanted = contains.strip().lower()
        rows = [
            {
                "name": entity.name,
                "type": entity.type,
                "chunks": len(set(entity.chunk_ids)),
                "related": len(self.edges.get(key, {})),
            }
            for key, entity in self.entities.items()
            if not wanted or wanted in key
        ]
        rows.sort(key=lambda row: (-row["chunks"], -row["related"], row["name"].lower()))
        return {
            "entities": rows[:limit],
            "total": len(rows),
            "entities_in_index": len(self.entities),
            "chunks_with_entities": len(self.chunks),
            "relations": sum(len(targets) for targets in self.edges.values()) // 2,
        }

    def detail(self, name: str) -> dict:
        """One entity: what it is related to and where it is mentioned."""

        key = _key(name)
        if key not in self.entities:
            raise KeyError(name)
        entity = self.entities[key]
        mentions = []
        for chunk_id in dict.fromkeys(entity.chunk_ids):
            chunk = self.chunks[chunk_id]
            preview = " ".join(chunk.text.split())
            mentions.append(
                {
                    "id": chunk.id,
                    "file": display_file_name(str(chunk.metadata.get("source_file", "?"))),
                    "document_id": str(chunk.metadata.get("document_id", "")),
                    "page": chunk.metadata.get("page_start"),
                    "section": str(chunk.metadata.get("section", "")),
                    "location": str(chunk.metadata.get("location", "")),
                    "preview": preview[: _PREVIEW_CHARS - 1] + "…" if len(preview) > _PREVIEW_CHARS else preview,
                }
            )
        return {
            "name": entity.name,
            "type": entity.type,
            "related": [
                {
                    "name": self.entities[other].name,
                    "type": self.entities[other].type,
                    "relations": sorted(verbs),
                    "chunks": len(set(self.entities[other].chunk_ids)),
                }
                for other, verbs in sorted(self.edges.get(key, {}).items())
            ],
            "chunks": mentions,
        }


# ---------------------------------------------------------------------------
# Cross-request cache (same policy as the BM25 index)
# ---------------------------------------------------------------------------

_GRAPH_CACHE: dict[tuple, EntityGraph] = {}
_GRAPH_LOCK = RLock()


def get_entity_graph(store) -> EntityGraph:
    """Build (or reuse) the entity graph over every chunk in `store`.

    Reading every chunk to assemble the graph is the expensive part, so the
    result is kept until the corpus changes. Raises NotImplementedError when
    the store cannot enumerate its chunks.
    """

    from rag_app.retrieval.bm25 import corpus_cache_key

    with _GRAPH_LOCK:
        key = corpus_cache_key(store)
        if key is not None and key in _GRAPH_CACHE:
            return _GRAPH_CACHE[key]
        graph = EntityGraph()
        graph.build(store.all_chunks())
        logger.info("Entity graph built: %d entities from %d chunks", len(graph), len(graph.chunks))
        if key is not None:
            for stale in [k for k in _GRAPH_CACHE if k[:2] == key[:2]]:
                del _GRAPH_CACHE[stale]
            _GRAPH_CACHE[key] = graph
        return graph
