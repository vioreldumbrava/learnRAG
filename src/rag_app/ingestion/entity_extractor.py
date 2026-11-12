"""Entity extraction at ingest: the raw material for entity-graph retrieval.

Vector search finds passages that *sound like* the question. Some questions
are about a *thing* — a register, an error code, a function — and what you
want is every passage that mentions that thing, plus the things it is tied
to. That is a lookup, not a similarity search, and it needs to know which
things each passage mentions.

So at ingest time the chat model reads each chunk and lists:

    ENTITY: NBRP | register
    ENTITY: CAN FD | protocol
    RELATION: NBRP | must match | DBRP

The result is stored in the chunk's own metadata (`entities`, `entity_types`,
`relations`). Nothing else has to be kept in step: when a document is
replaced or forgotten its chunks go, and their entities with them. The graph
itself is assembled from the stored chunks when a query first needs it (see
`retrieval/entity_graph.py`).

Cost warning: like contextual retrieval, this is one chat call **per chunk**,
paid at ingest. Queries make no extra model call. A failed or unreadable
reply leaves that chunk without entities; the document still ingests.

This is the entity half of Microsoft's GraphRAG. The other half — clustering
the graph into communities and writing a summary of each, so "what are the
main themes?" can be answered — is not implemented here.
"""

from __future__ import annotations

import logging
import re

from rag_app.models import ChatMessage
from rag_app.providers.base import ChatProvider


logger = logging.getLogger(__name__)

MAX_ENTITIES = 8
MAX_RELATIONS = 6
_NAME_CHARS = 60
_CHUNK_CHARS = 3000

_SYSTEM = (
    "You index a technical passage for search. List the specific named things it is about: "
    "registers, bit fields, signals, functions, error codes, part numbers, protocols, "
    "components, modes, files. Use the name exactly as written in the passage. "
    "Do not list generic words such as 'value', 'system' or 'data'.\n"
    "One line per entity, in exactly this form:\n"
    "ENTITY: <name> | <type>\n"
    "Then the relationships the passage states between those entities, if any:\n"
    "RELATION: <entity> | <short verb phrase> | <entity>\n"
    f"At most {MAX_ENTITIES} entities and {MAX_RELATIONS} relations. "
    "If the passage names nothing specific, reply with exactly: NONE"
)
_ENTITY = re.compile(r"^[\s>*`_-]*ENTITY[\s*`_]*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_RELATION = re.compile(r"^[\s>*`_-]*RELATION[\s*`_]*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
# Separators of the stored metadata strings; never allowed inside a name.
_RESERVED = re.compile(r"[|;>\n\r]+")


def _clean(text: str) -> str:
    return " ".join(_RESERVED.sub(" ", text).strip(" \t\"'`*").split())[:_NAME_CHARS]


def parse_extraction(reply: str) -> dict:
    """`{"entities": [(name, type)], "relations": [(a, verb, b)]}` from a reply.

    Lines that do not fit the format are ignored. A relation is kept only
    when both of its ends are listed entities, so the graph never points at
    a name that no passage is known to mention.
    """

    entities: dict[str, tuple[str, str]] = {}
    for match in _ENTITY.finditer(reply or ""):
        name, _, kind = match.group(1).partition("|")
        name, kind = _clean(name), _clean(kind).lower() or "thing"
        if len(name) < 2 or name.lower() in entities:
            continue
        entities[name.lower()] = (name, kind)
        if len(entities) >= MAX_ENTITIES:
            break
    relations: list[tuple[str, str, str]] = []
    for match in _RELATION.finditer(reply or ""):
        parts = [_clean(part) for part in match.group(1).split("|")]
        if len(parts) != 3 or not all(parts):
            continue
        source, verb, target = parts
        if source.lower() not in entities or target.lower() not in entities or source.lower() == target.lower():
            continue
        relations.append((entities[source.lower()][0], verb.lower(), entities[target.lower()][0]))
        if len(relations) >= MAX_RELATIONS:
            break
    return {"entities": list(entities.values()), "relations": relations}


def extract_entities(chunk_texts: list[str], chat_provider: ChatProvider) -> list[dict]:
    """One extraction per chunk (empty on failure), in chunk order."""

    results = []
    for index, text in enumerate(chunk_texts):
        try:
            reply = chat_provider.generate(
                [
                    ChatMessage(role="system", content=_SYSTEM),
                    ChatMessage(role="user", content=f"Passage:\n{text[:_CHUNK_CHARS]}"),
                ],
                temperature=0.0,
                max_tokens=300,
            )
        except Exception:
            logger.warning("Entity extraction failed for chunk %d - stored without entities", index)
            results.append({"entities": [], "relations": []})
            continue
        results.append(parse_extraction(reply))
    return results


def entity_metadata(extraction: dict) -> dict:
    """The extraction as flat metadata strings a vector store can hold.

    Vector stores keep scalar metadata only, so the lists are joined:
    `entities` "NBRP|DBRP", `entity_types` "register|register" (same order),
    `relations` "NBRP>must match>DBRP;...". Empty when nothing was found.
    """

    if not extraction["entities"]:
        return {}
    metadata = {
        "entities": "|".join(name for name, _ in extraction["entities"]),
        "entity_types": "|".join(kind for _, kind in extraction["entities"]),
    }
    if extraction["relations"]:
        metadata["relations"] = ";".join(">".join(relation) for relation in extraction["relations"])
    return metadata


def read_entity_metadata(metadata: dict) -> dict:
    """The inverse of `entity_metadata`, tolerant of chunks without entities."""

    names = [name for name in str(metadata.get("entities") or "").split("|") if name]
    kinds = str(metadata.get("entity_types") or "").split("|")
    entities = [(name, kinds[i] if i < len(kinds) and kinds[i] else "thing") for i, name in enumerate(names)]
    relations = []
    for item in str(metadata.get("relations") or "").split(";"):
        parts = item.split(">")
        if len(parts) == 3 and all(parts):
            relations.append((parts[0], parts[1], parts[2]))
    return {"entities": entities, "relations": relations}
