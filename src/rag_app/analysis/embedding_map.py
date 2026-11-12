"""A flat picture of the embedding space.

Every chunk is a point in a space with hundreds of dimensions; retrieval is
"find the points nearest the question". You cannot look at that space, but you
can project it onto the two directions along which the chunks differ most
(principal component analysis) and plot the result. Chunks about the same
topic land near each other, a question lands among the chunks it resembles,
and the retrieved chunks can be marked on the same picture.

Read it for what it is. Two components keep only a fraction of the variation
(reported as `explained_variance`), so points that look close on the map can be
far apart in the full space and the reverse. That is why the nearest
neighbours are computed in the *full* space and then drawn on the map — when
they do not look nearest in 2D, the map is showing you its own limits.

PCA here is three lines of linear algebra: centre the vectors, take the
singular value decomposition, keep the first two right-singular vectors.
"""

from __future__ import annotations

from rag_app.utils.naming import display_file_name


GROUP_FIELDS = ("module", "source_file", "file_type")
_NO_GROUP = "(none)"
_PREVIEW_CHARS = 160
# A scatter compares every group with every other, and only about three hues
# stay tellable apart for every pair (colour-blind readers included). The
# largest groups get a colour slot each; the rest are drawn as "other".
MAX_SERIES = 3
# Reading every chunk of a large index to draw a scatter is wasteful; beyond
# this many the map shows an evenly spaced sample.
_POOL_LIMIT = 20_000


def embedding_map(
    store,
    embedder=None,
    *,
    query: str | None = None,
    top_k: int = 5,
    limit: int = 1500,
    group_by: str = "module",
    where: dict | None = None,
) -> dict:
    """Project stored chunk vectors (and optionally a query) onto two axes."""

    import numpy as np  # needed only here; keeps start-up of the CLI lean

    if group_by not in GROUP_FIELDS:
        raise ValueError(f"group_by must be one of: {', '.join(GROUP_FIELDS)}")
    if not 1 <= limit <= 5000:
        raise ValueError("limit must be between 1 and 5000")

    pool = store.list_chunks(where=where, limit=_POOL_LIMIT)
    total = len(pool)
    if total > limit:
        step = total / limit
        pool = [pool[int(i * step)] for i in range(limit)]
    vectors = store.embeddings_for_ids([chunk.id for chunk in pool])
    chunks = [chunk for chunk in pool if vectors.get(chunk.id)]

    result = {
        "points": [],
        "groups": [],
        "query": None,
        "explained_variance": [0.0, 0.0],
        "dimension": 0,
        "shown": len(chunks),
        "total": total,
        "group_by": group_by,
    }
    if not chunks:
        return result

    matrix = _unit_rows(np.asarray([vectors[chunk.id] for chunk in chunks], dtype=float))
    mean = matrix.mean(axis=0)
    centred = matrix - mean
    # Rows of `axes` are the directions of greatest variation, strongest first.
    _, singular, axes = np.linalg.svd(centred, full_matrices=False)
    axes = axes[:2]
    if axes.shape[0] < 2:  # a single chunk: nothing varies
        axes = np.vstack([axes, np.zeros((2 - axes.shape[0], matrix.shape[1]))])
    # An axis and its mirror image explain the same variation, and which of the
    # two the decomposition returns depends on row order. Pin the direction
    # (largest component positive) so the same corpus always draws the same way
    # round instead of flipping between two draws.
    for axis in axes:
        if axis[np.argmax(np.abs(axis))] < 0:
            axis *= -1
    coords = centred @ axes.T
    energy = float((singular**2).sum())
    variance = [float(s**2) / energy if energy else 0.0 for s in singular[:2]]

    points = []
    for chunk, (x, y) in zip(chunks, coords):
        preview = " ".join(chunk.text.split())
        points.append(
            {
                "id": chunk.id,
                "x": round(float(x), 4),
                "y": round(float(y), 4),
                "group": str(chunk.metadata.get(group_by) or _NO_GROUP),
                "file": display_file_name(str(chunk.metadata.get("source_file", "?"))),
                "section": str(chunk.metadata.get("section", "")),
                "location": str(chunk.metadata.get("location", "")),
                # Enough to open the original at the right page from a point.
                "document_id": str(chunk.metadata.get("document_id", "")),
                "page": chunk.metadata.get("page_start"),
                "preview": preview[: _PREVIEW_CHARS - 1] + "…"
                if len(preview) > _PREVIEW_CHARS
                else preview,
            }
        )
    if group_by == "source_file":
        for point in points:
            point["group"] = display_file_name(point["group"])

    counts: dict[str, int] = {}
    for point in points:
        counts[point["group"]] = counts.get(point["group"], 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    # Slots go out in name order, not size order, so a group keeps its colour
    # when a few chunks more or less change which group is the largest.
    slots = {name: slot for slot, name in enumerate(sorted(name for name, _ in ordered[:MAX_SERIES]))}
    result.update(
        points=points,
        groups=[{"name": name, "count": count, "series": slots.get(name)} for name, count in ordered],
        explained_variance=[round(v, 4) for v in (variance + [0.0, 0.0])[:2]],
        dimension=int(matrix.shape[1]),
    )

    if query and query.strip() and embedder is not None:
        vector = np.asarray(embedder.embed_query(query), dtype=float)
        if vector.shape[0] != matrix.shape[1]:
            raise ValueError(
                f"The query embedding has {vector.shape[0]} dimensions but the stored "
                f"chunks have {matrix.shape[1]}. The index was built with a different "
                "embedding model."
            )
        unit = _unit_rows(vector.reshape(1, -1))[0]
        x, y = (unit - mean) @ axes.T
        result["query"] = {"text": query, "x": round(float(x), 4), "y": round(float(y), 4)}
        # Nearest in the FULL space (cosine), which is what retrieval uses.
        similarity = matrix @ unit
        order = np.argsort(-similarity)[: max(1, min(top_k, len(points)))]
        # ...and where each of those ranks by distance ON THE MAP. When the two
        # disagree, the projection has hidden or invented closeness.
        on_map = np.argsort(((coords - np.array([x, y])) ** 2).sum(axis=1))
        map_rank = {int(index): rank for rank, index in enumerate(on_map, start=1)}
        for rank, index in enumerate(order, start=1):
            points[int(index)]["rank"] = rank
            points[int(index)]["similarity"] = round(float(similarity[index]), 4)
            points[int(index)]["map_rank"] = map_rank[int(index)]
    return result


def _unit_rows(matrix):
    """Scale every row to length 1, so the map reflects direction (cosine)."""

    import numpy as np

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def map_csv(result: dict) -> str:
    """The map as CSV rows (one per point, then the query if there is one)."""

    import csv
    import io

    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(["kind", "x", "y", "group", "file", "location", "section", "rank", "similarity", "preview"])
    for point in result["points"]:
        writer.writerow(
            [
                "chunk", point["x"], point["y"], point["group"], point["file"],
                point["location"], point["section"], point.get("rank", ""),
                point.get("similarity", ""), point["preview"],
            ]
        )
    if result.get("query"):
        query = result["query"]
        writer.writerow(["query", query["x"], query["y"], "", "", "", "", "", "", query["text"]])
    return stream.getvalue()
