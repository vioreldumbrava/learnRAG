/* The Lab panel: chunking playground and embedding map.
   Everything from the server is placed with textContent / attributes; no
   markup is ever built from strings. */

const SVG = "http://www.w3.org/2000/svg";
/* Scatter is an all-pairs form: three hues stay distinguishable for every
   pair, including for colour-blind readers, on this dark surface. A fourth
   group and beyond are drawn in neutral grey as "Other". */
const SERIES = ["var(--series-1)", "var(--series-2)", "var(--series-3)"];
const OTHER = "var(--series-other)";
/* Dot radius. The 2px surface ring is drawn half inside the dot, so the
   coloured part that remains is 8px across. */
const DOT = 5;

function el(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = String(text);
  if (className) node.className = className;
  return node;
}
function svg(tag, attributes = {}) {
  const node = document.createElementNS(SVG, tag);
  for (const [name, value] of Object.entries(attributes))
    node.setAttribute(name, value);
  return node;
}

/* ---- Chunking playground ------------------------------------------------- */

const PLAYGROUND_FIELDS = [
  { name: "chunk_size", label: "Chunk size (chars)", min: 50, max: 20000 },
  { name: "chunk_overlap", label: "Overlap (chars)", min: 0, max: 10000 },
  {
    name: "strategy",
    label: "Strategy",
    choices: ["paragraph", "heading", "semantic"],
  },
  { name: "code_max_chars", label: "Code chunk limit", min: 200, max: 20000 },
];

/* One column of settings; `read()` returns the values to try. */
export function playgroundSettings(container, prefix, defaults) {
  const inputs = {};
  container.replaceChildren();
  for (const field of PLAYGROUND_FIELDS) {
    const label = el("label", undefined, "technique");
    const input = field.choices ? el("select") : el("input");
    input.id = `${prefix}-${field.name}`;
    if (field.choices) {
      for (const choice of field.choices) {
        const option = el("option", choice);
        option.value = choice;
        input.append(option);
      }
    } else {
      input.type = "number";
      input.min = field.min;
      input.max = field.max;
    }
    input.value = defaults[field.name];
    label.append(el("span", field.label), input);
    container.append(label);
    inputs[field.name] = input;
  }
  return {
    /* Fill the column from a lab or preset; fields it does not name stay. */
    apply(settings) {
      for (const [name, value] of Object.entries(settings ?? {}))
        if (inputs[name]) inputs[name].value = value;
    },
    read() {
      const settings = {};
      for (const field of PLAYGROUND_FIELDS) {
        const raw = inputs[field.name].value;
        if (field.choices) {
          settings[field.name] = raw;
          continue;
        }
        const value = Number(raw);
        if (raw.trim() === "" || !Number.isInteger(value))
          throw new Error(`${field.label} must be a whole number`);
        settings[field.name] = value;
      }
      return settings;
    },
  };
}

/* A row of bars, one per chunk, height = size. Shows at a glance whether a
   setting produces even chunks or a few giants and many crumbs. */
function sizeStrip(chunks, limit) {
  const width = 520,
    height = 72,
    base = height - 4;
  const figure = el("div", undefined, "viz size-strip");
  const drawing = svg("svg", {
    viewBox: `0 0 ${width} ${height}`,
    role: "img",
    "aria-label": `Sizes of the ${chunks.length} chunks; the list below has every value`,
  });
  const top = Math.max(limit, ...chunks.map((chunk) => chunk.chars), 1);
  const step = width / chunks.length;
  const gap = step > 6 ? 2 : 0;
  // The size limit, as a recessive reference line.
  const limitY = base - (limit / top) * (base - 6);
  drawing.append(
    svg("line", { x1: 0, x2: width, y1: limitY, y2: limitY, class: "viz-rule" }),
    svg("line", { x1: 0, x2: width, y1: base, y2: base, class: "viz-axis" }),
  );
  const bars = chunks.map((chunk, index) => {
    const barHeight = Math.max(1, (chunk.chars / top) * (base - 6));
    const bar = svg("rect", {
      x: index * step + gap / 2,
      y: base - barHeight,
      width: Math.max(1, step - gap),
      height: barHeight,
      rx: Math.min(4, (step - gap) / 2),
      class: "viz-bar",
    });
    drawing.append(bar);
    return bar;
  });
  const tip = el("div", undefined, "viz-tip");
  tip.hidden = true;
  drawing.addEventListener("pointermove", (event) => {
    const box = drawing.getBoundingClientRect();
    const index = Math.min(
      chunks.length - 1,
      Math.max(0, Math.floor(((event.clientX - box.left) / box.width) * chunks.length)),
    );
    bars.forEach((bar, at) => bar.classList.toggle("active", at === index));
    const chunk = chunks[index];
    tip.replaceChildren(
      el("strong", `${chunk.chars} chars`),
      el(
        "span",
        ` chunk ${chunk.index}${chunk.overlap ? `, ${chunk.overlap} repeated` : ""}`,
      ),
    );
    tip.hidden = false;
    tip.style.left = `${Math.min(event.clientX - box.left + 12, box.width - 150)}px`;
    tip.style.top = "0px";
  });
  drawing.addEventListener("pointerleave", () => {
    tip.hidden = true;
    bars.forEach((bar) => bar.classList.remove("active"));
  });
  figure.append(drawing, tip);
  return figure;
}

export function renderChunkPreview(container, title, result) {
  const stats = result.stats;
  const column = el("div");
  const settings = Object.entries(result.settings)
    .map(([name, value]) => `${name} ${value}`)
    .join(", ");
  column.append(
    el("h3", `${title}: ${stats.count} chunk(s)`),
    el("p", settings, "muted"),
    el(
      "p",
      `Smallest ${stats.smallest}, mean ${stats.mean}, largest ${stats.largest} characters. ` +
        (stats.repeated_chars
          ? `${stats.repeated_chars} characters are stored twice because of overlap (${Math.round((100 * stats.repeated_chars) / Math.max(stats.stored_chars, 1))}% of what is stored).`
          : "No text is stored twice."),
    ),
  );
  const limit =
    result.settings.chunk_size ?? result.settings.code_max_chars ?? stats.largest;
  if (result.chunks.length > 1 && result.chunks.length <= 260)
    column.append(sizeStrip(result.chunks, limit));
  for (const chunk of result.chunks) {
    const details = el("details", undefined, "chunk");
    const where = [chunk.location, chunk.section].filter(Boolean).join(" | ");
    details.append(
      el(
        "summary",
        `#${chunk.index} | ${chunk.chars} chars${chunk.overlap ? ` | ${chunk.overlap} repeated` : ""}${where ? ` | ${where}` : ""}`,
      ),
    );
    const text = el("pre");
    if (chunk.overlap) {
      // The part this chunk repeats from the one before it.
      const repeated = el("span", chunk.text.slice(0, chunk.overlap), "overlap-text");
      repeated.title = "Repeated from the previous chunk (overlap)";
      text.append(repeated);
    }
    text.append(document.createTextNode(chunk.text.slice(chunk.overlap)));
    details.append(text);
    column.append(details);
  }
  container.append(column);
}

/* ---- Entity graph -------------------------------------------------------- */

function plainTable(headers, rows) {
  const table = el("table", undefined, "stage-table"),
    head = el("tr");
  for (const title of headers) head.append(el("th", title));
  table.append(head);
  for (const cells of rows) {
    const row = el("tr");
    for (const cell of cells) {
      const td = el("td");
      if (cell instanceof Node) td.append(cell);
      else td.textContent = String(cell);
      row.append(td);
    }
    table.append(row);
  }
  const wrap = el("div", undefined, "table-wrap");
  wrap.append(table);
  return wrap;
}
function actionButton(text, action) {
  const node = el("button", text);
  node.type = "button";
  node.addEventListener("click", action);
  return node;
}

/* The most-mentioned entities; `openEntity(name)` shows one of them. */
export function renderGraphOverview(container, overview, openEntity) {
  container.replaceChildren();
  if (!overview.entities_in_index) {
    container.append(
      el(
        "p",
        overview.enabled
          ? "No entities yet. Run ingestion again: files indexed before the setting was turned on have none."
          : "The index has no entities. Set chunking.entity_graph: true in the configuration and ingest again (one chat call per chunk).",
        "muted",
      ),
    );
    return;
  }
  container.append(
    el(
      "p",
      `${overview.entities_in_index} entities and ${overview.relations} relations, from ${overview.chunks_with_entities} chunks.` +
        (overview.total > overview.entities.length
          ? ` Showing the ${overview.entities.length} most mentioned of ${overview.total}.`
          : ""),
    ),
    plainTable(
      ["Entity", "Type", "Chunks", "Related", ""],
      overview.entities.map((entity) => [
        entity.name,
        entity.type,
        entity.chunks,
        entity.related,
        actionButton("Look at it", () => openEntity(entity.name)),
      ]),
    ),
  );
}

/* One entity: its neighbours in the graph and the chunks that mention it. */
export function renderGraphEntity(container, detail, openEntity, openChunk) {
  container.replaceChildren(el("h3", `${detail.name} (${detail.type})`));
  if (detail.related.length)
    container.append(
      el("p", "Related entities, one step away. Their chunks are brought in at half weight.", "muted"),
      plainTable(
        ["Entity", "Type", "Relation", "Chunks", ""],
        detail.related.map((other) => [
          other.name,
          other.type,
          other.relations.join("; "),
          other.chunks,
          actionButton("Look at it", () => openEntity(other.name)),
        ]),
      ),
    );
  else container.append(el("p", "No relations were extracted for it.", "muted"));
  container.append(
    el("p", "Chunks that mention it:", "muted"),
    plainTable(
      ["File", "Where", "Preview", ""],
      detail.chunks.map((chunk) => [
        chunk.file,
        chunk.location || chunk.section,
        chunk.preview,
        actionButton("View", () => openChunk(chunk)),
      ]),
    ),
  );
}

/* ---- Embedding map ------------------------------------------------------- */

function groupColors(groups) {
  // The server hands each of the largest groups a colour slot (`series`);
  // every other group is drawn in the neutral "other" grey.
  const slots = new Map(groups.map((group) => [group.name, group.series]));
  return (name) => SERIES[slots.get(name)] ?? OTHER;
}

export function renderEmbeddingMap(container, result, openChunk) {
  container.replaceChildren();
  if (!result.points.length) {
    container.append(el("p", "The index is empty. Add documents first.", "muted"));
    return;
  }
  const [first, second] = result.explained_variance;
  const percent = (value) => `${Math.round(value * 100)}%`;
  container.append(
    el(
      "p",
      `${result.shown} of ${result.total} chunks. The map shows ${percent(first + second)} of how the chunks differ ` +
        `(axis 1: ${percent(first)}, axis 2: ${percent(second)}); the rest of the ${result.dimension} dimensions is not visible here, ` +
        "so closeness on the map is a hint, not a measurement.",
    ),
  );

  const colorOf = groupColors(result.groups);
  // Drawn at the size it is shown, so a dot is 8px and a label 12px on a
  // phone as well as on a wide screen instead of shrinking with the picture.
  const width = Math.max(300, Math.min(container.clientWidth || 760, 1100)),
    height = Math.round(Math.max(280, Math.min(width * 0.62, 620))),
    margin = width < 500 ? 26 : 40;
  const xs = result.points.map((p) => p.x),
    ys = result.points.map((p) => p.y);
  if (result.query) {
    xs.push(result.query.x);
    ys.push(result.query.y);
  }
  const minX = Math.min(...xs),
    maxX = Math.max(...xs),
    minY = Math.min(...ys),
    maxY = Math.max(...ys);
  // One scale for both axes, so distance on screen means the same in every
  // direction (the whole point of the picture is distance).
  const scale = Math.min(
    (width - 2 * margin) / Math.max(maxX - minX, 1e-9),
    (height - 2 * margin) / Math.max(maxY - minY, 1e-9),
  );
  const offsetX = (width - (maxX - minX) * scale) / 2,
    offsetY = (height - (maxY - minY) * scale) / 2;
  const sx = (x) => offsetX + (x - minX) * scale,
    sy = (y) => height - offsetY - (y - minY) * scale;

  const figure = el("div", undefined, "viz map");
  const drawing = svg("svg", {
    viewBox: `0 0 ${width} ${height}`,
    role: "img",
    "aria-label":
      "Chunks projected onto two axes; the table below lists every point",
  });
  // The origin lines: the centre of the corpus on each axis.
  if (minX < 0 && maxX > 0)
    drawing.append(svg("line", { x1: sx(0), x2: sx(0), y1: 8, y2: height - 8, class: "viz-rule" }));
  if (minY < 0 && maxY > 0)
    drawing.append(svg("line", { x1: 8, x2: width - 8, y1: sy(0), y2: sy(0), class: "viz-rule" }));
  const axisX = svg("text", { x: width - 8, y: height - 8, class: "viz-label", "text-anchor": "end" });
  axisX.textContent = `axis 1 (${percent(first)} of the variation)`;
  const axisY = svg("text", { x: 8, y: 16, class: "viz-label" });
  axisY.textContent = `axis 2 (${percent(second)})`;
  drawing.append(axisX, axisY);

  const marks = result.points.map((point) => {
    const mark = svg("circle", {
      cx: sx(point.x),
      cy: sy(point.y),
      r: DOT,
      fill: colorOf(point.group),
      class: "viz-dot",
    });
    drawing.append(mark);
    return mark;
  });
  // The true nearest chunks: a ring, and a rank number on the first few.
  const ranked = result.points
    .map((point, index) => ({ point, index }))
    .filter((entry) => entry.point.rank)
    .sort((a, b) => a.point.rank - b.point.rank);
  for (const { point } of ranked) {
    drawing.append(
      svg("circle", { cx: sx(point.x), cy: sy(point.y), r: 10, class: "viz-ring" }),
    );
    if (point.rank <= 5) {
      const label = svg("text", { x: sx(point.x) + 13, y: sy(point.y) + 4, class: "viz-mark-label" });
      label.textContent = String(point.rank);
      drawing.append(label);
    }
  }
  if (result.query) {
    const x = sx(result.query.x),
      y = sy(result.query.y);
    drawing.append(
      svg("path", {
        d: `M ${x} ${y - 9} L ${x + 9} ${y} L ${x} ${y + 9} L ${x - 9} ${y} Z`,
        class: "viz-query",
      }),
    );
    const label = svg("text", { x: x + 13, y: y - 8, class: "viz-mark-label" });
    label.textContent = "question";
    drawing.append(label);
  }

  // Hover: the nearest point within reach, so an 8px dot is not an 8px target.
  const tip = el("div", undefined, "viz-tip");
  tip.hidden = true;
  let active = -1;
  const nearest = (event) => {
    const box = drawing.getBoundingClientRect();
    const px = ((event.clientX - box.left) / box.width) * width,
      py = ((event.clientY - box.top) / box.height) * height;
    let best = -1,
      bestDistance = 24 * (width / box.width);
    result.points.forEach((point, index) => {
      const distance = Math.hypot(sx(point.x) - px, sy(point.y) - py);
      if (distance < bestDistance) {
        bestDistance = distance;
        best = index;
      }
    });
    return { best, box };
  };
  drawing.addEventListener("pointermove", (event) => {
    const { best, box } = nearest(event);
    if (active !== -1) marks[active].setAttribute("r", DOT);
    active = best;
    if (best === -1) {
      tip.hidden = true;
      drawing.style.cursor = "";
      return;
    }
    const point = result.points[best];
    marks[best].setAttribute("r", DOT + 2);
    drawing.style.cursor = "pointer";
    tip.replaceChildren(
      el("strong", point.file),
      el("span", ` ${[point.location, point.section].filter(Boolean).join(" | ")}`),
      el("div", `${result.group_by}: ${point.group}${point.rank ? ` | nearest #${point.rank} (similarity ${point.similarity.toFixed(3)})` : ""}`, "muted"),
      el("div", point.preview),
    );
    tip.hidden = false;
    const left = event.clientX - box.left + 14;
    tip.style.left = `${Math.max(0, Math.min(left, box.width - 300))}px`;
    tip.style.top = `${event.clientY - box.top + 14}px`;
  });
  drawing.addEventListener("pointerleave", () => {
    if (active !== -1) marks[active].setAttribute("r", DOT);
    active = -1;
    tip.hidden = true;
  });
  drawing.addEventListener("click", (event) => {
    const { best } = nearest(event);
    if (best !== -1) openChunk(result.points[best]);
  });
  figure.append(drawing, tip);

  // Legend: identity is never colour alone - every group is named here.
  const legend = el("ul", undefined, "viz-legend");
  const coloured = result.groups.filter((group) => SERIES[group.series]);
  const folded = result.groups.filter((group) => !SERIES[group.series]);
  const entry = (swatchClass, style, text) => {
    const item = el("li"),
      swatch = el("span", undefined, `viz-swatch ${swatchClass}`);
    if (style) swatch.style.background = style;
    item.append(swatch, el("span", text));
    legend.append(item);
  };
  for (const group of [...coloured].sort((a, b) => a.series - b.series))
    entry("", colorOf(group.name), `${group.name} (${group.count})`);
  if (folded.length)
    entry(
      "",
      OTHER,
      `Other: ${folded.map((group) => `${group.name} (${group.count})`).join(", ")}`,
    );
  if (result.query) {
    entry("diamond", "", "your question");
    entry("ring", "", "truly nearest chunks, numbered");
  }
  container.append(figure, legend);

  if (ranked.length) {
    container.append(
      el("h3", "Nearest chunks: all dimensions against the map"),
      el(
        "p",
        "True rank uses every dimension, which is what retrieval does. Rank on the map uses only the two drawn. Where they differ, the picture is misleading at that spot.",
        "muted",
      ),
    );
    const table = el("table", undefined, "stage-table");
    const head = el("tr");
    for (const title of ["True rank", "Rank on map", "Similarity", "File", "Where", "Preview"])
      head.append(el("th", title));
    table.append(head);
    for (const { point } of ranked) {
      const row = el("tr");
      for (const [column, text] of [
        ["rank", `#${point.rank}`],
        ["rank", `#${point.map_rank}`],
        ["rank", point.similarity.toFixed(3)],
        ["file", point.file],
        ["section", point.location || point.section],
        ["preview", point.preview],
      ]) {
        const cell = el("td", text);
        cell.dataset.col = column;
        row.append(cell);
      }
      table.append(row);
    }
    const wrap = el("div", undefined, "table-wrap");
    wrap.append(table);
    container.append(wrap);
  }

  // The table view: every point without needing a pointer.
  const all = el("details");
  all.append(el("summary", `View all ${result.points.length} points as a table`));
  const table = el("table", undefined, "stage-table");
  const head = el("tr");
  for (const title of ["File", "Where", result.group_by, "Axis 1", "Axis 2", "Passage"])
    head.append(el("th", title));
  table.append(head);
  for (const point of result.points) {
    const row = el("tr");
    for (const [column, text] of [
      ["file", point.file],
      ["section", point.location || point.section],
      ["rank", point.group],
      ["rank", point.x.toFixed(3)],
      ["rank", point.y.toFixed(3)],
    ]) {
      const cell = el("td", text);
      cell.dataset.col = column;
      row.append(cell);
    }
    const action = el("td"),
      view = el("button", "View");
    view.type = "button";
    view.addEventListener("click", () => openChunk(point));
    action.append(view);
    row.append(action);
    table.append(row);
  }
  const wrap = el("div", undefined, "table-wrap");
  wrap.append(table);
  all.append(wrap);
  container.append(all);
}
