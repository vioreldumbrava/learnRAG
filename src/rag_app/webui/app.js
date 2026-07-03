/* local-rag-learning web UI.
 *
 * Vanilla JS against the same-origin REST API. Everything received from the
 * server is rendered with textContent (never innerHTML) — retrieved chunks
 * and LLM output are untrusted text.
 */
"use strict";

const $ = (sel) => document.querySelector(sel);

// ---------------------------------------------------------------------------
// Tabs
// ---------------------------------------------------------------------------

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
    tab.classList.add("active");
    $(`#panel-${tab.dataset.panel}`).classList.add("active");
    if (tab.dataset.panel === "documents") refreshDocuments();
    if (tab.dataset.panel === "stats") refreshStats();
    if (tab.dataset.panel === "config") refreshConfig();
  });
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function setStatus(id, text, isError = false) {
  const el = $(id);
  el.textContent = text;
  el.classList.toggle("error", isError);
}

function clearChildren(el) {
  while (el.firstChild) el.removeChild(el.firstChild);
}

function td(text, className) {
  const cell = document.createElement("td");
  cell.textContent = text;
  if (className) cell.className = className;
  return cell;
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body.detail) detail = String(body.detail);
    } catch (e) { /* non-JSON error body */ }
    throw new Error(detail);
  }
  return response.json();
}

// Parse "module=CAN,vendor=NXP" into a Chroma where-clause (same rules as
// the CLI/GUI: single pair stays flat, multiple pairs need $and).
function parseFilter(text) {
  const pairs = {};
  for (const part of text.split(",")) {
    const idx = part.indexOf("=");
    if (idx <= 0) continue;
    pairs[part.slice(0, idx).trim()] = part.slice(idx + 1).trim();
  }
  const keys = Object.keys(pairs);
  if (keys.length === 0) return null;
  if (keys.length === 1) return pairs;
  return { $and: keys.map((k) => ({ [k]: pairs[k] })) };
}

// ---------------------------------------------------------------------------
// Health dot
// ---------------------------------------------------------------------------

(async () => {
  const dot = $("#health");
  try {
    await fetchJson("/health");
    dot.classList.add("ok");
    dot.title = "API reachable";
  } catch (e) {
    dot.classList.add("bad");
    dot.title = "API unreachable";
  }
})();

// ---------------------------------------------------------------------------
// Ask
// ---------------------------------------------------------------------------

const history = []; // [{role, content}] — sent back on every turn.

function updateHistoryLabel() {
  const turns = history.length / 2;
  $("#history-label").textContent = turns ? `History: ${turns} turn(s)` : "";
}

// Debug and Stream are mutually exclusive: the server streams whenever
// stream=true and ignores debug on that path.
$("#stream").addEventListener("change", () => {
  if ($("#stream").checked) $("#debug").checked = false;
});
$("#debug").addEventListener("change", () => {
  if ($("#debug").checked) $("#stream").checked = false;
});

$("#clear-history").addEventListener("click", () => {
  history.length = 0;
  updateHistoryLabel();
  setStatus("#ask-status", "History cleared.");
});

function renderSources(sources) {
  const tbody = $("#sources tbody");
  clearChildren(tbody);
  (sources || []).forEach((s, i) => {
    const row = document.createElement("tr");
    row.appendChild(td(String(i + 1), "num"));
    row.appendChild(td(String(s.file)));
    row.appendChild(td(s.section || ""));
    row.appendChild(td(String(s.chunk_index), "num"));
    row.appendChild(td(s.score == null ? "n/a" : Number(s.score).toFixed(4), "num"));
    tbody.appendChild(row);
  });
}

function renderDebug(debug) {
  const block = $("#debug-block");
  if (!debug) {
    block.hidden = true;
    return;
  }
  block.hidden = false;
  $("#debug-info").textContent = Object.entries(debug)
    .map(([k, v]) => `${k}: ${v}`)
    .join("\n");
}

async function streamAnswer(body, answerEl) {
  const response = await fetch("/api/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let answer = "";

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let sep;
    while ((sep = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      if (!frame.startsWith("data: ")) continue;
      const data = frame.slice(6);
      if (data === "[DONE]") continue;
      const event = JSON.parse(data);
      if (event.sources) renderSources(event.sources);
      if (event.token !== undefined) {
        answer += event.token;
        answerEl.textContent = answer;
      }
    }
  }
  return answer;
}

async function ask() {
  const question = $("#question").value.trim();
  if (!question) return;

  const stream = $("#stream").checked;
  const debug = $("#debug").checked;
  const answerEl = $("#answer");

  $("#ask").disabled = true;
  setStatus("#ask-status", stream ? "Retrieving + streaming..." : "Thinking...");
  answerEl.textContent = "";
  renderSources([]);
  renderDebug(null);

  const body = {
    question,
    stream,
    debug,
    filter: parseFilter($("#filter").value),
    history: history.length ? history : null,
  };

  try {
    let answer;
    if (stream) {
      answer = await streamAnswer(body, answerEl);
    } else {
      const result = await fetchJson("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      answer = result.answer;
      answerEl.textContent = answer;
      renderSources(result.sources);
      renderDebug(result.debug);
    }
    history.push({ role: "user", content: question });
    history.push({ role: "assistant", content: answer });
    updateHistoryLabel();
    setStatus("#ask-status", "");
  } catch (e) {
    setStatus("#ask-status", `Error: ${e.message}`, true);
  } finally {
    $("#ask").disabled = false;
  }
}

$("#ask").addEventListener("click", ask);
$("#question").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) ask();
});

// ---------------------------------------------------------------------------
// Ingest
// ---------------------------------------------------------------------------

function resultList(title, items, className) {
  const container = document.createElement("div");
  const heading = document.createElement("h2");
  heading.textContent = `${title} (${items.length})`;
  container.appendChild(heading);
  const list = document.createElement("ul");
  list.className = "result-list";
  for (const item of items) {
    const li = document.createElement("li");
    li.className = className;
    li.textContent = Array.isArray(item) ? `${item[0]} — ${item[1]}` : String(item);
    list.appendChild(li);
  }
  container.appendChild(list);
  return container;
}

$("#run-ingest").addEventListener("click", async () => {
  $("#run-ingest").disabled = true;
  setStatus("#ingest-status", "Ingesting... (embedding can take a while)");
  clearChildren($("#ingest-results"));
  const path = $("#ingest-path").value.trim();
  try {
    const result = await fetchJson("/api/ingest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ force: $("#force").checked, path: path || null }),
    });
    setStatus(
      "#ingest-status",
      `Done: ${result.total_chunks} chunk(s) total, embedding dim ${result.embedding_dim ?? "n/a"}.`,
    );
    const results = $("#ingest-results");
    if (result.indexed_files.length)
      results.appendChild(resultList("Indexed", result.indexed_files, "tag-indexed"));
    if (result.skipped_files.length)
      results.appendChild(resultList("Skipped (unchanged)", result.skipped_files, "tag-skipped"));
    if (result.failed_files.length)
      results.appendChild(resultList("Failed", result.failed_files, "tag-failed"));
  } catch (e) {
    setStatus("#ingest-status", `Error: ${e.message}`, true);
  } finally {
    $("#run-ingest").disabled = false;
  }
});

// ---------------------------------------------------------------------------
// Documents
// ---------------------------------------------------------------------------

async function refreshDocuments() {
  setStatus("#documents-status", "Loading...");
  $("#select-all").checked = false;
  try {
    const result = await fetchJson("/api/documents");
    const tbody = $("#documents tbody");
    clearChildren(tbody);
    for (const doc of result.documents) {
      const row = document.createElement("tr");

      const checkCell = document.createElement("td");
      const check = document.createElement("input");
      check.type = "checkbox";
      check.className = "doc-check";
      check.value = doc.path;
      checkCell.appendChild(check);
      row.appendChild(checkCell);

      row.appendChild(td(doc.source_file));
      row.appendChild(td(doc.path));
      row.appendChild(td(String(doc.chunks), "num"));
      row.appendChild(td(doc.document_hash.slice(0, 12)));

      const actions = document.createElement("td");
      const inspectBtn = document.createElement("button");
      inspectBtn.textContent = "Inspect";
      inspectBtn.className = "secondary mini";
      inspectBtn.addEventListener("click", () => inspectDocument(doc.source_file));
      const forgetBtn = document.createElement("button");
      forgetBtn.textContent = "Forget";
      forgetBtn.className = "danger mini";
      forgetBtn.addEventListener("click", () => forgetDocuments([doc.path]));
      actions.appendChild(inspectBtn);
      actions.appendChild(forgetBtn);
      row.appendChild(actions);
      tbody.appendChild(row);
    }
    setStatus(
      "#documents-status",
      result.documents.length ? `${result.documents.length} document(s)` : "No documents ingested yet.",
    );
  } catch (e) {
    setStatus("#documents-status", `Error: ${e.message}`, true);
  }
}

$("#select-all").addEventListener("change", (e) => {
  document.querySelectorAll(".doc-check").forEach((c) => { c.checked = e.target.checked; });
});

$("#forget-selected").addEventListener("click", () => {
  const paths = Array.from(document.querySelectorAll(".doc-check:checked")).map((c) => c.value);
  if (!paths.length) {
    setStatus("#documents-status", "Select one or more documents first.");
    return;
  }
  forgetDocuments(paths);
});

async function forgetDocuments(paths) {
  const label = paths.length === 1 ? `"${paths[0]}"` : `${paths.length} documents`;
  if (!confirm(`Forget ${label}?\n\nChunks are removed from the vector store; the files on disk are untouched and re-ingesting brings them back.`)) return;
  try {
    for (const path of paths) {
      await fetchJson(`/api/documents?path=${encodeURIComponent(path)}`, { method: "DELETE" });
    }
    await refreshDocuments();
  } catch (e) {
    setStatus("#documents-status", `Error: ${e.message}`, true);
  }
}

function renderChunks(chunks) {
  if (!chunks.length) {
    $("#chunks-view").textContent = "(no chunks)";
    return;
  }
  $("#chunks-view").textContent = chunks
    .map((c, i) =>
      `[${i + 1}] id=${c.id}\n` +
      `    source_file=${c.source_file}  chunk_index=${c.chunk_index}` +
      `  section=${c.section || "-"}  module=${c.module || "-"}\n` +
      `    ${c.text.trim()}`)
    .join("\n\n");
}

async function inspectDocument(sourceFile) {
  setStatus("#documents-status", `Loading chunks for ${sourceFile}...`);
  try {
    const result = await fetchJson(`/api/chunks?source_file=${encodeURIComponent(sourceFile)}&limit=50`);
    renderChunks(result.chunks);
    setStatus("#documents-status", `${result.chunks.length} chunk(s) from ${sourceFile}`);
  } catch (e) {
    setStatus("#documents-status", `Error: ${e.message}`, true);
  }
}

$("#show-random").addEventListener("click", async () => {
  const n = Number($("#sample-n").value) || 5;
  setStatus("#documents-status", "Sampling chunks...");
  try {
    const result = await fetchJson(`/api/chunks?sample=${n}`);
    renderChunks(result.chunks);
    setStatus("#documents-status", `${result.chunks.length} random chunk(s)`);
  } catch (e) {
    setStatus("#documents-status", `Error: ${e.message}`, true);
  }
});

$("#refresh-documents").addEventListener("click", refreshDocuments);

// ---------------------------------------------------------------------------
// Stats  (store stats + a summary of the effective config)
// ---------------------------------------------------------------------------

function featuresSummary(r) {
  const reranker = r.reranker_model ? `${r.reranker_backend}:${r.reranker_model}` : "off";
  return [
    `hybrid: ${r.hybrid ? "on" : "off"}`,
    `HyDE: ${r.use_hyde ? "on" : "off"}`,
    `decompose: ${r.query_decomposition ? "on" : "off"}`,
    `MMR: ${r.use_mmr ? "on" : "off"}`,
    `reranker: ${reranker}`,
    `multi-query: ${r.multi_query || "off"}`,
    `neighbors: ±${r.neighbor_radius}`,
  ].join("  |  ");
}

async function refreshStats() {
  setStatus("#stats-status", "Loading...");
  try {
    const [stats, cfg] = await Promise.all([
      fetchJson("/api/stats"),
      fetchJson("/api/config"),
    ]);
    const tbody = $("#stats tbody");
    clearChildren(tbody);
    const rows = [
      ["Collection", stats.collection_name],
      ["Chunks indexed", stats.chunks_indexed],
      ["Vector DB path", stats.persist_dir],
      ["Embedding dimension", stats.embedding_dim ?? "(empty)"],
      ["Embedding provider", `${cfg.embeddings.provider} / ${cfg.embeddings.model}`],
      ["Chat provider", `${cfg.chat.provider} / ${cfg.chat.model}`],
      ["Chunk size / overlap", `${cfg.chunking.chunk_size} / ${cfg.chunking.chunk_overlap}`],
      ["Chunk strategy", cfg.chunking.strategy],
      ["top_k", cfg.retrieval.top_k],
      ["Retrieval features", featuresSummary(cfg.retrieval)],
    ];
    for (const [key, value] of rows) {
      const row = document.createElement("tr");
      row.appendChild(td(key));
      row.appendChild(td(String(value)));
      tbody.appendChild(row);
    }
    setStatus("#stats-status", "");
  } catch (e) {
    setStatus("#stats-status", `Error: ${e.message}`, true);
  }
}

$("#refresh-stats").addEventListener("click", refreshStats);

$("#clear-index").addEventListener("click", async () => {
  if (!confirm("Wipe the ENTIRE vector store and ingestion index?\n\nYou will need to re-ingest everything.")) return;
  try {
    await fetchJson("/api/index", { method: "DELETE" });
    await refreshStats();
  } catch (e) {
    setStatus("#stats-status", `Error: ${e.message}`, true);
  }
});

// ---------------------------------------------------------------------------
// Config (read-only view of the effective server configuration)
// ---------------------------------------------------------------------------

// Order + which sections to show (skip nothing — the config is non-secret).
const CONFIG_SECTIONS = [
  ["chat", "Chat provider"],
  ["embeddings", "Embedding provider"],
  ["chunking", "Chunking"],
  ["retrieval", "Retrieval"],
  ["ocr", "OCR"],
  ["vector_store", "Vector store"],
  ["prompt", "Prompt"],
  ["paths", "Paths"],
  ["server", "Server"],
  ["app", "App"],
];

async function refreshConfig() {
  setStatus("#config-status", "Loading...");
  try {
    const cfg = await fetchJson("/api/config");
    const container = $("#config-tables");
    clearChildren(container);
    for (const [key, title] of CONFIG_SECTIONS) {
      const section = cfg[key];
      if (section == null || typeof section !== "object") continue;
      const heading = document.createElement("h2");
      heading.textContent = title;
      container.appendChild(heading);
      const table = document.createElement("table");
      const tbody = document.createElement("tbody");
      for (const [field, value] of Object.entries(section)) {
        const row = document.createElement("tr");
        row.appendChild(td(field));
        row.appendChild(td(value === null ? "(null)" : String(value)));
        tbody.appendChild(row);
      }
      table.appendChild(tbody);
      container.appendChild(table);
    }
    setStatus("#config-status", "Read-only. Edit the config file + restart to change.");
  } catch (e) {
    setStatus("#config-status", `Error: ${e.message}`, true);
  }
}

$("#refresh-config").addEventListener("click", refreshConfig);
