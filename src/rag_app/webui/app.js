/* Same-origin, local-only UI. Untrusted documents and model output stay text. */
import { consumeSSE } from "./sse.js";
const $ = (selector) => document.querySelector(selector);
const history = [];
let activeQuery = null;
let droppedFiles = null;
const selectedJobs = { ingest: null, experiment: null };
let polling = null;

function el(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = String(text);
  if (className) node.className = className;
  return node;
}
function status(selector, text, error = false) {
  const node = $(selector);
  node.textContent = text;
  node.classList.toggle("error", error);
}
function button(text, action, className) {
  const node = el("button", text, className);
  node.type = "button";
  node.addEventListener("click", action);
  return node;
}
async function request(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      detail =
        typeof body.detail === "string"
          ? body.detail
          : JSON.stringify(body.detail || body);
    } catch {}
    throw new Error(detail);
  }
  return response;
}
async function json(url, options) {
  return (await request(url, options)).json();
}
function post(url, data, options = {}) {
  return json(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
    ...options,
  });
}
function parseFilter(text) {
  if (!text.trim()) return null;
  const pairs = new Map();
  for (const part of text.split(",")) {
    const index = part.indexOf("=");
    const key = part.slice(0, index).trim(),
      value = part.slice(index + 1).trim();
    if (index < 1 || !key || key.startsWith("$") || !value || pairs.has(key))
      throw new Error("Use distinct key=value pairs separated by commas");
    pairs.set(key, value);
  }
  const conditions = [...pairs].map(([key, value]) => ({ [key]: value }));
  return conditions.length === 1 ? conditions[0] : { $and: conditions };
}

function showPanel(name, focus = false) {
  document.querySelectorAll(".tab").forEach((tab) => {
    const selected = tab.dataset.panel === name;
    tab.classList.toggle("active", selected);
    tab.setAttribute("aria-selected", String(selected));
    tab.tabIndex = selected ? 0 : -1;
    if (selected && focus) tab.focus();
  });
  document.querySelectorAll(".panel").forEach((panel) => {
    panel.hidden = panel.id !== `panel-${name}`;
  });
  if (name === "documents") refreshDocuments();
  if (name === "stats") refreshStats();
  if (name === "config") refreshConfig();
  if (name === "ingest" || name === "experiments") refreshJobs();
}
const tabs = [...document.querySelectorAll(".tab")];
tabs.forEach((tab, index) => {
  tab.addEventListener("click", () => showPanel(tab.dataset.panel));
  tab.addEventListener("keydown", (event) => {
    let next;
    if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
    if (event.key === "ArrowLeft")
      next = (index + tabs.length - 1) % tabs.length;
    if (event.key === "Home") next = 0;
    if (event.key === "End") next = tabs.length - 1;
    if (next !== undefined) {
      event.preventDefault();
      showPanel(tabs[next].dataset.panel, true);
    }
  });
});
$("#go-ingest").onclick = () => showPanel("ingest", true);

async function inspectSource(sourceRef) {
  const dialog = $("#source-dialog");
  $("#source-title").textContent = "Source evidence";
  $("#source-text").textContent = "Loading...";
  dialog.showModal();
  try {
    const ids = sourceRef.context_ids?.length
      ? sourceRef.context_ids
      : [sourceRef.id];
    const passages = await Promise.all(
      ids.map((id) => json(`/api/chunks/${encodeURIComponent(id)}`)),
    );
    $("#source-title").textContent = `${sourceRef.file} | retrieved evidence`;
    $("#source-text").textContent = passages
      .map(
        (source) =>
          `${source.source_path} | chunk ${source.chunk_index}\n${source.section || ""}\n\n${source.text}`,
      )
      .join("\n\n---\n\n");
  } catch (error) {
    $("#source-text").textContent = error.message;
  }
}
$("#close-source").onclick = () => $("#source-dialog").close();
function sourceButtons(container, sources) {
  container.replaceChildren();
  for (const [index, source] of (sources || []).entries()) {
    const node = button(
      `[${index + 1}] ${source.file} | ${source.section || `chunk ${source.chunk_index}`}`,
      () => inspectSource(source),
    );
    node.title = `${source.score_type}: ${source.score ?? "n/a"}\n${source.preview}`;
    container.append(node);
  }
  if (!sources?.length)
    container.append(el("span", "No retrieved evidence", "muted"));
}
function showDebug(container, debug) {
  container.replaceChildren();
  if (!debug) return;
  const details = el("details");
  details.append(
    el("summary", "Inspect prompt, timings, and settings"),
    el("pre", JSON.stringify(debug, null, 2)),
  );
  container.append(details);
}
async function ask(event) {
  event?.preventDefault();
  if (activeQuery) return;
  const question = $("#question").value.trim();
  if (!question) {
    status("#ask-status", "Enter a question.", true);
    return;
  }
  let filter;
  try {
    filter = parseFilter($("#filter").value);
  } catch (error) {
    status("#ask-status", error.message, true);
    return;
  }
  const controller = (activeQuery = new AbortController());
  $("#ask").disabled = $("#clear-history").disabled = true;
  $("#stop").hidden = false;
  const turn = el("article", undefined, "turn");
  const answer = el("div", "Retrieving evidence...", "answer-text");
  const sources = el("div", undefined, "sources"),
    debug = el("div"),
    outcome = el("p", "", "muted");
  turn.append(el("h3", question), answer, sources, debug, outcome);
  $("#transcript").append(turn);
  const body = {
    question,
    filter,
    history: [...history],
    stream: $("#stream").checked,
    debug: $("#debug").checked,
  };
  status("#ask-status", "Retrieving and answering...");
  let text = "";
  try {
    if (body.stream) {
      const response = await request("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      await consumeSSE(
        response,
        (message) => {
          if (message.sources) sourceButtons(sources, message.sources);
          if (message.token !== undefined) {
            text += message.token;
            answer.textContent = text;
          }
          if (message.debug) showDebug(debug, message.debug);
        },
        controller.signal,
      );
    } else {
      const result = await post("/api/query", body, {
        signal: controller.signal,
      });
      text = result.answer;
      answer.textContent = text;
      sourceButtons(sources, result.sources);
      showDebug(debug, result.debug);
    }
    if (controller.signal.aborted)
      throw new DOMException("Stopped", "AbortError");
    history.push(
      { role: "user", content: question },
      { role: "assistant", content: text },
    );
    if (history.length > 200) history.splice(0, history.length - 200);
    $("#history-label").textContent =
      `${history.length / 2} completed turn(s). Ctrl+Enter to ask.`;
    outcome.textContent = "Completed";
    status("#ask-status", "Answer complete.");
  } catch (error) {
    turn.classList.add("incomplete");
    if (!text) answer.textContent = "No complete answer was received.";
    const message = controller.signal.aborted
      ? "Stopped. This turn will not be sent as conversation history."
      : `Incomplete: ${error.message}`;
    outcome.textContent = message;
    status("#ask-status", message, !controller.signal.aborted);
  } finally {
    activeQuery = null;
    $("#ask").disabled = $("#clear-history").disabled = false;
    $("#stop").hidden = true;
  }
}
$("#ask-form").addEventListener("submit", ask);
$("#question").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) ask(event);
});
$("#stop").onclick = () => activeQuery?.abort();
$("#clear-history").onclick = () => {
  if (activeQuery) return;
  history.length = 0;
  $("#transcript").replaceChildren();
  status("#ask-status", "Conversation cleared.");
  $("#history-label").textContent = "Ctrl+Enter to ask.";
};

function renderJob(group, job) {
  const container = $(`#${group}-job`);
  container.replaceChildren();
  container.append(
    el(
      "strong",
      `${job.kind} | ${job.state}${job.cancel_requested ? " (stop requested)" : ""}`,
    ),
  );
  if (job.progress) {
    const progress = el("progress");
    progress.max = Math.max(1, job.progress.total);
    progress.value = job.progress.current;
    progress.setAttribute("aria-label", `${job.kind} progress`);
    container.append(
      progress,
      el(
        "p",
        `${job.progress.current}/${job.progress.total} | ${job.progress.status} | ${job.progress.path}`,
      ),
    );
  }
  if (
    !["completed", "cancelled", "failed", "interrupted"].includes(job.state)
  ) {
    const cancel = button("Cancel job", async () => {
      try {
        renderJob(group, await post(`/api/jobs/${job.id}/cancel`, {}));
      } catch (error) {
        status(
          `#${group === "ingest" ? "upload" : "experiment"}-status`,
          error.message,
          true,
        );
      }
    });
    cancel.disabled = job.cancel_requested;
    container.append(cancel);
  }
  if (job.error) container.append(el("p", job.error, "error"));
  if (group === "ingest" && job.result) renderIngest(job.result);
  if (group === "experiment" && job.result) renderExperiment(job);
}
function terminal(job) {
  return ["completed", "cancelled", "failed", "interrupted"].includes(
    job.state,
  );
}
async function trackJob(group, job) {
  selectedJobs[group] = job.id;
  renderJob(group, job);
  if (!polling) polling = setTimeout(pollJobs, 700);
  await refreshJobs();
}
async function pollJobs() {
  polling = null;
  let running = false;
  for (const [group, id] of Object.entries(selectedJobs)) {
    if (!id) continue;
    try {
      const job = await json(`/api/jobs/${id}`);
      renderJob(group, job);
      running ||= !terminal(job);
      if (terminal(job) && group === "ingest") updateEmptyState();
    } catch (error) {
      running = true;
      status(
        group === "ingest" ? "#upload-status" : "#experiment-status",
        `Progress unavailable: ${error.message}`,
        true,
      );
    }
  }
  if (running) polling = setTimeout(pollJobs, 1000);
  else refreshJobs();
}
async function refreshJobs() {
  try {
    const { jobs } = await json("/api/jobs");
    for (const group of ["ingest", "experiment"]) {
      const candidates = jobs.filter(
        (job) => (job.kind === "experiment") === (group === "experiment"),
      );
      const list = $(`#${group}-jobs`);
      list.replaceChildren();
      list.className = "job-list";
      for (const job of candidates.slice(0, 8))
        list.append(
          button(
            `${job.created_at.slice(0, 16).replace("T", " ")} | ${job.state}`,
            async () => {
              try {
                await trackJob(group, await json(`/api/jobs/${job.id}`));
              } catch (error) {
                list.append(el("p", error.message, "error"));
              }
            },
          ),
        );
      if (!selectedJobs[group] && candidates.length) {
        selectedJobs[group] = candidates[0].id;
        if (!polling) polling = setTimeout(pollJobs, 0);
      }
    }
  } catch (error) {
    status("#upload-status", `Job history unavailable: ${error.message}`, true);
  }
}
function renderIngest(result) {
  const container = $("#ingest-results");
  container.replaceChildren(
    el("p", `${result.total_chunks} chunks indexed in this job.`),
  );
  for (const [key, title] of [
    ["indexed_files", "Indexed"],
    ["skipped_files", "Unchanged"],
    ["failed_files", "Failed"],
  ]) {
    if (!result[key]?.length) continue;
    const details = el("details");
    details.open = key === "failed_files";
    details.append(el("summary", `${title} (${result[key].length})`));
    const list = el("ul");
    for (const item of result[key])
      list.append(el("li", Array.isArray(item) ? item.join(" | ") : item));
    details.append(list);
    container.append(details);
  }
}
$("#ingest-form").onsubmit = async (event) => {
  event.preventDefault();
  $("#run-ingest").disabled = true;
  try {
    await trackJob(
      "ingest",
      await post("/api/ingest/jobs", {
        path: $("#ingest-path").value.trim() || null,
        force: $("#force").checked,
      }),
    );
    status("#upload-status", "Ingestion queued.");
  } catch (error) {
    status("#upload-status", error.message, true);
  } finally {
    $("#run-ingest").disabled = false;
  }
};
const drop = $("#drop-zone");
for (const name of ["dragover", "dragenter"])
  drop.addEventListener(name, (event) => {
    event.preventDefault();
    drop.classList.add("dragover");
  });
drop.addEventListener("dragleave", () => drop.classList.remove("dragover"));
drop.addEventListener("drop", (event) => {
  event.preventDefault();
  drop.classList.remove("dragover");
  droppedFiles = [...event.dataTransfer.files];
  status(
    "#upload-status",
    `${droppedFiles.length} file(s) selected. Press Upload and index.`,
  );
  $("#upload-files").required = false;
});
$("#upload-files").onchange = () => {
  droppedFiles = null;
  $("#upload-files").required = true;
};
$("#upload-form").onsubmit = async (event) => {
  event.preventDefault();
  const files = droppedFiles || [...$("#upload-files").files];
  if (!files.length) return;
  $("#upload").disabled = true;
  status("#upload-status", "Uploading...");
  try {
    const data = new FormData();
    files.forEach((file) => data.append("files", file));
    const job = await json("/api/uploads", { method: "POST", body: data });
    status(
      "#upload-status",
      `Saved ${job.saved_paths.length} file(s). ${job.rejected.length} rejected. Indexing queued.`,
    );
    await trackJob("ingest", job);
    $("#upload-files").value = "";
    droppedFiles = null;
    $("#upload-files").required = true;
  } catch (error) {
    status("#upload-status", error.message, true);
  } finally {
    $("#upload").disabled = false;
  }
};

async function refreshDocuments() {
  try {
    const { documents } = await json("/api/documents");
    const tbody = $("#documents tbody");
    tbody.replaceChildren();
    $("#select-all").checked = false;
    for (const doc of documents) {
      const row = el("tr"),
        check = el("input");
      check.type = "checkbox";
      check.className = "doc-check";
      check.value = doc.path;
      check.setAttribute("aria-label", `Select ${doc.source_file}`);
      const first = el("td");
      first.append(check);
      const actions = el("td");
      actions.append(
        button("Inspect", () => inspectDocument(doc)),
        button("Forget", () => forgetDocuments([doc.path]), "danger"),
      );
      row.append(
        first,
        el("td", doc.source_file),
        el("td", doc.path),
        el("td", doc.chunks, "num"),
        actions,
      );
      tbody.append(row);
    }
    status(
      "#documents-status",
      documents.length
        ? `${documents.length} documents`
        : "No documents indexed. Add documents to start.",
    );
  } catch (error) {
    status("#documents-status", error.message, true);
  }
}
async function forgetDocuments(paths) {
  if (
    !paths.length ||
    !confirm(
      `Forget ${paths.length} document(s)? Original files will remain on disk.`,
    )
  )
    return;
  try {
    for (const path of paths)
      await json(`/api/documents?path=${encodeURIComponent(path)}`, {
        method: "DELETE",
      });
    await refreshDocuments();
    await updateEmptyState();
  } catch (error) {
    await refreshDocuments();
    status("#documents-status", error.message, true);
  }
}
$("#select-all").onchange = (event) =>
  document.querySelectorAll(".doc-check").forEach((check) => {
    check.checked = event.target.checked;
  });
$("#forget-selected").onclick = () =>
  forgetDocuments(
    [...document.querySelectorAll(".doc-check:checked")].map(
      (check) => check.value,
    ),
  );
$("#refresh-documents").onclick = refreshDocuments;
function renderChunks(chunks) {
  const container = $("#chunks-view");
  container.replaceChildren();
  for (const chunk of chunks) {
    const details = el("details");
    details.append(
      el(
        "summary",
        `${chunk.source_file} | chunk ${chunk.chunk_index} | ${chunk.section || ""}`,
      ),
      el("pre", chunk.text),
    );
    container.append(details);
  }
  if (!chunks.length) container.append(el("p", "No chunks found."));
}
async function inspectDocument(doc) {
  try {
    const query = doc.document_id
      ? `document_id=${encodeURIComponent(doc.document_id)}`
      : `source_file=${encodeURIComponent(doc.source_file)}`;
    renderChunks((await json(`/api/chunks?${query}&limit=1000`)).chunks);
  } catch (error) {
    status("#documents-status", error.message, true);
  }
}
$("#show-random").onclick = async () => {
  try {
    renderChunks(
      (await json(`/api/chunks?sample=${$("#sample-n").value}`)).chunks,
    );
  } catch (error) {
    status("#documents-status", error.message, true);
  }
};

$("#experiment-form").onsubmit = async (event) => {
  event.preventDefault();
  const presets = [
    ...document.querySelectorAll('input[name="preset"]:checked'),
  ].map((check) => check.value);
  if (!presets.length) {
    status("#experiment-status", "Choose at least one preset.", true);
    return;
  }
  $("#run-experiment").disabled = true;
  try {
    await trackJob(
      "experiment",
      await post("/api/experiments", {
        presets,
        mode: $("#experiment-mode").value,
      }),
    );
    status(
      "#experiment-status",
      "Comparison queued. Ingestion commits wait until this corpus snapshot is released.",
    );
  } catch (error) {
    status("#experiment-status", error.message, true);
  } finally {
    $("#run-experiment").disabled = false;
  }
};
function renderExperiment(job) {
  const report = job.result,
    container = $("#experiment-results");
  container.replaceChildren();
  const links = el("div", undefined, "download-links");
  for (const format of ["json", "csv"]) {
    const link = el("a", `Download ${format.toUpperCase()}`);
    link.href = `/api/experiments/${job.id}/report?format=${format}`;
    links.append(link);
  }
  container.append(
    links,
    el(
      "p",
      `Mode: ${report.mode} | Corpus revision: ${report.corpus_revision}`,
      "muted",
    ),
  );
  const table = el("table"),
    head = el("tr");
  for (const label of [
    "Preset",
    "Recall",
    "MRR",
    "nDCG",
    "Keyword coverage",
    "Time (ms)",
  ])
    head.append(el("th", label));
  table.append(head);
  for (const run of report.runs) {
    const row = el("tr");
    row.append(el("td", run.preset));
    for (const value of [
      run.summary.recall,
      run.summary.mrr,
      run.summary.ndcg,
      run.summary.keyword_coverage,
    ])
      row.append(el("td", value == null ? "Not scored" : value.toFixed(3)));
    row.append(el("td", run.elapsed_ms));
    table.append(row);
  }
  const wrap = el("div", undefined, "table-wrap report-summary");
  wrap.append(table);
  container.append(wrap);
  for (const run of report.runs) {
    const details = el("details");
    details.append(
      el(
        "summary",
        `${run.preset}: ${run.summary.questions} questions | inspect individual results`,
      ),
    );
    for (const result of run.results) {
      const entry = el("details");
      entry.append(
        el(
          "summary",
          `${!result.scored ? "Not scored" : result.passed ? "Pass" : "Check"}: ${result.question.question}`,
        ),
        el("pre", JSON.stringify(result, null, 2)),
      );
      details.append(entry);
    }
    container.append(details);
  }
}
async function updateEmptyState() {
  try {
    $("#empty-index").hidden = (await json("/api/stats")).chunks_indexed > 0;
  } catch {}
}
async function refreshStats() {
  try {
    const [stats, cfg] = await Promise.all([
      json("/api/stats"),
      json("/api/config"),
    ]);
    const body = $("#stats tbody");
    body.replaceChildren();
    for (const [key, value] of Object.entries({
      ...stats,
      chat: cfg.chat,
      embeddings: cfg.embeddings,
      retrieval: cfg.retrieval,
      cache: cfg.cache,
      observability: cfg.observability,
    })) {
      const row = el("tr");
      row.append(
        el("td", key),
        el("td", typeof value === "object" ? JSON.stringify(value) : value),
      );
      body.append(row);
    }
    status("#stats-status", "Status refreshed.");
  } catch (error) {
    status("#stats-status", error.message, true);
  }
}
$("#refresh-stats").onclick = refreshStats;
$("#check-readiness").onclick = async () => {
  $("#check-readiness").disabled = true;
  $("#readiness").textContent = "Checking model availability...";
  try {
    $("#readiness").textContent = JSON.stringify(
      await json("/api/readiness"),
      null,
      2,
    );
  } catch (error) {
    $("#readiness").textContent = error.message;
  } finally {
    $("#check-readiness").disabled = false;
  }
};
$("#clear-index").onclick = async () => {
  if (
    !confirm(
      "Clear the active index? Original files and rebuild backups are retained.",
    )
  )
    return;
  try {
    await json("/api/index", { method: "DELETE" });
    await refreshStats();
    await updateEmptyState();
  } catch (error) {
    status("#stats-status", error.message, true);
  }
};
async function refreshConfig() {
  try {
    const cfg = await json("/api/config"),
      container = $("#config-tables");
    container.replaceChildren();
    for (const [key, value] of Object.entries(cfg)) {
      const details = el("details");
      details.open = ["chat", "embeddings"].includes(key);
      details.append(
        el("summary", key),
        el("pre", JSON.stringify(value, null, 2)),
      );
      container.append(details);
    }
    status("#config-status", "Loaded all configuration sections.");
  } catch (error) {
    status("#config-status", error.message, true);
  }
}
$("#refresh-config").onclick = refreshConfig;
(async () => {
  try {
    await json("/health");
    $("#health").textContent = "API reachable";
    $("#health").classList.add("ok");
  } catch {
    $("#health").textContent = "API unavailable";
    $("#health").classList.add("error");
  }
  await updateEmptyState();
  await refreshJobs();
  try {
    const cfg = await json("/api/config");
    $("#upload-limits").textContent =
      `Up to ${cfg.server.upload_max_files} files, ${Math.round(cfg.server.upload_max_bytes / 1024 / 1024)} MiB each. TXT, MD, PDF, DOCX, HTML, CSV${cfg.ocr.enabled ? ", images (OCR)" : ""}.`;
  } catch {}
})();
