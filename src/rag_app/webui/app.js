/* Same-origin, local-only UI. Untrusted documents and model output stay text. */
import { consumeSSE } from "./sse.js";
import { splitCitations } from "./citations.js";
import {
  playgroundSettings,
  renderChunkPreview,
  renderEmbeddingMap,
  renderGraphEntity,
  renderGraphOverview,
} from "./lab.js";
import { initEvalSets } from "./evalsets.js";
import { initLearn } from "./learn.js";
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
  if (name === "lab") initLab();
  if (name === "learn") learn.refresh();
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
    $("#source-title").textContent =
      `${displayName(sourceRef.file)} | retrieved evidence`;
    // The original file, opened at the cited page when the format has pages.
    const link = $("#source-open");
    link.hidden = !sourceRef.document_id;
    if (sourceRef.document_id) {
      link.href =
        `/api/documents/${encodeURIComponent(sourceRef.document_id)}/file` +
        (sourceRef.page ? `#page=${sourceRef.page}` : "");
      link.textContent = sourceRef.page
        ? `Open the original at page ${sourceRef.page}`
        : "Open the original file";
    }
    $("#source-text").textContent = passages
      .map(
        (source) =>
          `${source.source_path} | ${source.location || `chunk ${source.chunk_index}`}\n${source.section || ""}\n\n${source.text}`,
      )
      .join("\n\n---\n\n");
  } catch (error) {
    $("#source-text").textContent = error.message;
  }
}
$("#close-source").onclick = () => $("#source-dialog").close();
/* Uploaded files are stored as "<32 hex>_<name>"; show the name. */
function displayName(file) {
  return String(file ?? "?").replace(/^[0-9a-f]{32}_/, "");
}
function sourceButtons(container, sources) {
  container.replaceChildren();
  for (const [index, source] of (sources || []).entries()) {
    const node = button(
      // "datasheet.pdf, p. 41 | 3.2 Bit timing" or "driver.c, lines 12-40 | Spi_Init"
      `[${index + 1}] ${displayName(source.file)}${source.location ? `, ${source.location}` : ""} | ${source.section || `chunk ${source.chunk_index}`}`,
      () => inspectSource(source),
    );
    node.title = `${source.score_type}: ${source.score ?? "n/a"}\n${source.preview}`;
    container.append(node);
  }
  if (!sources?.length)
    container.append(el("span", "No retrieved evidence", "muted"));
}
/* ---- Citations: [n] in the answer opens source n ------------------------- */
function renderAnswer(node, text, sources) {
  node.replaceChildren();
  for (const part of splitCitations(text)) {
    if (part.text !== undefined) {
      node.append(document.createTextNode(part.text));
      continue;
    }
    for (const number of part.numbers) {
      const source = sources?.[number - 1];
      if (!source) {
        const unknown = el("span", `[${number}]`, "cite-invalid");
        unknown.title = "No source has this number";
        node.append(unknown);
        continue;
      }
      const cite = button(`[${number}]`, () => inspectSource(source), "cite");
      cite.title = `${displayName(source.file)}${source.location ? `, ${source.location}` : ""}`;
      node.append(cite);
    }
  }
}
/* Mark the source buttons the answer cited, and say what the check found. */
function markCitations(sourcesNode, noteNode, citations, sourceCount) {
  if (!citations || !sourceCount) return;
  const cited = new Set(citations.cited || []);
  [...sourcesNode.querySelectorAll("button")].forEach((node, index) => {
    node.classList.toggle("cited", cited.has(index + 1));
    node.classList.toggle("uncited", !cited.has(index + 1));
  });
  const notes = [];
  if (citations.invalid?.length)
    notes.push(
      `The answer cites ${citations.invalid.map((n) => `[${n}]`).join(", ")}, which match no source.`,
    );
  if (!cited.size && !citations.invalid?.length)
    notes.push(
      "The answer cites none of its sources, so no statement in it can be traced to a passage.",
    );
  else if (cited.size)
    notes.push(
      `Cited: ${[...cited].map((n) => `[${n}]`).join(" ")}. Click a number in the answer to read the passage.`,
    );
  noteNode.textContent = notes.join(" ");
}
/* ---- Pipeline view: the server sends each stage with its facts + table ---- */
function renderTable(headers, rows, viewRow) {
  const table = el("table", undefined, "stage-table"),
    head = el("tr"),
    body = el("tbody");
  for (const header of headers) head.append(el("th", header));
  if (viewRow) head.append(el("th", "Passage"));
  rows.forEach((cells, index) => {
    const row = el("tr");
    cells.forEach((text, column) => {
      const cell = el("td", text);
      // Lets the stylesheet keep short columns on one line and wrap the rest.
      cell.dataset.col = headers[column].toLowerCase();
      if (cell.dataset.col === "file") cell.title = text;
      row.append(cell);
    });
    if (viewRow) {
      const cell = el("td");
      cell.append(button("View", () => viewRow(index)));
      row.append(cell);
    }
    body.append(row);
  });
  const thead = el("thead");
  thead.append(head);
  table.append(thead, body);
  const wrap = el("div", undefined, "table-wrap");
  wrap.append(table);
  return wrap;
}
function renderPipeline(stages) {
  const list = el("ol", undefined, "pipeline");
  for (const stage of stages) {
    const item = el("li"),
      details = el("details"),
      summary = el("summary");
    const notes = [];
    if (stage.items.length)
      notes.push(`${stage.detail.total ?? stage.items.length} chunk(s)`);
    if (stage.ms != null) notes.push(`${stage.ms.toFixed(1)} ms`);
    summary.append(
      el("strong", stage.title),
      el("span", notes.length ? ` ${notes.join(" | ")}` : "", "muted"),
    );
    details.append(summary);
    if (stage.facts.length) {
      const facts = el("dl", undefined, "facts");
      for (const [name, text] of stage.facts)
        facts.append(el("dt", name), el("dd", text));
      details.append(facts);
    }
    if (stage.table.rows.length)
      details.append(
        renderTable(stage.table.headers, stage.table.rows, (index) => {
          const chunk = stage.items[index];
          inspectSource({
            id: chunk.id,
            file: chunk.file,
            context_ids: chunk.context_ids,
          });
        }),
      );
    item.append(details);
    list.append(item);
  }
  return list;
}
function showDebug(container, debug) {
  container.replaceChildren();
  if (!debug) return;
  const { trace, ...rest } = debug;
  if (trace?.length) {
    const pipeline = el("details", undefined, "pipeline-box");
    pipeline.open = true;
    pipeline.append(
      el("summary", "Pipeline: what each stage did"),
      renderPipeline(trace),
    );
    container.append(pipeline);
  }
  const details = el("details");
  details.append(
    el("summary", "Inspect prompt, timings, and settings"),
    el("pre", JSON.stringify(rest, null, 2)),
  );
  container.append(details);
}

/* ---- Techniques: retrieval settings changed for one question only -------- */
let controlSpecs = null;
let askTechniques = null,
  compareA = null,
  compareB = null;
function configuredValue(control) {
  return control.kind === "switch" ? Boolean(control.value) : control.value;
}
/* One set of inputs built from /api/retrieval-controls. `read()` returns only
   the settings that differ from the configuration - or, with `baseline`, from
   the schema defaults (the dense baseline an experiment preset starts from). */
function techniquesForm(container, prefix, onChange, baseline = false) {
  const inputs = [];
  const startValue = (control) =>
    baseline
      ? control.kind === "switch"
        ? false
        : (control.default ?? null)
      : configuredValue(control);
  container.replaceChildren();
  for (const control of controlSpecs) {
    if (!control.editable) continue;
    const label = el("label", undefined, "technique"),
      input = el("input"),
      name = el("span", control.label);
    label.title = `${control.help} (${control.stage})`;
    input.id = `${prefix}-${control.field}`;
    const toggle = control.kind === "bool" || control.kind === "switch";
    if (toggle) {
      input.type = "checkbox";
      label.append(input, name);
    } else {
      input.type = "number";
      if (control.min !== undefined) input.min = control.min;
      if (control.max !== undefined) input.max = control.max;
      input.step = control.step ?? 1;
      if (control.nullable) input.placeholder = "auto";
      label.append(name, input);
    }
    if (control.needs_chat) {
      const badge = el("span", "LLM", "badge");
      badge.title = "Needs the chat model during retrieval";
      label.append(badge);
    }
    input.addEventListener("input", () => onChange?.());
    inputs.push({ control, input, toggle });
    container.append(label);
  }
  const reset = () => {
    for (const { control, input, toggle } of inputs) {
      if (toggle) input.checked = Boolean(startValue(control));
      else input.value = startValue(control) ?? "";
    }
    onChange?.();
  };
  const read = () => {
    const overrides = {};
    for (const { control, input, toggle } of inputs) {
      let value;
      if (toggle) value = input.checked;
      else if (input.value.trim() === "") {
        if (!control.nullable)
          throw new Error(`${control.label} needs a value`);
        value = null;
      } else {
        value = Number(input.value);
        if (
          !Number.isFinite(value) ||
          (control.kind === "int" && !Number.isInteger(value))
        )
          throw new Error(`${control.label} must be a valid number`);
        // The forms are `novalidate`, so a value hidden in a closed drawer
        // cannot silently block submission; ranges are checked here instead.
        if (
          (control.min !== undefined && value < control.min) ||
          (control.max !== undefined && value > control.max)
        )
          throw new Error(
            `${control.label} must be between ${control.min ?? "-inf"} and ${control.max ?? "inf"}`,
          );
      }
      if (value === startValue(control)) continue;
      overrides[control.field] =
        control.kind === "switch" ? (value ? control.on_value : null) : value;
    }
    return overrides;
  };
  /* Start again from the reference values, then set what `overrides` names
     (used by guided labs to preconfigure a form). */
  const apply = (overrides) => {
    reset();
    for (const { control, input, toggle } of inputs) {
      if (!(control.field in (overrides ?? {}))) continue;
      const value = overrides[control.field];
      if (toggle) input.checked = Boolean(value);
      else input.value = value ?? "";
    }
    onChange?.();
  };
  reset();
  return { read, reset, apply };
}
function overridesOrNull(form) {
  const overrides = form ? form.read() : {};
  return Object.keys(overrides).length ? overrides : null;
}
function updateAskTechniquesCount() {
  let text = "";
  try {
    const changed = Object.keys(askTechniques.read()).length;
    if (changed) text = `(${changed} changed)`;
  } catch {
    text = "(check values)";
  }
  $("#ask-techniques-count").textContent = text;
}
async function initTechniques() {
  try {
    controlSpecs = (await json("/api/retrieval-controls")).controls;
    askTechniques = techniquesForm(
      $("#ask-techniques-form"),
      "ask-tech",
      () => askTechniques && updateAskTechniquesCount(),
    );
    compareA = techniquesForm($("#compare-a-form"), "cmp-a");
    compareB = techniquesForm($("#compare-b-form"), "cmp-b");
  } catch (error) {
    for (const id of ["#ask-techniques-form", "#compare-a-form"])
      $(id).replaceChildren(
        el("p", `Techniques unavailable: ${error.message}`, "error"),
      );
  }
}
$("#ask-techniques-reset").onclick = () => askTechniques?.reset();
async function ask(event) {
  event?.preventDefault();
  if (activeQuery) return;
  const question = $("#question").value.trim();
  if (!question) {
    status("#ask-status", "Enter a question.", true);
    return;
  }
  let filter, retrieval;
  try {
    filter = parseFilter($("#filter").value);
    retrieval = overridesOrNull(askTechniques);
  } catch (error) {
    status("#ask-status", error.message, true);
    return;
  }
  const controller = (activeQuery = new AbortController());
  $("#ask").disabled = $("#clear-history").disabled = true;
  $("#conversation-select").disabled = true;
  $("#stop").hidden = false;
  const turnParts = addTurn(question);
  const { turn, answer, sources, debug, note, outcome } = turnParts;
  const body = {
    question,
    filter,
    retrieval,
    stream: $("#stream").checked,
    debug: $("#debug").checked,
  };
  // A saved conversation supplies the history on the server and records the
  // answer. If one cannot be created, fall back to history kept in this tab.
  try {
    body.conversation_id = await ensureConversation();
  } catch {
    body.history = [...history];
  }
  status("#ask-status", "Retrieving and answering...");
  let text = "",
    sourceList = [],
    citations = null;
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
          if (message.step) {
            showAgentStep(turnParts.steps, message.step);
            status(
              "#ask-status",
              `Agent step ${message.step.step}: ${message.step.action}...`,
            );
          }
          if (message.sources) {
            sourceList = message.sources;
            sourceButtons(sources, sourceList);
          }
          if (message.token !== undefined) {
            text += message.token;
            answer.textContent = text;
          }
          if (message.citations) citations = message.citations;
          if (message.debug) showDebug(debug, message.debug);
        },
        controller.signal,
      );
    } else {
      const result = await post("/api/query", body, {
        signal: controller.signal,
      });
      text = result.answer;
      sourceList = result.sources;
      citations = result.citations;
      for (const step of result.agent_steps ?? [])
        showAgentStep(turnParts.steps, step);
      sourceButtons(sources, sourceList);
      showDebug(debug, result.debug);
    }
    if (controller.signal.aborted)
      throw new DOMException("Stopped", "AbortError");
    // The answer is complete: turn its [n] markers into links to the sources.
    renderAnswer(answer, text, sourceList);
    markCitations(sources, note, citations, sourceList.length);
    history.push(
      { role: "user", content: question },
      { role: "assistant", content: text },
    );
    if (history.length > 200) history.splice(0, history.length - 200);
    $("#history-label").textContent =
      `${history.length / 2} completed turn(s). Ctrl+Enter to ask.`;
    outcome.textContent = "Completed";
    offerSaveQuestion(turnParts, question, text, sourceList, citations);
    status("#ask-status", "Answer complete.");
    if (body.conversation_id) refreshConversations();
  } catch (error) {
    turn.classList.add("incomplete");
    if (!text) answer.textContent = "No complete answer was received.";
    const message = controller.signal.aborted
      ? "Stopped. This turn will not be sent as conversation history."
      : `Incomplete: ${error.message}`;
    outcome.textContent = message;
    status("#ask-status", message, !controller.signal.aborted);
    if (error.message === "Conversation not found") forgetConversation();
  } finally {
    activeQuery = null;
    $("#ask").disabled = $("#clear-history").disabled = false;
    $("#conversation-select").disabled = false;
    $("#stop").hidden = true;
  }
}

/* One question/answer card in the transcript. */
function addTurn(question) {
  const turn = el("article", undefined, "turn");
  const parts = {
    turn,
    // Filled while an agent works: one line per step it took.
    steps: el("ol", undefined, "agent-steps"),
    answer: el("div", "Retrieving evidence...", "answer-text"),
    sources: el("div", undefined, "sources"),
    note: el("p", "", "muted citation-note"),
    debug: el("div"),
    outcome: el("p", "", "muted"),
    actions: el("div", undefined, "turn-actions"),
  };
  parts.steps.hidden = true;
  turn.append(
    el("h3", question),
    parts.steps,
    parts.answer,
    parts.sources,
    parts.note,
    parts.debug,
    parts.outcome,
    parts.actions,
  );
  $("#transcript").append(turn);
  return parts;
}
/* One step of agentic retrieval: what the model chose and what came back. */
function showAgentStep(list, step) {
  const item = el("li");
  let what = step.action.toUpperCase();
  if (step.action === "search") what += ` "${step.argument}"`;
  else if (step.argument) what += ` [${step.argument}]`;
  item.append(el("strong", what));
  if (step.thought) item.append(el("span", ` ${step.thought}`));
  if (step.observation && step.action !== "finish")
    item.append(el("div", step.observation, "muted"));
  list.hidden = false;
  list.append(item);
}
/* A completed answer can become a test: the question, the files that answered
   it and the answer itself go into a question set (Experiments tab). */
function offerSaveQuestion(parts, question, answer, sources, citations) {
  parts.actions.replaceChildren(
    button("Save as evaluation question", () =>
      evalSets.saveQuestion({ question, answer, sources, citations }),
    ),
  );
}

/* ---- Saved conversations ------------------------------------------------- */
let conversationId = null;
function rememberConversation(id) {
  conversationId = id;
  try {
    if (id) localStorage.setItem("rag.conversation", id);
    else localStorage.removeItem("rag.conversation");
  } catch {}
  $("#conversation-rename").disabled = $("#conversation-delete").disabled = !id;
  const link = $("#conversation-export");
  link.hidden = !id;
  if (id) {
    link.href = `/api/conversations/${id}/export`;
    link.setAttribute("download", "conversation.md");
  }
}
async function ensureConversation() {
  if (!conversationId)
    rememberConversation((await post("/api/conversations", {})).id);
  return conversationId;
}
function forgetConversation() {
  rememberConversation(null);
  history.length = 0;
  $("#transcript").replaceChildren();
  $("#conversation-select").value = "";
  $("#history-label").textContent = "Ctrl+Enter to ask.";
}
async function refreshConversations() {
  try {
    const { conversations } = await json("/api/conversations");
    const select = $("#conversation-select");
    select.replaceChildren(el("option", "New conversation"));
    select.firstChild.value = "";
    for (const item of conversations) {
      const option = el("option", `${item.title} (${item.turns})`);
      option.value = item.id;
      select.append(option);
    }
    const known = conversations.some((item) => item.id === conversationId);
    select.value = known ? conversationId : "";
    return conversations;
  } catch {
    return [];
  }
}
async function loadConversation(id) {
  const data = await json(`/api/conversations/${id}`);
  rememberConversation(id);
  history.length = 0;
  $("#transcript").replaceChildren();
  for (const saved of data.turns) {
    const parts = addTurn(saved.question);
    sourceButtons(parts.sources, saved.sources);
    renderAnswer(parts.answer, saved.answer, saved.sources);
    markCitations(
      parts.sources,
      parts.note,
      saved.citations,
      saved.sources.length,
    );
    parts.outcome.textContent = `Saved ${saved.created_at.slice(0, 16).replace("T", " ")}`;
    offerSaveQuestion(parts, saved.question, saved.answer, saved.sources, saved.citations);
    history.push(
      { role: "user", content: saved.question },
      { role: "assistant", content: saved.answer },
    );
  }
  $("#history-label").textContent =
    `${data.turns.length} completed turn(s). Ctrl+Enter to ask.`;
  status("#ask-status", `Continuing "${data.title || "conversation"}".`);
}
$("#conversation-select").onchange = async (event) => {
  if (activeQuery) return;
  const id = event.target.value;
  if (!id) {
    forgetConversation();
    status("#ask-status", "New conversation.");
    return;
  }
  try {
    await loadConversation(id);
  } catch (error) {
    status("#ask-status", error.message, true);
    forgetConversation();
    refreshConversations();
  }
};
$("#conversation-rename").onclick = async () => {
  if (!conversationId) return;
  const select = $("#conversation-select");
  const current = select.selectedOptions[0]?.textContent.replace(/ \(\d+\)$/, "");
  const title = prompt("Conversation title", current || "");
  if (!title?.trim()) return;
  try {
    await json(`/api/conversations/${conversationId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title }),
    });
    await refreshConversations();
  } catch (error) {
    status("#ask-status", error.message, true);
  }
};
$("#conversation-delete").onclick = async () => {
  if (!conversationId || activeQuery) return;
  if (!confirm("Delete this saved conversation? Your documents are not affected."))
    return;
  try {
    await json(`/api/conversations/${conversationId}`, { method: "DELETE" });
    forgetConversation();
    await refreshConversations();
    status("#ask-status", "Conversation deleted.");
  } catch (error) {
    status("#ask-status", error.message, true);
  }
};
async function initConversations() {
  const conversations = await refreshConversations();
  let last = null;
  try {
    last = localStorage.getItem("rag.conversation");
  } catch {}
  if (last && conversations.some((item) => item.id === last)) {
    try {
      await loadConversation(last);
      $("#conversation-select").value = last;
    } catch {
      forgetConversation();
    }
  }
}
$("#ask-form").addEventListener("submit", ask);
$("#question").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) ask(event);
});
$("#stop").onclick = () => activeQuery?.abort();
$("#clear-history").onclick = () => {
  if (activeQuery) return;
  // The conversation that was open stays saved; the next question starts a
  // new one.
  forgetConversation();
  status("#ask-status", "New conversation.");
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
      const candidates = jobs.filter((job) =>
        group === "experiment"
          ? job.kind === "experiment"
          : job.kind !== "experiment" && job.kind !== "eval-generate",
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
    ["pruned_files", "Forgotten (source file is gone)"],
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
        prune: $("#prune").checked,
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
        `${chunk.source_file} | ${chunk.location || `chunk ${chunk.chunk_index}`} | ${chunk.section || ""}`,
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

/* ---- Compare: one question, two settings, no answer generated ------------ */
function settingText(value) {
  return value === null || value === undefined ? "off" : String(value);
}
function renderCompare(result) {
  const container = $("#compare-results"),
    summary = result.summary;
  container.replaceChildren();
  const changed = Object.entries(summary.settings_changed);
  container.append(
    el(
      "p",
      changed.length
        ? `Settings that differ: ${changed
            .map(
              ([name, v]) =>
                `${name} (A: ${settingText(v.a)}, B: ${settingText(v.b)})`,
            )
            .join("; ")}`
        : "A and B use the same settings, so both sides retrieve the same chunks.",
    ),
    el(
      "p",
      `${summary.shared} chunk(s) in both | ${summary.only_a} only in A | ${summary.only_b} only in B | overlap ${Math.round(summary.overlap * 100)}% | ${summary.same_top_result ? "same top result" : "different top result"}`,
    ),
  );
  const found = { both: "both", only_a: "only A", only_b: "only B" };
  const rows = result.rows.map((row) => [
    row.rank_a === null ? "-" : `#${row.rank_a}`,
    row.rank_b === null ? "-" : `#${row.rank_b}`,
    found[row.status],
    row.file,
    row.section || `chunk ${row.chunk_index}`,
    row.preview,
  ]);
  const table = renderTable(
    ["A", "B", "Found by", "File", "Section", "Preview"],
    rows,
    (index) =>
      inspectSource({ id: result.rows[index].id, file: result.rows[index].file }),
  );
  table.querySelectorAll("tbody tr").forEach((node, index) => {
    node.classList.add(result.rows[index].status.replace("_", "-"));
  });
  container.append(table);
  // Full width, one after the other: a stage table needs the room.
  for (const side of ["a", "b"])
    container.append(
      el("h3", `Pipeline ${side.toUpperCase()}`),
      renderPipeline(result[side].trace),
    );
}
$("#compare-form").onsubmit = async (event) => {
  event.preventDefault();
  const question = $("#compare-question").value.trim();
  if (!question) {
    status("#compare-status", "Enter a question.", true);
    return;
  }
  let body;
  try {
    body = {
      question,
      filter: parseFilter($("#compare-filter").value),
      a: overridesOrNull(compareA),
      b: overridesOrNull(compareB),
      use_llm: $("#compare-llm").checked,
    };
  } catch (error) {
    status("#compare-status", error.message, true);
    return;
  }
  $("#run-compare").disabled = true;
  status("#compare-status", "Retrieving with both settings...");
  try {
    renderCompare(await post("/api/compare", body));
    status("#compare-status", "Comparison complete.");
  } catch (error) {
    status("#compare-status", error.message, true);
  } finally {
    $("#run-compare").disabled = false;
  }
};
$("#compare-reset").onclick = () => {
  compareA?.reset();
  compareB?.reset();
};

/* ---- Lab: chunking playground and embedding map --------------------------- */
let playgroundA = null,
  playgroundB = null;
async function initLab() {
  try {
    const { documents } = await json("/api/documents");
    const select = $("#playground-document"),
      chosen = select.value;
    select.replaceChildren();
    for (const doc of documents) {
      const option = el(
        "option",
        `${displayName(doc.source_file)} (${doc.chunks} chunks indexed)`,
      );
      option.value = doc.document_id;
      select.append(option);
    }
    if (documents.some((doc) => doc.document_id === chosen))
      select.value = chosen;
    if (!documents.length)
      status("#playground-status", "Add a document first; the playground chunks indexed documents.");
    if (!playgroundA) {
      const cfg = await json("/api/config");
      const configured = {
        chunk_size: cfg.chunking.chunk_size,
        chunk_overlap: cfg.chunking.chunk_overlap,
        strategy: cfg.chunking.strategy,
        code_max_chars: cfg.extraction.code_max_chars,
      };
      // A starts as configured; B starts at a third of the size, which is
      // usually enough of a change to see a difference straight away.
      playgroundA = playgroundSettings($("#playground-a"), "pg-a", configured);
      playgroundB = playgroundSettings($("#playground-b"), "pg-b", {
        ...configured,
        chunk_size: Math.max(100, Math.round(configured.chunk_size / 3)),
        chunk_overlap: Math.round(configured.chunk_overlap / 3),
      });
    }
  } catch (error) {
    status("#playground-status", error.message, true);
  }
}
$("#playground-form").onsubmit = async (event) => {
  event.preventDefault();
  const documentId = $("#playground-document").value;
  if (!documentId || !playgroundA) {
    status("#playground-status", "Add a document first.", true);
    return;
  }
  let settings;
  try {
    settings = [playgroundA.read(), playgroundB.read()];
  } catch (error) {
    status("#playground-status", error.message, true);
    return;
  }
  $("#run-playground").disabled = true;
  status("#playground-status", "Chunking...");
  try {
    const [first, second] = await Promise.all(
      settings.map((choice) =>
        post("/api/chunk-preview", { document_id: documentId, settings: choice }),
      ),
    );
    const container = $("#playground-results");
    container.replaceChildren();
    renderChunkPreview(container, "A", first);
    renderChunkPreview(container, "B", second);
    status(
      "#playground-status",
      `A makes ${first.stats.count} chunk(s), B makes ${second.stats.count}. Nothing was stored.`,
    );
  } catch (error) {
    status("#playground-status", error.message, true);
  } finally {
    $("#run-playground").disabled = false;
  }
};
async function showGraphEntity(name) {
  try {
    const detail = await json(`/api/graph?entity=${encodeURIComponent(name)}`);
    renderGraphEntity($("#graph-detail"), detail, showGraphEntity, (chunk) =>
      inspectSource({
        id: chunk.id,
        file: chunk.file,
        document_id: chunk.document_id,
        page: chunk.page,
      }),
    );
    $("#graph-detail").scrollIntoView({ block: "nearest" });
    status("#graph-status", `Showing ${detail.name}.`);
  } catch (error) {
    status("#graph-status", error.message, true);
  }
}
$("#graph-form").onsubmit = async (event) => {
  event.preventDefault();
  $("#run-graph").disabled = true;
  $("#graph-detail").replaceChildren();
  try {
    const contains = $("#graph-contains").value.trim();
    const overview = await json(
      `/api/graph?limit=50&contains=${encodeURIComponent(contains)}`,
    );
    renderGraphOverview($("#graph-results"), overview, showGraphEntity);
    status(
      "#graph-status",
      overview.entities_in_index
        ? `${overview.total} matching entit${overview.total === 1 ? "y" : "ies"}.`
        : "",
    );
  } catch (error) {
    status("#graph-status", error.message, true);
  } finally {
    $("#run-graph").disabled = false;
  }
};
$("#map-form").onsubmit = async (event) => {
  event.preventDefault();
  $("#run-map").disabled = true;
  status("#map-status", "Projecting the chunk vectors...");
  try {
    const result = await post("/api/embedding-map", {
      query: $("#map-query").value.trim() || null,
      top_k: Number($("#map-top-k").value) || 5,
      group_by: $("#map-group").value,
    });
    renderEmbeddingMap($("#map-results"), result, (point) =>
      inspectSource({
        id: point.id,
        file: point.file,
        document_id: point.document_id,
        page: point.page,
      }),
    );
    status(
      "#map-status",
      result.points.length
        ? "Map drawn. Hover a point to read it; click to open the passage."
        : "Nothing to draw yet.",
    );
  } catch (error) {
    status("#map-status", error.message, true);
  } finally {
    $("#run-map").disabled = false;
  }
};

/* ---- Experiments: presets, question sets, the judge ------------------------ */
const evalSets = initEvalSets({
  $,
  el,
  button,
  json,
  post,
  request,
  status,
  inspectSource,
});
const EXPERIMENT_MODES = {
  retrieval: { mode: "retrieval", use_llm: false, judge: false },
  "retrieval-llm": { mode: "retrieval", use_llm: true, judge: false },
  full: { mode: "full", use_llm: false, judge: false },
  "full-judge": { mode: "full", use_llm: false, judge: true },
};
let customPresetForm = null,
  maxRuns = 12;
const customPresets = [];
function renderCustomPresets() {
  const container = $("#custom-presets");
  container.replaceChildren();
  for (const preset of customPresets) {
    const item = el("li");
    const settings = Object.entries(preset.retrieval)
      .map(([name, value]) => `${name}=${settingText(value)}`)
      .join(", ");
    item.append(
      el("span", `${preset.name}: ${settings || "the dense baseline"}`),
      button("Remove", () => {
        customPresets.splice(customPresets.indexOf(preset), 1);
        renderCustomPresets();
      }),
    );
    container.append(item);
  }
  $("#custom-preset-count").textContent = customPresets.length
    ? `(${customPresets.length} added)`
    : "";
}
async function initExperiments() {
  try {
    const catalog = await json("/api/experiment-presets");
    maxRuns = catalog.max_runs;
    const container = $("#experiment-presets");
    container.replaceChildren();
    for (const preset of catalog.presets) {
      const label = el("label"),
        check = el("input");
      check.type = "checkbox";
      check.name = "preset";
      check.value = preset.name;
      check.checked = preset.default;
      label.title =
        Object.entries(preset.settings)
          .map(([name, value]) => `${name}=${settingText(value)}`)
          .join(", ") || "top 5, every optional technique off";
      label.append(check, el("span", preset.label));
      if (preset.needs_chat) {
        const badge = el("span", "LLM", "badge");
        badge.title =
          "Uses the chat model during retrieval: choose a mode other than Retrieval only";
        label.append(badge);
      }
      container.append(label);
    }
    $("#judge-note").textContent =
      `The judge is ${catalog.judge.model}` +
      (catalog.judge.same_as_chat
        ? ", the same model that writes the answers, so expect flattering scores (set evaluation.judge_model to use another). "
        : ". ") +
      "Its scores are a model's opinion and are noisy on small models: compare averages and spot-check rows by hand.";
    if (controlSpecs)
      customPresetForm = techniquesForm(
        $("#custom-preset-form"),
        "custom",
        null,
        true,
      );
  } catch (error) {
    status("#experiment-status", error.message, true);
  }
  await evalSets.refresh();
}
$("#add-custom-preset").onclick = () => {
  const name = $("#custom-preset-name").value.trim();
  try {
    if (!customPresetForm) throw new Error("The settings are not loaded yet.");
    if (!/^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$/.test(name))
      throw new Error(
        "Name the preset with letters, digits, '-' or '_' (at most 32 characters).",
      );
    if (
      customPresets.some((preset) => preset.name === name) ||
      document.querySelector(`input[name="preset"][value="${name}"]`)
    )
      throw new Error(`There is already a preset called "${name}".`);
    customPresets.push({ name, retrieval: customPresetForm.read() });
  } catch (error) {
    status("#experiment-status", error.message, true);
    return;
  }
  $("#custom-preset-name").value = "";
  customPresetForm.reset();
  renderCustomPresets();
  status("#experiment-status", `Preset "${name}" added to the next run.`);
};
$("#experiment-form").onsubmit = async (event) => {
  event.preventDefault();
  const presets = [
    ...document.querySelectorAll('input[name="preset"]:checked'),
  ].map((check) => check.value);
  if (!presets.length && !customPresets.length) {
    status("#experiment-status", "Choose at least one preset.", true);
    return;
  }
  if (presets.length + customPresets.length > maxRuns) {
    status(
      "#experiment-status",
      `At most ${maxRuns} presets per comparison.`,
      true,
    );
    return;
  }
  $("#run-experiment").disabled = true;
  try {
    await trackJob(
      "experiment",
      await post("/api/experiments", {
        presets,
        custom: customPresets,
        question_set: $("#experiment-set").value || "sample",
        ...EXPERIMENT_MODES[$("#experiment-mode").value],
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
  const full = report.mode === "full",
    judge = report.judge;
  container.append(
    links,
    el(
      "p",
      `Questions: ${report.question_set ?? "sample"} (${report.questions ?? "?"}) | ` +
        `Mode: ${report.mode}${!full && report.use_llm ? " with LLM help" : ""} | ` +
        `Corpus revision: ${report.corpus_revision}` +
        (judge
          ? ` | Judge: ${judge.model}${judge.same_as_chat ? " (grading its own answers)" : ""}`
          : ""),
      "muted",
    ),
  );
  // Only the columns this run can fill: no answers, no answer metrics.
  const columns = [
    ["Recall", "recall"],
    ["MRR", "mrr"],
    ["nDCG", "ndcg"],
  ];
  if (full)
    columns.push(
      ["Keyword coverage", "keyword_coverage"],
      ["Answers citing", "citation_rate"],
      ["Cited from expected file", "citation_precision"],
    );
  if (judge)
    columns.push(
      ["Faithfulness", "faithfulness"],
      ["Answer relevance", "answer_relevance"],
      ["Correctness", "correctness"],
    );
  const table = el("table"),
    head = el("tr");
  for (const label of ["Preset", ...columns.map(([label]) => label), "Time (ms)"])
    head.append(el("th", label));
  table.append(head);
  for (const run of report.runs) {
    const row = el("tr"),
      name = el("td", run.preset);
    name.title =
      Object.entries(run.settings ?? {})
        .map(([key, value]) => `${key}=${settingText(value)}`)
        .join(", ") || "the dense baseline";
    row.append(name);
    for (const [, key] of columns) {
      const value = run.summary[key];
      row.append(el("td", value == null ? "Not scored" : value.toFixed(3)));
    }
    row.append(el("td", run.elapsed_ms));
    table.append(row);
  }
  const wrap = el("div", undefined, "table-wrap report-summary");
  wrap.append(table);
  container.append(wrap);
  for (const note of report.notes ?? [])
    container.append(el("p", note, "muted"));
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
      const scores = result.judge
        ? ` | faithful ${result.judge.faithfulness?.toFixed(2) ?? "-"}, relevant ${result.judge.relevance?.toFixed(2) ?? "-"}`
        : "";
      entry.append(
        el(
          "summary",
          `${!result.scored ? "Not scored" : result.passed ? "Pass" : "Check"}: ${result.question.question}${scores}`,
        ),
      );
      // What the judge decided, claim by claim, before the raw record.
      for (const claim of result.judge?.claims ?? [])
        entry.append(
          el(
            "p",
            `${claim.supported ? "Supported" : "NOT supported"}: ${claim.claim}`,
            claim.supported ? "muted" : "error",
          ),
        );
      entry.append(el("pre", JSON.stringify(result, null, 2)));
      details.append(entry);
    }
    container.append(details);
  }
}
/* ---- Learn: progress, quiz, guided labs ----------------------------------- */
/* Take the learner to the tool a lab uses, with its question and settings
   filled in. Nothing runs until they press the button there. */
async function openLab(lab) {
  const ready = "Lab loaded. Read the settings, then press";
  if (lab.tool === "compare") {
    showPanel("compare", true);
    $("#compare-question").value = lab.question;
    compareA?.apply(lab.a);
    compareB?.apply(lab.b);
    $("#compare-llm").checked = Boolean(lab.use_llm);
    status("#compare-status", `${ready} Compare.`);
  } else if (lab.tool === "ask") {
    showPanel("ask", true);
    $("#question").value = lab.question;
    askTechniques?.apply(lab.a);
    $("#ask-techniques").open = Object.keys(lab.a ?? {}).length > 0;
    $("#debug").checked = true;
    status("#ask-status", `${ready} Ask.`);
  } else if (lab.tool === "map") {
    showPanel("lab", true);
    $("#map-query").value = lab.question;
    $("#map-form").scrollIntoView({ block: "start" });
    status("#map-status", `${ready} Draw map.`);
  } else if (lab.tool === "playground") {
    showPanel("lab", true);
    await initLab();
    const name = lab.document.split("/").pop();
    const option = [...$("#playground-document").options].find((item) =>
      item.textContent.startsWith(`${name} `),
    );
    if (option) $("#playground-document").value = option.value;
    playgroundA?.apply(lab.a);
    playgroundB?.apply(lab.b);
    status(
      "#playground-status",
      option
        ? `${ready} Chunk it.`
        : `This lab uses ${name} from the sample corpus, which is not indexed here. Pick another document, or ingest the samples.`,
      !option,
    );
  } else if (lab.tool === "experiment") {
    showPanel("experiments", true);
    for (const check of document.querySelectorAll('input[name="preset"]'))
      check.checked = lab.presets.includes(check.value);
    $("#experiment-set").value = "sample";
    $("#experiment-mode").value = lab.use_llm ? "retrieval-llm" : "retrieval";
    status("#experiment-status", `${ready} Run comparison.`);
  }
}
const learn = initLearn({ $, el, button, json, post, status, openLab });

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
      extraction: cfg.extraction,
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
  await initTechniques();
  await initConversations();
  await initExperiments();
  try {
    const cfg = await json("/api/config");
    $("#upload-limits").textContent =
      `Up to ${cfg.server.upload_max_files} files, ${Math.round(cfg.server.upload_max_bytes / 1024 / 1024)} MiB each. TXT, MD, PDF, DOCX, PPTX, XLSX, HTML, CSV${cfg.extraction.code_files ? ", source code (C, C++, Python, JS/TS, Java, C#, Go, Rust)" : ""}${cfg.ocr.enabled ? ", images (OCR)" : ""}.`;
  } catch {}
})();
