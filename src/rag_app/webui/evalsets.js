/* Question sets: the part of the Experiments panel where you build the
   questions an experiment asks, and the dialog that saves a question from
   the Ask tab. Everything from the server is placed with textContent. */

const NEW_SET = "\u0000new";
const SAMPLE = "sample";
const TERMINAL = ["completed", "cancelled", "failed", "interrupted"];

function list(text) {
  return text
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean);
}

/* `deps` are the page's own helpers: $, el, button, json, post, request,
   status, inspectSource. Returns `{ refresh, saveQuestion }`. */
export function initEvalSets(deps) {
  const { $, el, button, json, post, request, status, inspectSource } = deps;
  let sets = [];
  let lastSavedTo = "";

  const say = (text, error = false) => status("#evalset-status", text, error);
  const put = (url, data) =>
    json(url, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
  const url = (name, rest = "") =>
    `/api/eval-sets/${encodeURIComponent(name)}${rest}`;

  /* The set that add / draft / upload write to: the selected one, or the
     name typed for a new one. */
  function target() {
    const selected = $("#evalset-select").value;
    const name =
      selected === NEW_SET ? $("#evalset-new-name").value.trim() : selected;
    if (!name) throw new Error("Type a name for the new set.");
    if (name === SAMPLE)
      throw new Error(
        'The bundled "sample" set is read-only. Choose "+ New set" and give yours a name.',
      );
    return name;
  }

  function fillSelects(preferred) {
    const experiment = $("#experiment-set"),
      manage = $("#evalset-select"),
      datalist = $("#evalq-sets");
    const chosenRun = experiment.value || SAMPLE,
      chosen = preferred ?? manage.value;
    experiment.replaceChildren();
    manage.replaceChildren();
    datalist.replaceChildren();
    for (const set of sets) {
      const label =
        `${set.name} (${set.questions} question${set.questions === 1 ? "" : "s"}` +
        `${set.drafts ? `, ${set.drafts} draft${set.drafts === 1 ? "" : "s"}` : ""})`;
      for (const select of [experiment, manage]) {
        const option = el("option", label);
        option.value = set.name;
        // A set with nothing accepted cannot be measured yet.
        if (select === experiment) option.disabled = !set.questions;
        select.append(option);
      }
      if (!set.builtin) {
        const option = el("option");
        option.value = set.name;
        datalist.append(option);
      }
    }
    const fresh = el("option", "+ New set");
    fresh.value = NEW_SET;
    manage.append(fresh);
    const names = sets.map((set) => set.name);
    experiment.value = names.includes(chosenRun) ? chosenRun : SAMPLE;
    manage.value = names.includes(chosen) || chosen === NEW_SET ? chosen : SAMPLE;
  }

  function renderRows(data) {
    const container = $("#evalset-rows");
    container.replaceChildren();
    const writable = !data.builtin;
    $("#evalset-export").hidden = false;
    $("#evalset-export").href = url(data.name, "/export");
    $("#evalset-delete").hidden = !writable;
    $("#evalset-accept-all").hidden = !(writable && data.drafts);
    if (!data.items.length) {
      container.append(
        el("p", "This set has no questions yet. Add one below.", "muted"),
      );
      return;
    }
    const table = el("table", undefined, "stage-table"),
      head = el("tr");
    for (const title of ["#", "Status", "Origin", "Question", "Expected files", ""])
      head.append(el("th", title));
    table.append(head);
    data.items.forEach((item, index) => {
      const row = el("tr");
      if (item.status === "draft") row.className = "draft";
      const cells = [
        ["rank", String(index)],
        ["rank", item.status === "draft" ? "Draft" : "Accepted"],
        ["rank", item.origin],
        ["preview", item.question],
        [
          "file",
          (item.expected_sources.length
            ? item.expected_sources
            : Object.keys(item.expected_relevance)
          ).join(", "),
        ],
      ];
      for (const [column, text] of cells) {
        const cell = el("td", text);
        cell.dataset.col = column;
        row.append(cell);
      }
      if (item.reference_answer) row.title = `Reference answer: ${item.reference_answer}`;
      const actions = el("td");
      actions.dataset.col = "actions";
      if (item.source_chunk_id)
        actions.append(
          button("Passage", () =>
            inspectSource({ id: item.source_chunk_id, file: item.expected_sources[0] ?? "?" }),
          ),
        );
      if (writable && item.status === "draft")
        actions.append(
          button("Accept", () =>
            change(() => post(url(data.name, "/accept"), { indexes: [index] }), "Accepted."),
          ),
        );
      if (writable)
        actions.append(
          button("Remove", () =>
            change(
              () => json(url(data.name, `/questions/${index}`), { method: "DELETE" }),
              "Removed.",
            ),
          ),
        );
      row.append(actions);
      table.append(row);
    });
    const wrap = el("div", undefined, "table-wrap");
    wrap.append(table);
    container.append(wrap);
  }

  async function show(name) {
    $("#evalset-new-name").hidden = name !== NEW_SET;
    if (name === NEW_SET) {
      $("#evalset-rows").replaceChildren(
        el("p", "Name the new set, then add, draft or upload questions below.", "muted"),
      );
      for (const id of ["#evalset-export", "#evalset-delete", "#evalset-accept-all"])
        $(id).hidden = true;
      return;
    }
    try {
      renderRows(await json(url(name)));
    } catch (error) {
      say(error.message, true);
    }
  }

  async function refresh(preferred) {
    try {
      sets = (await json("/api/eval-sets")).sets;
      fillSelects(preferred);
      await show($("#evalset-select").value);
    } catch (error) {
      say(error.message, true);
    }
  }

  /* Run one change, then show the set it changed. */
  async function change(action, done, name) {
    try {
      const result = await action();
      await refresh(name ?? result?.name);
      say(done);
      return true;
    } catch (error) {
      say(error.message, true);
      return false;
    }
  }

  $("#evalset-select").onchange = () => {
    say("");
    show($("#evalset-select").value);
  };
  $("#evalset-accept-all").onclick = () => {
    const name = $("#evalset-select").value;
    change(() => post(url(name, "/accept"), {}), "All drafts accepted.");
  };
  $("#evalset-delete").onclick = async () => {
    const name = $("#evalset-select").value;
    if (!confirm(`Delete question set "${name}"? This cannot be undone.`)) return;
    try {
      await request(url(name), { method: "DELETE" });
      await refresh(SAMPLE);
      say(`Deleted "${name}".`);
    } catch (error) {
      say(error.message, true);
    }
  };

  $("#evalset-add-form").onsubmit = async (event) => {
    event.preventDefault();
    let name;
    try {
      name = target();
    } catch (error) {
      say(error.message, true);
      return;
    }
    const question = $("#evalset-question").value.trim();
    if (!question) {
      say("Write the question first.", true);
      return;
    }
    const saved = await change(
      () =>
        post(url(name, "/questions"), {
          question,
          expected_sources: list($("#evalset-sources").value),
          expected_contains: list($("#evalset-contains").value),
          reference_answer: $("#evalset-reference").value.trim(),
        }),
      `Added to "${name}".`,
      name,
    );
    if (saved)
      for (const id of ["#evalset-question", "#evalset-sources", "#evalset-contains", "#evalset-reference"])
        $(id).value = "";
  };

  $("#evalset-upload-form").onsubmit = async (event) => {
    event.preventDefault();
    const file = $("#evalset-upload-file").files[0];
    let name, rows;
    try {
      name = target();
      if (!file) throw new Error("Choose a JSON file.");
      rows = JSON.parse(await file.text());
      if (rows && !Array.isArray(rows) && Array.isArray(rows.questions)) rows = rows.questions;
      if (!Array.isArray(rows) || !rows.length)
        throw new Error("The file must hold a non-empty JSON list of questions.");
    } catch (error) {
      say(error instanceof SyntaxError ? "That file is not valid JSON." : error.message, true);
      return;
    }
    const replace = $("#evalset-upload-replace").checked;
    const saved = await change(
      () =>
        replace
          ? put(url(name), { questions: rows })
          : post(url(name, "/questions"), rows),
      `${rows.length} question(s) ${replace ? "now make up" : "added to"} "${name}".`,
      name,
    );
    if (saved) $("#evalset-upload-form").reset();
  };

  $("#evalset-generate-form").onsubmit = async (event) => {
    event.preventDefault();
    let name;
    try {
      name = target();
    } catch (error) {
      say(error.message, true);
      return;
    }
    const count = Number($("#evalset-generate-count").value);
    if (!Number.isInteger(count) || count < 1 || count > 50) {
      say("Choose between 1 and 50 questions.", true);
      return;
    }
    const box = $("#evalset-generate-job"),
      run = $("#evalset-generate-run");
    run.disabled = true;
    say("The chat model is writing questions...");
    try {
      let job = await post(url(name, "/generate"), { count });
      await refresh(name); // a new set exists as soon as the job is accepted
      while (!TERMINAL.includes(job.state)) {
        const progress = job.progress;
        box.replaceChildren(
          el(
            "p",
            progress
              ? `${progress.current}/${progress.total} | ${progress.path}`
              : "Starting...",
          ),
          button("Stop", async () => {
            try {
              await post(`/api/jobs/${job.id}/cancel`, {});
            } catch (error) {
              say(error.message, true);
            }
          }),
        );
        await new Promise((resolve) => setTimeout(resolve, 700));
        job = await json(`/api/jobs/${job.id}`);
      }
      box.replaceChildren();
      await refresh(name);
      if (job.state !== "completed")
        say(`Drafting ${job.state}${job.error ? `: ${job.error}` : "."}`, true);
      else if (!job.result.asked)
        say("There are no chunks to write questions from. Add documents first.", true);
      else
        say(
          `${job.result.added} draft(s) added` +
            (job.result.skipped
              ? `, ${job.result.skipped} unusable reply/replies skipped`
              : "") +
            ". Read each one, then accept or remove it.",
        );
    } catch (error) {
      box.replaceChildren();
      say(error.message, true);
    } finally {
      run.disabled = false;
    }
  };

  /* ---- Save a question from the Ask tab ---------------------------------- */
  const dialog = $("#evalq-dialog");
  $("#evalq-cancel").onclick = () => dialog.close();

  /* `turn` = { question, answer, sources, citations } of a completed answer. */
  function saveQuestion(turn) {
    $("#evalq-set").value = lastSavedTo || "my_questions";
    $("#evalq-question").value = turn.question;
    $("#evalq-reference").value = turn.answer;
    status("#evalq-status", "");
    const cited = new Set(
      (turn.citations?.cited ?? []).map((number) => turn.sources[number - 1]?.file),
    );
    const files = [...new Set(turn.sources.map((source) => source.file))];
    const box = $("#evalq-sources");
    box.replaceChildren();
    if (!files.length) box.append(el("span", "No sources were retrieved.", "muted"));
    for (const file of files) {
      const label = el("label"),
        check = el("input");
      check.type = "checkbox";
      check.value = file;
      // Start from what the answer cited; with no citations, from everything.
      check.checked = cited.size ? cited.has(file) : true;
      label.append(check, el("span", file.replace(/^[0-9a-f]{32}_/, "")));
      box.append(label);
    }
    dialog.showModal();
  }

  $("#evalq-form").onsubmit = async (event) => {
    event.preventDefault();
    const name = $("#evalq-set").value.trim(),
      question = $("#evalq-question").value.trim();
    if (!name || !question) {
      status("#evalq-status", "A set name and a question are needed.", true);
      return;
    }
    try {
      await post(url(name, "/questions"), {
        question,
        expected_sources: [
          ...document.querySelectorAll("#evalq-sources input:checked"),
        ].map((check) => check.value),
        reference_answer: $("#evalq-reference").value.trim(),
        origin: "conversation",
      });
      lastSavedTo = name;
      dialog.close();
      status("#ask-status", `Saved to question set "${name}".`);
      refresh();
    } catch (error) {
      status("#evalq-status", error.message, true);
    }
  };

  return { refresh, saveQuestion };
}
