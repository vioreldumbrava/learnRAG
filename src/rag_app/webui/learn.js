/* The Learn panel: progress per stage, the quiz, and guided labs.
   Everything from the server is placed with textContent. */

const VERDICTS = [
  ["knew", "I knew it"],
  ["partly", "Partly"],
  ["missed", "I missed it"],
];
const percent = (value) =>
  value === null || value === undefined ? "-" : `${Math.round(value * 100)}%`;

/* `deps`: $, el, button, json, post, status, openLab(lab).
   Returns `{ refresh }`, called whenever the panel is shown. */
export function initLearn(deps) {
  const { $, el, button, json, post, status, openLab } = deps;
  let round = [],
    position = 0,
    answered = 0;

  const put = (url, data) =>
    json(url, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });

  /* ---- progress ---------------------------------------------------------- */
  function renderProgress(summary) {
    $("#learn-summary").textContent =
      `Quiz: ${summary.known} known of ${summary.attempted} tried, ${summary.questions} in all` +
      (summary.mastery === null ? "" : ` (average ${percent(summary.mastery)})`) +
      `. Checkpoints: ${summary.checkpoints_done} of ${summary.checkpoints}. ` +
      `Labs opened: ${summary.labs_opened} of ${summary.labs}.`;
    const container = $("#learn-stages");
    // Keep the stages the learner has open, open.
    const open = new Set(
      [...container.querySelectorAll("details[open]")].map((node) => node.dataset.stage),
    );
    container.replaceChildren();
    for (const stage of summary.stages) {
      const details = el("details", undefined, "learn-stage");
      details.dataset.stage = String(stage.stage);
      details.open = open.has(String(stage.stage));
      const quiz = stage.questions
        ? `quiz ${stage.known}/${stage.questions} known${stage.mastery === null ? "" : `, ${percent(stage.mastery)}`}`
        : "no quiz questions";
      details.append(
        el(
          "summary",
          `${stage.stage}. ${stage.title} | ${quiz} | checkpoints ${stage.checkpoints_done}/${stage.checkpoints.length}` +
            (stage.labs ? ` | labs ${stage.labs_opened}/${stage.labs}` : ""),
        ),
      );
      for (const item of stage.checkpoints) {
        const label = el("label", undefined, "checkpoint"),
          box = el("input");
        box.type = "checkbox";
        box.checked = item.done;
        box.addEventListener("change", async () => {
          try {
            renderProgress(
              await put(`/api/learn/checkpoints/${stage.stage}/${item.index}`, {
                done: box.checked,
              }),
            );
          } catch (error) {
            box.checked = !box.checked;
            status("#quiz-status", error.message, true);
          }
        });
        label.append(box, el("span", item.text));
        details.append(label);
      }
      if (stage.questions)
        details.append(
          button(`Quiz me on stage ${stage.stage}`, () => {
            $("#quiz-scope").value = `stage:${stage.stage}`;
            startRound();
            $("#quiz-form").scrollIntoView({ block: "start" });
          }),
        );
      container.append(details);
    }
  }

  /* ---- quiz -------------------------------------------------------------- */
  function fillScopes(filters) {
    const select = $("#quiz-scope"),
      chosen = select.value;
    select.replaceChildren();
    const add = (value, text) => {
      const option = el("option", text);
      option.value = value;
      select.append(option);
    };
    add("", "Everything (weakest first)");
    for (const stage of filters.stages)
      add(`stage:${stage.stage}`, `Stage ${stage.stage}: ${stage.title} (${stage.questions})`);
    for (const section of filters.sections)
      add(`section:${section.name}`, `Section: ${section.name} (${section.questions})`);
    if ([...select.options].some((option) => option.value === chosen)) select.value = chosen;
  }

  function showQuestion() {
    const card = $("#quiz-card");
    if (position >= round.length) {
      card.hidden = true;
      status(
        "#quiz-status",
        answered
          ? `Round finished: ${answered} answered. Weak answers come back first next time.`
          : "Round finished.",
      );
      refresh();
      return;
    }
    const item = round[position];
    card.hidden = false;
    $("#quiz-position").textContent =
      `Question ${position + 1} of ${round.length} | ${item.section} #${item.id}`;
    $("#quiz-question").textContent = item.question;
    $("#quiz-answer").value = "";
    $("#quiz-feedback").replaceChildren();
    for (const id of ["#quiz-check", "#quiz-reveal", "#quiz-skip"]) $(id).disabled = false;
    $("#quiz-answer").focus();
  }

  function nextButton() {
    return button(position + 1 < round.length ? "Next question" : "Finish", () => {
      position += 1;
      showQuestion();
    });
  }

  function showFeedback(result) {
    const box = $("#quiz-feedback");
    box.replaceChildren(el("h4", "Reference answer"), el("p", result.reference, "reference"));
    for (const id of ["#quiz-check", "#quiz-reveal", "#quiz-skip"]) $(id).disabled = true;
    if (result.score !== null) {
      // The model's verdict: which key points were there, which were not.
      box.append(el("p", `Model's verdict: ${percent(result.score)} of the key points covered.`));
      const points = el("ul", undefined, "quiz-points");
      for (const point of result.covered) points.append(el("li", `Covered: ${point}`, "ok"));
      for (const point of result.missing) points.append(el("li", `Missing: ${point}`, "error"));
      box.append(points, nextButton());
      answered += 1;
      return;
    }
    if (result.note) box.append(el("p", result.note, "muted"));
    box.append(el("p", "Compare with the reference. How did you do?"));
    const row = el("div", undefined, "controls");
    for (const [verdict, text] of VERDICTS)
      row.append(
        button(text, async () => {
          try {
            await post(`/api/learn/quiz/${encodeURIComponent(result.id)}/self`, { verdict });
            answered += 1;
            position += 1;
            showQuestion();
          } catch (error) {
            status("#quiz-status", error.message, true);
          }
        }),
      );
    box.append(row);
  }

  async function check(useModel) {
    const item = round[position],
      answer = $("#quiz-answer").value.trim();
    if (useModel && !answer) {
      status("#quiz-status", "Write an answer first, or use Show the answer.", true);
      return;
    }
    $("#quiz-check").disabled = $("#quiz-reveal").disabled = true;
    status("#quiz-status", useModel ? "The model is reading your answer..." : "");
    try {
      showFeedback(
        await post(`/api/learn/quiz/${encodeURIComponent(item.id)}/check`, {
          answer,
          use_model: useModel,
        }),
      );
      status("#quiz-status", "");
    } catch (error) {
      $("#quiz-check").disabled = $("#quiz-reveal").disabled = false;
      status("#quiz-status", error.message, true);
    }
  }

  async function startRound() {
    const scope = $("#quiz-scope").value,
      count = Number($("#quiz-count").value);
    if (!Number.isInteger(count) || count < 1 || count > 50) {
      status("#quiz-status", "Choose between 1 and 50 questions.", true);
      return;
    }
    const query = new URLSearchParams({ n: String(count) });
    if (scope.startsWith("stage:")) query.set("stage", scope.slice(6));
    if (scope.startsWith("section:")) query.set("section", scope.slice(8));
    try {
      round = (await json(`/api/learn/quiz?${query}`)).questions;
      position = answered = 0;
      status("#quiz-status", round.length ? "" : "No questions match.", !round.length);
      showQuestion();
    } catch (error) {
      status("#quiz-status", error.message, true);
    }
  }
  $("#quiz-form").onsubmit = (event) => {
    event.preventDefault();
    startRound();
  };
  $("#quiz-check").onclick = () => check(true);
  $("#quiz-reveal").onclick = () => check(false);
  $("#quiz-skip").onclick = () => {
    position += 1;
    showQuestion();
  };

  /* ---- guided labs ------------------------------------------------------- */
  function renderLabs(labs) {
    const container = $("#learn-labs");
    container.replaceChildren();
    for (const lab of labs) {
      const details = el("details", undefined, "learn-lab");
      details.append(
        el(
          "summary",
          `Stage ${lab.stage} | ${lab.title}${lab.opened ? " (opened)" : ""}`,
        ),
        el("p", lab.goal),
        el("p", "What to look for:", "muted"),
      );
      const list = el("ul");
      for (const point of lab.look_for) list.append(el("li", point));
      details.append(
        list,
        el("p", `From a terminal: ${lab.command}`, "muted"),
        button("Open this lab", async () => {
          try {
            await post(`/api/learn/labs/${encodeURIComponent(lab.id)}/open`, {});
            await openLab(lab);
          } catch (error) {
            status("#quiz-status", error.message, true);
          }
        }),
      );
      container.append(details);
    }
  }

  async function refresh() {
    try {
      const [summary, labs, quiz] = await Promise.all([
        json("/api/learn/progress"),
        json("/api/learn/labs"),
        json("/api/learn/quiz?n=1"),
      ]);
      renderProgress(summary);
      renderLabs(labs.labs);
      fillScopes(quiz);
    } catch (error) {
      status("#quiz-status", error.message, true);
    }
  }

  return { refresh };
}
