import assert from "node:assert/strict";
import test from "node:test";
import {
  citationNumbers,
  splitCitations,
} from "../../src/rag_app/webui/citations.js";

const cited = (text) =>
  splitCitations(text)
    .filter((part) => part.numbers)
    .flatMap((part) => part.numbers);

/* The same cases as tests/test_citations.py: the browser and the server must
   agree on what a citation is. */
test("finds the markers the server finds", () => {
  assert.deepEqual(cited("SMPR is ignored while EN is set [1]."), [1]);
  assert.deepEqual(cited("Two facts [1][3]."), [1, 3]);
  assert.deepEqual(cited("Grouped [1, 2] and ranged [2-3]."), [1, 2, 2, 3]);
  assert.deepEqual(cited("Older style [Source 2] and [sources 1; 3]."), [2, 1, 3]);
  assert.deepEqual(cited("Made up [7] next to real [2]."), [7, 2]);
  assert.deepEqual(cited("No citations at all."), []);
});

test("code and bit ranges are not citations", () => {
  const answer =
    "Read `table[2]` and buf[0], then check bits [5:4] of RES. " +
    "The array index arr[1] is not a source. Only this is [3].";
  assert.deepEqual(cited(answer), [3]);
});

test("ranges expand only when they are plausible", () => {
  assert.deepEqual(citationNumbers("1, 3-5"), [1, 3, 4, 5]);
  assert.deepEqual(citationNumbers("Source 2; source 4"), [2, 4]);
  assert.deepEqual(citationNumbers("1-900"), [1, 900]);
  assert.deepEqual(citationNumbers("5-3"), [5, 3]);
});

test("splitting keeps every character of the answer", () => {
  const text = "First [1], then [2, 3].\nA new line and code[4].";
  const parts = splitCitations(text);
  assert.equal(parts.map((part) => part.text ?? part.marker).join(""), text);
  assert.deepEqual(parts[1], { marker: "[1]", numbers: [1] });
  assert.deepEqual(parts[3], { marker: "[2, 3]", numbers: [2, 3] });
  assert.equal(parts.length, 5); // "code[4]" stays inside the last text part
  assert.deepEqual(splitCitations(""), []);
});
