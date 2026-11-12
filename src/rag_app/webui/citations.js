/* Find citation markers in an answer so they can be drawn as links.
   Same rule as retrieval/citations.py: "[1]", "[2, 3]", "[1-3]" and the older
   "[Source 1]"; a bracket straight after a letter, digit or underscore is code
   (table[3]), not a citation. tests/js/test_citations.mjs keeps the two in
   step. */
const CITATION =
  /(?<![A-Za-z0-9_])\[\s*(?:sources?\s*)?(\d+(?:\s*[,;\-]\s*(?:sources?\s*)?\d+)*)\s*\]/gi;
const RANGE_LIMIT = 50;

export function citationNumbers(body) {
  const numbers = [];
  for (const part of body.replace(/sources?/gi, "").split(/[,;]/)) {
    const bounds = (part.match(/\d+/g) || []).map(Number);
    const span = bounds[1] - bounds[0];
    if (
      bounds.length === 2 &&
      part.includes("-") &&
      span > 0 &&
      span <= RANGE_LIMIT
    ) {
      for (let n = bounds[0]; n <= bounds[1]; n++) numbers.push(n);
    } else numbers.push(...bounds);
  }
  return numbers;
}

/* "A [1] b" -> [{text: "A "}, {marker: "[1]", numbers: [1]}, {text: " b"}] */
export function splitCitations(text) {
  const parts = [];
  let cursor = 0;
  for (const match of text.matchAll(CITATION)) {
    if (match.index > cursor)
      parts.push({ text: text.slice(cursor, match.index) });
    parts.push({ marker: match[0], numbers: citationNumbers(match[1]) });
    cursor = match.index + match[0].length;
  }
  if (cursor < text.length) parts.push({ text: text.slice(cursor) });
  return parts;
}
