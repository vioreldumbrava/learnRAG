import test from "node:test";
import assert from "node:assert/strict";
import { SSEParser, consumeSSE } from "../../src/rag_app/webui/sse.js";

function response(text, width = 1) {
  const bytes = new TextEncoder().encode(text);
  return new Response(
    new ReadableStream({
      start(controller) {
        for (let i = 0; i < bytes.length; i += width)
          controller.enqueue(bytes.slice(i, i + width));
        controller.close();
      },
    }),
  );
}

test("arbitrary UTF-8 and CRLF boundaries preserve tokens", async () => {
  for (const width of [1, 2, 3, 9, 100]) {
    const events = [];
    await consumeSSE(
      response(
        ': heartbeat\r\ndata: {"token":"Grüße 👋\\n"}\r\n\r\ndata: [DONE]\r\n\r\n',
        width,
      ),
      (event) => events.push(event),
    );
    assert.deepEqual(events, [{ token: "Grüße 👋\n" }]);
  }
});

test("a single CRLF is not an event separator", () => {
  const events = [],
    parser = new SSEParser((event) => events.push(event));
  parser.feed("data: one\r\n");
  assert.equal(events.length, 0);
  parser.feed("data: two\r");
  parser.feed("\n\r");
  parser.feed("\n");
  assert.deepEqual(events, ["one\ntwo"]);
});

test("missing completion is an error even after partial tokens", async () => {
  await assert.rejects(
    consumeSSE(response('data: {"token":"partial"}\n\n'), () => {}),
    /before the answer completed/,
  );
});

test("provider error propagates and is never mistaken for success", async () => {
  await assert.rejects(
    consumeSSE(
      response('data: {"error":{"message":"model offline"}}\n\n'),
      () => {},
    ),
    /model offline/,
  );
});

test("cancellation does not produce a successful answer", async () => {
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(
    consumeSSE(response("data: [DONE]\n\n"), () => {}, controller.signal),
    { name: "AbortError" },
  );
});
