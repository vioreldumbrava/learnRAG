/* Incremental SSE decoding: arbitrary byte boundaries, UTF-8 and CRLF. */
export class SSEParser {
  constructor(onEvent) {
    this.buffer = "";
    this.lines = [];
    this.onEvent = onEvent;
  }
  feed(text) {
    this.buffer += text;
    let match;
    while ((match = /[\r\n]/.exec(this.buffer))) {
      const index = match.index;
      if (this.buffer[index] === "\r" && index === this.buffer.length - 1)
        break;
      const line = this.buffer.slice(0, index);
      const width =
        this.buffer[index] === "\r" && this.buffer[index + 1] === "\n" ? 2 : 1;
      this.buffer = this.buffer.slice(index + width);
      if (!line) {
        const data = this.lines.join("\n");
        this.lines = [];
        if (data) this.onEvent(data);
      } else if (line.startsWith("data:")) {
        this.lines.push(line.slice(5).replace(/^ /, ""));
      }
    }
  }
}

export async function consumeSSE(response, onEvent, signal) {
  if (!response.body) throw new Error("Streaming response has no body");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let completed = false;
  const parser = new SSEParser((data) => {
    if (completed) return;
    if (data === "[DONE]") {
      completed = true;
      return;
    }
    const event = JSON.parse(data);
    if (event.error)
      throw new Error(event.error.message || String(event.error));
    onEvent(event);
  });
  try {
    for (;;) {
      if (signal?.aborted) throw new DOMException("Stopped", "AbortError");
      const { done, value } = await reader.read();
      if (done) {
        parser.feed(decoder.decode());
        break;
      }
      parser.feed(decoder.decode(value, { stream: true }));
      if (completed) break;
    }
    if (!completed)
      throw new Error(
        "Connection ended before the answer completed. Please retry.",
      );
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}
