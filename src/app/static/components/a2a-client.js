export const GENERIC_AGENT_ERROR =
  "The agent could not complete the request. Check the server logs and configuration.";

export class A2AStreamingClient {
  constructor(card) {
    this.card = card;
    this.url = sameOriginA2AUrl(card.url);
  }

  static async discover() {
    return A2AStreamingClient.discoverFromUrl("/.well-known/agent-card.json");
  }

  static async discoverFromUrl(cardUrl) {
    const response = await fetch(cardUrl);
    if (!response.ok) {
      throw new Error("Agent discovery failed.");
    }
    return new A2AStreamingClient(await response.json());
  }

  async *streamRequest(payload) {
    const response = await fetch(this.url, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(messageStreamRequest(payload)),
    });
    if (!response.ok || !response.body) {
      throw new Error(await streamErrorMessage(response));
    }
    yield* parseSse(response.body);
  }

  async *streamReview(payload) {
    yield* this.streamRequest(payload);
  }
}

async function streamErrorMessage(response) {
  await response.text();
  return GENERIC_AGENT_ERROR;
}

export function consumeA2AEvent(eventData) {
  const result = eventData.result || eventData;
  const update =
    result.statusUpdate ||
    result.status_update ||
    result.artifactUpdate ||
    result.artifact_update ||
    result;
  const message = update.status?.message || result.message || result.msg;
  const artifact = update.artifact || result.artifact;
  if (message?.role === "user") return null;
  const text = extractText(message) || extractText(artifact);
  if (!text) return null;
  const displayText = sanitizeAgentText(text);
  const receivedAt = new Date().toLocaleTimeString();
  const streamTrace = traceFromStateDelta(update, result, receivedAt);

  if (isFailure(update, result)) {
    return {
      type: "error",
      error: GENERIC_AGENT_ERROR,
      trace: traceEvent(update, result, GENERIC_AGENT_ERROR, receivedAt),
      traceBatch: streamTrace,
    };
  }

  const parsed = tryJson(text);
  if (parsed && isFinalPayload(parsed)) {
    return { type: "result", result: parsed, traceBatch: streamTrace };
  }
  return {
    type: "trace",
    trace: traceEvent(update, result, displayText, receivedAt),
    traceBatch: streamTrace,
  };
}

function messageStreamRequest(payload) {
  return {
    jsonrpc: "2.0",
    id: crypto.randomUUID(),
    method: "message/stream",
    params: {
      message: {
        kind: "message",
        messageId: crypto.randomUUID(),
        role: "user",
        parts: [{ kind: "text", text: JSON.stringify(payload) }],
      },
      configuration: {
        acceptedOutputModes: ["text/plain", "application/json"],
      },
    },
  };
}

function sameOriginA2AUrl(cardUrl) {
  const endpoint = new URL(cardUrl, window.location.origin);
  return new URL(endpoint.pathname, window.location.origin).toString();
}

async function* parseSse(body) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
    const events = buffer.split("\n\n");
    buffer = events.pop() || "";
    for (const event of events) {
      const data = event
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trim())
        .join("\n");
      if (data) {
        yield JSON.parse(data);
      }
    }
  }
}

function extractText(container) {
  if (!container || !Array.isArray(container.parts)) return "";
  return container.parts
    .map((part) => part.text || part.root?.text || part.data?.text || "")
    .filter(Boolean)
    .join("\n");
}

function isFailure(update, result) {
  return update.status?.state === "failed" || result.status?.state === "failed";
}

function sanitizeAgentText(text) {
  if (/OPENROUTER_API_KEY|OpenRouter|LiteLLM|API key|Traceback|RuntimeError/i.test(text)) {
    return GENERIC_AGENT_ERROR;
  }
  return text;
}

function traceEvent(update, result, text, receivedAt) {
  return {
    stage:
      extractStage(update.metadata || result.metadata) ||
      update.status?.state ||
      result.kind ||
      "working",
    message: text,
    received_at: receivedAt,
  };
}

function traceFromStateDelta(update, result, receivedAt) {
  const metadata = update.metadata || result.metadata || {};
  const trace = metadata.adk_actions?.stateDelta?.trace;
  if (!Array.isArray(trace)) return [];
  return trace
    .filter((event) => event && event.stage)
    .map((event) => ({
      stage: String(event.stage),
      message: String(event.message || ""),
      received_at: event.received_at || receivedAt,
      duration_ms: Number(event.duration_ms) || null,
    }));
}

function tryJson(text) {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

function extractStage(metadata) {
  const raw = metadata?.adk_custom_metadata;
  if (!raw) return "";
  return String(raw).match(/'stage': '([^']+)'/)?.[1] || "";
}

function isFinalPayload(parsed) {
  return (
    (parsed.rating && parsed.review) ||
    Array.isArray(parsed.recommendations)
  );
}
