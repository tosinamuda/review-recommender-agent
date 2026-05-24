import { A2AStreamingClient, GENERIC_AGENT_ERROR, consumeA2AEvent } from "../a2a-client.js";
import "./task-one-form.js";
import "./task-one-result.js";

export class TaskOnePanel extends HTMLElement {
  constructor() {
    super();
    this.client = null;
    this.trace = [];
    this.result = null;
    this.error = "";
    this.running = false;
    this.elapsedMs = 0;
  }

  connectedCallback() {
    this.render();
    this.discoverAgent();
    this.addEventListener("task-one-submit", (event) => this.runSimulation(event.detail));
  }

  async discoverAgent() {
    try {
      this.client = await A2AStreamingClient.discover();
      this.setStatus("Agent · ready", true, "ready");
    } catch (error) {
      this.setStatus("Agent unavailable", false, "unavailable");
    }
  }

  async runSimulation(payload) {
    if (!this.client) {
      await this.discoverAgent();
    }
    if (!this.client) {
      this.result = null;
      this.error = GENERIC_AGENT_ERROR;
      this.trace = [{ stage: "a2a_error", message: GENERIC_AGENT_ERROR }];
      this.running = false;
      this.elapsedMs = 0;
      this.setStatus("Agent unavailable", false, "unavailable");
      this.syncChildren();
      return;
    }
    this.trace = [];
    this.result = null;
    this.error = "";
    this.running = true;
    this.elapsedMs = 0;
    const startedAt = performance.now();
    this.setStatus("Agent · running", true, "running");
    this.syncChildren();

    try {
      for await (const eventData of this.client.streamReview(payload)) {
        this.consumeStreamEvent(eventData);
        this.syncChildren();
      }
    } catch (error) {
      this.error = GENERIC_AGENT_ERROR;
      this.trace.push({ stage: "a2a_error", message: GENERIC_AGENT_ERROR });
    } finally {
      this.running = false;
      this.elapsedMs = Math.max(0, performance.now() - startedAt);
      this.setStatus("Agent · ready", true, "ready");
      this.syncChildren();
    }
  }

  consumeStreamEvent(eventData) {
    const consumed = consumeA2AEvent(eventData);
    if (!consumed) return;
    const traceBatch = consumed.traceBatch?.length ? consumed.traceBatch : [];
    if (consumed.type === "result") {
      this.result = consumed.result;
      this.trace = mergeTrace(this.trace, traceBatch);
      this.trace = mergeTrace(this.trace, consumed.result.trace || []);
      return;
    }
    if (consumed.type === "error") {
      this.error = consumed.error;
      this.trace = mergeTrace(this.trace, traceBatch);
      this.trace = mergeTrace(this.trace, [consumed.trace]);
      return;
    }
    this.trace = mergeTrace(this.trace, traceBatch);
    this.trace = mergeTrace(this.trace, [consumed.trace]);
  }

  setStatus(message, ready, state = "ready") {
    this.dispatchEvent(
      new CustomEvent("agent-status-change", {
        bubbles: true,
        detail: { message, ready, state },
      }),
    );
  }

  syncChildren() {
    const form = this.querySelector("bct-task-one-form");
    const result = this.querySelector("bct-task-one-result");
    if (form) {
      form.busy = this.running;
      if (this.result && !this.running) {
        form.completed = {
          elapsed: (this.elapsedMs / 1000).toFixed(2),
          attempts: this.result.evidence?.verifier_attempts || 1,
        };
      }
    }
    if (result) {
      result.updateState({
        error: this.error,
        running: this.running,
        elapsedMs: this.elapsedMs,
        trace: this.trace,
        result: this.result,
      });
    }
  }

  render() {
    this.innerHTML = `
      <section class="task-page">
        ${breadcrumb("Tasks", "Task 1 · User modelling")}
        <header class="page-header">
          <h1 class="page-title">Review simulation</h1>
          <p class="page-subtitle">
            Given a raw user persona and product details, generate the review and calibrate the rating from the selected review.
          </p>
        </header>
        <div class="dashboard-grid dashboard-grid--task-one">
          <bct-task-one-form></bct-task-one-form>
          <bct-task-one-result></bct-task-one-result>
        </div>
      </section>
    `;
  }
}

function mergeTrace(liveTrace, finalTrace) {
  const merged = [];
  const stageIndex = new Map();
  for (const event of [...liveTrace, ...finalTrace]) {
    if (!event?.stage) continue;
    const existingIndex = stageIndex.get(event.stage);
    if (existingIndex === undefined) {
      stageIndex.set(event.stage, merged.length);
      merged.push(event);
      continue;
    }
    const existing = merged[existingIndex];
    merged[existingIndex] = {
      ...existing,
      ...event,
      received_at: existing.received_at || event.received_at || "",
    };
  }
  return merged;
}

customElements.define("bct-task-one-panel", TaskOnePanel);

function breadcrumb(section, current) {
  return `
    <nav class="breadcrumb" aria-label="Breadcrumb">
      <span>${section}</span>
      <span class="breadcrumb__sep">/</span>
      <span class="breadcrumb__current">${current}</span>
    </nav>
  `;
}
