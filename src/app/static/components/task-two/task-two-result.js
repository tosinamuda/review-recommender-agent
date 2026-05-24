import { escapeHtml } from "../dom.js";

const RETRIEVE_STAGE = {
  id: "retrieve_candidate_items",
  title: "Retrieve candidate items",
  subtitle: "FAISS over persona, context, and joint catalogue axes",
};

const COVERAGE_STAGE = {
  id: "judge_candidate_coverage",
  title: "Judge candidate coverage",
  subtitle: "LLM checks whether retrieved items fit the requested context",
};

const RANK_STAGE = {
  id: "rank_and_reason",
  title: "Rank and reason about candidates",
  subtitle: "single LLM call ranks the coverage-approved items",
};

const LIMITED_STAGE = {
  id: "generate_contextual_recommendations",
  title: "Generate contextual recommendations",
  subtitle: "LLM generates non-catalogue recommendations when coverage is insufficient",
};

const VALIDATE_STAGE = {
  id: "validate_and_build_response",
  title: "Validate and build response",
  subtitle: "deterministic guardrail validates ids and assembles the response",
};

const PIPELINE_STAGE_IDS = [
  RETRIEVE_STAGE.id,
  COVERAGE_STAGE.id,
  RANK_STAGE.id,
  LIMITED_STAGE.id,
  VALIDATE_STAGE.id,
];

const STAGE_IDS = new Set(PIPELINE_STAGE_IDS);

const DEFAULT_PIPELINE_STAGES = [
  RETRIEVE_STAGE,
  COVERAGE_STAGE,
  RANK_STAGE,
  VALIDATE_STAGE,
];

const COVERAGE_LIMITED_PIPELINE_STAGES = [
  RETRIEVE_STAGE,
  COVERAGE_STAGE,
  LIMITED_STAGE,
  VALIDATE_STAGE,
];

const RESULT_SETTLING_MS = 700;

export class TaskTwoResult extends HTMLElement {
  constructor() {
    super();
    this.handleClick = this.handleClick.bind(this);
    this.handleKeydown = this.handleKeydown.bind(this);
    this._evidenceOpen = false;
    this._settling = false;
    this._settlingTimer = 0;
  }

  updateState(nextState) {
    const hadResult = Boolean(this._result);
    if (Object.prototype.hasOwnProperty.call(nextState, "error")) {
      this._error = nextState.error || "";
    }
    if (Object.prototype.hasOwnProperty.call(nextState, "running")) {
      this._running = Boolean(nextState.running);
    }
    if (Object.prototype.hasOwnProperty.call(nextState, "elapsedMs")) {
      this._elapsedMs = Number(nextState.elapsedMs) || 0;
    }
    if (Object.prototype.hasOwnProperty.call(nextState, "trace")) {
      this._trace = Array.isArray(nextState.trace) ? nextState.trace : [];
    }
    if (!Object.prototype.hasOwnProperty.call(nextState, "result")) {
      this.render();
      return;
    }

    this._result = nextState.result;
    if (!nextState.result) {
      this._evidenceOpen = false;
      this._settling = false;
      window.clearTimeout(this._settlingTimer);
    } else if (!hadResult) {
      this._evidenceOpen = false;
      this._settling = true;
      window.clearTimeout(this._settlingTimer);
      this._settlingTimer = window.setTimeout(() => {
        this._settling = false;
        this.render();
      }, RESULT_SETTLING_MS);
    }
    this.render();
  }

  connectedCallback() {
    this._trace = this._trace || [];
    this.addEventListener("click", this.handleClick);
    document.addEventListener("keydown", this.handleKeydown);
    this.render();
  }

  disconnectedCallback() {
    this.removeEventListener("click", this.handleClick);
    document.removeEventListener("keydown", this.handleKeydown);
    window.clearTimeout(this._settlingTimer);
  }

  handleClick(event) {
    const target = event.target;
    if (!(target instanceof Element)) return;
    if (target.closest("[data-open-evidence]")) {
      this._evidenceOpen = true;
      this.render();
      return;
    }
    const clickedBackdrop = target.classList.contains("evidence-modal");
    if (target.closest("[data-close-evidence]") || clickedBackdrop) {
      this._evidenceOpen = false;
      this.render();
    }
  }

  handleKeydown(event) {
    if (event.key !== "Escape" || !this._evidenceOpen) return;
    this._evidenceOpen = false;
    this.render();
  }

  render() {
    const trace = this._trace || [];
    const result = this._result;
    const error = this._error;
    const running = this._running;
    const everStarted = running || trace.length > 0 || Boolean(result) || Boolean(error);
    const state = outputState({ error, result, running, settling: this._settling });

    this.innerHTML = `
      <section class="dashboard-card output-card output-card--${state}" aria-labelledby="task-two-output-title">
        <div class="card__header">
          <div>
            <h2 class="card__title" id="task-two-output-title">Recommendations</h2>
          </div>
          <span class="card__meta">${headerMeta(trace, result, running, this._elapsedMs)}</span>
        </div>
        <div class="output-card__body">
          ${error ? errorBlock(error) : ""}
          ${resultBody({
            trace,
            result,
            error,
            running,
            everStarted,
            settling: this._settling,
          })}
        </div>
        <div class="output-card__footer">
          <span class="footer-meta">${footerMeta(state, trace, result, this._elapsedMs)}</span>
          ${
            state === "complete"
              ? `
                <cds-button kind="ghost" size="sm" type="button" data-open-evidence>
                  ${reasoningIcon()}
                  <span>View reasoning</span>
                </cds-button>
              `
              : ""
          }
        </div>
        ${this._evidenceOpen ? evidenceModal(trace, result, this._elapsedMs) : ""}
      </section>
    `;
  }
}

customElements.define("bct-task-two-result", TaskTwoResult);

function outputState({ error, result, running, settling }) {
  if (error) return "error";
  if (running) return "running";
  if (result && settling) return "settling";
  if (result) return "complete";
  return "idle";
}

function resultBody({ trace, result, error, running, everStarted, settling }) {
  if (error) return everStarted ? timelineBlock(trace, null, false) : "";
  if (!everStarted) return emptyBlock();
  if (running && !result) return progressBlock(trace, running);
  if (result && settling) return settlingBlock(trace);
  if (result) return recommendationsBlock(result);
  return progressBlock(trace, running);
}

function emptyBlock() {
  return `
    <div class="empty-state__body">
      <div class="empty-state__icon" aria-hidden="true">
        <svg focusable="false" preserveAspectRatio="xMidYMid meet" width="24" height="24" viewBox="0 0 32 32">
          <path d="M6 6h20v2H6zm0 6h20v2H6zm0 6h14v2H6zm0 6h10v2H6z"></path>
        </svg>
      </div>
      <h3>Ready to run</h3>
      <p>Press Get recommendations to retrieve candidate items, judge catalogue coverage, then rank concrete matches or generate non-catalogue recommendations.</p>
    </div>
  `;
}

function progressBlock(trace, running) {
  return `
    <div class="output-stage output-stage--progress" aria-live="polite">
      ${timelineBlock(trace, null, running)}
    </div>
  `;
}

function settlingBlock(trace) {
  return `
    <div class="output-stage output-stage--settling" aria-live="polite">
      <div class="settling-steps">
        ${timelineBlock(trace, null, false)}
      </div>
    </div>
  `;
}

function recommendationsBlock(result) {
  const items = result.recommendations || [];
  const generatedItems = result.generated_recommendations || [];
  if (!items.length && result.recommendation_mode === "llm_generated") {
    return generatedRecommendationsBlock(generatedItems);
  }
  if (!items.length) return artifactEmpty("No recommendations returned.");
  const evidence = result.evidence || {};
  return `
    <div class="recommendations-summary">
      <p class="recommendations-headline">
        Surfaced <strong>${items.length}</strong> recommendation${items.length === 1 ? "" : "s"}
        from <strong>${escapeHtml(evidence.candidate_count || 0)}</strong> candidates, ranked by fit with the persona.
      </p>
    </div>
    <ul class="recommendations-list">
      ${items.map(recommendationCard).join("")}
    </ul>
  `;
}

function generatedRecommendationsBlock(generatedItems) {
  return `
    ${generatedItems.length ? `
      <div class="recommendations-summary recommendations-summary--generated">
        <p class="recommendations-headline">
          Generated <strong>${generatedItems.length}</strong> non-catalogue recommendation${generatedItems.length === 1 ? "" : "s"}.
          These are not counted in catalogue ranking metrics.
        </p>
      </div>
      <ul class="recommendations-list">
        ${generatedItems.map((item) => recommendationCard(item, { generated: true })).join("")}
      </ul>
    ` : artifactEmpty("No generated recommendations returned.")}
  `;
}

function recommendationCard(item, options = {}) {
  const product = item.product || {};
  const rank = String(item.rank || 0).padStart(2, "0");
  const score = typeof item.fit_score === "number" ? item.fit_score.toFixed(2) : "--";
  return `
    <li class="recommendation-card${options.generated ? " recommendation-card--generated" : ""}">
      <details class="recommendation-card__details">
        <summary class="recommendation-card__head">
          <span class="recommendation-card__rank">${escapeHtml(rank)}</span>
          <div class="recommendation-card__title-block">
            <h3 class="recommendation-card__name">${escapeHtml(product.name || "Unnamed item")}</h3>
            <p class="recommendation-card__headline">${escapeHtml(item.headline || "")}</p>
            ${options.generated ? `<p class="recommendation-card__mode">Generated · not catalogue-grounded</p>` : ""}
            ${productMeta(product) ? `<p class="recommendation-card__meta">${productMeta(product)}</p>` : ""}
          </div>
          <span class="recommendation-card__score" aria-label="Fit score">${escapeHtml(score)}</span>
          <span class="recommendation-card__chevron" aria-hidden="true">›</span>
        </summary>
        <div class="recommendation-card__body">
          <p class="recommendation-card__reasoning">${escapeHtml(item.reasoning || "No reasoning supplied.")}</p>
          ${product.description ? `<p class="recommendation-card__description">${escapeHtml(product.description)}</p>` : ""}
        </div>
      </details>
    </li>
  `;
}

function evidenceModal(trace, result, elapsedMs) {
  const stages = pipelineStages(trace, result);
  return `
    <div class="evidence-modal" role="presentation">
      <section
        class="evidence-modal__panel"
        role="dialog"
        aria-modal="true"
        aria-labelledby="task-two-evidence-title"
      >
        <header class="evidence-modal__header">
          <div>
            <h3 id="task-two-evidence-title">Trace + evidence</h3>
            <p>${completedStageCount(trace, result)} / ${stages.length} stages · ${formatElapsed(elapsedMs)}</p>
          </div>
          <cds-button kind="ghost" size="sm" type="button" data-close-evidence>
            Close
          </cds-button>
        </header>
        <div class="evidence-modal__body">
          ${traceAccordion(trace, result)}
          ${auxiliaryEvents(trace)}
        </div>
      </section>
    </div>
  `;
}

function traceAccordion(trace, result) {
  const eventsByStage = stageEventMap(trace);
  const stages = pipelineStages(trace, result);
  return `
    <div class="trace-accordion">
      ${stages.map((stage, index) => {
        const event = eventsByStage.get(stage.id);
        return `
          <details class="trace-accordion__item" ${index === 0 ? "open" : ""}>
            <summary class="trace-accordion__summary">
              <span>${escapeHtml(stage.title)}</span>
              <span>${eventTime(event)}</span>
            </summary>
            <div class="trace-accordion__body">
              <p class="trace-accordion__message">${escapeHtml(event?.message || stage.subtitle)}</p>
              ${renderArtifact(stage.id, result)}
            </div>
          </details>
        `;
      }).join("")}
    </div>
  `;
}

function timelineBlock(trace, result, running) {
  const eventsByStage = stageEventMap(trace);
  const stages = pipelineStages(trace, result);
  const firstPendingIndex = stages.findIndex((stage) => !eventsByStage.has(stage.id));
  return `
    <ol class="pipeline-timeline">
      ${stages.map((stage, index) => {
        const event = eventsByStage.get(stage.id);
        let status = "pending";
        if (event) status = "complete";
        else if (running && index === firstPendingIndex) status = "running";
        return stageCard(stage, status, event, result);
      }).join("")}
    </ol>
  `;
}

function stageCard(stage, status, event, result) {
  const artifact = status === "complete" && result ? renderArtifact(stage.id, result) : "";
  return `
    <li class="pipeline-step pipeline-step--${status}">
      <div class="pipeline-step__head">
        <span class="pipeline-step__status" aria-label="${statusLabel(status)}">
          ${statusGlyph(status)}
        </span>
        <div class="pipeline-step__heading">
          <div class="pipeline-step__title-row">
            <span class="pipeline-step__title">${escapeHtml(stage.title)}</span>
            <span class="pipeline-step__name">${escapeHtml(stage.id)}</span>
          </div>
          <p class="pipeline-step__message">${escapeHtml(event?.message || stage.subtitle)}</p>
        </div>
        <span class="pipeline-step__time">${eventTime(event) || pendingLabel(status)}</span>
      </div>
      ${artifact ? `<div class="pipeline-step__artifact">${artifact}</div>` : ""}
    </li>
  `;
}

function renderArtifact(stageId, result) {
  if (!result) return "";
  const evidence = result.evidence || {};
  if (stageId === "retrieve_candidate_items") {
    return candidatesArtifact(evidence.candidates || [], evidence.candidate_count || 0);
  }
  if (stageId === "judge_candidate_coverage") {
    return coverageArtifact(result.coverage || {});
  }
  if (stageId === "rank_and_reason") {
    return personaNeedsArtifact(evidence.persona_needs || []);
  }
  if (stageId === "generate_contextual_recommendations") {
    return generatedRecommendationsArtifact(
      result.generated_recommendations || [],
      evidence.note || "",
    );
  }
  if (stageId === "validate_and_build_response") {
    return returnedArtifact(
      result.recommendations || [],
      result.generated_recommendations || [],
      evidence.note || "",
    );
  }
  return "";
}

function candidatesArtifact(candidates, count) {
  if (!candidates.length) return artifactEmpty("No candidate items retrieved.");
  const visible = candidates.slice(0, 10);
  return `
    <ul class="candidate-products">
      ${visible
        .map((candidate) => {
          const product = candidate.product || {};
          return `
            <li class="candidate-products__item${candidate.shortlisted ? " candidate-products__item--shortlisted" : ""}">
              <span>${escapeHtml(product.name || "")}</span>
              <span>${candidateSignalLabel(candidate, product)}</span>
              ${candidate.shortlisted ? "<strong>shortlisted</strong>" : ""}
            </li>
          `;
        })
        .join("")}
    </ul>
    ${count > visible.length ? `<p class="artifact-empty">+ ${escapeHtml(count - visible.length)} more candidates</p>` : ""}
  `;
}

function candidateSignalLabel(candidate, product) {
  const signals = candidate.axis_scores || {};
  const parts = [];
  if (product.category) parts.push(product.category);
  if (product.metadata?.source) parts.push(product.metadata.source);
  if (typeof signals.similar_persona === "number") {
    parts.push(`similar persona ${signals.similar_persona.toFixed(2)}`);
  }
  if (typeof signals.product_text === "number") {
    parts.push(`product ${signals.product_text.toFixed(2)}`);
  }
  return escapeHtml(parts.join(" · "));
}

function coverageArtifact(coverage) {
  const signals = coverage.unsupported_signals || [];
  return `
    <div class="coverage-artifact">
      <span class="persona-needs__label">Coverage decision</span>
      <p>${escapeHtml(coverage.status || "unknown")} · ${coverage.allow_concrete_recommendations ? "concrete recommendations allowed" : "generated recommendation route"}</p>
      ${coverage.reason ? `<p>${escapeHtml(coverage.reason)}</p>` : ""}
      ${signals.length ? `
        <ul>
          ${signals.map((signal) => `<li>${escapeHtml(signal)}</li>`).join("")}
        </ul>
      ` : ""}
    </div>
  `;
}

function personaNeedsArtifact(needs) {
  if (!needs.length) return artifactEmpty("No persona needs returned.");
  return `
    <div class="persona-needs">
      <span class="persona-needs__label">Persona needs identified</span>
      <ul>
        ${needs.map((need) => `<li>${escapeHtml(need)}</li>`).join("")}
      </ul>
    </div>
  `;
}

function generatedRecommendationsArtifact(generatedRecommendations, note) {
  if (!generatedRecommendations.length) {
    return artifactEmpty(note || "No generated recommendations returned.");
  }
  return `
    <div class="persona-needs">
      <span class="persona-needs__label">Generated recommendations</span>
      <ul>
        ${generatedRecommendations
          .map((item) => `<li>${escapeHtml(item.product?.name || "")}</li>`)
          .join("")}
      </ul>
      ${note ? `<p>${escapeHtml(note)}</p>` : ""}
    </div>
  `;
}

function returnedArtifact(recommendations, generatedRecommendations, note) {
  const count = recommendations.length + generatedRecommendations.length;
  if (!count) return artifactEmpty(note || "No recommendations returned.");
  return `
    <div class="format-summary">
      <span class="format-summary-label">Recommendations returned</span>
      <strong>${escapeHtml(count)}</strong>
      ${note ? `<p>${escapeHtml(note)}</p>` : ""}
    </div>
  `;
}

function errorBlock(error) {
  return `
    <div class="output-error">
      <h3>Agent call failed</h3>
      <p>${escapeHtml(error)}</p>
    </div>
  `;
}

function productMeta(product) {
  const parts = [];
  if (product.category) parts.push(escapeHtml(product.category));
  if (product.location) parts.push(escapeHtml(product.location));
  if (product.price && product.currency) parts.push(`${escapeHtml(product.currency)} ${escapeHtml(product.price)}`);
  return parts.join(" · ");
}

function statusLabel(status) {
  if (status === "complete") return "Complete";
  if (status === "running") return "Working";
  return "Pending";
}

function pendingLabel(status) {
  if (status === "running") return "running";
  if (status === "pending") return "pending";
  return "";
}

function statusGlyph(status) {
  if (status === "complete") return carbonCheckmarkIcon();
  if (status === "running") return '<span class="pipeline-step__spinner" aria-hidden="true"></span>';
  return '<span class="pipeline-step__pending-dot" aria-hidden="true"></span>';
}

function carbonCheckmarkIcon() {
  return `
    <svg class="pipeline-step__icon" focusable="false" preserveAspectRatio="xMidYMid meet" aria-hidden="true" width="16" height="16" viewBox="0 0 16 16">
      <path d="M6 11L2 7l.7-.7L6 9.6l7.3-7.3.7.7z"></path>
    </svg>
  `;
}

function headerMeta(trace, result, running, elapsedMs) {
  if (!running && !trace.length) return "Ready";
  const done = completedStageCount(trace, result);
  const stages = pipelineStages(trace, result);
  const elapsed = elapsedMs ? ` · ${formatElapsed(elapsedMs)}` : "";
  const suffix = running ? " · running..." : "";
  return `${done}/${stages.length} stages${elapsed}${suffix}`;
}

function footerMeta(state, trace, result, elapsedMs) {
  if (state === "complete") {
    const elapsed = formatElapsed(elapsedMs);
    const stages = pipelineStages(trace, result);
    return `Trace + evidence · ${completedStageCount(trace, result)}/${
      stages.length
    } stages${elapsed ? ` · ${elapsed}` : ""}`;
  }
  if (state === "running") return "Pipeline running...";
  if (state === "error") return "Run failed.";
  return "No run yet.";
}

function reasoningIcon() {
  return `
    <svg slot="icon" focusable="false" preserveAspectRatio="xMidYMid meet" aria-hidden="true" width="16" height="16" viewBox="0 0 16 16">
      <path d="M2 2h12v2H2V2zm0 4h12v2H2V6zm0 4h8v2H2v-2z"></path>
    </svg>
  `;
}

function stageEventMap(trace) {
  const eventsByStage = new Map();
  trace.forEach((event) => {
    if (event && event.stage && STAGE_IDS.has(event.stage)) {
      eventsByStage.set(event.stage, event);
    }
  });
  return eventsByStage;
}

function completedStageCount(trace, result) {
  return pipelineStages(trace, result).filter((stage) =>
    trace.some((event) => event && event.stage === stage.id),
  ).length;
}

function auxiliaryEvents(trace) {
  const extras = trace.filter((event) => event && event.stage && !STAGE_IDS.has(event.stage));
  if (!extras.length) return "";
  return `
    <div class="pipeline-aux">
      <h3 class="pipeline-aux__title">Other events</h3>
      <ul class="pipeline-aux__list">
        ${extras
          .map(
            (event) => `
          <li class="pipeline-aux__item">
            <span class="pipeline-aux__name">${escapeHtml(event.stage)}</span>
            <span class="pipeline-aux__message">${escapeHtml(event.message || "")}</span>
            <span class="pipeline-aux__time">${eventTime(event)}</span>
          </li>`,
          )
          .join("")}
      </ul>
    </div>
  `;
}

function pipelineStages(trace, result) {
  const hasCoverageLimitedRoute =
    result?.recommendation_mode === "coverage_limited" ||
    result?.recommendation_mode === "llm_generated" ||
    trace.some((event) => event?.stage === LIMITED_STAGE.id);
  return hasCoverageLimitedRoute
    ? COVERAGE_LIMITED_PIPELINE_STAGES
    : DEFAULT_PIPELINE_STAGES;
}

function eventTime(event) {
  if (!event) return "";
  if (Number.isFinite(Number(event.duration_ms))) return `+${(Number(event.duration_ms) / 1000).toFixed(1)}s`;
  return escapeHtml(event.received_at || "");
}

function artifactEmpty(text) {
  return `<p class="artifact-empty">${escapeHtml(text)}</p>`;
}

function formatElapsed(ms) {
  if (!ms) return "";
  return `${(ms / 1000).toFixed(2)}s`;
}
