import { escapeHtml } from "../dom.js";

const PIPELINE_STAGES = [
  {
    id: "retrieve_similar_user_reviews",
    title: "Retrieve similar reviews",
    subtitle: "behavioural retrieval over the review corpus",
  },
  {
    id: "generate_review_with_refinement",
    title: "Generate candidate drafts",
    subtitle: "refinement boundary over n=3 drafts",
  },
  {
    id: "select_best_review_for_persona",
    title: "Select best draft",
    subtitle: "ranked against persona signals",
  },
  {
    id: "calibrate_rating_from_review",
    title: "Calibrate rating",
    subtitle: "ordinal threshold model over the chosen review",
  },
];

const STAGE_IDS = new Set(PIPELINE_STAGES.map((stage) => stage.id));
const RESULT_SETTLING_MS = 950;

export class TaskOneResult extends HTMLElement {
  constructor() {
    super();
    this.handleClick = this.handleClick.bind(this);
    this.handleKeydown = this.handleKeydown.bind(this);
    this._evidenceOpen = false;
    this._settling = false;
    this._settlingTimer = 0;
  }

  set elapsedMs(value) {
    this.updateState({ elapsedMs: Number(value) || 0 });
  }

  set error(value) {
    this.updateState({ error: value || "" });
  }

  set result(value) {
    this.updateState({ result: value });
  }

  set running(value) {
    this.updateState({ running: Boolean(value) });
  }

  set trace(value) {
    this.updateState({ trace: Array.isArray(value) ? value : [] });
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
      <section class="dashboard-card output-card output-card--${state}" aria-labelledby="task-one-output-title">
        <div class="card__header">
          <div>
            <h2 class="card__title" id="task-one-output-title">Ratings & Review</h2>
          </div>
          <span class="card__meta">${headerMeta(trace, running, this._elapsedMs)}</span>
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
          <span class="footer-meta">${footerMeta(state, trace, this._elapsedMs)}</span>
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

function outputState({ error, result, running, settling }) {
  if (error) return "error";
  if (running) return "running";
  if (result && settling) return "settling";
  if (result) return "complete";
  return "idle";
}

function resultBody({ trace, result, error, running, everStarted, settling }) {
  if (error) {
    return everStarted ? timelineBlock(trace, null, false) : "";
  }
  if (!everStarted) return emptyBlock();
  if (running && !result) return progressBlock(trace, running);
  if (result && settling) return settlingBlock(trace);
  if (result) return finalOutputBlock(result);
  return progressBlock(trace, running);
}

function summaryBlock(result) {
  const rating = Number(result.rating) || 0;
  const evidence = result.evidence || {};
  const continuous =
    typeof evidence.rating_continuous === "number" ? evidence.rating_continuous.toFixed(2) : "";
  const distribution = Array.isArray(evidence.rating_distribution)
    ? evidence.rating_distribution
    : [];
  const confidence = distribution.length
    ? Math.round(Math.max(...distribution.map((value) => Number(value) || 0)) * 100)
    : null;
  return `
    <div class="output-summary">
      <div class="output-rating">
        <div class="rating">
          <span class="rating__number">${escapeHtml(rating)}</span>
          <span class="rating__stars">${starsFor(rating)}</span>
        </div>
        <div class="rating-side">
          <p class="rating-headline">
            Calibrated from selected review${
              confidence !== null ? ` · <strong>${confidence}% confident</strong>` : ""
            }${continuous ? ` · expected ${escapeHtml(continuous)}` : ""}
          </p>
        </div>
      </div>
      <div class="review-text">${paragraphsFor(result.review)}</div>
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

function finalOutputBlock(result) {
  return `
    <div class="output-stage output-stage--final">
      ${summaryBlock(result)}
    </div>
  `;
}

function evidenceModal(trace, result, elapsedMs) {
  return `
    <div class="evidence-modal" role="presentation">
      <section
        class="evidence-modal__panel"
        role="dialog"
        aria-modal="true"
        aria-labelledby="evidence-modal-title"
      >
        <header class="evidence-modal__header">
          <div>
            <h3 id="evidence-modal-title">Trace + evidence</h3>
            <p>${completedStageCount(trace)} / ${PIPELINE_STAGES.length} nodes · ${formatElapsed(elapsedMs)}</p>
          </div>
          <cds-button kind="ghost" size="sm" type="button" data-close-evidence>
            Close
          </cds-button>
        </header>
        <div class="evidence-modal__body">
          ${timelineBlock(trace, result, false)}
          ${auxiliaryEvents(trace)}
        </div>
      </section>
    </div>
  `;
}

function emptyBlock() {
  return `
    <div class="empty-state__body">
      <div class="empty-state__icon" aria-hidden="true">
        <svg
          focusable="false"
          preserveAspectRatio="xMidYMid meet"
          width="24"
          height="24"
          viewBox="0 0 32 32"
        >
          <path d="M16 30A14 14 0 1 1 30 16 14 14 0 0 1 16 30Zm0-26a12 12 0 1 0 12 12A12 12 0 0 0 16 4Z"></path>
          <path d="M17 8h-2v11.2l-4.6-4.6L9 16l7 7 7-7-1.4-1.4-4.6 4.6V8z"></path>
        </svg>
      </div>
      <h3>Ready to run</h3>
      <p>Press Run simulation to generate a review for this persona and product. The pipeline takes around 3 seconds across 4 nodes.</p>
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

function timelineBlock(trace, result, running) {
  const eventsByStage = new Map();
  trace.forEach((event) => {
    if (event && event.stage && STAGE_IDS.has(event.stage)) {
      eventsByStage.set(event.stage, event);
    }
  });

  const firstPendingIndex = PIPELINE_STAGES.findIndex(
    (stage) => !eventsByStage.has(stage.id),
  );

  const cards = PIPELINE_STAGES.map((stage, index) => {
    const event = eventsByStage.get(stage.id);
    let status = "pending";
    if (event) status = "complete";
    else if (running && index === firstPendingIndex) status = "running";
    return stageCard(stage, status, event, result);
  });

  return `<ol class="pipeline-timeline">${cards.join("")}</ol>`;
}

function stageCard(stage, status, event, result) {
  const artifact = status === "complete" && result ? renderArtifact(stage.id, result) : "";
  const description = event ? escapeHtml(event.message) : escapeHtml(stage.subtitle);
  const time = event ? escapeHtml(event.received_at || "") : "";
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
          <p class="pipeline-step__message">${description}</p>
        </div>
        <span class="pipeline-step__time">${time || pendingLabel(status)}</span>
      </div>
      ${artifact ? `<div class="pipeline-step__artifact">${artifact}</div>` : ""}
    </li>
  `;
}

function statusLabel(status) {
  if (status === "complete") return "Complete";
  if (status === "running") return "Working";
  return "Pending";
}

function pendingLabel(status) {
  if (status === "running") return "working...";
  if (status === "pending") return "pending";
  return "";
}

function statusGlyph(status) {
  if (status === "complete") return carbonCheckmarkIcon();
  if (status === "running") {
    return '<span class="pipeline-step__spinner" aria-hidden="true"></span>';
  }
  return '<span class="pipeline-step__pending-dot" aria-hidden="true"></span>';
}

function carbonCheckmarkIcon() {
  return `
    <svg
      class="pipeline-step__icon"
      focusable="false"
      preserveAspectRatio="xMidYMid meet"
      aria-hidden="true"
      width="16"
      height="16"
      viewBox="0 0 16 16"
    >
      <path d="M6 11L2 7l.7-.7L6 9.6l7.3-7.3.7.7z"></path>
    </svg>
  `;
}

function renderArtifact(stageId, result) {
  const evidence = result.evidence || {};
  if (stageId === "retrieve_similar_user_reviews") {
    return similarReviewsArtifact(evidence.similar_user_reviews || []);
  }
  if (stageId === "generate_review_with_refinement") {
    return candidateDraftsArtifact(evidence.candidate_reviews || [], result.review);
  }
  if (stageId === "select_best_review_for_persona") {
    return reasonArtifact(evidence.reason || "");
  }
  if (stageId === "calibrate_rating_from_review") {
    return ratingArtifact(Number(result.rating) || 0);
  }
  return "";
}

function similarReviewsArtifact(items) {
  if (!items.length) return artifactEmpty("No similar reviews retrieved.");
  return `
    <ul class="similar-reviews">
      ${items
        .map(
          (item) => `
        <li class="similar-reviews__item">
          <span class="similar-reviews__id">${escapeHtml(item.exemplar_id || "")}</span>
          <p class="similar-reviews__text">${escapeHtml(item.review_text || "")}</p>
          <span class="similar-reviews__rating similar-reviews__rating--${ratingClass(
            item.rating,
          )}">★ ${escapeHtml(item.rating || "")}</span>
        </li>`,
        )
        .join("")}
    </ul>
  `;
}

function candidateDraftsArtifact(items, chosenReview) {
  if (!items.length) return artifactEmpty("No candidate drafts produced.");
  const chosenText = (chosenReview || "").trim();
  const chosenIndex = items.findIndex((text) => text && text.trim() === chosenText);
  return `
    <ul class="candidate-drafts">
      ${items
        .map((text, index) => {
          const chosen = index === chosenIndex;
          return `
        <li class="candidate-drafts__item${chosen ? " candidate-drafts__item--chosen" : ""}">
          <details ${chosen ? "open" : ""}>
            <summary>
              <span class="candidate-drafts__index">Draft ${index + 1}</span>
              ${chosen ? '<span class="candidate-drafts__badge">selected</span>' : ""}
              <span class="candidate-drafts__preview">${escapeHtml(truncate(text, 140))}</span>
            </summary>
            <p class="candidate-drafts__body">${escapeHtml(text)}</p>
          </details>
        </li>`;
        })
        .join("")}
    </ul>
  `;
}

function reasonArtifact(reason) {
  if (!reason) return artifactEmpty("No selection reason returned.");
  return `<p class="reason-text">${escapeHtml(reason)}</p>`;
}

function ratingArtifact(rating) {
  return `
    <div class="rating-artifact">
      <span class="rating-artifact__number">${escapeHtml(rating)}</span>
      <span class="rating-artifact__stars">${starsFor(rating)}</span>
      <span class="rating-artifact__label">calibrated rating · / 5</span>
    </div>
  `;
}

function artifactEmpty(text) {
  return `<p class="artifact-empty">${escapeHtml(text)}</p>`;
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
            <span class="pipeline-aux__time">${escapeHtml(event.received_at || "")}</span>
          </li>`,
          )
          .join("")}
      </ul>
    </div>
  `;
}

function headerMeta(trace, running, elapsedMs) {
  if (!running && !trace.length) return "Ready";
  const done = completedStageCount(trace);
  const elapsed = elapsedMs ? ` · ${formatElapsed(elapsedMs)}` : "";
  const suffix = running ? " · streaming" : "";
  return `${done}/${PIPELINE_STAGES.length} nodes${elapsed}${suffix}`;
}

function footerMeta(state, trace, elapsedMs) {
  if (state === "complete") {
    const elapsed = formatElapsed(elapsedMs);
    return `Trace + evidence · ${completedStageCount(trace)}/${
      PIPELINE_STAGES.length
    } nodes${elapsed ? ` · ${elapsed}` : ""}`;
  }
  if (state === "running") return "Pipeline running...";
  if (state === "error") return "Run failed.";
  return "No run yet.";
}

function reasoningIcon() {
  return `
    <svg
      slot="icon"
      focusable="false"
      preserveAspectRatio="xMidYMid meet"
      aria-hidden="true"
      width="16"
      height="16"
      viewBox="0 0 16 16"
    >
      <path d="M2 2h12v2H2V2zm0 4h12v2H2V6zm0 4h8v2H2v-2z"></path>
    </svg>
  `;
}

function completedStageCount(trace) {
  return PIPELINE_STAGES.filter((stage) =>
    trace.some((event) => event && event.stage === stage.id),
  ).length;
}

function truncate(text, length) {
  if (!text) return "";
  if (text.length <= length) return text;
  return text.slice(0, length).trimEnd() + "...";
}

function starsFor(rating) {
  const bounded = Math.max(0, Math.min(5, Math.round(rating)));
  return `${"★".repeat(bounded)}${"☆".repeat(5 - bounded)}`;
}

function paragraphsFor(text) {
  return String(text || "")
    .split(/\n{2,}/)
    .map((paragraph) => paragraph.trim())
    .filter(Boolean)
    .map((paragraph) => `<p>${escapeHtml(paragraph)}</p>`)
    .join("");
}

function ratingClass(rating) {
  const value = Number(rating) || 0;
  if (value <= 2) return "low";
  if (value === 3) return "mid";
  return "high";
}

function formatElapsed(ms) {
  if (!ms) return "";
  return `${(ms / 1000).toFixed(2)}s`;
}

customElements.define("bct-task-one-result", TaskOneResult);
