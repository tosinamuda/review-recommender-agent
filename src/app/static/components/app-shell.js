import "./task-one/task-one-panel.js";
import "./task-two/task-two-panel.js";

export class ReviewSimulationApp extends HTMLElement {
  constructor() {
    super();
    this.activeView = "task1";
    this.agentStatus = "Agent · ready";
    this.agentReady = true;
    this.agentState = "ready";
  }

  connectedCallback() {
    this.renderShell();
    this.bindNavigation();
    this.renderActiveView();
    this.addEventListener("agent-status-change", (event) => {
      this.agentStatus = event.detail.message;
      this.agentReady = event.detail.ready;
      this.agentState = event.detail.state || "ready";
      this.updateHeaderStatus();
    });
  }

  bindNavigation() {
    this.querySelectorAll("[data-view]").forEach((button) => {
      button.addEventListener("click", () => {
        this.setActiveView(button.dataset.view);
      });
    });
  }

  setActiveView(view) {
    if (!view || view === this.activeView) return;
    this.activeView = view;
    this.renderActiveView();
    this.updateNavActive();
  }

  renderActiveView() {
    const main = this.querySelector(".main-content__inner");
    if (!main) return;
    let panel = main.querySelector(`[data-panel="${this.activeView}"]`);
    if (!panel) {
      main.insertAdjacentHTML(
        "beforeend",
        `<div class="view-panel" data-panel="${this.activeView}">${panelMarkup(
          this.activeView,
        )}</div>`,
      );
      panel = main.querySelector(`[data-panel="${this.activeView}"]`);
    }
    main.querySelectorAll("[data-panel]").forEach((item) => {
      item.toggleAttribute("hidden", item !== panel);
    });
  }

  updateNavActive() {
    this.querySelectorAll("[data-view]").forEach((button) => {
      button.classList.toggle("active", button.dataset.view === this.activeView);
    });
  }

  updateHeaderStatus() {
    const status = this.querySelector(".ui-shell-header__status");
    if (!status) return;
    status.classList.toggle("ready", this.agentReady);
    status.classList.toggle("running", this.agentState === "running");
    status.querySelector("span:last-child").textContent = this.agentStatus;
  }

  renderShell() {
    this.innerHTML = `
      <div class="app-shell">
        <header class="ui-shell-header cds-theme-zone-g100">
          <button class="ui-shell-header__menu" aria-label="Open menu">
            <svg width="20" height="20" viewBox="0 0 32 32" aria-hidden="true">
              <path d="M4 6h24v2H4zm0 9h24v2H4zm0 9h24v2H4z"></path>
            </svg>
          </button>
          <div class="ui-shell-header__brand">
            <span class="ui-shell-header__name">DSN × BCT Challenge</span>
            <span class="ui-shell-header__platform">Review Simulation Agent</span>
          </div>
          <div class="ui-shell-header__actions">
            <div class="ui-shell-header__status ${this.agentReady ? "ready" : ""} ${
              this.agentState === "running" ? "running" : ""
            }">
              <span class="status-dot"></span>
              <span>${this.agentStatus}</span>
            </div>
            <div class="ui-shell-header__user" aria-label="User menu">
              <div class="avatar">TO</div>
            </div>
          </div>
        </header>
        <div class="dashboard-body">
          <nav class="side-nav cds-theme-zone-g100" aria-label="Primary">
            <div class="side-nav__heading">Tasks</div>
            ${navButton("task1", "Task 1 · User modelling", circleCheckIcon(), this.activeView)}
            ${navButton("task2", "Task 2 · Recommendation", listIcon(), this.activeView)}
            <div class="side-nav__heading">Reference</div>
            ${navButton("architecture", "Architecture", cubeIcon(), this.activeView)}
            ${navButton("datasets", "Datasets", databaseIcon(), this.activeView)}
            ${navButton("about", "About", infoIcon(), this.activeView)}
          </nav>
          <main class="main-content">
            <div class="main-content__inner"></div>
          </main>
        </div>
      </div>
    `;
  }
}

function navButton(view, label, icon, activeView) {
  return `
    <button class="side-nav__item ${view === activeView ? "active" : ""}" data-view="${view}" type="button">
      ${icon}
      <span>${label}</span>
    </button>
  `;
}

function referenceView(section, current, title, subtitle) {
  return `
    <section class="reference-view">
      ${breadcrumb(section, current)}
      <header class="page-header">
        <h1 class="page-title">${title}</h1>
        <p class="page-subtitle">${subtitle}</p>
      </header>
    </section>
  `;
}

function panelMarkup(view) {
  if (view === "task1") return "<bct-task-one-panel></bct-task-one-panel>";
  if (view === "task2") return "<bct-task-two-panel></bct-task-two-panel>";
  if (view === "architecture") {
    return referenceView(
      "Reference",
      "Architecture",
      "Architecture",
      "ADK 2.0 workflows, A2A exposure, DSPy modules, retrieval, ranking, and response evidence.",
    );
  }
  if (view === "datasets") {
    return referenceView(
      "Reference",
      "Datasets",
      "Datasets",
      "Review exemplars and a curated Nigerian product catalogue feed the task-specific retrieval paths.",
    );
  }
  if (view === "about") {
    return referenceView(
      "Reference",
      "About",
      "About",
      "DSN × BCT LLM Agent Challenge entry with separate Task 1 review simulation and Task 2 recommendation agents.",
    );
  }
  return "";
}

export function breadcrumb(section, current) {
  return `
    <nav class="breadcrumb" aria-label="Breadcrumb">
      <span>${section}</span>
      <span class="breadcrumb__sep">/</span>
      <span class="breadcrumb__current">${current}</span>
    </nav>
  `;
}

function circleCheckIcon() {
  return `
    <svg class="side-nav__icon" viewBox="0 0 32 32" aria-hidden="true">
      <path d="M16 4a12 12 0 1 0 12 12A12 12 0 0 0 16 4Zm0 22a10 10 0 1 1 10-10 10 10 0 0 1-10 10Z"></path>
      <path d="M20.6 11.7 14 18.3l-2.6-2.6L10 17.1l4 4 8-8Z"></path>
    </svg>
  `;
}

function listIcon() {
  return `
    <svg class="side-nav__icon" viewBox="0 0 32 32" aria-hidden="true">
      <path d="M28 6H4a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h24a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2Zm0 18H4V8h24Z"></path>
      <path d="M7 12h18v2H7zm0 4h18v2H7zm0 4h12v2H7z"></path>
    </svg>
  `;
}

function cubeIcon() {
  return `
    <svg class="side-nav__icon" viewBox="0 0 32 32" aria-hidden="true">
      <path d="M16 4 4 10v12l12 6 12-6V10Zm10 16.3-9 4.5v-9.6l9-4.5Zm-10-6.8L7 9l9-4.5L25 9Zm-10 11.3-9-4.5v-9.6l9 4.5Z"></path>
    </svg>
  `;
}

function databaseIcon() {
  return `
    <svg class="side-nav__icon" viewBox="0 0 32 32" aria-hidden="true">
      <path d="M16 4C10 4 4 5.5 4 8v16c0 2.5 6 4 12 4s12-1.5 12-4V8c0-2.5-6-4-12-4Zm10 20c0 .5-3.5 2-10 2S6 24.5 6 24v-2.8a23.9 23.9 0 0 0 10 1.8 23.9 23.9 0 0 0 10-1.8Zm0-6c0 .5-3.5 2-10 2S6 18.5 6 18v-2.8A23.9 23.9 0 0 0 16 17a23.9 23.9 0 0 0 10-1.8ZM16 12C10 12 6 10.5 6 10s4-2 10-2 10 1.5 10 2-4 2-10 2Z"></path>
    </svg>
  `;
}

function infoIcon() {
  return `
    <svg class="side-nav__icon" viewBox="0 0 32 32" aria-hidden="true">
      <path d="M16 2a14 14 0 1 0 14 14A14 14 0 0 0 16 2Zm0 26a12 12 0 1 1 12-12 12 12 0 0 1-12 12Z"></path>
      <path d="M17 14h-2v9h2zm0-5h-2v3h2z"></path>
    </svg>
  `;
}

customElements.define("bct-review-app", ReviewSimulationApp);
