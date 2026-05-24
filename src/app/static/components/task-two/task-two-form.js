import { fieldValue } from "../dom.js";

const QUICK_FILLS = {
  student: {
    persona: "Lagos-based student, budget conscious, likes spicy food.",
    context: "weekday lunch, near Yaba",
  },
  professional: {
    persona:
      "Abuja-based professional, values quality and presentation, willing to pay a premium for reliable service.",
    context: "dinner with colleagues",
  },
  family: {
    persona:
      "Mother of two in Ibadan, balances cost and child-friendly options, prefers familiar Nigerian dishes.",
    context: "Saturday family lunch",
  },
};

export class TaskTwoForm extends HTMLElement {
  connectedCallback() {
    this.render();
    this.querySelector("form").addEventListener("submit", (event) => this.submit(event));
    this.querySelector("#send-prompt").addEventListener("click", (event) => this.submit(event));
    this.applyQuickFill("student");
    this.querySelectorAll("[data-quick-fill]").forEach((tag) => {
      tag.addEventListener("click", () => this.applyQuickFill(tag.dataset.quickFill));
      tag.addEventListener("keydown", (event) => {
        if (event.key !== "Enter" && event.key !== " ") return;
        event.preventDefault();
        this.applyQuickFill(tag.dataset.quickFill);
      });
    });
  }

  applyQuickFill(name) {
    const preset = QUICK_FILLS[name];
    if (!preset) return;
    setCarbonTextareaValue(this.querySelector("#user-persona"), preset.persona);
    setCarbonTextareaValue(this.querySelector("#user-context"), preset.context);
    this.querySelectorAll("cds-tag[data-quick-fill]").forEach((tag) => {
      const active = tag.dataset.quickFill === name;
      tag.setAttribute("aria-checked", String(active));
      tag.setAttribute("type", active ? "blue" : "gray");
    });
  }

  set busy(value) {
    this._busy = Boolean(value);
    const button = this.querySelector("#send-prompt");
    const label = button?.querySelector(".button-label") || button;
    if (button) button.toggleAttribute("disabled", this._busy);
    if (this._busy) this._completed = false;
    if (label) {
      label.textContent = this._busy
        ? "Finding recommendations..."
        : this._completed
          ? "Re-run recommendations"
          : "Get recommendations";
    }
    const meta = this.querySelector("#form-footer-meta");
    if (meta && this._busy) {
      meta.textContent = "Running pipeline...";
    } else if (meta && !this._busy && !this._completed) {
      meta.textContent = "Edit inputs, then press Get recommendations.";
    }
  }

  set completed(value) {
    if (!value) return;
    this._completed = true;
    const label = this.querySelector("#send-prompt .button-label") || this.querySelector("#send-prompt");
    if (label) label.textContent = "Re-run recommendations";
    const meta = this.querySelector("#form-footer-meta");
    if (meta) {
      meta.textContent = `Last run · ${value.elapsed}s · ${value.recommendation_count} items`;
    }
  }

  submit(event) {
    event.preventDefault();
    if (this._busy) return;
    this.dispatchEvent(
      new CustomEvent("task-two-submit", {
        bubbles: true,
        detail: this.payload(),
      }),
    );
  }

  payload() {
    return {
      user_persona: fieldValue(this, "#user-persona"),
      context: fieldValue(this, "#user-context"),
      k: 5,
    };
  }

  render() {
    this.innerHTML = `
      <section class="dashboard-card input-card" aria-labelledby="task-two-input-title">
        <div class="card__header">
          <div>
            <h2 class="card__title" id="task-two-input-title">User Persona &amp; Context</h2>
          </div>
        </div>
        <form class="task-form">
          <div class="task-form__body">
            <div class="quick-fill-row" role="radiogroup" aria-label="Quick fill presets">
              <span class="quick-fill-label">Quick fill</span>
              ${quickFillTag("student", "Student", "blue", true, "student")}
              ${quickFillTag("professional", "Professional", "gray", false, "professional")}
              ${quickFillTag("family", "Family", "gray", false, "family")}
            </div>
            <div class="form-group form-group--persona">
              <cds-textarea
                id="user-persona"
                aria-label="User persona"
                rows="4"
                placeholder="Describe the user persona."
              >
                <span slot="label-text">User persona</span>
                <span slot="helper-text">Describe the user's preferences, context, and behaviour.</span>
              </cds-textarea>
            </div>
            <div class="form-group form-group--context">
              <cds-textarea
                id="user-context"
                aria-label="Context"
                rows="3"
                placeholder="Optional context like location, time, or occasion."
              >
                <span slot="label-text">Context (optional)</span>
                <span slot="helper-text">Any situational signals like time, location, or occasion.</span>
              </cds-textarea>
            </div>
          </div>
          <div class="task-form__footer">
            <span class="footer-meta" id="form-footer-meta">Edit inputs, then press Get recommendations.</span>
            <cds-button id="send-prompt" type="submit">
              <span class="button-label">Get recommendations</span>
              <svg
                slot="icon"
                focusable="false"
                preserveAspectRatio="xMidYMid meet"
                aria-hidden="true"
                width="16"
                height="16"
                viewBox="0 0 16 16"
              >
                <path d="M9.3 3.7 8.6 4.4 11.2 7H2v1h9.2l-2.6 2.6.7.7L13.2 7.5 9.3 3.7z"></path>
              </svg>
            </cds-button>
          </div>
        </form>
      </section>
    `;
  }
}

customElements.define("bct-task-two-form", TaskTwoForm);

function quickFillTag(name, label, type, checked, dotClass) {
  return `
    <cds-tag
      type="${type}"
      size="md"
      data-quick-fill="${name}"
      role="radio"
      aria-checked="${checked}"
      tabindex="0"
    >
      <span slot="icon" class="preset-dot preset-dot--${dotClass}"></span>
      ${label}
    </cds-tag>
  `;
}

function setCarbonTextareaValue(field, value) {
  if (!field) return;
  field.setAttribute("value", value);
  const textarea = field.shadowRoot?.querySelector("textarea");
  if (!textarea) return;
  textarea.value = value;
  textarea.dispatchEvent(new Event("input", { bubbles: true, composed: true }));
}
