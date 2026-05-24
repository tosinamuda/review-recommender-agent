import { fieldValue } from "../dom.js";

const QUICK_FILLS = {
  positive: {
    persona:
      "Ondo-based corper, budget conscious, likes peppery food, quick service, and generous portions.",
    product: `Product:
Jollof Bowl with Grilled Chicken

Category: food
Description: Smoky jollof rice with grilled chicken, fried plantain, and pepper sauce.
Price: 4500
Currency: NGN
Delivery minutes: 35
Portion: large
Spice level: high`,
  },
  mixed: {
    persona:
      "Lagos-based product designer, values aesthetics and packaging, mild spice tolerance, willing to pay more but impatient with slow delivery.",
    product: `Product:
Suya Wrap Combo

Category: food
Description: Spicy beef suya in flatbread with onions and tomato salsa, served with chips.
Price: 5200
Currency: NGN
Delivery minutes: 55
Portion: medium
Spice level: high`,
  },
  negative: {
    persona:
      "Abuja-based banker, strict about hygiene and punctuality, dislikes oily food, low spice tolerance, expects premium presentation for the price.",
    product: `Product:
Budget Fried Rice Pack

Category: food
Description: Basic fried rice with two pieces of fried chicken, packed in foil. No sauce included.
Price: 6500
Currency: NGN
Delivery minutes: 75
Portion: small
Spice level: high`,
  },
};

const TEXTAREA_ROWS = {
  persona: 3,
  product: 6,
};

export class TaskOneForm extends HTMLElement {
  connectedCallback() {
    this.render();
    this.querySelector("form").addEventListener("submit", (event) => this.submit(event));
    this.applyQuickFill("positive");
    this.querySelector("#send-prompt").addEventListener("click", (event) => this.submit(event));
    this.querySelectorAll("[data-quick-fill]").forEach((tag) => {
      tag.addEventListener("click", () => this.applyQuickFill(tag.dataset.quickFill));
      tag.addEventListener("keydown", (event) => {
        if (event.key !== "Enter" && event.key !== " ") return;
        event.preventDefault();
        this.applyQuickFill(tag.dataset.quickFill);
      });
    });
  }

  applyQuickFill(tone) {
    const preset = QUICK_FILLS[tone];
    if (!preset) return;
    setCarbonTextareaValue(this.querySelector("#user-persona"), preset.persona);
    setCarbonTextareaValue(this.querySelector("#product-details"), preset.product);
    this.querySelectorAll("cds-tag[data-quick-fill]").forEach((tag) => {
      const active = tag.dataset.quickFill === tone;
      tag.setAttribute("aria-checked", String(active));
      tag.setAttribute("type", active ? "blue" : "gray");
    });
  }

  set busy(value) {
    this._busy = Boolean(value);
    const button = this.querySelector("#send-prompt");
    const label = button?.querySelector(".button-label") || button;
    if (button) {
      button.toggleAttribute("disabled", this._busy);
    }
    if (this._busy) {
      this._completed = false;
    }
    if (label) {
      label.textContent = this._busy
        ? "Running..."
        : this._completed
          ? "Re-run simulation"
          : "Run simulation";
    }
    const meta = this.querySelector("#form-footer-meta");
    if (meta && this._busy) {
      meta.textContent = "Pipeline running...";
    } else if (meta && !this._busy && !this._completed) {
      meta.textContent = "Edit inputs, then press Run simulation.";
    }
  }

  set completed(value) {
    if (!value) return;
    this._completed = true;
    const label = this.querySelector("#send-prompt .button-label") || this.querySelector("#send-prompt");
    if (label) label.textContent = "Re-run simulation";
    const meta = this.querySelector("#form-footer-meta");
    if (!meta) return;
    const attempts = Number(value.attempts) || 1;
    meta.textContent = `Last run · ${value.elapsed}s · ${attempts} verifier attempt${
      attempts > 1 ? "s" : ""
    }`;
  }

  submit(event) {
    event.preventDefault();
    if (this._busy) return;
    this.dispatchEvent(
      new CustomEvent("task-one-submit", {
        bubbles: true,
        detail: this.payload(),
      }),
    );
  }

  payload() {
    return {
      user_persona: fieldValue(this, "#user-persona"),
      product_details: fieldValue(this, "#product-details"),
      options: { sample_count: 3 },
    };
  }

  render() {
    this.innerHTML = `
      <section class="dashboard-card input-card" aria-labelledby="task-one-input-title">
        <div class="card__header">
          <div>
            <h2 class="card__title" id="task-one-input-title">User Persona & Product Detail</h2>
          </div>
        </div>
        <form class="task-form">
          <div class="task-form__body">
            <div class="quick-fill-row" role="radiogroup" aria-label="Quick fill presets">
              <span class="quick-fill-label">Quick fill</span>
              <cds-tag
                type="gray"
                size="md"
                data-quick-fill="positive"
                role="radio"
                aria-checked="false"
                tabindex="0"
              >
                <span slot="icon" class="preset-dot preset-dot--positive"></span>
                Positive
              </cds-tag>
              <cds-tag
                type="gray"
                size="md"
                data-quick-fill="mixed"
                role="radio"
                aria-checked="false"
                tabindex="0"
              >
                <span slot="icon" class="preset-dot preset-dot--mixed"></span>
                Mixed
              </cds-tag>
              <cds-tag
                type="gray"
                size="md"
                data-quick-fill="negative"
                role="radio"
                aria-checked="false"
                tabindex="0"
              >
                <span slot="icon" class="preset-dot preset-dot--negative"></span>
                Negative
              </cds-tag>
            </div>
            <div class="form-group form-group--persona">
              <cds-textarea
                id="user-persona"
                aria-label="User persona"
                rows="${TEXTAREA_ROWS.persona}"
                placeholder="Describe the user persona."
              >
                <span slot="label-text">User persona</span>
                <span slot="helper-text">Describe the user's preferences, context, and behaviour.</span>
              </cds-textarea>
            </div>
            <div class="form-group form-group--product">
              <cds-textarea
                id="product-details"
                aria-label="Product details"
                rows="${TEXTAREA_ROWS.product}"
                placeholder="Describe the product details."
              >
                <span slot="label-text">Product details</span>
                <span slot="helper-text">Describe the unseen item, including category and useful metadata.</span>
              </cds-textarea>
            </div>
          </div>
          <div class="task-form__footer">
            <span class="footer-meta" id="form-footer-meta">Edit inputs, then press Run simulation.</span>
            <cds-button id="send-prompt" type="submit">
              <span class="button-label">Run simulation</span>
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

customElements.define("bct-task-one-form", TaskOneForm);

function setCarbonTextareaValue(field, value) {
  if (!field) return;
  field.setAttribute("value", value);
  const textarea = field.shadowRoot?.querySelector("textarea");
  if (!textarea) return;
  textarea.value = value;
  textarea.dispatchEvent(new Event("input", { bubbles: true, composed: true }));
}
