export function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => {
    const entities = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" };
    return entities[char];
  });
}

export function fieldValue(root, selector) {
  const element = root.querySelector(selector);
  if (!element) return "";
  const carbonTextarea = element.matches?.("cds-textarea")
    ? element.shadowRoot?.querySelector("textarea")
    : null;
  if (carbonTextarea) return carbonTextarea.value;
  if ("value" in element) {
    return element.value;
  }
  return element.getAttribute("value") || "";
}

export function numberFieldValue(root, selector) {
  const value = Number(fieldValue(root, selector));
  return Number.isFinite(value) ? value : 0;
}
