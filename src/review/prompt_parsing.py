from __future__ import annotations

PRODUCT_CATEGORY_KEYWORDS = {
    "food": ("food", "meal", "restaurant", "delivery", "rice", "jollof"),
    "banking app": ("bank", "wallet", "transfer", "payment", "loan", "credit"),
    "government app": ("passport", "immigration", "identity", "verification", "nin"),
    "ecommerce app": ("marketplace", "shopping", "order", "refund", "return"),
    "telecom app": ("telecom", "airtime", "data bundle", "otp", "network"),
}


def split_labeled_sections(text: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    current_label = ""
    current_lines: list[str] = []

    for raw_line in text.splitlines():
        label, value = split_label(raw_line)
        if label:
            if current_label:
                sections[current_label] = "\n".join(current_lines).strip()
            current_label = normalize_label(label)
            current_lines = [value]
            continue
        if current_label:
            current_lines.append(raw_line.strip())

    if current_label:
        sections[current_label] = "\n".join(current_lines).strip()
    return sections


def split_label(line: str) -> tuple[str, str]:
    if ":" not in line:
        return "", ""
    label, value = line.split(":", 1)
    if not label.strip() or len(label.split()) > 4:
        return "", ""
    return label.strip(), value.strip()


def normalize_label(label: str) -> str:
    return label.strip().lower().replace(" ", "_")


def infer_category(product_details: str) -> str:
    normalized = product_details.lower()
    for category, keywords in PRODUCT_CATEGORY_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            return category
    return "general product"
