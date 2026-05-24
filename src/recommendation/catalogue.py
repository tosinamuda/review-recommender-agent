from __future__ import annotations

import json
from pathlib import Path

from .schemas import CandidateProduct


def load_product_catalogue(path: Path) -> tuple[CandidateProduct, ...]:
    if not path.exists():
        raise FileNotFoundError(f"Recommendation catalogue not found: {path}")
    products: list[CandidateProduct] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        product = CandidateProduct.model_validate(json.loads(line))
        if product.product_id in seen_ids:
            raise ValueError(
                f"Duplicate product_id in {path} line {line_number}: {product.product_id}"
            )
        seen_ids.add(product.product_id)
        products.append(product)
    if not products:
        raise ValueError(f"Recommendation catalogue is empty: {path}")
    return tuple(products)


def product_search_text(product: CandidateProduct, axis: str) -> str:
    tags = product.metadata.get("tags", [])
    occasions = product.metadata.get("occasions", [])
    audience = product.metadata.get("audience", [])
    fields = [
        product.name,
        product.category,
        product.description,
        product.location or "",
        " ".join(str(tag) for tag in tags),
    ]
    if axis == "persona":
        fields.extend(str(item) for item in audience)
    elif axis == "context":
        fields.extend(str(item) for item in occasions)
        fields.append(str(product.metadata.get("delivery_minutes", "")))
    elif axis == "joint":
        fields.extend(str(item) for item in audience)
        fields.extend(str(item) for item in occasions)
        fields.extend(f"{key}:{value}" for key, value in product.metadata.items())
    else:
        raise ValueError(f"Unknown recommendation retrieval axis: {axis}")
    return " ".join(str(field) for field in fields if field)
