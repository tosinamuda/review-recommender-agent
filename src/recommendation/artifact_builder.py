from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import faiss

from shared.embeddings import EmbeddingModel

from .case_data import (
    RecommendationEvalCase,
    RecommendationInteraction,
    RecommendationSampleManifest,
    StagedRecommendationCase,
    load_manifest,
    load_recommendation_cases,
)
from .case_validation import validate_recommendation_cases
from .catalogue import product_search_text
from .schemas import CandidateProduct


def build_recommendation_artifacts(
    *,
    source_path: Path,
    manifest_path: Path,
    catalogue_path: Path,
    interactions_path: Path,
    eval_path: Path,
    index_dir: Path,
    embedding_model: EmbeddingModel,
    persona_cases_path: Path | None = None,
    manual_catalogue_path: Path | None = None,
    manual_cases_path: Path | None = None,
    heldout_source_path: Path | None = None,
    heldout_manifest_path: Path | None = None,
) -> dict:
    validation = validate_recommendation_cases(
        source_path=source_path,
        manifest_path=manifest_path,
        require_personas=True,
    )
    if not validation.ok:
        raise ValueError(
            "Recommendation cases are not valid. "
            f"First failure: {validation.failures[0]}"
        )

    cases = load_recommendation_cases(source_path)
    manifest = load_manifest(manifest_path)
    heldout_cases = load_optional_cases(heldout_source_path)
    heldout_manifest = load_optional_manifest(heldout_manifest_path)
    if heldout_cases and heldout_manifest_path is None:
        raise ValueError("heldout_manifest_path is required when heldout_source_path is set")
    if heldout_cases:
        heldout_validation = validate_recommendation_cases(
            source_path=heldout_source_path or Path(),
            manifest_path=heldout_manifest_path or Path(),
            require_personas=True,
            enforce_context_match=True,
        )
        if not heldout_validation.ok:
            raise ValueError(
                "Held-out recommendation cases are not valid. "
                f"First failure: {heldout_validation.failures[0]}"
            )

    referenced_product_ids = referenced_products(cases)
    heldout_product_ids = referenced_products(heldout_cases)
    products = manifest_products_for_references(
        manifest=manifest,
        product_ids=referenced_product_ids,
    )
    if heldout_manifest is not None:
        products.extend(
            manifest_products_for_references(
                manifest=heldout_manifest,
                product_ids=heldout_product_ids,
            )
        )
    manual_products = load_manual_catalogue(manual_catalogue_path)
    products.extend(manual_products)
    duplicate_products = duplicates([product.product_id for product in products])
    if duplicate_products:
        raise ValueError(
            "Duplicate recommendation product_id: "
            f"{', '.join(duplicate_products[:5])}"
        )
    products_by_id = {product.product_id: product for product in products}
    if referenced_product_ids - set(products_by_id):
        missing = sorted(referenced_product_ids - set(products_by_id))
        raise ValueError(f"Referenced products missing from manifest: {', '.join(missing[:5])}")
    if heldout_product_ids - set(products_by_id):
        missing = sorted(heldout_product_ids - set(products_by_id))
        raise ValueError(
            f"Held-out referenced products missing from manifest: {', '.join(missing[:5])}"
        )

    manual_cases = load_manual_cases(manual_cases_path)
    validate_manual_cases(manual_cases, products_by_id)
    persona_source_cases = cases + manual_cases
    interactions = build_interactions(persona_source_cases)
    persona_cases = recommendation_eval_cases(persona_source_cases)
    eval_source_cases = heldout_cases or persona_source_cases
    eval_cases = recommendation_eval_cases(eval_source_cases)
    duplicate_cases = duplicates([case.case_id for case in eval_cases])
    if duplicate_cases:
        raise ValueError(
            "Duplicate recommendation eval case_id: "
            f"{', '.join(duplicate_cases[:5])}"
        )

    write_jsonl(catalogue_path, [product.model_dump(exclude_none=True) for product in products])
    write_jsonl(
        interactions_path,
        [interaction.model_dump(exclude_none=True) for interaction in interactions],
    )
    resolved_persona_cases_path = persona_cases_path or eval_path.with_name("persona_cases.jsonl")
    write_jsonl(
        resolved_persona_cases_path,
        [case.model_dump(exclude_none=True) for case in persona_cases],
    )
    write_jsonl(eval_path, [case.model_dump(exclude_none=True) for case in eval_cases])
    write_indexes(
        products=products,
        persona_cases=persona_cases,
        embedding_model=embedding_model,
        index_dir=index_dir,
        catalogue_path=catalogue_path,
        interactions_path=interactions_path,
        persona_cases_path=resolved_persona_cases_path,
    )
    return {
        "product_count": len(products),
        "interaction_count": len(interactions),
        "persona_case_count": len(persona_cases),
        "eval_case_count": len(eval_cases),
        "catalogue": str(catalogue_path),
        "interactions": str(interactions_path),
        "persona_cases": str(resolved_persona_cases_path),
        "eval": str(eval_path),
        "index_dir": str(index_dir),
        "embedding_model": embedding_model.model_name,
        "manual_product_count": len(manual_products),
        "manual_case_count": len(manual_cases),
        "heldout_case_count": len(heldout_cases),
    }


def load_optional_cases(path: Path | None) -> list[StagedRecommendationCase]:
    if path is None or not path.exists():
        return []
    return load_recommendation_cases(path)


def load_optional_manifest(path: Path | None) -> RecommendationSampleManifest | None:
    if path is None or not path.exists():
        return None
    return load_manifest(path)


def manifest_products_for_references(
    *,
    manifest: RecommendationSampleManifest,
    product_ids: set[str],
) -> list[CandidateProduct]:
    return [
        manifest_product_to_candidate(product)
        for product in manifest.products
        if product.product_id in product_ids
    ]


def recommendation_eval_cases(
    cases: list[StagedRecommendationCase],
) -> list[RecommendationEvalCase]:
    return [
        RecommendationEvalCase(
            case_id=case.case_id,
            user_persona=case.user_persona,
            context=case.context,
            relevant_product_ids=case.relevant_product_ids,
            source=case.source,
        )
        for case in cases
    ]


def load_manual_catalogue(path: Path | None) -> list[CandidateProduct]:
    if path is None or not path.exists():
        return []
    products: list[CandidateProduct] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            products.append(CandidateProduct.model_validate(json.loads(line)))
        except ValueError as exc:
            raise ValueError(
                f"Invalid manual recommendation catalogue row in {path} line {line_number}: {exc}"
            ) from exc
    return products


def load_manual_cases(path: Path | None) -> list[StagedRecommendationCase]:
    if path is None or not path.exists():
        return []
    records = [
        StagedRecommendationCase.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return records


def validate_manual_cases(
    cases: list[StagedRecommendationCase],
    products_by_id: dict[str, CandidateProduct],
) -> None:
    product_ids = set(products_by_id)
    for case in cases:
        if not case.user_persona.strip():
            raise ValueError(f"{case.case_id}: manual recommendation persona must not be empty")
        if not case.relevant_product_ids:
            raise ValueError(f"{case.case_id}: relevant_product_ids must not be empty")
        unresolved = [
            product_id
            for product_id in [*case.history_product_ids, *case.relevant_product_ids]
            if product_id not in product_ids
        ]
        if unresolved:
            raise ValueError(
                f"{case.case_id}: product ids do not resolve: {', '.join(unresolved[:5])}"
            )


def duplicates(values: list[str]) -> list[str]:
    seen: set[str] = set()
    repeated: list[str] = []
    for value in values:
        if value in seen and value not in repeated:
            repeated.append(value)
        seen.add(value)
    return repeated


def referenced_products(cases: list[StagedRecommendationCase]) -> set[str]:
    product_ids: set[str] = set()
    for case in cases:
        product_ids.update(case.history_product_ids)
        product_ids.update(case.relevant_product_ids)
    return product_ids


def manifest_product_to_candidate(product) -> CandidateProduct:
    category = product.categories[0] if product.categories else "business"
    location = ", ".join(part for part in [product.city, product.state] if part)
    description = (
        f"{product.name} is a {', '.join(product.categories) or 'local business'} "
        f"in {location or 'its local market'}."
    )
    if product.stars is not None and product.review_count is not None:
        description += (
            f" Yelp metadata lists a {product.stars:.1f} average rating across "
            f"{product.review_count} reviews."
        )
    return CandidateProduct(
        product_id=product.product_id,
        name=product.name,
        category=category.lower(),
        description=description,
        price=None,
        currency="USD",
        location=location,
        metadata={
            "source": "yelp",
            "source_business_id": product.source_business_id,
            "categories": product.categories,
            "stars": product.stars,
            "review_count": product.review_count,
            "is_open": product.is_open,
        },
    )


def build_interactions(cases: list[StagedRecommendationCase]) -> list[RecommendationInteraction]:
    interactions: list[RecommendationInteraction] = []
    for case in cases:
        for item in case.persona_context.history:
            interactions.append(
                RecommendationInteraction(
                    case_id=case.case_id,
                    product_id=item.product_id,
                    rating=item.rating,
                    date=item.date,
                    liked=item.rating >= 4.0,
                    split="history",
                    source=case.source,
                )
            )
    return interactions


def write_indexes(
    *,
    products: list[CandidateProduct],
    persona_cases: list[RecommendationEvalCase],
    embedding_model: EmbeddingModel,
    index_dir: Path,
    catalogue_path: Path,
    interactions_path: Path,
    persona_cases_path: Path,
) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    product_vectors = embedding_model.encode(
        [product_search_text(product, "joint") for product in products]
    )
    persona_vectors = embedding_model.encode(
        [f"{case.user_persona} {case.context}" for case in persona_cases]
    )
    faiss.write_index(build_faiss_index(product_vectors), str(index_dir / "product.faiss"))
    faiss.write_index(build_faiss_index(persona_vectors), str(index_dir / "persona.faiss"))
    metadata = {
        "embedding_model": embedding_model.model_name,
        "product_count": len(products),
        "case_count": len(persona_cases),
        "catalogue_path": str(catalogue_path),
        "interactions_path": str(interactions_path),
        "persona_cases_path": str(persona_cases_path),
        "indexes": ["product.faiss", "persona.faiss"],
    }
    (index_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_faiss_index(vectors) -> faiss.Index:
    index = faiss.IndexFlatIP(vectors.shape[1])
    cast(Any, index).add(vectors)
    return index


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
