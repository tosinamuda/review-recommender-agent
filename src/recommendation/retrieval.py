from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, cast

import faiss

from app.settings import get_settings
from shared.embeddings import EmbeddingModel, SentenceTransformerEmbeddingModel

from .case_data import RecommendationEvalCase, RecommendationInteraction
from .catalogue import load_product_catalogue, product_search_text
from .schemas import CandidateProductSet, RetrievedCandidate

AXES = ("persona", "context", "joint")
AXIS_WEIGHTS = {
    "persona": 0.40,
    "context": 0.25,
    "joint": 0.35,
}
DEFAULT_CANDIDATE_COUNT = 28
ARTIFACT_PRODUCT_WEIGHT = 0.55
ARTIFACT_SIMILAR_PERSONA_WEIGHT = 0.30
ARTIFACT_QUALITY_WEIGHT = 0.15
ARTIFACT_QUERY_DOMAIN_WEIGHT = 0.45
ARTIFACT_QUERY_SUBCATEGORY_WEIGHT = 0.12
ARTIFACT_QUERY_LOCATION_WEIGHT = 0.40
ARTIFACT_QUERY_TOKEN_WEIGHT = 0.20
TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a",
    "about",
    "after",
    "all",
    "an",
    "and",
    "are",
    "around",
    "at",
    "based",
    "for",
    "from",
    "in",
    "into",
    "is",
    "near",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "who",
    "with",
    "wants",
    "want",
}
DOMAIN_QUERY_TERMS = {
    "digital_service": {
        "app",
        "apps",
        "delivery",
        "ecommerce",
        "online",
        "ride",
        "transport",
        "utility",
    },
    "education": {
        "admission",
        "admissions",
        "applicant",
        "bootcamp",
        "campus",
        "career",
        "certification",
        "coding",
        "college",
        "cybersecurity",
        "data",
        "degree",
        "developer",
        "higher",
        "ict",
        "job",
        "learning",
        "mentorship",
        "parent",
        "polytechnic",
        "programme",
        "programmes",
        "professional",
        "remote",
        "school",
        "skills",
        "software",
        "student",
        "technology",
        "training",
        "undergraduate",
        "university",
        "upskilling",
    },
    "entertainment": {
        "activity",
        "arcade",
        "afrobeats",
        "bowling",
        "cinema",
        "date",
        "entertainment",
        "game",
        "hangout",
        "leisure",
        "movie",
        "music",
        "outing",
        "podcast",
        "series",
        "song",
        "stream",
        "streaming",
        "theater",
        "theatre",
        "video",
    },
    "finance": {
        "bank",
        "banking",
        "card",
        "merchant",
        "payment",
        "pos",
        "settlement",
        "transfer",
        "wallet",
    },
    "food": {
        "amala",
        "bole",
        "budget",
        "buka",
        "bukka",
        "cafe",
        "canteen",
        "central",
        "cheap",
        "continental",
        "couple",
        "date",
        "dinner",
        "eba",
        "ewedu",
        "family",
        "fish",
        "food",
        "gbegiri",
        "grill",
        "indian",
        "jollof",
        "local",
        "lunch",
        "meal",
        "market",
        "nigerian",
        "pepper",
        "professional",
        "restaurant",
        "rice",
        "seafood",
        "soup",
        "spicy",
        "swallow",
        "team",
    },
    "health": {
        "care",
        "clinic",
        "diagnostic",
        "diagnostics",
        "emergency",
        "family",
        "fitness",
        "gym",
        "health",
        "hospital",
        "outpatient",
        "pharmacist",
        "pharmacy",
        "routine",
        "specialist",
        "wellness",
    },
    "religion": {
        "church",
        "fellowship",
        "service",
        "worship",
    },
    "retail": {
        "clothes",
        "clothing",
        "fabric",
        "fashion",
        "foodstuff",
        "goods",
        "grocery",
        "household",
        "leather",
        "market",
        "provision",
        "provisions",
        "retail",
        "shoe",
        "shopping",
        "sourcing",
        "textile",
        "textiles",
        "thrift",
        "trader",
        "wholesale",
    },
    "service": {
        "boardroom",
        "barber",
        "convenience",
        "coworking",
        "device",
        "dry",
        "errand",
        "grooming",
        "laundry",
        "meeting",
        "office",
        "phone",
        "pickup",
        "printing",
        "repair",
        "salon",
        "service",
        "workspace",
    },
    "telecom": {
        "airtime",
        "calls",
        "connectivity",
        "data",
        "mobile",
        "network",
        "telecom",
    },
}
DOMAIN_PRODUCT_ALIASES = {
    "entertainment": {"cinema", "entertainment", "film", "movie_theatre", "place"},
}
SUBCATEGORY_QUERY_TERMS = {
    "career_learning": {
        "bootcamp",
        "career",
        "certification",
        "coding",
        "cybersecurity",
        "data",
        "developer",
        "ict",
        "job",
        "mentorship",
        "professional",
        "remote",
        "skills",
        "software",
        "training",
        "upskilling",
    },
    "amala_local_canteen": {
        "amala",
        "budget",
        "bukka",
        "canteen",
        "cheap",
        "ewedu",
        "gbegiri",
        "local",
        "lunch",
        "quick",
        "student",
        "worker",
    },
    "bole_casual_restaurant": {
        "after",
        "bole",
        "casual",
        "coworker",
        "fish",
        "friend",
        "local",
        "lunch",
        "plantain",
        "rivers",
        "worker",
    },
    "casual_lounge_restaurant": {
        "access",
        "ambience",
        "business",
        "casual",
        "children",
        "comfortable",
        "delivery",
        "dinner",
        "family",
        "friend",
        "group",
        "island",
        "late",
        "lounge",
        "lunch",
        "mainland",
        "nigerian",
        "office",
        "outing",
        "planned",
        "professional",
        "space",
        "varied",
        "weekend",
    },
    "contemporary_nigerian_restaurant": {
        "business",
        "children",
        "comfortable",
        "contemporary",
        "date",
        "dining",
        "family",
        "group",
        "island",
        "lunch",
        "nigerian",
        "office",
        "planned",
        "private",
        "professional",
        "varied",
        "weekend",
    },
    "family_restaurant": {
        "casual",
        "children",
        "dinner",
        "family",
        "group",
        "lunch",
        "meal",
        "restaurant",
        "swallow",
    },
    "gym_wellness": {
        "exercise",
        "fitness",
        "gym",
        "routine",
        "wellness",
        "workout",
    },
    "hospital_clinic": {
        "care",
        "clinic",
        "diagnostic",
        "diagnostics",
        "emergency",
        "hospital",
        "outpatient",
        "routine",
        "specialist",
    },
    "indian_restaurant": {
        "couple",
        "date",
        "dinner",
        "group",
        "indian",
        "international",
        "planned",
        "restaurant",
    },
    "lebanese_intercontinental_restaurant": {
        "casual",
        "date",
        "dining",
        "dinner",
        "friend",
        "group",
        "intercontinental",
        "international",
        "lebanese",
        "planned",
        "professional",
    },
    "coworking_space": {
        "boardroom",
        "conference",
        "coworking",
        "desk",
        "internet",
        "meeting",
        "office",
        "printing",
        "remote",
        "workspace",
    },
    "laundry_pickup": {
        "cleaning",
        "dry",
        "errand",
        "express",
        "laundry",
        "pickup",
        "stain",
        "workwear",
    },
    "local_food_canteen": {
        "amala",
        "budget",
        "bukka",
        "canteen",
        "cheap",
        "ewedu",
        "gbegiri",
        "local",
        "market",
        "quick",
        "suya",
    },
    "multi_service_concierge": {
        "concierge",
        "coworking",
        "errand",
        "event",
        "grooming",
        "laundry",
        "lifestyle",
        "workspace",
    },
    "multi_specialist_hospital": {
        "care",
        "clinic",
        "diagnostic",
        "diagnostics",
        "emergency",
        "hospital",
        "laboratory",
        "outpatient",
        "specialist",
    },
    "multi_branch_nigerian_restaurant": {
        "business",
        "children",
        "comfortable",
        "continental",
        "delivery",
        "family",
        "lunch",
        "meal",
        "nigerian",
        "office",
        "pickup",
        "quick",
        "rice",
        "space",
        "varied",
        "weekend",
    },
    "nigerian_continental_restaurant": {
        "central",
        "continental",
        "date",
        "dinner",
        "group",
        "international",
        "nigerian",
        "professional",
    },
    "polytechnic": {
        "admission",
        "admissions",
        "applicant",
        "campus",
        "college",
        "degree",
        "diploma",
        "higher",
        "polytechnic",
        "programme",
        "programmes",
        "student",
        "technology",
        "undergraduate",
    },
    "pharmacy": {
        "errand",
        "medicine",
        "ordering",
        "pharmacist",
        "pharmacy",
        "pickup",
        "prescription",
        "products",
        "wellness",
    },
    "quick_service_restaurant": {
        "delivery",
        "family",
        "lunch",
        "meal",
        "pickup",
        "quick",
        "rice",
        "service",
        "swallow",
        "takeout",
        "team",
        "workday",
    },
    "seafood_family_restaurant": {
        "children",
        "comfortable",
        "family",
        "group",
        "international",
        "meal",
        "platter",
        "planned",
        "seafood",
        "space",
        "sushi",
        "varied",
        "weekend",
    },
    "phone_repair_service": {
        "battery",
        "customer",
        "device",
        "phone",
        "repair",
        "screen",
        "service",
        "warranty",
    },
    "private_university": {
        "admission",
        "admissions",
        "applicant",
        "campus",
        "degree",
        "higher",
        "programme",
        "programmes",
        "student",
        "undergraduate",
        "university",
    },
    "restaurant_cafe": {
        "bakery",
        "cafe",
        "casual",
        "dining",
        "family",
        "group",
        "intercontinental",
        "light",
        "lunch",
        "restaurant",
    },
    "science_technology_university": {
        "admission",
        "admissions",
        "applicant",
        "campus",
        "degree",
        "higher",
        "programme",
        "programmes",
        "science",
        "student",
        "technology",
        "undergraduate",
        "university",
    },
    "university": {
        "admission",
        "admissions",
        "applicant",
        "campus",
        "degree",
        "higher",
        "programme",
        "programmes",
        "student",
        "undergraduate",
        "university",
    },
    "waterfront_family_restaurant": {
        "breakfast",
        "children",
        "comfortable",
        "dinner",
        "family",
        "international",
        "lunch",
        "office",
        "restaurant",
        "space",
        "varied",
        "waterfront",
        "weekend",
    },
}
LOCATION_TOKENS = {
    "abuja",
    "akoka",
    "aba",
    "adesola",
    "asokoro",
    "basorun",
    "bodija",
    "cbd",
    "dugbe",
    "dline",
    "garki",
    "gra",
    "ibadan",
    "ikeja",
    "iwo",
    "isolo",
    "kano",
    "lago",
    "lagos",
    "lekki",
    "mabushi",
    "maitama",
    "mbadiwe",
    "mokola",
    "nigeria",
    "ogudu",
    "ojota",
    "oshodi",
    "oju",
    "olobun",
    "ozumba",
    "port",
    "harcourt",
    "river",
    "rivers",
    "sango",
    "tombia",
    "trans",
    "amadi",
    "victoria",
    "wuse",
    "yaba",
}
BROAD_LOCATION_TOKENS = {
    "abuja",
    "harcourt",
    "ibadan",
    "kano",
    "lago",
    "lagos",
    "nigeria",
    "port",
    "river",
}


class ProductRetriever:
    def __init__(
        self,
        catalogue_path: Path | None = None,
        index_dir: Path | None = None,
        interactions_path: Path | None = None,
        persona_cases_path: Path | None = None,
        eval_cases_path: Path | None = None,
        embedding_model: EmbeddingModel | None = None,
    ):
        settings = get_settings()
        self._catalogue_path = (catalogue_path or settings.recommendation_catalogue_path).resolve()
        self._index_dir = (index_dir or settings.recommendation_index_dir).resolve()
        self._interactions_path = (
            interactions_path or settings.recommendation_interactions_path
        ).resolve()
        if persona_cases_path is not None:
            resolved_persona_cases_path = persona_cases_path
        elif eval_cases_path is not None:
            resolved_persona_cases_path = eval_cases_path
        else:
            resolved_persona_cases_path = settings.recommendation_persona_cases_path
        self._persona_cases_path = resolved_persona_cases_path.resolve()
        self._embedding_model = embedding_model or SentenceTransformerEmbeddingModel(
            settings.embedding_model_name
        )
        self._products = load_product_catalogue(self._catalogue_path)
        self._artifact_index = self._load_artifact_index()
        self._indexes = None if self._artifact_index else self._build_axis_indexes()

    def retrieve(
        self,
        user_persona: str,
        context: str = "",
        category: str | None = None,
        k: int = DEFAULT_CANDIDATE_COUNT,
    ) -> CandidateProductSet:
        if self._artifact_index:
            return self._retrieve_from_artifacts(
                user_persona=user_persona,
                context=context,
                category=category,
                k=k,
            )
        product_indexes = self._candidate_indexes(category)
        if not product_indexes:
            return CandidateProductSet(candidates=[], retrieved_via=list(AXES), category=category)
        axis_indexes = self._indexes
        if axis_indexes is None:
            raise RuntimeError("Recommendation retrieval indexes are not loaded.")

        axis_queries = {
            "persona": user_persona,
            "context": context or user_persona,
            "joint": " ".join(part for part in [user_persona, context] if part),
        }
        combined_scores: dict[int, float] = {}
        axis_scores: dict[int, dict[str, float]] = {}
        per_axis_k = min(len(self._products), max(k * 3, k))

        for axis, query in axis_queries.items():
            query_vector = self._embedding_model.encode([query])
            distances, indexes = cast(Any, axis_indexes[axis]).search(query_vector, per_axis_k)
            for raw_score, raw_index in zip(distances[0], indexes[0], strict=True):
                if raw_index < 0:
                    continue
                index = int(raw_index)
                if index not in product_indexes:
                    continue
                score = max(float(raw_score), 0.0)
                axis_scores.setdefault(index, {})[axis] = round(score, 6)
                combined_scores[index] = combined_scores.get(index, 0.0) + (
                    AXIS_WEIGHTS[axis] * score
                )

        ranked_indexes = sorted(
            combined_scores,
            key=lambda index: (combined_scores[index], self._products[index].name),
            reverse=True,
        )
        if len(ranked_indexes) < min(k, len(product_indexes)):
            remaining = [index for index in product_indexes if index not in combined_scores]
            ranked_indexes.extend(remaining)
        candidates = [
            RetrievedCandidate(
                product=self._products[index],
                score=max(round(combined_scores.get(index, 0.0), 6), 0.0),
                axis_scores=axis_scores.get(index, {}),
            )
            for index in ranked_indexes[:k]
        ]
        return CandidateProductSet(
            candidates=candidates,
            retrieved_via=[axis for axis, query in axis_queries.items() if query],
            category=category,
        )

    def _retrieve_from_artifacts(
        self,
        *,
        user_persona: str,
        context: str,
        category: str | None,
        k: int,
    ) -> CandidateProductSet:
        artifact_index = self._artifact_index
        if artifact_index is None:
            raise RuntimeError("Artifact index is not loaded.")
        product_indexes = self._candidate_indexes(category)
        if not product_indexes:
            return CandidateProductSet(
                candidates=[],
                retrieved_via=["product_text", "similar_persona", "quality"],
                category=category,
            )

        query = " ".join(part for part in [user_persona, context] if part)
        query_vector = self._embedding_model.encode([query])
        product_scores: dict[int, float] = {}
        axis_scores: dict[int, dict[str, float]] = {}
        self._add_product_text_scores(
            artifact_index=artifact_index,
            query_vector=query_vector,
            product_indexes=product_indexes,
            product_scores=product_scores,
            axis_scores=axis_scores,
            k=k,
        )
        self._add_similar_persona_scores(
            artifact_index=artifact_index,
            query_vector=query_vector,
            product_indexes=product_indexes,
            product_scores=product_scores,
            axis_scores=axis_scores,
        )
        self._add_query_match_scores(
            query=query,
            product_indexes=product_indexes,
            product_scores=product_scores,
            axis_scores=axis_scores,
        )
        self._add_quality_scores(
            product_indexes=product_indexes,
            product_scores=product_scores,
            axis_scores=axis_scores,
        )

        ranked_indexes = sorted(
            product_scores,
            key=lambda index: (product_scores[index], self._products[index].name),
            reverse=True,
        )
        candidates = [
            RetrievedCandidate(
                product=self._products[index],
                score=max(round(product_scores.get(index, 0.0), 6), 0.0),
                axis_scores=axis_scores.get(index, {}),
            )
            for index in ranked_indexes[:k]
        ]
        return CandidateProductSet(
            candidates=candidates,
            retrieved_via=["product_text", "similar_persona", "quality"],
            category=category,
        )

    def _add_product_text_scores(
        self,
        *,
        artifact_index: RecommendationArtifactIndex,
        query_vector,
        product_indexes: set[int],
        product_scores: dict[int, float],
        axis_scores: dict[int, dict[str, float]],
        k: int,
    ) -> None:
        per_axis_k = min(len(self._products), max(k * 4, k))
        distances, indexes = cast(Any, artifact_index.product_index).search(
            query_vector,
            per_axis_k,
        )
        for raw_score, raw_index in zip(distances[0], indexes[0], strict=True):
            if raw_index < 0:
                continue
            index = int(raw_index)
            if index not in product_indexes:
                continue
            score = max(float(raw_score), 0.0)
            product_scores[index] = product_scores.get(index, 0.0) + (
                ARTIFACT_PRODUCT_WEIGHT * score
            )
            axis_scores.setdefault(index, {})["product_text"] = round(score, 6)

    def _add_similar_persona_scores(
        self,
        *,
        artifact_index: RecommendationArtifactIndex,
        query_vector,
        product_indexes: set[int],
        product_scores: dict[int, float],
        axis_scores: dict[int, dict[str, float]],
    ) -> None:
        if not artifact_index.persona_cases:
            return
        persona_distances, persona_indexes = cast(Any, artifact_index.persona_index).search(
            query_vector,
            min(len(artifact_index.persona_cases), 20),
        )
        for raw_score, raw_index in zip(persona_distances[0], persona_indexes[0], strict=True):
            if raw_index < 0:
                continue
            similar_case = artifact_index.persona_cases[int(raw_index)]
            similar_score = max(float(raw_score), 0.0)
            self._add_liked_product_scores(
                artifact_index=artifact_index,
                case_id=similar_case.case_id,
                similar_score=similar_score,
                product_indexes=product_indexes,
                product_scores=product_scores,
                axis_scores=axis_scores,
            )

    def _add_liked_product_scores(
        self,
        *,
        artifact_index: RecommendationArtifactIndex,
        case_id: str,
        similar_score: float,
        product_indexes: set[int],
        product_scores: dict[int, float],
        axis_scores: dict[int, dict[str, float]],
    ) -> None:
        for product_id in artifact_index.liked_products_by_case_id.get(case_id, []):
            product_index = artifact_index.product_index_by_id.get(product_id)
            if product_index is None or product_index not in product_indexes:
                continue
            product_scores[product_index] = product_scores.get(product_index, 0.0) + (
                ARTIFACT_SIMILAR_PERSONA_WEIGHT * similar_score
            )
            previous = axis_scores.setdefault(product_index, {}).get("similar_persona", 0.0)
            axis_scores[product_index]["similar_persona"] = round(
                max(previous, similar_score),
                6,
            )

    def _add_quality_scores(
        self,
        *,
        product_indexes: set[int],
        product_scores: dict[int, float],
        axis_scores: dict[int, dict[str, float]],
    ) -> None:
        for index in product_indexes:
            quality = product_quality_score(self._products[index])
            if quality <= 0:
                continue
            product_scores[index] = product_scores.get(index, 0.0) + (
                ARTIFACT_QUALITY_WEIGHT * quality
            )
            axis_scores.setdefault(index, {})["quality"] = round(quality, 6)

    def _add_query_match_scores(
        self,
        *,
        query: str,
        product_indexes: set[int],
        product_scores: dict[int, float],
        axis_scores: dict[int, dict[str, float]],
    ) -> None:
        query_tokens = text_tokens(query)
        if not query_tokens:
            return
        inferred_domains = infer_query_domains(query_tokens)
        requested_locations = query_tokens & LOCATION_TOKENS
        requested_specific_locations = requested_locations - BROAD_LOCATION_TOKENS
        for index in product_indexes:
            product = self._products[index]
            product_tokens = product_match_tokens(product)
            score = 0.0
            if inferred_domains & product_domains(product):
                score += ARTIFACT_QUERY_DOMAIN_WEIGHT
            if query_tokens & product_subcategory_terms(product):
                score += ARTIFACT_QUERY_SUBCATEGORY_WEIGHT
            if requested_locations:
                location_score = location_match_score(requested_locations, product_tokens)
                score += location_score
                if (
                    requested_specific_locations
                    and location_score == 0.0
                    and product_tokens & LOCATION_TOKENS
                ):
                    score -= ARTIFACT_QUERY_LOCATION_WEIGHT
            overlap = query_tokens & product_tokens
            if overlap:
                score += ARTIFACT_QUERY_TOKEN_WEIGHT * min(len(overlap) / 6.0, 1.0)
            if score == 0:
                continue
            product_scores[index] = product_scores.get(index, 0.0) + score
            axis_scores.setdefault(index, {})["query_match"] = round(score, 6)

    def _candidate_indexes(self, category: str | None) -> set[int]:
        if not category:
            return set(range(len(self._products)))
        normalized = category.strip().lower()
        return {
            index
            for index, product in enumerate(self._products)
            if product.category.lower() == normalized or normalized in product_domains(product)
        }

    def _build_axis_indexes(self) -> dict[str, faiss.Index]:
        return {
            axis: build_faiss_index(
                self._embedding_model.encode(
                    [product_search_text(product, axis) for product in self._products]
                )
            )
            for axis in AXES
        }

    def _load_artifact_index(self) -> RecommendationArtifactIndex | None:
        paths = RecommendationIndexPaths(self._index_dir)
        artifact_files = [
            paths.metadata,
            paths.product,
            paths.persona,
            self._interactions_path,
            self._persona_cases_path,
        ]
        if not all(path.exists() for path in artifact_files):
            return None
        metadata = json.loads(paths.metadata.read_text(encoding="utf-8"))
        if metadata.get("embedding_model") != self._embedding_model.model_name:
            raise ValueError(
                "Recommendation index embedding model mismatch: "
                f"artifact={metadata.get('embedding_model')!r}, "
                f"runtime={self._embedding_model.model_name!r}"
            )
        if int(metadata["product_count"]) != len(self._products):
            raise ValueError(
                "Recommendation index product count mismatch: "
                f"artifact={metadata['product_count']}, catalogue={len(self._products)}"
            )
        return RecommendationArtifactIndex(
            product_index=faiss.read_index(str(paths.product)),
            persona_index=faiss.read_index(str(paths.persona)),
            persona_cases=load_eval_cases(self._persona_cases_path),
            liked_products_by_case_id=load_liked_products(self._interactions_path),
            product_index_by_id={
                product.product_id: index
                for index, product in enumerate(self._products)
            },
        )


def build_faiss_index(vectors) -> faiss.Index:
    index = faiss.IndexFlatIP(vectors.shape[1])
    cast(Any, index).add(vectors)
    return index


class RecommendationIndexPaths:
    def __init__(self, root: Path):
        self.root = root

    @property
    def metadata(self) -> Path:
        return self.root / "metadata.json"

    @property
    def product(self) -> Path:
        return self.root / "product.faiss"

    @property
    def persona(self) -> Path:
        return self.root / "persona.faiss"


class RecommendationArtifactIndex:
    def __init__(
        self,
        *,
        product_index: faiss.Index,
        persona_index: faiss.Index,
        persona_cases: list[RecommendationEvalCase],
        liked_products_by_case_id: dict[str, list[str]],
        product_index_by_id: dict[str, int],
    ):
        self.product_index = product_index
        self.persona_index = persona_index
        self.persona_cases = persona_cases
        self.liked_products_by_case_id = liked_products_by_case_id
        self.product_index_by_id = product_index_by_id


def load_eval_cases(path: Path) -> list[RecommendationEvalCase]:
    return [
        RecommendationEvalCase.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_liked_products(path: Path) -> dict[str, list[str]]:
    liked_products: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        interaction = RecommendationInteraction.model_validate(json.loads(line))
        if interaction.split != "history" or not interaction.liked:
            continue
        liked_products.setdefault(interaction.case_id, [])
        if interaction.product_id not in liked_products[interaction.case_id]:
                liked_products[interaction.case_id].append(interaction.product_id)
    return liked_products


def infer_query_domains(query_tokens: set[str]) -> set[str]:
    return {
        domain
        for domain, domain_terms in DOMAIN_QUERY_TERMS.items()
        if query_tokens & {normalize_token(term) for term in domain_terms}
    }


def product_domains(product) -> set[str]:
    metadata = product.metadata
    raw_domains = {
        product.category,
        str(metadata.get("domain", "")),
        str(metadata.get("subcategory", "")),
    }
    domains = {
        value.strip().lower().replace(" ", "_")
        for value in raw_domains
        if value.strip()
    }
    for domain, aliases in DOMAIN_PRODUCT_ALIASES.items():
        alias_keys = {
            alias.strip().lower().replace(" ", "_")
            for alias in aliases
            if alias.strip()
        }
        if domains & alias_keys:
            domains.add(domain)
    return {domain for domain in domains if domain}


def product_subcategory_terms(product) -> set[str]:
    raw_subcategory = str(product.metadata.get("subcategory", "")).strip().lower()
    subcategory_keys = {
        raw_subcategory,
        raw_subcategory.replace(" ", "_"),
        normalize_token(raw_subcategory),
    }
    return {
        normalize_token(term)
        for subcategory in subcategory_keys
        for term in SUBCATEGORY_QUERY_TERMS.get(subcategory, set())
    }


def product_match_tokens(product) -> set[str]:
    values: list[Any] = [
        product.name,
        product.category,
        product.description,
        product.location or "",
    ]
    metadata = product.metadata
    for key in (
        "domain",
        "subcategory",
        "area",
        "city",
        "country",
        "tags",
        "audience",
        "occasions",
        "menu_highlights",
        "service_options",
    ):
        value = metadata.get(key)
        if isinstance(value, list):
            values.extend(value)
        elif value:
            values.append(value)
    return text_tokens(" ".join(str(value) for value in values))


def location_match_score(requested_locations: set[str], product_tokens: set[str]) -> float:
    requested_broad_locations = requested_locations & BROAD_LOCATION_TOKENS
    if requested_broad_locations and not (requested_broad_locations & product_tokens):
        return 0.0
    requested_specific_locations = requested_locations - BROAD_LOCATION_TOKENS
    if requested_specific_locations:
        if requested_specific_locations & product_tokens:
            return ARTIFACT_QUERY_LOCATION_WEIGHT
        if requested_locations & product_tokens:
            return ARTIFACT_QUERY_LOCATION_WEIGHT * 0.75
        return 0.0
    if requested_locations & product_tokens:
        return ARTIFACT_QUERY_LOCATION_WEIGHT
    return 0.0


def text_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for match in TOKEN_RE.finditer(text.lower()):
        raw_token = match.group(0)
        candidate_tokens = [raw_token]
        if "_" in raw_token:
            candidate_tokens.extend(raw_token.split("_"))
        for candidate in candidate_tokens:
            token = normalize_token(candidate)
            if token and token not in STOPWORDS and len(token) > 1:
                tokens.add(token)
    return tokens


def normalize_token(token: str) -> str:
    normalized = token.strip().lower().replace("_", " ")
    normalized = normalized.split()[-1] if " " in normalized else normalized
    if len(normalized) > 4 and normalized.endswith("ies"):
        return f"{normalized[:-3]}y"
    if len(normalized) > 3 and normalized.endswith("s"):
        return normalized[:-1]
    return normalized


def product_quality_score(product) -> float:
    stars = product.metadata.get("stars")
    review_count = product.metadata.get("review_count")
    if stars is None:
        return 0.0
    rating_score = max(0.0, min(float(stars) / 5.0, 1.0))
    if review_count is None:
        return rating_score
    count_score = min(float(review_count) / 500.0, 1.0)
    return (0.75 * rating_score) + (0.25 * count_score)
