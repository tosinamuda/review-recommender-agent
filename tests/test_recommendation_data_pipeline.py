from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from recommendation.artifact_builder import build_recommendation_artifacts
from recommendation.case_data import (
    load_recommendation_cases,
    normalize_persona,
    write_recommendation_cases,
)
from recommendation.case_validation import validate_recommendation_cases
from recommendation.catalogue import load_product_catalogue
from recommendation.manual_personas import (
    NO_EMPTY_RECOMMENDATION_PERSONA_MESSAGE,
    display_case,
    set_case_persona,
)
from recommendation.persona_leakage import audit_persona_business_leakage
from recommendation.retrieval import ProductRetriever
from recommendation.yelp_sampling import sample_yelp_recommendation_cases
from tests.fakes import HashingEmbeddingModel


def test_sampler_creates_empty_persona_cases_and_manifest(tmp_path: Path) -> None:
    paths = write_yelp_fixture(tmp_path)

    cases, manifest = sample_yelp_recommendation_cases(
        business_path=paths["business"],
        review_path=paths["reviews"],
        output_path=paths["cases"],
        manifest_path=paths["manifest"],
        sample_size=2,
        seed=20260522,
        oversample_factor=2,
    )

    assert len(cases) == 2
    assert all(case.user_persona == "" for case in cases)
    assert len({case.case_id for case in cases}) == 2
    assert all(case.relevant_product_ids for case in cases)
    assert manifest.seed == 20260522
    assert manifest.sample_size == 2
    assert manifest.selected_case_ids == [case.case_id for case in cases]
    assert paths["cases"].exists()
    assert paths["manifest"].exists()


def test_sampler_keeps_heldout_labels_in_the_query_context(tmp_path: Path) -> None:
    paths = write_yelp_fixture(tmp_path)

    cases, manifest = sample_fixture(paths)
    relevant_product_ids = {
        product_id for case in cases for product_id in case.relevant_product_ids
    }
    product_names = {product.product_id: product.name for product in manifest.products}

    assert {case.context for case in cases} == {
        "restaurants in Philadelphia, PA",
        "food in Tampa, FL",
    }
    assert "Wrong City Cleaner" not in product_names.values()
    assert "Wrong Domain Spa" not in product_names.values()
    assert all(product_id in product_names for product_id in relevant_product_ids)


def test_show_case_hides_heldout_product_details(tmp_path: Path) -> None:
    paths = write_yelp_fixture(tmp_path)
    cases, _ = sample_fixture(paths)
    case = cases[0]

    shown = display_case(paths["cases"], case_id=case.case_id)

    shown_text = json.dumps(shown)
    assert "relevant_product_ids" not in shown
    assert "Heldout" not in shown_text
    assert shown["relevant_product_count"] >= 1
    assert shown["history"]


def test_set_persona_updates_only_target_case(tmp_path: Path) -> None:
    paths = write_yelp_fixture(tmp_path)
    cases, _ = sample_fixture(paths)
    target = cases[0]
    before = [case.model_dump() for case in load_recommendation_cases(paths["cases"])]

    updated = set_case_persona(
        paths["cases"],
        case_id=target.case_id,
        persona="Practical casual diner who prefers reliable restaurants with quick service.",
    )
    after = [case.model_dump() for case in load_recommendation_cases(paths["cases"])]

    assert updated.user_persona.startswith("Practical casual diner")
    changed_rows = [
        (old, new)
        for old, new in zip(before, after, strict=True)
        if old != new
    ]
    assert len(changed_rows) == 1
    old, new = changed_rows[0]
    assert old["case_id"] == new["case_id"] == target.case_id
    assert old | {"user_persona": new["user_persona"]} == new

    with pytest.raises(ValueError, match="already has a persona"):
        set_case_persona(paths["cases"], case_id=target.case_id, persona="Another persona.")


def test_persona_leakage_audit_flags_business_names_not_generic_traits(
    tmp_path: Path,
) -> None:
    catalogue_path = tmp_path / "catalogue.jsonl"
    cases_path = tmp_path / "cases.jsonl"
    write_jsonl(
        catalogue_path,
        [
            recommendation_product(
                product_id="ng_food_lagos_korede_spags_001",
                name="Korede Spaghetti",
                category="food",
            ),
            recommendation_product(
                product_id="ng_cinema_lagos_ozone_001",
                name="Ozone Cinemas",
                category="cinema",
            ),
        ],
    )
    write_jsonl(
        cases_path,
        [
            recommendation_case(
                case_id="generic_food_case",
                persona=(
                    "Young Lagos worker who wants cheap saucy spaghetti and quick "
                    "casual food."
                ),
                product_ids=["ng_food_lagos_korede_spags_001"],
            ),
            recommendation_case(
                case_id="leaky_cinema_case",
                persona="Lagos movie lover specifically asking for Ozone outings.",
                product_ids=["ng_cinema_lagos_ozone_001"],
            ),
        ],
    )

    findings = audit_persona_business_leakage(
        cases_path=cases_path,
        catalogue_path=catalogue_path,
    )

    assert [(finding.case_id, finding.matched_text) for finding in findings] == [
        ("leaky_cinema_case", "ozone")
    ]


def test_set_recommendation_case_persona_script_updates_manual_ng_source(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "manual_ng_cases.jsonl"
    write_jsonl(
        source_path,
        [
            recommendation_case(
                case_id="case_one",
                persona="Old persona for the first case.",
                product_ids=["product_one"],
            ),
            recommendation_case(
                case_id="case_two",
                persona="Second persona must remain unchanged.",
                product_ids=["product_two"],
            ),
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/set_recommendation_case_persona.py",
            "case_one",
            "--source",
            str(source_path),
            "--persona",
            "Manual reviewer writes a cleaner persona without business names.",
            "--force",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    cases = load_recommendation_cases(source_path)
    assert cases[0].user_persona == (
        "Manual reviewer writes a cleaner persona without business names."
    )
    assert cases[1].user_persona == "Second persona must remain unchanged."


def test_persona_sentence_validation_allows_common_abbreviations() -> None:
    assert normalize_persona(
        "St. Petersburg diner who likes lively restaurants and attentive service."
    ).startswith("St. Petersburg")

    with pytest.raises(ValueError, match="one sentence"):
        normalize_persona("Local diner who likes quick meals. Also wants strong service.")


def test_validator_rejects_empty_personas_then_accepts_filled_cases(tmp_path: Path) -> None:
    paths = write_yelp_fixture(tmp_path)
    cases, _ = sample_fixture(paths)

    rejected = validate_recommendation_cases(
        source_path=paths["cases"],
        manifest_path=paths["manifest"],
    )

    assert not rejected.ok
    assert any("Persona must not be empty" in failure for failure in rejected.failures)

    for index, case in enumerate(cases, 1):
        set_case_persona(
            paths["cases"],
            case_id=case.case_id,
            persona=f"Local diner {index} who values reliable food and convenient service.",
        )

    accepted = validate_recommendation_cases(
        source_path=paths["cases"],
        manifest_path=paths["manifest"],
    )

    assert accepted.ok
    assert accepted.case_count == 2


def test_show_next_empty_reports_completion_without_stack_trace(tmp_path: Path) -> None:
    paths = write_yelp_fixture(tmp_path)
    cases, _ = sample_fixture(paths)
    for index, case in enumerate(cases, 1):
        set_case_persona(
            paths["cases"],
            case_id=case.case_id,
            persona=f"Local diner {index} who values reliable food and convenient service.",
        )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/show_recommendation_case.py",
            "--next-empty",
            "--source",
            str(paths["cases"]),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert result.stderr == ""
    assert json.loads(result.stdout) == {
        "complete": True,
        "message": NO_EMPTY_RECOMMENDATION_PERSONA_MESSAGE,
    }


def test_artifact_builder_and_hybrid_retriever_use_generated_data(tmp_path: Path) -> None:
    paths = write_yelp_fixture(tmp_path)
    cases, _ = sample_fixture(paths)
    for index, case in enumerate(cases, 1):
        set_case_persona(
            paths["cases"],
            case_id=case.case_id,
            persona=f"Restaurant regular {index} who likes dependable service and casual meals.",
        )

    summary = build_recommendation_artifacts(
        source_path=paths["cases"],
        manifest_path=paths["manifest"],
        catalogue_path=paths["catalogue"],
        interactions_path=paths["interactions"],
        persona_cases_path=paths["persona_cases"],
        eval_path=paths["eval"],
        index_dir=paths["index_dir"],
        embedding_model=HashingEmbeddingModel(),
    )
    retriever = ProductRetriever(
        catalogue_path=paths["catalogue"],
        index_dir=paths["index_dir"],
        interactions_path=paths["interactions"],
        persona_cases_path=paths["persona_cases"],
        embedding_model=HashingEmbeddingModel(),
    )

    result = retriever.retrieve(
        user_persona="Restaurant regular who likes dependable service and casual meals.",
        context="restaurants in Philadelphia, PA",
        k=5,
    )

    assert summary["eval_case_count"] == 2
    assert paths["index_dir"].joinpath("product.faiss").exists()
    assert paths["index_dir"].joinpath("persona.faiss").exists()
    assert result.candidates
    assert "similar_persona" in result.retrieved_via
    assert any("similar_persona" in candidate.axis_scores for candidate in result.candidates)


def test_artifact_builder_keeps_heldout_eval_cases_out_of_persona_index(
    tmp_path: Path,
) -> None:
    paths = write_yelp_fixture(tmp_path)
    cases, _ = sample_fixture(paths)
    for index, case in enumerate(cases, 1):
        set_case_persona(
            paths["cases"],
            case_id=case.case_id,
            persona=f"Restaurant regular {index} who likes dependable service and casual meals.",
        )
    filled_cases = load_recommendation_cases(paths["cases"])
    train_path = tmp_path / "train_cases.jsonl"
    heldout_path = tmp_path / "heldout_cases.jsonl"
    write_recommendation_cases(train_path, [filled_cases[0]])
    write_recommendation_cases(heldout_path, [filled_cases[1]])

    summary = build_recommendation_artifacts(
        source_path=train_path,
        manifest_path=paths["manifest"],
        catalogue_path=paths["catalogue"],
        interactions_path=paths["interactions"],
        persona_cases_path=paths["persona_cases"],
        eval_path=paths["eval"],
        index_dir=paths["index_dir"],
        embedding_model=HashingEmbeddingModel(),
        heldout_source_path=heldout_path,
        heldout_manifest_path=paths["manifest"],
    )
    persona_case_ids = {
        json.loads(line)["case_id"]
        for line in paths["persona_cases"].read_text(encoding="utf-8").splitlines()
    }
    eval_case_ids = {
        json.loads(line)["case_id"]
        for line in paths["eval"].read_text(encoding="utf-8").splitlines()
    }
    metadata = json.loads(paths["index_dir"].joinpath("metadata.json").read_text())

    assert summary["persona_case_count"] == 1
    assert summary["eval_case_count"] == 1
    assert persona_case_ids == {filled_cases[0].case_id}
    assert eval_case_ids == {filled_cases[1].case_id}
    assert metadata["case_count"] == 1
    assert metadata["persona_cases_path"] == str(paths["persona_cases"])


def test_manual_nigerian_recommendation_sources_match_runtime_schema() -> None:
    catalogue_path = Path("data/recommendation/nigerian_catalogue_manual.jsonl")
    cases_path = Path("data/generated/recommendation_cases_manual_ng.jsonl")

    products = load_product_catalogue(catalogue_path)
    cases = load_recommendation_cases(cases_path)
    product_ids = {product.product_id for product in products}

    assert len(products) >= 10
    assert len({product.product_id for product in products}) == len(products)
    assert all(product.metadata.get("source") for product in products)
    assert cases
    assert len({case.case_id for case in cases}) == len(cases)
    assert all(case.source == "manual_ng" for case in cases)
    for case in cases:
        assert case.user_persona.strip()
        assert case.relevant_product_ids
        assert set(case.relevant_product_ids).issubset(product_ids)
        assert set(case.history_product_ids).issubset(product_ids)


def test_validator_accepts_manual_nigerian_cases_with_manual_catalogue() -> None:
    products = load_product_catalogue(Path("data/recommendation/nigerian_catalogue_manual.jsonl"))

    result = validate_recommendation_cases(
        source_path=Path("data/generated/recommendation_cases_manual_ng.jsonl"),
        manifest_path=Path("data/generated/recommendation_sample_manifest.json"),
        extra_product_ids={product.product_id for product in products},
        extra_product_names={product.product_id: product.name for product in products},
        require_manifest_case_ids=False,
    )

    assert result.ok
    assert result.case_count >= 20


def test_checked_in_recommendation_eval_is_held_out_from_persona_index() -> None:
    eval_cases = [
        json.loads(line)
        for line in Path("data/eval/recommendation_eval_cases.jsonl").read_text().splitlines()
        if line.strip()
    ]
    persona_cases = [
        json.loads(line)
        for line in Path("data/recommendation/persona_cases.jsonl").read_text().splitlines()
        if line.strip()
    ]
    products = load_product_catalogue(Path("data/recommendation/product_catalogue.jsonl"))
    product_ids = {product.product_id for product in products}
    metadata = json.loads(Path("data/index/recommendation/metadata.json").read_text())
    holdout_validation = validate_recommendation_cases(
        source_path=Path("data/generated/recommendation_cases_holdout_yelp.jsonl"),
        manifest_path=Path("data/generated/recommendation_holdout_sample_manifest.json"),
        enforce_context_match=True,
    )
    expected_persona_case_count = len(
        load_recommendation_cases(Path("data/generated/recommendation_cases_manual.jsonl"))
    ) + len(
        load_recommendation_cases(Path("data/generated/recommendation_cases_manual_ng.jsonl"))
    )

    assert holdout_validation.ok, holdout_validation.failures[:5]
    assert len(eval_cases) == 100
    assert len(persona_cases) == expected_persona_case_count
    assert {case["case_id"] for case in eval_cases}.isdisjoint(
        {case["case_id"] for case in persona_cases}
    )
    assert all(
        product_id in product_ids
        for case in eval_cases
        for product_id in case["relevant_product_ids"]
    )
    assert metadata["case_count"] == len(persona_cases)
    assert metadata["persona_cases_path"] == "data/recommendation/persona_cases.jsonl"


def test_validator_rejects_relevant_product_name_in_persona(tmp_path: Path) -> None:
    paths = write_yelp_fixture(tmp_path)
    cases, _ = sample_fixture(paths)
    target = cases[0]
    manifest = json.loads(paths["manifest"].read_text())
    leaked_product_name = next(
        product["name"]
        for product in manifest["products"]
        if product["product_id"] == target.relevant_product_ids[0]
    )

    set_case_persona(
        paths["cases"],
        case_id=target.case_id,
        persona=f"Local diner who wants {leaked_product_name} for lunch.",
    )
    for case in cases[1:]:
        set_case_persona(
            paths["cases"],
            case_id=case.case_id,
            persona="Local diner who values reliable food and convenient service.",
        )

    result = validate_recommendation_cases(
        source_path=paths["cases"],
        manifest_path=paths["manifest"],
    )

    assert not result.ok
    assert any("relevant product name leaked" in failure for failure in result.failures)


def test_validator_rejects_relevant_product_outside_context(tmp_path: Path) -> None:
    paths = write_yelp_fixture(tmp_path)
    cases, _ = sample_fixture(paths)
    bad_case = cases[0].model_copy(
        update={"relevant_product_ids": cases[1].relevant_product_ids}
    )
    write_recommendation_cases(paths["cases"], [bad_case, cases[1]])

    result = validate_recommendation_cases(
        source_path=paths["cases"],
        manifest_path=paths["manifest"],
        require_personas=False,
        enforce_context_match=True,
    )

    assert not result.ok
    assert any("relevant product does not match context" in failure for failure in result.failures)


def test_artifact_builder_merges_manual_nigerian_sources(tmp_path: Path) -> None:
    paths = write_yelp_fixture(tmp_path)
    cases, _ = sample_fixture(paths)
    for index, case in enumerate(cases, 1):
        set_case_persona(
            paths["cases"],
            case_id=case.case_id,
            persona=f"Restaurant regular {index} who likes dependable service and casual meals.",
        )

    summary = build_recommendation_artifacts(
        source_path=paths["cases"],
        manifest_path=paths["manifest"],
        catalogue_path=paths["catalogue"],
        interactions_path=paths["interactions"],
        persona_cases_path=paths["persona_cases"],
        eval_path=paths["eval"],
        index_dir=paths["index_dir"],
        embedding_model=HashingEmbeddingModel(),
        manual_catalogue_path=Path("data/recommendation/nigerian_catalogue_manual.jsonl"),
        manual_cases_path=Path("data/generated/recommendation_cases_manual_ng.jsonl"),
    )
    products = [json.loads(line) for line in paths["catalogue"].read_text().splitlines()]
    eval_cases = [json.loads(line) for line in paths["eval"].read_text().splitlines()]

    assert summary["manual_product_count"] >= 10
    assert summary["manual_case_count"] >= 5
    assert any(product["product_id"] == "ng_food_lagos_yaba_bukka_hut_001" for product in products)
    assert any(case["case_id"] == "ng_rec_food_yaba_mixed_nigerian_001" for case in eval_cases)


def test_recommendation_coverage_slices_define_100_item_target() -> None:
    path = Path("data/curation/recommendation_coverage_slices.jsonl")
    slices = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    domains = {item["domain"] for item in slices}
    coverage_groups = {item["coverage_group"] for item in slices}
    cities = {item["city"] for item in slices}
    persona_archetypes = {
        archetype
        for item in slices
        for archetype in item.get("persona_archetypes", [])
    }
    food_slice_count = sum(item["domain"] == "food" for item in slices)
    national_slice_count = sum(item["city"].lower() == "national" for item in slices)
    non_lagos_slice_count = sum("lagos" not in item["region"].lower() for item in slices)
    serialized = "\n".join(json.dumps(item).lower() for item in slices)

    assert len(slices) == 20
    assert len({item["slice_id"] for item in slices}) == len(slices)
    assert sum(item["target_business_count"] for item in slices) == 100
    assert all(item["target_business_count"] == 5 for item in slices)
    assert all(item["search_queries"] for item in slices)
    assert all(item["domain"] for item in slices)
    assert all(item["region"] for item in slices)
    assert all(item["city"] for item in slices)
    assert all(item.get("persona_archetypes") for item in slices)
    assert "yaba" not in serialized
    assert food_slice_count <= 4
    assert len(domains) >= 8
    assert len(coverage_groups) >= 7
    assert len(cities) >= 6
    assert len(persona_archetypes) >= 18
    assert national_slice_count >= 5
    assert non_lagos_slice_count >= 10


def test_search_curated_manual_catalogue_rows_have_source_evidence() -> None:
    products = load_product_catalogue(Path("data/recommendation/nigerian_catalogue_manual.jsonl"))

    for product in products:
        source = str(product.metadata.get("source", ""))
        if source not in {
            "manual_exa_search_curation",
            "manual_google_search_curation",
            "manual_google_maps_curation",
        }:
            continue
        assert product.metadata.get("source_url"), product.product_id
        assert product.metadata.get("source_note"), product.product_id
        assert product.metadata.get("evidence_quality"), product.product_id


def test_lagos_entertainment_query_surfaces_verified_manual_candidates() -> None:
    retriever = ProductRetriever()

    result = retriever.retrieve(
        user_persona=(
            "Lagos family looking for safe indoor activities, games, movement, "
            "and enough variety for children and adults on a weekend."
        ),
        context="weekend family activity in Lagos",
        k=5,
    )
    returned_ids = [candidate.product.product_id for candidate in result.candidates]

    assert returned_ids[:2] == [
        "ng_entertainment_lagos_upbeat_centre_001",
        "ng_entertainment_lagos_rufus_and_bee_001",
    ]


def test_abuja_entertainment_query_surfaces_verified_manual_candidates() -> None:
    retriever = ProductRetriever()

    result = retriever.retrieve(
        user_persona=(
            "Abuja couple or young professionals looking for a relaxed evening "
            "with movies, bowling, dining, and easy planning."
        ),
        context="weekend date night or group hangout in Abuja",
        category="entertainment",
        k=6,
    )
    returned_ids = [candidate.product.product_id for candidate in result.candidates]

    assert returned_ids[:3] == [
        "ng_entertainment_abuja_jabi_lake_mall_001",
        "ng_entertainment_abuja_the_dome_001",
        "ng_cinema_abuja_silverbird_jabi_lake_001",
    ]


def test_national_streaming_queries_surface_verified_manual_candidates() -> None:
    retriever = ProductRetriever()

    video_result = retriever.retrieve(
        user_persona=(
            "Nigerian movie lover who wants Nollywood, international series, "
            "family viewing, and easy subscription streaming at home or on mobile."
        ),
        context="movie and series streaming in Nigeria",
        category="entertainment",
        k=5,
    )
    music_result = retriever.retrieve(
        user_persona=(
            "Nigerian music listener who wants Afrobeats, playlists, offline downloads, "
            "and mobile-friendly listening without managing local files."
        ),
        context="music streaming and offline listening in Nigeria",
        category="entertainment",
        k=5,
    )

    assert [candidate.product.product_id for candidate in video_result.candidates[:3]] == [
        "ng_entertainment_streaming_showmax_001",
        "ng_entertainment_streaming_netflix_001",
        "ng_entertainment_streaming_prime_video_001",
    ]
    assert [candidate.product.product_id for candidate in music_result.candidates[:2]] == [
        "ng_entertainment_music_audiomack_001",
        "ng_entertainment_music_boomplay_001",
    ]


def test_kano_retail_queries_surface_verified_manual_candidates() -> None:
    retriever = ProductRetriever()

    textile_result = retriever.retrieve(
        user_persona=(
            "Kano trader who sources fabrics, leather goods, and traditional craft items "
            "for resale, tailoring, or customer orders."
        ),
        context="textile and leather sourcing in Kano",
        category="retail",
        k=5,
    )
    wholesale_result = retriever.retrieve(
        user_persona=(
            "Kano small business owner or family buyer who needs wholesale foodstuffs, "
            "provisions, household goods, and reliable market variety."
        ),
        context="bulk market shopping in Kano",
        category="retail",
        k=5,
    )

    assert [candidate.product.product_id for candidate in textile_result.candidates[:3]] == [
        "ng_retail_kano_kurmi_market_001",
        "ng_retail_kano_kantin_kwari_market_001",
        "ng_retail_kano_sabon_gari_market_001",
    ]
    assert [candidate.product.product_id for candidate in wholesale_result.candidates[:3]] == [
        "ng_retail_kano_singer_market_001",
        "ng_retail_kano_yankaba_market_001",
        "ng_retail_kano_sabon_gari_market_001",
    ]


def test_lagos_education_queries_surface_verified_manual_candidates() -> None:
    retriever = ProductRetriever()

    public_result = retriever.retrieve(
        user_persona=(
            "Prospective Lagos undergraduate comparing public university options, "
            "recognized programmes, admission pathways, and campus access for "
            "family decision-making."
        ),
        context="public university admission options in Lagos",
        category="education",
        k=5,
    )
    practical_result = retriever.retrieve(
        user_persona=(
            "Lagos applicant seeking practical technology-focused study, diploma or "
            "degree pathways, flexible learning, and a structured campus environment."
        ),
        context="technology-focused higher education in Lagos",
        category="education",
        k=5,
    )

    expected_ids = {
        "ng_education_lagos_lasu_001",
        "ng_education_lagos_lasustech_001",
        "ng_education_lagos_pan_atlantic_university_001",
        "ng_education_lagos_unilag_001",
        "ng_education_lagos_yabatech_001",
    }

    assert {
        candidate.product.product_id for candidate in public_result.candidates[:5]
    } == expected_ids
    assert {
        candidate.product.product_id for candidate in practical_result.candidates[:5]
    } == expected_ids
    assert practical_result.candidates[0].product.product_id == "ng_education_lagos_yabatech_001"


def test_national_career_learning_queries_surface_verified_manual_candidates() -> None:
    retriever = ProductRetriever()

    software_result = retriever.retrieve(
        user_persona=(
            "Nigerian early-career professional switching into technology who needs "
            "structured software learning, mentorship, practical projects, and career support."
        ),
        context="software engineering and tech career transition in Nigeria",
        category="education",
        k=5,
    )
    flexible_result = retriever.retrieve(
        user_persona=(
            "Nigerian working learner seeking flexible online skills training in data, "
            "product, cybersecurity, or business without leaving full-time work."
        ),
        context="online professional upskilling in Nigeria",
        category="education",
        k=5,
    )
    certification_result = retriever.retrieve(
        user_persona=(
            "Nigerian professional seeking recognized ICT certification, digital literacy, "
            "project management, or web development training with online or centre-based options."
        ),
        context="professional certification and ICT training in Nigeria",
        category="education",
        k=5,
    )

    expected_ids = {
        "ng_education_career_altschool_africa_001",
        "ng_education_career_decagon_institute_001",
        "ng_education_career_hiit_plc_001",
        "ng_education_career_semicolon_africa_001",
        "ng_education_career_utiva_001",
    }

    assert {
        candidate.product.product_id for candidate in software_result.candidates[:5]
    } == expected_ids
    assert {
        candidate.product.product_id for candidate in flexible_result.candidates[:5]
    } == expected_ids
    assert {
        candidate.product.product_id for candidate in certification_result.candidates[:5]
    } == expected_ids
    assert software_result.candidates[0].product.product_id in {
        "ng_education_career_semicolon_africa_001",
        "ng_education_career_altschool_africa_001",
        "ng_education_career_decagon_institute_001",
    }
    assert certification_result.candidates[0].product.product_id == (
        "ng_education_career_hiit_plc_001"
    )


def test_lagos_health_queries_surface_verified_manual_candidates() -> None:
    retriever = ProductRetriever()

    pharmacy_result = retriever.retrieve(
        user_persona=(
            "Lagos parent or caregiver who needs accessible pharmacy errands, family "
            "health products, wellness items, and options for ordering or pickup."
        ),
        context="family pharmacy and wellness product errands in Lagos",
        category="health",
        k=5,
    )
    hospital_result = retriever.retrieve(
        user_persona=(
            "Busy Lagos professional comparing outpatient care, diagnostics, specialist "
            "services, emergency-ready branches, and clear hospital locations."
        ),
        context="hospital and clinic service discovery in Lagos",
        category="health",
        k=5,
    )
    wellness_result = retriever.retrieve(
        user_persona=(
            "Lagos professional trying to keep a consistent wellness routine with "
            "convenient gym access and practical health-product errands across the city."
        ),
        context="fitness and wellness service discovery in Lagos",
        category="health",
        k=5,
    )

    assert [candidate.product.product_id for candidate in pharmacy_result.candidates[:2]] == [
        "ng_health_lagos_medplus_pharmacy_001",
        "ng_health_lagos_healthplus_pharmacy_001",
    ]
    assert [candidate.product.product_id for candidate in hospital_result.candidates[:2]] == [
        "ng_health_lagos_reddington_hospital_001",
        "ng_health_lagos_lagoon_hospitals_001",
    ]
    assert wellness_result.candidates[0].product.product_id == "ng_health_lagos_ifitness_001"
    assert {
        candidate.product.product_id for candidate in wellness_result.candidates[:3]
    } == {
        "ng_health_lagos_healthplus_pharmacy_001",
        "ng_health_lagos_ifitness_001",
        "ng_health_lagos_medplus_pharmacy_001",
    }


def test_abuja_service_queries_surface_verified_manual_candidates() -> None:
    retriever = ProductRetriever()

    workspace_result = retriever.retrieve(
        user_persona=(
            "Abuja remote worker or small-team founder who needs reliable workspace, "
            "meeting rooms, internet, printing, and a professional environment."
        ),
        context="coworking and meeting room service discovery in Abuja",
        category="service",
        k=8,
    )
    errand_result = retriever.retrieve(
        user_persona=(
            "Busy Abuja professional who wants practical help with laundry, grooming, "
            "personal errands, and occasional workspace needs around the city."
        ),
        context="convenience errands and grooming services in Abuja",
        category="service",
        k=8,
    )
    device_result = retriever.retrieve(
        user_persona=(
            "Abuja professional dependent on a phone for work who needs device repair "
            "support and backup places to work while handling errands."
        ),
        context="phone repair and practical workday services in Abuja",
        category="service",
        k=8,
    )

    expected_ids = {
        "ng_service_abuja_carlcare_001",
        "ng_service_abuja_concierge_solutions_place_001",
        "ng_service_abuja_tushup_laundry_001",
        "ng_service_abuja_ventures_park_001",
        "ng_service_abuja_work_and_connect_001",
    }

    assert {
        candidate.product.product_id for candidate in workspace_result.candidates
    } == expected_ids
    assert {candidate.product.product_id for candidate in errand_result.candidates} == expected_ids
    assert {candidate.product.product_id for candidate in device_result.candidates} == expected_ids
    assert [candidate.product.product_id for candidate in workspace_result.candidates[:3]] == [
        "ng_service_abuja_ventures_park_001",
        "ng_service_abuja_work_and_connect_001",
        "ng_service_abuja_concierge_solutions_place_001",
    ]
    assert [candidate.product.product_id for candidate in errand_result.candidates[:2]] == [
        "ng_service_abuja_concierge_solutions_place_001",
        "ng_service_abuja_tushup_laundry_001",
    ]
    assert device_result.candidates[0].product.product_id == "ng_service_abuja_carlcare_001"


def test_abuja_food_queries_surface_verified_manual_candidates() -> None:
    retriever = ProductRetriever()

    professional_result = retriever.retrieve(
        user_persona=(
            "Abuja professional or civil servant who needs a reliable central lunch option "
            "with Nigerian meals, easy access, and enough structure for workday planning."
        ),
        context="professional lunch in central Abuja or Wuse",
        category="food",
        k=10,
    )
    family_result = retriever.retrieve(
        user_persona=(
            "Abuja family looking for a comfortable restaurant or cafe with varied meals, "
            "casual service, and enough flexibility for adults and children."
        ),
        context="family lunch or early dinner in Abuja",
        category="food",
        k=10,
    )
    date_result = retriever.retrieve(
        user_persona=(
            "Abuja couple or friend group planning a more intentional meal with central "
            "access, pleasant ambience, and either international or Nigerian-continental food."
        ),
        context="date night or group dinner in Abuja",
        category="food",
        k=10,
    )

    expected_ids = {
        "ng_food_abuja_blucabana_001",
        "ng_food_abuja_charcoal_grill_001",
        "ng_food_abuja_cilantro_maitama_001",
        "ng_food_abuja_jevinik_001",
        "ng_food_abuja_nkoyo_001",
    }

    assert {
        candidate.product.product_id for candidate in professional_result.candidates[:5]
    } == expected_ids
    assert {
        candidate.product.product_id for candidate in family_result.candidates[:5]
    } == expected_ids
    assert {
        candidate.product.product_id for candidate in date_result.candidates[:5]
    } == expected_ids
    assert professional_result.candidates[0].product.product_id in {
        "ng_food_abuja_charcoal_grill_001",
        "ng_food_abuja_jevinik_001",
        "ng_food_abuja_nkoyo_001",
    }
    assert family_result.candidates[0].product.product_id == "ng_food_abuja_blucabana_001"
    assert date_result.candidates[0].product.product_id == (
        "ng_food_abuja_cilantro_maitama_001"
    )


def test_ibadan_food_queries_surface_verified_manual_candidates() -> None:
    retriever = ProductRetriever()

    budget_result = retriever.retrieve(
        user_persona=(
            "Ibadan budget diner who wants affordable amala, ewedu, gbegiri, "
            "and filling local canteen food for a quick lunch."
        ),
        context="budget amala and local bukka lunch in Ibadan",
        category="food",
        k=10,
    )
    family_result = retriever.retrieve(
        user_persona=(
            "Ibadan family looking for casual Nigerian meals, swallow options, "
            "and a comfortable restaurant or cafe for lunch or early dinner."
        ),
        context="family lunch or early dinner in Ibadan",
        category="food",
        k=10,
    )
    student_result = retriever.retrieve(
        user_persona=(
            "Ibadan student or young worker seeking cheap filling local food near "
            "campus, market, or central neighbourhood routes."
        ),
        context="quick affordable local lunch in Ibadan",
        category="food",
        k=10,
    )

    expected_ids = {
        "ng_food_ibadan_cafe_chrysalis_001",
        "ng_food_ibadan_inastrait_canteen_001",
        "ng_food_ibadan_iya_meta_canteen_001",
        "ng_food_ibadan_kokodome_001",
        "ng_food_ibadan_ose_olohun_amala_001",
    }

    assert {
        candidate.product.product_id for candidate in budget_result.candidates[:5]
    } == expected_ids
    assert {
        candidate.product.product_id for candidate in family_result.candidates[:5]
    } == expected_ids
    assert {
        candidate.product.product_id for candidate in student_result.candidates[:5]
    } == expected_ids
    assert budget_result.candidates[0].product.product_id in {
        "ng_food_ibadan_iya_meta_canteen_001",
        "ng_food_ibadan_inastrait_canteen_001",
    }
    assert family_result.candidates[0].product.product_id == "ng_food_ibadan_kokodome_001"
    assert student_result.candidates[0].product.product_id == (
        "ng_food_ibadan_inastrait_canteen_001"
    )


def test_port_harcourt_food_queries_surface_verified_manual_candidates() -> None:
    retriever = ProductRetriever()

    bole_result = retriever.retrieve(
        user_persona=(
            "Port Harcourt local food lover who wants bole, fish, plantain, "
            "and casual Rivers-style lunch for friends or coworkers."
        ),
        context="casual local lunch or after-work food in Port Harcourt",
        category="food",
        k=10,
    )
    family_result = retriever.retrieve(
        user_persona=(
            "Port Harcourt family or group seeking comfortable Nigerian meals, "
            "swallows, rice, and reliable sit-down dining for lunch or dinner."
        ),
        context="family lunch or group dinner in Port Harcourt",
        category="food",
        k=10,
    )
    professional_result = retriever.retrieve(
        user_persona=(
            "Port Harcourt professional or oil-and-gas worker planning a casual "
            "team meal with quick service, varied menus, and central GRA access."
        ),
        context="team lunch or relaxed group meal around Port Harcourt GRA",
        category="food",
        k=10,
    )

    expected_ids = {
        "ng_food_port_harcourt_blue_elephant_001",
        "ng_food_port_harcourt_bole_king_001",
        "ng_food_port_harcourt_jevinik_001",
        "ng_food_port_harcourt_kilimanjaro_001",
        "ng_food_port_harcourt_native_tray_001",
    }

    assert {
        candidate.product.product_id for candidate in bole_result.candidates[:5]
    } == expected_ids
    assert {
        candidate.product.product_id for candidate in family_result.candidates[:5]
    } == expected_ids
    assert {
        candidate.product.product_id for candidate in professional_result.candidates[:5]
    } == expected_ids
    assert bole_result.candidates[0].product.product_id == (
        "ng_food_port_harcourt_bole_king_001"
    )
    assert family_result.candidates[0].product.product_id == (
        "ng_food_port_harcourt_kilimanjaro_001"
    )
    assert professional_result.candidates[0].product.product_id == (
        "ng_food_port_harcourt_kilimanjaro_001"
    )


def test_lagos_office_family_food_queries_surface_verified_manual_candidates() -> None:
    retriever = ProductRetriever()

    office_result = retriever.retrieve(
        user_persona=(
            "Lagos office worker who needs reliable sit-down or delivery-friendly "
            "lunch around business districts, with Nigerian and continental options."
        ),
        context="office lunch in Victoria Island or Ikeja Lagos",
        category="food",
        k=10,
    )
    family_result = retriever.retrieve(
        user_persona=(
            "Lagos family looking for comfortable weekend dining with varied meals, "
            "seafood or Nigerian dishes, and enough space for adults and children."
        ),
        context="family lunch or early dinner across Lagos",
        category="food",
        k=10,
    )
    group_result = retriever.retrieve(
        user_persona=(
            "Lagos professionals or friends planning a group meal with seafood, "
            "contemporary Nigerian dishes, casual ambience, and easy Island or "
            "mainland access."
        ),
        context="planned group lunch or dinner in Lagos",
        category="food",
        k=10,
    )

    expected_ids = {
        "ng_food_lagos_cactus_001",
        "ng_food_lagos_farmcity_001",
        "ng_food_lagos_ocean_basket_001",
        "ng_food_lagos_the_place_001",
        "ng_food_lagos_yellow_chilli_001",
    }

    for result in (office_result, family_result, group_result):
        assert {candidate.product.product_id for candidate in result.candidates[:7]} >= (
            expected_ids
        )
        assert all(
            "Lagos" in (candidate.product.location or "")
            for candidate in result.candidates[:10]
        )

    assert office_result.candidates[0].product.product_id == "ng_food_lagos_the_place_001"


def sample_fixture(paths: dict[str, Path]):
    return sample_yelp_recommendation_cases(
        business_path=paths["business"],
        review_path=paths["reviews"],
        output_path=paths["cases"],
        manifest_path=paths["manifest"],
        sample_size=2,
        seed=20260522,
        oversample_factor=2,
    )


def write_yelp_fixture(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "business": tmp_path / "business.json",
        "reviews": tmp_path / "reviews.json",
        "cases": tmp_path / "recommendation_cases_manual.jsonl",
        "manifest": tmp_path / "recommendation_sample_manifest.json",
        "catalogue": tmp_path / "product_catalogue.jsonl",
        "interactions": tmp_path / "interactions.jsonl",
        "persona_cases": tmp_path / "persona_cases.jsonl",
        "eval": tmp_path / "recommendation_eval_cases.jsonl",
        "index_dir": tmp_path / "index",
    }
    businesses = [
        business("b1", "History Diner One", "Restaurants, Food", "Philadelphia", "PA", 4.1),
        business("b2", "History Cafe One", "Restaurants, Coffee", "Philadelphia", "PA", 3.8),
        business("b3", "History Pizza One", "Restaurants, Food", "Philadelphia", "PA", 4.4),
        business("b4", "Heldout Tacos One", "Restaurants, Food", "Philadelphia", "PA", 4.7),
        business("b9", "Wrong City Cleaner", "Local Services", "Tampa", "FL", 4.9),
        business("b5", "History Diner Two", "Restaurants, Food", "Tampa", "FL", 4.0),
        business("b6", "History Spa Two", "Beauty, Health", "Tampa", "FL", 3.5),
        business("b7", "History Coffee Two", "Food, Coffee", "Tampa", "FL", 4.2),
        business("b8", "Heldout Brunch Two", "Restaurants, Food", "Tampa", "FL", 4.8),
        business("b10", "Wrong Domain Spa", "Beauty, Health", "Tampa", "FL", 4.9),
    ]
    reviews = [
        review("r1", "u1", "b1", 5, "2020-01-01", "Great breakfast and quick service."),
        review("r2", "u1", "b2", 3, "2020-02-01", "Coffee was fine but service dragged."),
        review("r3", "u1", "b3", 4, "2020-03-01", "Reliable casual pizza dinner."),
        review("r4", "u1", "b4", 5, "2020-04-01", "Heldout taco review should not show."),
        review(
            "r9",
            "u1",
            "b9",
            5,
            "2020-05-01",
            "Wrong-city cleaner should not label a restaurant query.",
        ),
        review("r5", "u2", "b5", 4, "2021-01-01", "Good diner for a practical lunch."),
        review("r6", "u2", "b6", 2, "2021-02-01", "Spa was not worth the wait."),
        review("r7", "u2", "b7", 4, "2021-03-01", "Coffee stop was convenient."),
        review("r8", "u2", "b8", 5, "2021-04-01", "Heldout brunch review should not show."),
        review(
            "r10",
            "u2",
            "b10",
            5,
            "2021-05-01",
            "Wrong-domain spa should not label a restaurant query.",
        ),
    ]
    write_jsonl(paths["business"], businesses)
    write_jsonl(paths["reviews"], reviews)
    return paths


def recommendation_product(
    *,
    product_id: str,
    name: str,
    category: str,
) -> dict:
    return {
        "product_id": product_id,
        "name": name,
        "category": category,
        "description": f"{name} recommendation item.",
        "price": None,
        "currency": "NGN",
        "location": "Lagos, Nigeria",
        "metadata": {"domain": category, "source": "test"},
    }


def recommendation_case(
    *,
    case_id: str,
    persona: str,
    product_ids: list[str],
) -> dict:
    return {
        "case_id": case_id,
        "source": "manual_ng",
        "user_persona": persona,
        "context": "test context",
        "persona_context": {"history": []},
        "history_product_ids": [],
        "relevant_product_ids": product_ids,
        "sampling": {
            "seed": 20260524,
            "source_user_id": f"test_{case_id}",
            "history_review_ids": [],
            "heldout_review_ids": [],
        },
    }


def business(
    business_id: str,
    name: str,
    categories: str,
    city: str,
    state: str,
    stars: float,
) -> dict:
    return {
        "business_id": business_id,
        "name": name,
        "address": "1 Main St",
        "city": city,
        "state": state,
        "postal_code": "00000",
        "latitude": 0.0,
        "longitude": 0.0,
        "stars": stars,
        "review_count": 10,
        "is_open": 1,
        "attributes": {},
        "categories": categories,
        "hours": {},
    }


def review(
    review_id: str,
    user_id: str,
    business_id: str,
    stars: float,
    date: str,
    text: str,
) -> dict:
    return {
        "review_id": review_id,
        "user_id": user_id,
        "business_id": business_id,
        "stars": stars,
        "useful": 0,
        "funny": 0,
        "cool": 0,
        "text": text,
        "date": f"{date} 12:00:00",
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
