from insurance_rag_assistant.evaluation.evaluator import (
    normalize_evaluation_text,
)


def test_normalization_handles_unicode_dashes() -> None:
    assert (
        normalize_evaluation_text("Five‑year cyber incident")
        == "five year cyber incident"
    )


def test_normalization_handles_currency_punctuation() -> None:
    assert normalize_evaluation_text("CAD 2,500") == "cad 2 500"


def test_normalization_handles_accents() -> None:
    assert (
        normalize_evaluation_text("Critères d'admissibilité")
        == "criteres d admissibilite"
    )
