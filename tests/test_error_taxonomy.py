import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "genpilot"))

from utils_all.error_taxonomy import (  # noqa: E402
    ACTION_ERROR,
    ATTRIBUTE_ERROR,
    COUNTING_ERROR,
    OBJECT_ERROR,
    OTHER_ERROR,
    SPATIAL_RELATION_ERROR,
    STYLE_ERROR,
    TEXT_RENDERING_ERROR,
    aggregate_errors,
    build_strategy_block,
    classify_error,
    count_error_items,
    normalize_categories,
)


def test_classify_error_examples():
    examples = [
        ("The dragon is missing from the sky.", OBJECT_ERROR),
        ("The cabbage heads are yellow instead of green.", ATTRIBUTE_ERROR),
        ("The cat is behind the chair, not in front of it.", SPATIAL_RELATION_ERROR),
        ("The image shows six boats instead of exactly three boats.", COUNTING_ERROR),
        ("The person is standing instead of riding the horse.", ACTION_ERROR),
        ("The scene is a cartoon, not a cinematic photograph.", STYLE_ERROR),
        ("The sign text 'OPEN' is misspelled and not legible.", TEXT_RENDERING_ERROR),
        ("The generated result does not match the intended description.", OTHER_ERROR),
    ]
    for error_text, expected in examples:
        assert expected in classify_error(error_text)


def test_normalize_categories_aliases():
    assert normalize_categories("sytle error") == [STYLE_ERROR]
    assert normalize_categories("counting error, text rendering error") == [COUNTING_ERROR, TEXT_RENDERING_ERROR]
    assert normalize_categories({"categories": ["object", "attribute"]}) == [OBJECT_ERROR, ATTRIBUTE_ERROR]


def test_build_strategy_block():
    block = build_strategy_block([COUNTING_ERROR, TEXT_RENDERING_ERROR])
    assert "exact cardinality" in block
    assert "quote the exact visible text" in block
    assert "minimal" in block


def test_unknown_strategy_falls_back():
    block = build_strategy_block("unknown")
    assert "smallest unambiguous edit" in block


def test_aggregate_errors_keeps_short_lists():
    errors = ["The dragon is missing.", "The cabbage is yellow instead of green."]
    assert aggregate_errors(errors, max_errors_per_fragment=5) == errors


def test_aggregate_errors_groups_and_prioritizes():
    errors = [
        "The image uses cartoon style instead of cinematic photography.",
        "The dragon is missing from the sky.",
        "No visible dragon appears in the scene, even though the prompt requires a large flying dragon.",
        "The sign text 'OPEN' is misspelled.",
        "There are six boats instead of exactly three boats.",
        "The cat is behind the chair, not in front of it.",
        "The cabbage is yellow instead of green.",
    ]
    aggregated = aggregate_errors(errors, max_errors_per_fragment=3)
    assert len(aggregated) == 3
    assert "dragon" in aggregated[0]
    assert "three boats" in aggregated[1]
    assert "OPEN" in aggregated[2]


def test_aggregate_errors_handles_nested_inputs_and_counts():
    errors = {
        "1": [
            {"type": "object_error", "explanation": "The horse is missing."},
            {"category": "attribute_error", "error": "The rider's coat is red instead of blue."},
            "None",
        ],
        "2": "The text 'SALE' is not legible.\nThe text 'SALE' is not legible.",
    }
    assert count_error_items(errors) == 3
    aggregated = aggregate_errors(errors, max_errors_per_fragment=5)
    assert len(aggregated) == 3
    assert any("horse" in item for item in aggregated)
    assert any("SALE" in item for item in aggregated)


if __name__ == "__main__":
    test_classify_error_examples()
    test_normalize_categories_aliases()
    test_build_strategy_block()
    test_unknown_strategy_falls_back()
    test_aggregate_errors_keeps_short_lists()
    test_aggregate_errors_groups_and_prioritizes()
    test_aggregate_errors_handles_nested_inputs_and_counts()
    print("taxonomy tests passed")
