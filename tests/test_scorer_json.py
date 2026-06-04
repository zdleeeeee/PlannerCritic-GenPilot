import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "genpilot"))

from utils_all.scorer import parse_rating_response  # noqa: E402


def _expected_scores(result):
    return result["scores"]["Attribute-Binding"], result["scores"]["Object-Relationship"], result["scores"]["Background-Consistency"]


def test_parse_fenced_json():
    response = '''```json
{
  "scores": {
    "Attribute-Binding": 5,
    "Object-Relationship": 4,
    "Background-Consistency": 3
  },
  "reasons": {
    "Attribute-Binding": "Attributes match.",
    "Object-Relationship": "Relationships mostly match.",
    "Background-Consistency": "Background is acceptable."
  }
}
```'''
    result = parse_rating_response(response)
    assert _expected_scores(result) == (5, 4, 3)


def test_parse_raw_json():
    response = '''{
  "scores": {
    "Attribute-Binding": "4",
    "Object-Relationship": "3",
    "Background-Consistency": "2"
  },
  "reasons": {
    "Attribute-Binding": "Good.",
    "Object-Relationship": "Partial.",
    "Background-Consistency": "Weak."
  }
}'''
    result = parse_rating_response(response)
    assert _expected_scores(result) == (4, 3, 2)


def test_parse_json_embedded_in_text_and_trailing_commas():
    response = '''Here is the rating:
{
  "scores": {
    "Attribute-Binding": [3],
    "Object-Relationship": 2,
    "Background-Consistency": 1,
  },
  "reasons": {
    "Attribute-Binding": "Some attributes match.",
    "Object-Relationship": "Relationships are wrong.",
    "Background-Consistency": "Background is wrong.",
  }
}
Thanks.'''
    result = parse_rating_response(response)
    assert _expected_scores(result) == (3, 2, 1)


def test_invalid_response_returns_none():
    assert parse_rating_response("I cannot provide that rating.") is None
    assert parse_rating_response(None) is None


if __name__ == "__main__":
    test_parse_fenced_json()
    test_parse_raw_json()
    test_parse_json_embedded_in_text_and_trailing_commas()
    test_invalid_response_returns_none()
    print("scorer json tests passed")
