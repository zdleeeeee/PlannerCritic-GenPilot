import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_baseline import parse_score_line  # noqa: E402


def test_parse_np_float_score_line():
    assert parse_score_line("[np.float64(12.0), np.float64(11.0), np.float64(13.0)]") == [12.0, 11.0, 13.0]


def test_parse_plain_score_line():
    assert parse_score_line("[12.0, 11, 13]") == [12.0, 11.0, 13.0]


def test_ignore_history_lines_with_np_float64_dicts():
    line = "[\"text{'Attribute-Binding': np.float64(3.0), 'Object-Relationship': np.float64(4.0)}\"]"
    assert parse_score_line(line) == []


def test_ignore_non_score_text():
    assert parse_score_line("candidate_prompts in 0-1-0-retry0-[...") == []


if __name__ == "__main__":
    test_parse_np_float_score_line()
    test_parse_plain_score_line()
    test_ignore_history_lines_with_np_float64_dicts()
    test_ignore_non_score_text()
    print("compare metric tests passed")
