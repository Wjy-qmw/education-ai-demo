from src.personalization import build_learning_strategies, build_system_prompt
from src.questionnaire import normalize_answers, top_values
from src.markdown_utils import normalize_markdown_math


def test_answers_are_clamped() -> None:
    vector = normalize_answers({"self_direction": 8, "security": -1})
    assert vector["self_direction"] == 5.0
    assert vector["security"] == 1.0
    assert len(vector) == 10


def test_high_autonomy_changes_strategy() -> None:
    vector = normalize_answers({"self_direction": 5})
    strategies = build_learning_strategies(vector)
    assert any("自主引导" in item for item in strategies)


def test_prompt_contains_structure_and_vector() -> None:
    prompt = build_system_prompt(normalize_answers({"achievement": 5}))
    assert "## 分步讲解" in prompt
    assert "成就: 5.0/5" in prompt


def test_top_values_are_descending() -> None:
    vector = normalize_answers({"achievement": 5, "security": 4})
    values = top_values(vector, 2)
    assert values[0][0] == "achievement"
    assert values[0][1] >= values[1][1]


def test_streamlit_math_delimiters_are_normalized() -> None:
    source = r"行内 \(x+1\)，块级 \[\frac{3}{5}\]"
    assert normalize_markdown_math(source) == r"行内 $x+1$，块级 $$\frac{3}{5}$$"


def test_math_delimiters_inside_code_are_preserved() -> None:
    source = r"正文 \(x\)，代码 `\(y\)`"
    assert normalize_markdown_math(source) == r"正文 $x$，代码 `\(y\)`"
