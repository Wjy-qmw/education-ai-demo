import copy, json
from uuid import uuid4
import pytest
from src.questionnaire import normalize_answers, top_values
from src.knowledge import update_probability, apply_evidence
from src.memory import add_preference, resolve_preference, revoke_preference
from src.math_rules import expression, fraction_problem, grade
from src.decision import decide, explicit_style
from src.storage import SQLiteStore, empty_state, validate_state, StateConflict
from src.service import run_turn
from src.learning_engine import prepare, validate_model_response


@pytest.mark.parametrize(
    "bad",
    [
        [],
        None,
        {"security": None},
        {"security": float("nan")},
        {"security": True},
        {"security": "5"},
    ],
)
def test_invalid_profile(bad):
    with pytest.raises(ValueError):
        normalize_answers(bad)


def test_flat_profile():
    assert top_values(normalize_answers({})) == []


def test_bkt_values():
    assert update_probability(0.2, True) == pytest.approx(0.5764705882352942)
    assert update_probability(0.2, False) == pytest.approx(0.1272727272727273)


@pytest.mark.parametrize(
    "source,independent", [("student", False), ("model", True), ("model", False)]
)
def test_non_independent(source, independent):
    k = {}
    assert not apply_evidence(
        k, "1", {"fraction_simplify": True}, independent=independent, source=source
    )["updated"]
    assert not k.get("skills")


def test_dedup():
    k = {}
    apply_evidence(k, "same", {"fraction_simplify": True}, independent=True)
    saved = copy.deepcopy(k)
    assert (
        apply_evidence(k, "same", {"fraction_simplify": True}, independent=True)[
            "reason"
        ]
        == "duplicate"
    )
    assert k == saved


def test_current_no_overwrite():
    records = []
    add_preference(records, "hint", confirmed=True)
    saved = copy.deepcopy(records)
    assert decide(records, "完整讲解").action == "full"
    assert records == saved
    assert decide(records, "").action == "hint"


def test_conflict_revoke():
    records = []
    a = add_preference(records, "hint", confirmed=True)
    add_preference(records, "full", confirmed=True)
    assert decide(records, "").action == "clarify"
    assert len(records) == 2
    assert revoke_preference(records, a["id"])
    assert a["version"] == 2
    assert decide(records, "").action == "full"


def test_scope_source():
    records = []
    add_preference(records, "full", scope="session")
    assert resolve_preference(records)["value"] is None
    add_preference(records, "hint", confirmed=True, subject="全部")
    assert resolve_preference(records, "英语")["value"] == "hint"
    with pytest.raises(ValueError):
        add_preference(records, "full", source="model", confirmed=True)


@pytest.mark.parametrize(
    "goal", ["只给提示", "少给直接答案", "不要完整答案", "不要最终答案"]
)
def test_negative_request(goal):
    assert explicit_style(goal) == "hint"


@pytest.mark.parametrize(
    "bad",
    [
        "__import__('os').system('x')",
        "x.__class__",
        "x**100000",
        "sin(x)",
        "1/0",
        "9" * 300,
        "x" * 300,
    ],
)
def test_parser_bounds(bad):
    with pytest.raises(ValueError):
        expression(bad)


@pytest.mark.parametrize(
    "question,answer,conditions",
    [
        ("(x^2-1)/(x-1)", "x+1", "x!=1"),
        ("(x^2-4)/(x+2)", "x-2", "x!=-2"),
        ("(x^2-x)/(x^2-1)", "x/(x+1)", "x!=1,x!=-1"),
        ("(2*x^2-x)/(2*x-1)", "x", "x!=1/2"),
        ("(x^2+1)/(1)", "x^2+1", ""),
    ],
)
def test_exact_fraction(question, answer, conditions):
    assert grade(fraction_problem(question), answer, conditions)["correct"]


@pytest.mark.parametrize(
    "answer,condition",
    [
        ("x+1", ""),
        ("x+1", "x!=1,x!=2"),
        ("x+1", "x!=1 and x>2"),
        ("x-1", "x!=1"),
        ("(x^2-1)/(x-1)", "x!=2"),
    ],
)
def test_wrong_domain(answer, condition):
    assert not grade(fraction_problem("(x^2-1)/(x-1)"), answer, condition)["correct"]


@pytest.mark.parametrize(
    "bad", ["(x)/(0)", "(x)/(x^2-2)", "(x^7)/(x-1)", "[无法识别] (x)/(x-1)"]
)
def test_unsupported(bad):
    with pytest.raises(ValueError):
        fraction_problem(bad)


@pytest.mark.parametrize(
    "history",
    [
        [None],
        [{}],
        [{"role": "system", "content": "bad"}],
        [{"role": "user", "content": 3}],
    ],
)
def test_bad_history(history):
    with pytest.raises(ValueError):
        validate_state({"history": history})


def test_store_cas_isolation(tmp_path):
    store = SQLiteStore(tmp_path / "test.db")
    a = uuid4().hex
    b = uuid4().hex
    s = empty_state()
    s["question"] = "A"
    assert store.save(a, s, 0) == 1
    assert store.load(b)[0]["question"] == ""
    with pytest.raises(StateConflict):
        store.save(a, s, 0)
    assert store.load(a)[1] == 1


def test_single_history(tmp_path):
    store = SQLiteStore(tmp_path / "test.db")
    identifier = uuid4().hex
    s = empty_state()
    s["history"] = [{"role": "user", "content": "hello"}]
    store.save(identifier, s, 0)
    assert store.load(identifier)[0]["history"] == s["history"]


def test_real_local_continuity(tmp_path):
    store = SQLiteStore(tmp_path / "test.db")
    identifier = uuid4().hex
    s = empty_state()
    v = 0
    for i in range(5):
        s, v = run_turn(
            store,
            identifier,
            s,
            v,
            "(x^2-1)/(x-1)",
            "完整讲解",
            student_answer="x+1",
            restriction="x!=1",
            independent=True,
            evidence_id="same",
        )
    assert len(s["history"]) == 10
    assert len(s["knowledge"]["evidence"]) == 1
    assert s["knowledge"]["skills"]["fraction_simplify"] == pytest.approx(
        0.5764705882352942
    )
    assert "x≠1" in s["history"][-1]["content"]
    assert store.load(identifier) == (s, v)


def test_api_failure_no_commit(tmp_path):
    store = SQLiteStore(tmp_path / "test.db")
    identifier = uuid4().hex
    s = empty_state()
    v = store.save(identifier, s, 0)

    class FakeTutor:
        configured = True
        last_metrics = {}

        def answer(self, *a, **kw):
            return json.dumps(
                {
                    "explanation": "错误",
                    "final_expression": "x-1",
                    "restrictions": "x!=1",
                }
            )

    with pytest.raises(ValueError):
        run_turn(
            store, identifier, s, v, "(x^2-1)/(x-1)", "完整讲解", tutor=FakeTutor()
        )
    assert store.load(identifier) == (s, v)
    assert s == empty_state()


def test_hint_output_guard():
    p = prepare(empty_state(), "(x^2-1)/(x-1)", "只给提示")
    with pytest.raises(ValueError):
        validate_model_response(
            json.dumps(
                {
                    "explanation": "x+1",
                    "final_expression": "x+1",
                    "restrictions": "x!=1",
                }
            ),
            p,
        )


def test_ui_flow_reload(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("EDU_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    app = AppTest.from_file(
        __import__("pathlib").Path(__file__).resolve().parents[1] / "app.py",
        default_timeout=15,
    ).run()
    assert not app.exception
    app.text_area[0].set_value("(x^2-1)/(x-1)")
    next(t for t in app.text_input if t.label == "今天希望怎样帮你？").set_value(
        "完整讲解"
    )
    app.run()
    next(b for b in app.button if b.label == "获得帮助 / 继续追问").click().run()
    assert not app.exception
    assert len(app.session_state.saved["history"]) == 2
    identifier = app.session_state.learner
    reload = AppTest.from_file(
        __import__("pathlib").Path(__file__).resolve().parents[1] / "app.py",
        default_timeout=15,
    )
    reload.query_params["session"] = identifier
    reload.run()
    assert not reload.exception
    assert len(reload.session_state.saved["history"]) == 2
    assert reload.text_area[0].value == "(x^2-1)/(x-1)"


def test_api_json_configuration_history_and_usage(monkeypatch):
    from types import SimpleNamespace
    import src.deepseek_client as client

    monkeypatch.setenv("DEEPSEEK_API_KEY", "SYNTHETIC-NOT-A-KEY")
    capture = {}

    class FakeClient:
        def __init__(self, **kw):
            capture["config"] = kw
            self.chat = SimpleNamespace(completions=self)

        def create(self, **kw):
            capture["request"] = kw
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))],
                usage=None,
            )

    monkeypatch.setattr(client, "OpenAI", FakeClient)
    tutor = client.DeepSeekTutor()
    assert (
        tutor.answer(
            "JSON",
            "question",
            [None, {"role": "user", "content": "old"}],
            structured=True,
        )
        == "{}"
    )
    assert capture["request"]["response_format"] == {"type": "json_object"}
    assert len(capture["request"]["messages"]) == 3
    assert capture["config"]["timeout"] == 45
    assert tutor.last_metrics["elapsed_s"] >= 0


def test_network_error_no_commit(tmp_path):
    store = SQLiteStore(tmp_path / "test.db")
    ident = uuid4().hex
    state = empty_state()

    class FailedTutor:
        configured = True

        def answer(self, *a, **kw):
            raise TimeoutError("synthetic timeout")

    with pytest.raises(TimeoutError):
        run_turn(
            store, ident, state, 0, "(x^2-1)/(x-1)", "完整讲解", tutor=FailedTutor()
        )
    assert store.load(ident) == (empty_state(), 0)


def test_memory_ui_save_revoke(tmp_path, monkeypatch):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("EDU_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    app = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "app.py", default_timeout=15
    ).run()
    next(c for c in app.checkbox if "我确认将" in c.label).check().run()
    next(b for b in app.button if b.label == "保存长期偏好").click().run()
    assert not app.exception
    assert len(app.session_state.saved["memory"]) == 1
    next(b for b in app.button if b.label == "撤销这项记忆").click().run()
    assert not app.exception
    assert app.session_state.saved["memory"][0]["active"] is False
