"""Prototype capability-link storage with transactional version checks.
The random session token is a bearer capability, not authenticated identity.
"""

import copy, json, math, os, re, sqlite3
from pathlib import Path
from src.questionnaire import normalize_answers
from src.memory import STYLES, SOURCES

STATE_PATH = Path(__file__).resolve().parent.parent / "data" / "state.json"


class StateConflict(ValueError):
    pass


def empty_state():
    return {
        "profile": {},
        "question": "",
        "history": [],
        "memory": [],
        "knowledge": {"skills": {}, "evidence": {}},
        "audit": [],
    }


def validate_state(raw):
    if not isinstance(raw, dict):
        raise ValueError("根节点须为对象。")
    s = empty_state()
    profile = raw.get("profile", {})
    if not isinstance(profile, dict):
        raise ValueError("画像须为对象。")
    s["profile"] = normalize_answers(profile) if profile else {}
    if not isinstance(raw.get("question", ""), str):
        raise ValueError("题目须为文字。")
    s["question"] = raw.get("question", "")[:12000]
    history = raw.get("history", [])
    if not isinstance(history, list) or len(history) > 1000:
        raise ValueError("历史格式或长度无效。")
    for h in history:
        if (
            not isinstance(h, dict)
            or h.get("role") not in {"user", "assistant"}
            or not isinstance(h.get("content"), str)
            or len(h["content"]) > 50000
        ):
            raise ValueError("历史条目须包含合法角色和文字。")
    s["history"] = copy.deepcopy(history)
    memory = raw.get("memory", [])
    if not isinstance(memory, list) or len(memory) > 200:
        raise ValueError("记忆格式或长度无效。")
    for m in memory:
        if (
            not isinstance(m, dict)
            or not isinstance(m.get("id"), str)
            or m.get("value") not in STYLES
            or m.get("source") not in SOURCES
            or m.get("scope") != "long_term"
            or m.get("subject") not in {"数学", "全部"}
            or type(m.get("active")) is not bool
            or type(m.get("version")) is not int
            or m["version"] < 1
        ):
            raise ValueError("记忆格式无效；临时偏好不能持久化。")
    s["memory"] = copy.deepcopy(memory)
    k = raw.get("knowledge", s["knowledge"])
    if (
        not isinstance(k, dict)
        or not isinstance(k.get("skills", {}), dict)
        or not isinstance(k.get("evidence", {}), dict)
    ):
        raise ValueError("知识状态格式无效。")
    for name, v in k.get("skills", {}).items():
        if (
            name not in {"fraction_simplify", "domain_restriction"}
            or type(v) not in (float, int)
            or not math.isfinite(v)
            or not 0 <= v <= 1
        ):
            raise ValueError("知识概率无效。")
    for ident, e in k.get("evidence", {}).items():
        if (
            not isinstance(e, dict)
            or e.get("source") not in {"student", "model"}
            or type(e.get("independent")) is not bool
            or not isinstance(e.get("correctness"), dict)
            or any(type(v) is not bool for v in e["correctness"].values())
        ):
            raise ValueError("知识证据格式无效。")
    if len(k.get("evidence", {})) > 2000:
        raise ValueError("证据超过原型容量。")
    s["knowledge"] = copy.deepcopy(k)
    audit = raw.get("audit", [])
    if not isinstance(audit, list) or any(not isinstance(a, dict) for a in audit):
        raise ValueError("审计格式无效。")
    s["audit"] = copy.deepcopy(audit[-1000:])
    if len(json.dumps(s, ensure_ascii=False)) > 2000000:
        raise ValueError("文件超过原型容量。")
    return s


class SQLiteStore:
    def __init__(self, path=None):
        self.path = Path(
            path or Path(os.getenv("EDU_DATA_DIR", "data")) / "learners.sqlite3"
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS learners (id TEXT PRIMARY KEY, version INTEGER NOT NULL, payload TEXT NOT NULL)"
            )

    def _id(self, identifier):
        if not re.fullmatch(r"[a-f0-9]{32}", identifier):
            raise ValueError("学习链接无效。")

    def load(self, identifier):
        self._id(identifier)
        with sqlite3.connect(self.path) as db:
            row = db.execute(
                "SELECT version,payload FROM learners WHERE id=?", (identifier,)
            ).fetchone()
        return (
            (validate_state(json.loads(row[1])), row[0]) if row else (empty_state(), 0)
        )

    def save(self, identifier, state, expected_version):
        self._id(identifier)
        safe = validate_state(state)
        with sqlite3.connect(self.path, timeout=10) as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT version FROM learners WHERE id=?", (identifier,)
            ).fetchone()
            if (row[0] if row else 0) != expected_version:
                raise StateConflict("记录已在另一个窗口更新，请刷新后重试。")
            version = expected_version + 1
            db.execute(
                "INSERT INTO learners VALUES (?,?,?) ON CONFLICT(id) DO UPDATE SET version=excluded.version,payload=excluded.payload",
                (identifier, version, json.dumps(safe, ensure_ascii=False)),
            )
        return version


# Explicit legacy file helpers; app never reads the old shared file automatically.
def load_state(path=STATE_PATH):
    try:
        return validate_state(json.loads(Path(path).read_text(encoding="utf-8")))
    except (ValueError, OSError):
        return empty_state()


def save_state(state, path=STATE_PATH):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(state_as_json(validate_state(state)), encoding="utf-8")


def state_as_json(state):
    return json.dumps(state, ensure_ascii=False, indent=2)
