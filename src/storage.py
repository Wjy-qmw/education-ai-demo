from __future__ import annotations

import json
from pathlib import Path
from typing import Any


STATE_PATH = Path(__file__).resolve().parent.parent / "data" / "state.json"


def empty_state() -> dict[str, Any]:
    return {"profile": {}, "question": "", "history": []}


def load_state(path: Path = STATE_PATH) -> dict[str, Any]:
    if not path.exists():
        return empty_state()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return empty_state()
    if not isinstance(data, dict):
        return empty_state()
    return {
        "profile": data.get("profile", {}) if isinstance(data.get("profile"), dict) else {},
        "question": data.get("question", "") if isinstance(data.get("question"), str) else "",
        "history": data.get("history", []) if isinstance(data.get("history"), list) else [],
    }


def save_state(state: dict[str, Any], path: Path = STATE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_state = {
        "profile": state.get("profile", {}),
        "question": state.get("question", ""),
        "history": state.get("history", []),
    }
    path.write_text(json.dumps(safe_state, ensure_ascii=False, indent=2), encoding="utf-8")


def state_as_json(state: dict[str, Any]) -> str:
    return json.dumps(state, ensure_ascii=False, indent=2)

