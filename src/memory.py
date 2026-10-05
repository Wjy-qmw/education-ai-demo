"""Conditioned preferences: user statements only, alternatives retained, versioned revocation."""

from uuid import uuid4

STYLES = {"hint", "full", "structured"}
SOURCES = {"user": 3, "teacher": 2, "model": 0}


def add_preference(
    records, value, *, subject="数学", scope="long_term", source="user", confirmed=False
):
    if (
        value not in STYLES
        or scope not in {"long_term", "session"}
        or source not in SOURCES
    ):
        raise ValueError("偏好格式无效。")
    if scope == "long_term" and (source != "user" or not confirmed):
        raise ValueError("长期偏好需要用户明确确认，不能从模型摘要自动写入。")
    item = {
        "id": uuid4().hex,
        "value": value,
        "subject": subject,
        "scope": scope,
        "source": source,
        "active": True,
        "version": 1,
    }
    records.append(item)
    return item


def revoke_preference(records, record_id):
    for item in records:
        if item["id"] == record_id and item.get("active", True):
            item["active"] = False
            item["version"] = item.get("version", 1) + 1
            return True
    return False


def resolve_preference(records, subject="数学", current=None, session_id=None):
    if current in STYLES:
        return {
            "value": current,
            "conflict": False,
            "record_ids": [],
            "source": "current_request",
        }
    candidates = [
        r
        for r in records
        if r.get("active", True)
        and (
            r.get("scope") == "long_term"
            or (session_id is not None and r.get("session_id") == session_id)
        )
        and r.get("subject") in {subject, "全部"}
        and r.get("source") in {"user", "teacher"}
    ]
    if not candidates:
        return {"value": None, "conflict": False, "record_ids": [], "source": "none"}
    best = max(SOURCES[r["source"]] for r in candidates)
    candidates = [r for r in candidates if SOURCES[r["source"]] == best]
    # A specific subject has priority over general records, without deleting either.
    if any(r["subject"] == subject for r in candidates):
        candidates = [r for r in candidates if r["subject"] == subject]
    values = {r["value"] for r in candidates}
    return {
        "value": next(iter(values)) if len(values) == 1 else None,
        "conflict": len(values) > 1,
        "record_ids": [r["id"] for r in candidates],
        "source": "conditioned_memory",
    }
