"""Stage, verify and persist a turn atomically; failures leave the saved state intact."""

import copy, time
from datetime import datetime, timezone
from uuid import uuid4
from src.learning_engine import (
    prepare,
    local_response,
    evidence_prompt,
    validate_model_response,
    commit_evidence,
)


def run_turn(
    store, identifier, state, version, question, goal, *, tutor=None, **evidence
):
    start = time.perf_counter()
    prepared = prepare(state, question, goal, **evidence)
    mode = "local_rules"
    validation = "本地规则；不自动评分自然语言讲解"
    metrics = {}
    if tutor is not None and tutor.configured:
        raw = tutor.answer(
            evidence_prompt(prepared, state["profile"]),
            "当前完整题目：" + question + "\n本次要求：" + goal,
            state["history"],
            structured=True,
        )
        response, validation = validate_model_response(raw, prepared)
        mode = "live_api"
        metrics = tutor.last_metrics
    else:
        response = local_response(prepared)
    candidate = copy.deepcopy(state)
    update = commit_evidence(candidate, prepared)
    candidate["question"] = question
    candidate["history"].extend(
        [
            {"role": "user", "content": question + "\n" + goal},
            {"role": "assistant", "content": response},
        ]
    )
    decision = prepared["decision"]
    candidate["audit"].append(
        {
            "id": uuid4().hex,
            "utc": datetime.now(timezone.utc).isoformat(),
            "mode": mode,
            "action": decision.action,
            "reason": decision.reason,
            "validation": validation,
            "memory": decision.memory,
            "rules": [r["id"] for r in prepared["rules"]],
            "candidates": decision.candidates,
            "grade": prepared["grade"],
            "knowledge_update": update,
            "metrics": metrics,
            "elapsed_s": time.perf_counter() - start,
        }
    )
    new_version = store.save(identifier, candidate, version)
    return candidate, new_version
