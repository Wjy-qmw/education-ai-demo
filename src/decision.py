"""MWM-inspired inspectable candidate transitions, NOT a fitted student simulator."""

from dataclasses import dataclass, field
from src.memory import resolve_preference

ACTION_NAMES = {
    "hint": "先给线索",
    "full": "分步讲解",
    "diagnose": "核验作答",
    "clarify": "先确认信息",
}


@dataclass
class TeachingDecision:
    action: str
    reason: str
    memory: dict
    candidates: list = field(default_factory=list)


def explicit_style(goal):
    if any(
        w in goal
        for w in (
            "线索",
            "提示",
            "少给直接答案",
            "不要答案",
            "不要最终答案",
            "不要完整答案",
        )
    ):
        return "hint"
    if any(w in goal for w in ("完整", "赶时间", "直接答案")):
        return "full"
    if "步骤" in goal:
        return "structured"
    return None


def decide(
    records,
    goal,
    *,
    subject="数学",
    grade_result=None,
    reliable=True,
    knowledge_probability=0.2,
    use_rollout=True
):
    memory = resolve_preference(records, subject, explicit_style(goal))
    if not reliable:
        return TeachingDecision("clarify", "题文尚未确认", memory, [])
    if memory["conflict"]:
        return TeachingDecision("clarify", "同情境下偏好仍有冲突", memory, [])
    diagnosis = grade_result is not None and not grade_result["correct"]
    desired = (
        "diagnose"
        if diagnosis
        else ("full" if memory["value"] in {"full", "structured"} else "hint")
    )
    if not use_rollout:
        return TeachingDecision(desired, "条件与作答控制规则", memory, [])
    if not 0 <= knowledge_probability <= 1:
        raise ValueError("知识概率无效。")
    candidates = []
    for action in ("hint", "diagnose", "full"):
        alignment = 1.0 if action == desired else 0.15
        independent = {"hint": 0.8, "diagnose": 0.7, "full": 0.2}[action]
        cost = {"hint": 0.1, "diagnose": 0.15, "full": 0.3}[action]
        # Transparent design assumptions; must be fitted before claims of prediction quality.
        predicted = min(
            1.0, knowledge_probability + (1 - knowledge_probability) * independent * 0.1
        )
        score = (
            3 * alignment
            + 0.5 * independent
            + 0.2 * (predicted - knowledge_probability)
            - cost
        )
        candidates.append(
            {
                "action": action,
                "goal_alignment": alignment,
                "independent_opportunity": independent,
                "predicted_knowledge": predicted,
                "relative_cost": cost,
                "score": score,
                "parameter_status": "design_assumption_not_fitted",
            }
        )
    selected = max(candidates, key=lambda c: c["score"])
    return TeachingDecision(
        selected["action"],
        "候选动作按目标、独立作答机会与相对成本比较",
        memory,
        candidates,
    )
