"""Auditable BKT mechanics. Default parameters are design values, not fitted data."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class BKTParameters:
    initial: float = 0.2
    learn: float = 0.1
    slip: float = 0.1
    guess: float = 0.2

    def __post_init__(self):
        if not all(
            math.isfinite(x) and 0 < x < 1
            for x in (self.initial, self.learn, self.slip, self.guess)
        ):
            raise ValueError("BKT参数须介于0与1之间。")


def update_probability(prior, correct, parameters=None):
    p = parameters or BKTParameters()
    if not math.isfinite(prior) or not 0 <= prior <= 1:
        raise ValueError("先验概率无效。")
    if correct:
        posterior = (
            prior * (1 - p.slip) / (prior * (1 - p.slip) + (1 - prior) * p.guess)
        )
    else:
        posterior = prior * p.slip / (prior * p.slip + (1 - prior) * (1 - p.guess))
    return posterior + (1 - posterior) * p.learn


def apply_evidence(
    knowledge,
    evidence_id,
    correctness,
    *,
    independent,
    source="student",
    parameters=None
):
    """One attempt updates multiple skills once. Assisted or model evidence is logged only."""
    p = parameters or BKTParameters()
    seen = knowledge.setdefault("evidence", {})
    if evidence_id in seen:
        return {"updated": False, "reason": "duplicate", "changes": []}
    changes = []
    if independent and source == "student":
        skills = knowledge.setdefault("skills", {})
        for skill, correct in correctness.items():
            before = float(skills.get(skill, p.initial))
            after = update_probability(before, bool(correct), p)
            skills[skill] = after
            changes.append(
                {
                    "skill": skill,
                    "before": before,
                    "after": after,
                    "correct": bool(correct),
                }
            )
    seen[evidence_id] = {
        "source": source,
        "independent": independent,
        "correctness": correctness,
    }
    return {
        "updated": bool(changes),
        "reason": "independent" if changes else "not_independent_student_evidence",
        "changes": changes,
    }
