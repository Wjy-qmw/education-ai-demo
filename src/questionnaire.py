from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ValueQuestion:
    key: str
    name: str
    english: str
    prompt: str
    hint: str


QUESTIONS: tuple[ValueQuestion, ...] = (
    ValueQuestion("self_direction", "自主", "Self-Direction", "我喜欢自己探索解题方法，而不是只照着固定步骤做。", "独立思考、选择与创造"),
    ValueQuestion("stimulation", "刺激", "Stimulation", "遇到新题型和有挑战的学习任务时，我通常会感到兴奋。", "新奇、变化与挑战"),
    ValueQuestion("hedonism", "享乐", "Hedonism", "我希望学习过程有趣、轻松，并能带来愉快体验。", "愉悦与学习体验"),
    ValueQuestion("achievement", "成就", "Achievement", "完成高难度目标、看到成绩进步会强烈激励我。", "目标、能力与成功"),
    ValueQuestion("power", "权力", "Power", "在小组学习中，我愿意主导任务并影响最终决定。", "影响力、地位与资源"),
    ValueQuestion("security", "安全", "Security", "我更喜欢有清晰计划、可预期结果和较低出错风险的学习方式。", "稳定、秩序与可预期"),
    ValueQuestion("conformity", "遵从", "Conformity", "学习时，我重视遵守老师要求、规范步骤和课堂规则。", "规范、自律与规则"),
    ValueQuestion("tradition", "传统", "Tradition", "我重视经过长期验证的经典方法和学习习惯。", "传统、习惯与延续"),
    ValueQuestion("benevolence", "仁爱", "Benevolence", "我愿意和同学互相帮助，也希望讲解耐心、友善。", "关心身边的人"),
    ValueQuestion("universalism", "普世关怀", "Universalism", "我希望学习内容能联系公平、社会与更广泛的人类福祉。", "公平、包容与公共关怀"),
)

VALUE_NAMES = {q.key: q.name for q in QUESTIONS}


def normalize_answers(answers: dict[str, int | float]) -> dict[str, float]:
    """Clamp questionnaire answers to the five-point scale."""
    return {
        q.key: round(min(5.0, max(1.0, float(answers.get(q.key, 3)))), 1)
        for q in QUESTIONS
    }


def top_values(vector: dict[str, float], limit: int = 3) -> list[tuple[str, float]]:
    return sorted(vector.items(), key=lambda item: item[1], reverse=True)[:limit]
