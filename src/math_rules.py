"""Bounded, non-eval expression parser and exact single-variable rational checks."""

import ast, re
from dataclasses import dataclass
import sympy as sp

x = sp.Symbol("x", real=True)


def expression(text):
    text = (
        text.strip()
        .replace("²", "**2")
        .replace("³", "**3")
        .replace("^", "**")
        .replace("−", "-")
        .replace("×", "*")
        .replace("÷", "/")
    )
    text = re.sub(r"(\d)x", r"\1*x", text)
    if not text or len(text) > 250:
        raise ValueError("表达式为空或过长。")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ValueError("请使用x、数字及四则运算。") from exc
    if sum(1 for _ in ast.walk(tree)) > 80:
        raise ValueError("表达式过于复杂。")

    def visit(node, depth=0):
        if depth > 16:
            raise ValueError("表达式嵌套过深。")
        if (
            isinstance(node, ast.Constant)
            and type(node.value) in (int, float)
            and abs(node.value) < 1e7
        ):
            return sp.Rational(str(node.value))
        if isinstance(node, ast.Name) and node.id == "x":
            return x
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            v = visit(node.operand, depth + 1)
            return v if isinstance(node.op, ast.UAdd) else -v
        if isinstance(node, ast.BinOp):
            a, b = visit(node.left, depth + 1), visit(node.right, depth + 1)
            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            if isinstance(node.op, ast.Mult):
                return a * b
            if isinstance(node.op, ast.Div) and b != 0:
                return a / b
            if isinstance(node.op, ast.Pow) and b.is_Integer and abs(b) <= 6:
                return a**b
        raise ValueError("不支持该表达式；函数、属性和代码调用被禁止。")

    return visit(tree.body)


@dataclass(frozen=True)
class FractionProblem:
    numerator: object
    denominator: object
    simplified: object
    excluded: tuple

    def record(self):
        return {
            "numerator": str(self.numerator),
            "denominator": str(self.denominator),
            "simplified": str(self.simplified),
            "excluded": [str(v) for v in self.excluded],
        }


def fraction_problem(question):
    if "无法识别" in question:
        raise ValueError("请先确认题目中看不清的符号。")
    match = re.search(r"\(([^()]*)\)\s*[/÷]\s*\(([^()]*)\)", question)
    if not match:
        raise ValueError("本地核验目前支持(x^2-1)/(x-1)形式的单变量分式。")
    num, den = expression(match[1]), expression(match[2])
    try:
        if sp.Poly(num, x).degree() > 6 or sp.Poly(den, x).degree() > 4:
            raise ValueError("超过核验次数范围。")
    except sp.PolynomialError as exc:
        raise ValueError("分子分母须为多项式。") from exc
    if den == 0:
        raise ValueError("分母不能恒为0。")
    roots = sp.solveset(den, x, domain=sp.S.Reals)
    if roots is sp.S.EmptySet:
        excluded = ()
    elif isinstance(roots, sp.FiniteSet) and all(v.is_Rational for v in roots):
        excluded = tuple(sorted(roots, key=str))
    else:
        raise ValueError("无法有限核验该分母。")
    return FractionProblem(num, den, sp.cancel(num / den), excluded)


def grade(problem, answer, restriction):
    try:
        equivalent = sp.cancel(expression(answer) - problem.simplified) == 0
    except (ValueError, ZeroDivisionError):
        equivalent = False
    values = re.findall(r"x\s*(?:!=|≠|不等于|\\ne)\s*(-?\d+(?:/\d+)?)", restriction)
    remainder = re.sub(r"x\s*(?:!=|≠|不等于|\\ne)\s*(-?\d+(?:/\d+)?)", "", restriction)
    clean = not remainder.strip(" ,，;；、\n")
    try:
        specified = {expression(v) for v in values}
    except (ValueError, ZeroDivisionError):
        specified = set()
        clean = False
    domain_ok = (
        clean
        and specified == set(problem.excluded)
        and (bool(values) or not problem.excluded)
    )
    # An equivalent expression must also not introduce extra singularities.
    if equivalent:
        answer_den = sp.denom(sp.together(expression(answer)))
        if sp.Poly(answer_den, x).degree() > 4:
            return {
                "fraction_simplify": False,
                "domain_restriction": bool(domain_ok),
                "correct": False,
            }
        additional = sp.solveset(answer_den, x, domain=sp.S.Reals)
        equivalent = additional.is_subset(sp.FiniteSet(*problem.excluded)) is True
    return {
        "fraction_simplify": bool(equivalent),
        "domain_restriction": bool(domain_ok),
        "correct": bool(equivalent and domain_ok),
    }


RULES = [
    {
        "id": "F01",
        "skill": "domain_restriction",
        "text": "原式分母不能为零；约分后仍需保留原式排除值。",
    },
    {
        "id": "F02",
        "skill": "fraction_simplify",
        "text": "只能约去公因式，不能把加减项分别约去。",
    },
    {
        "id": "F03",
        "skill": "fraction_simplify",
        "text": "平方差公式：a²−b²=(a−b)(a+b)。",
    },
]


def retrieve_rules(problem):
    return RULES[:2] + (
        [RULES[2]] if sp.Poly(problem.numerator, x).degree() == 2 else []
    )
