"""Reproducible synthetic controller and printed-OCR checks; no learner outcomes."""

import argparse, copy, csv, io, json, platform, time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from src.storage import empty_state, SQLiteStore
from src.memory import add_preference
from src.learning_engine import prepare, commit_evidence
from src.knowledge import update_probability
from src.vision import get_vision_adapter


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)


def distance(a, b):
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        new = [i]
        for j, cb in enumerate(b, 1):
            new.append(min(new[-1] + 1, row[j] + 1, row[j - 1] + (ca != cb)))
        row = new
    return row[-1]


def main(out):
    out.mkdir(parents=True, exist_ok=True)
    fixtures = json.loads(
        (Path(__file__).parent / "fixtures/scenarios.json").read_text()
    )
    results = []
    expected = {
        "稳定偏好": "hint",
        "本轮需求变化": "full",
        "情境差异": "hint",
        "来源矛盾": "hint",
        "作答与自述不一致": "diagnose",
        "识别不确定性": "clarify",
    }
    for case in fixtures:
        kind = case["scenario_type"]
        index = int(case["base_item_id"][1:])
        state = empty_state()
        add_preference(state["memory"], "hint", confirmed=True)
        if kind == "来源矛盾":
            state["memory"].append(
                {
                    "id": "synthetic-model-summary",
                    "scope": "long_term",
                    "source": "model",
                    "subject": "数学",
                    "value": "full",
                    "active": True,
                    "version": 1,
                }
            )
        kwargs = {
            "reliable": kind != "识别不确定性",
            "student_answer": f"x+{index}" if kind == "作答与自述不一致" else "",
            "restriction": "",
            "independent": False,
            "evidence_id": case["case_id"],
        }
        start = time.perf_counter()
        p = prepare(state, case["question_for_model"], case["current_goal"], **kwargs)
        elapsed = time.perf_counter() - start
        latest = (
            "full"
            if kind == "来源矛盾"
            else ("full" if kind == "本轮需求变化" else "hint")
        )
        actions = {
            "固定线索规则": "hint",
            "最近描述规则": latest,
            "条件控制规则": prepare(
                state,
                case["question_for_model"],
                case["current_goal"],
                use_rollout=False,
                **kwargs,
            )["decision"].action,
            "候选动作比较": p["decision"].action,
        }
        before = copy.deepcopy(state["knowledge"])
        update = commit_evidence(state, p)
        for policy, action in actions.items():
            results.append(
                {
                    "case_id": case["case_id"],
                    "scenario_type": kind,
                    "policy": policy,
                    "expected_action": expected[kind],
                    "actual_action": action,
                    "pass": action == expected[kind],
                    "engine_ms": elapsed * 1000,
                    "knowledge_unchanged": state["knowledge"].get("skills", {})
                    == before.get("skills", {}),
                    "sample_type": "synthetic_rule_fixture_not_llm_trial",
                }
            )
    write_csv(out / "controller_results.csv", results)
    trajectories = []
    p = 0.2
    for i, correct in enumerate([True, True, False, True, True, False, True, True], 1):
        before = p
        p = update_probability(p, correct)
        trajectories.append(
            {
                "attempt": i,
                "independent_correct": correct,
                "before": before,
                "after": p,
                "assisted_only_probability": 0.2,
                "sample_type": "synthetic_BKT_calculation",
            }
        )
    write_csv(out / "bkt_trajectory.csv", trajectories)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 30)
    ocr = []
    for i in range(1, 7):
        expected_text = f"Simplify (x^2-{i*i})/(x-{i})"
        im = Image.new("RGB", (850, 100), "white")
        ImageDraw.Draw(im).text((20, 25), expected_text, font=font, fill="#172033")
        if i > 3:
            im = im.rotate(
                (i - 4) * 1.5, resample=Image.Resampling.BICUBIC, fillcolor="white"
            )
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        (out / f"ocr_fixture_{i}.png").write_bytes(buf.getvalue())
        start = time.perf_counter()
        res = get_vision_adapter().extract(buf.getvalue())
        sec = time.perf_counter() - start
        a = "".join(expected_text.split())
        b = "".join(res.text.split())
        d = distance(a, b)
        ocr.append(
            {
                "id": i,
                "expected_text": expected_text,
                "actual_text": res.text,
                "edit_distance": d,
                "reference_characters": len(a),
                "CER": d / len(a),
                "elapsed_s": sec,
                "sample_type": "synthetic_printed_english_fixture_not_camera_photo",
            }
        )
    write_csv(out / "ocr_results.csv", ocr)
    summary = {
        "utc": __import__("datetime")
        .datetime.now(__import__("datetime").timezone.utc)
        .isoformat(),
        "python": platform.python_version(),
        "synthetic_cases": 120,
        "unique_base_questions": 20,
        "policy_observations": 480,
        "policy_pass_counts": {
            k: sum(r["pass"] for r in results if r["policy"] == k) for k in actions
        },
        "knowledge_guard_passes": sum(
            r["knowledge_unchanged"] for r in results if r["policy"] == "候选动作比较"
        ),
        "ocr_fixture_count": 6,
        "ocr_CER_weighted": sum(r["edit_distance"] for r in ocr)
        / sum(r["reference_characters"] for r in ocr),
        "human_participants": 0,
        "new_branch_live_api_calls": 0,
    }
    (out / "validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("validation-output"))
    main(parser.parse_args().output)
