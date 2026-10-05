from __future__ import annotations
import copy, json, re, hashlib
from uuid import uuid4
import streamlit as st
from dotenv import load_dotenv
from src.deepseek_client import DeepSeekTutor
from src.questionnaire import QUESTIONS, normalize_answers, top_values, VALUE_NAMES
from src.storage import SQLiteStore, validate_state, state_as_json
from src.memory import add_preference, revoke_preference
from src.service import run_turn
from src.vision import get_vision_adapter
from src.markdown_utils import normalize_markdown_math

load_dotenv()
st.set_page_config(page_title="知向教育 · 证据引导辅学", page_icon="✦", layout="wide")
CSS = """
<style>
:root { --purple:#6350e8; --purple-dark:#4936d2; --ink:#172033; --muted:#7d8698; --line:#ebeaf3; }
.stApp { background: radial-gradient(circle at 16% 4%, #efefff 0, #f7f8fc 31%, #f7f8fc 100%); }
.block-container { max-width: 1120px; padding-top: 2rem; padding-bottom: 4rem; }
[data-testid="stSidebar"] { background: #fbfbfe; border-right: 1px solid #ececf4; }
.eyebrow { display:inline-flex; align-items:center; gap:.45rem; color:#5d4de0; background:#ece9ff; border-radius:999px; padding:.48rem .9rem; font-weight:700; font-size:.87rem; }
.hero { text-align:center; padding: 1.3rem 0 1.55rem; }
.hero h1 { color:var(--ink); font-size:clamp(2.1rem,5vw,3.65rem); letter-spacing:-.055em; margin:.9rem 0 .55rem; }
.hero p { color:var(--muted); font-size:1.08rem; line-height:1.75; max-width:720px; margin:auto; }
.steps { display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; margin:1rem 0 1.6rem; }
.step-card { background:rgba(255,255,255,.82); border:1px solid rgba(230,230,240,.95); border-radius:20px; padding:1.25rem; box-shadow:0 12px 35px rgba(60,52,112,.055); min-height:165px; }
.step-card.active { border-color:#b8afff; box-shadow:0 14px 36px rgba(99,80,232,.12); transform:translateY(-2px); }
.step-top { display:flex; align-items:center; gap:.7rem; color:#8a91a0; font-weight:650; }
.step-icon { width:42px; height:42px; display:grid; place-items:center; border-radius:12px; background:#efedff; color:#5c49e7; font-size:1.25rem; }
.step-card h3 { color:var(--ink); font-size:1.08rem; margin:1rem 0 .35rem; }
.step-card p { color:var(--muted); line-height:1.55; margin:0; font-size:.91rem; }
.section-card { background:rgba(255,255,255,.9); border:1px solid var(--line); border-radius:24px; padding:1.3rem 1.45rem; box-shadow:0 16px 42px rgba(44,39,91,.06); margin:.7rem 0; }
.section-card h2 { margin:.1rem 0 .25rem; color:var(--ink); font-size:1.45rem; }
.section-card p { color:var(--muted); margin:.2rem 0 .5rem; }
.profile-chip { display:inline-block; background:#f0edff; color:#5140d4; border-radius:999px; padding:.34rem .7rem; margin:.16rem; font-size:.84rem; font-weight:650; }
.strategy { border-left:3px solid #6b57ec; padding:.52rem .8rem; margin:.45rem 0; background:#faf9ff; border-radius:0 10px 10px 0; color:#4d5362; }
.value-row { display:grid; grid-template-columns:72px 1fr 35px; gap:.55rem; align-items:center; margin:.42rem 0; font-size:.86rem; }
.value-track { height:8px; background:#ececf4; border-radius:99px; overflow:hidden; }
.value-fill { height:100%; background:linear-gradient(90deg,#8e7cf2,#5742df); border-radius:99px; }
.small-note { color:#888fa0; font-size:.82rem; line-height:1.55; }
div.stButton > button[kind="primary"], div.stFormSubmitButton > button { background:linear-gradient(90deg,#6652ec,#4f3bdd); border:0; border-radius:12px; min-height:3rem; font-weight:700; }
div.stButton > button { border-radius:12px; min-height:2.75rem; }
[data-testid="stFileUploaderDropzone"] { border:1.5px dashed #b9b1ef; background:#fbfaff; border-radius:16px; }
@media (max-width:760px) { .steps { grid-template-columns:1fr; } .step-card { min-height:auto; } .block-container{padding:1rem .9rem 3rem;} }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


store = SQLiteStore()
identifier = st.query_params.get("session", "")
if not re.fullmatch(r"[a-f0-9]{32}", identifier):
    identifier = uuid4().hex
    st.query_params["session"] = identifier
if "learner" not in st.session_state or st.session_state.learner != identifier:
    st.session_state.learner = identifier
    st.session_state.saved, st.session_state.version = store.load(identifier)
    st.session_state.ocr_text = ""


def persist(candidate):
    version = store.save(identifier, candidate, st.session_state.version)
    st.session_state.saved = candidate
    st.session_state.version = version


state = st.session_state.saved
st.markdown(
    '<div class="hero"><span class="eyebrow">✦ 偏好与学习证据共同引导</span><h1>知向教育 AI 辅学</h1><p>今天想要线索，还是完整步骤？让讲解适合当下的你。</p></div>',
    unsafe_allow_html=True,
)
with st.sidebar:
    st.subheader("我的学习空间")
    st.caption("保留此私有学习链接可恢复记录。链接具有访问权限，请勿公开分享。")
    if state["profile"]:
        top = top_values(state["profile"])
        st.write(
            "、".join(f"{VALUE_NAMES[k]} {v:.1f}" for k, v in top)
            or "各项回答相同，暂无突出的偏好。"
        )
    st.download_button(
        "导出学习记录",
        state_as_json(state),
        "zhixiang-learning.json",
        "application/json",
    )
    imported = st.file_uploader("导入记录", type=["json"])
    if imported is not None and st.button("确认导入"):
        try:
            candidate = validate_state(json.loads(imported.getvalue().decode("utf-8")))
            persist(candidate)
            st.rerun()
        except (ValueError, UnicodeDecodeError):
            st.error("记录格式无效或版本已更新，请检查文件并刷新。")
    st.caption("记录按私有链接隔离。云端本机数据库可能随容器重建丢失，请定期导出。")
learn, preferences, records = st.tabs(["学习", "画像与记忆", "学习记录"])
with preferences:
    st.subheader("学习表达偏好")
    st.caption("问卷借鉴价值观维度，为项目自编题目；不用于心理诊断或能力预测。")
    with st.form("questionnaire"):
        answers = {}
        for q in QUESTIONS:
            answers[q.key] = st.slider(
                q.prompt,
                1,
                5,
                int(state["profile"].get(q.key, 3)),
                key="value_" + q.key,
            )
        submitted = st.form_submit_button("保存学习画像")
    if submitted:
        try:
            candidate = copy.deepcopy(state)
            candidate["profile"] = normalize_answers(answers)
            persist(candidate)
            st.rerun()
        except ValueError:
            st.error("记录有变化，请刷新后重试。")
    st.subheader("我允许记住的偏好")
    style = st.selectbox(
        "长期数学讲解偏好",
        ["hint", "full", "structured"],
        format_func=lambda v: {
            "hint": "先给线索",
            "full": "完整讲解",
            "structured": "分步结构",
        }[v],
    )
    confirmed = st.checkbox("我确认将这项偏好保存到我的学习记录")
    if st.button("保存长期偏好", disabled=not confirmed):
        try:
            candidate = copy.deepcopy(state)
            add_preference(candidate["memory"], style, confirmed=True)
            persist(candidate)
            st.rerun()
        except ValueError:
            st.error("记录有变化，请刷新后重试。")
    for m in state["memory"]:
        st.write(
            f"数学：{m['value']} · 版本{m['version']} · {'生效' if m['active'] else '已撤销'}"
        )
        if m["active"] and st.button("撤销这项记忆", key=m["id"]):
            try:
                candidate = copy.deepcopy(state)
                revoke_preference(candidate["memory"], m["id"])
                persist(candidate)
                st.rerun()
            except ValueError:
                st.error("记录有变化，请刷新后重试。")
    st.caption("今天的要求优先于长期偏好；同条件的冲突偏好会保留并请求确认。")
with learn:
    tutor = DeepSeekTutor()
    st.caption(
        f"模型接口已配置 · {tutor.model}；成功响应后才记录调用结果。"
        if tutor.configured
        else "本地数学规则模式 · 支持单变量分式，其他题目需模型接口。"
    )
    uploaded = st.file_uploader("题目图片（可选）", type=["png", "jpg", "jpeg", "webp"])
    if uploaded is not None and st.button("识别题目文字"):
        try:
            result = get_vision_adapter().extract(uploaded.getvalue(), uploaded.type)
            st.session_state.ocr_text = result.text
            st.info(result.notice)
        except Exception:
            st.error("文字识别未完成，请手动录入并核对题目。")
    question = st.text_area(
        "完整题目",
        value=st.session_state.ocr_text or state["question"],
        placeholder="化简 (x^2-1)/(x-1)，写出原式条件。",
    )
    reliable = st.checkbox(
        "我已核对题目文字及公式", value=not bool(st.session_state.ocr_text)
    )
    goal = st.text_input(
        "今天希望怎样帮你？", placeholder="例如：只给提示，不要答案；或：完整讲解"
    )
    with st.expander("提交自己的作答（可选）"):
        answer = st.text_input("我的最终表达式", placeholder="x+1")
        restriction = st.text_input("原式限制", placeholder="x!=1；多个条件用逗号分隔")
        independent = st.checkbox("这次作答是在未查看本题答案或提示时独立完成的")
        st.caption(
            "只有核验范围内的独立作答才更新模型知识概率；这是状态估计，不是成绩。"
        )
    if st.button("获得帮助 / 继续追问", type="primary", disabled=not question.strip()):
        try:
            with st.spinner("正在组织讲解并核验…"):
                candidate, version = run_turn(
                    store,
                    identifier,
                    state,
                    st.session_state.version,
                    question.strip(),
                    goal,
                    tutor=tutor,
                    student_answer=answer,
                    restriction=restriction,
                    independent=independent,
                    evidence_id=hashlib.sha256(
                        (
                            question.strip()
                            + "\n"
                            + answer.strip()
                            + "\n"
                            + restriction.strip()
                        ).encode()
                    ).hexdigest(),
                    reliable=reliable,
                )
            st.session_state.saved = candidate
            st.session_state.version = version
            st.rerun()
        except Exception:
            st.error(
                "本轮未完成：接口、输出核验或记录版本检查未通过。记录未覆盖；请核对题目后重试，或刷新页面。"
            )
    for item in state["history"]:
        with st.chat_message(item["role"]):
            st.markdown(normalize_markdown_math(item["content"]))
with records:
    st.subheader("知识状态与处理记录")
    st.caption("以下概率来自设计参数的 BKT 更新，尚未经真实学生数据校准。")
    for skill, label in [
        ("fraction_simplify", "分式化简"),
        ("domain_restriction", "原式条件"),
    ]:
        st.write(label)
        st.progress(state["knowledge"]["skills"].get(skill, 0.2))
    if state["audit"]:
        st.dataframe(
            [
                {
                    "教学动作": a["action"],
                    "模式": a["mode"],
                    "核验范围": a["validation"],
                    "耗时（秒）": round(a["elapsed_s"], 3),
                }
                for a in state["audit"]
            ],
            hide_index=True,
        )
        with st.expander("查看可追溯的决策依据"):
            st.json(state["audit"][-1])
    else:
        st.info("完成一轮帮助后显示记录。")
