from __future__ import annotations

import json
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from src.deepseek_client import DeepSeekConfigurationError, DeepSeekTutor, demo_answer
from src.markdown_utils import normalize_markdown_math
from src.personalization import (
    build_learning_strategies,
    build_system_prompt,
    build_user_prompt,
    profile_summary,
)
from src.questionnaire import QUESTIONS, VALUE_NAMES, normalize_answers, top_values
from src.storage import load_state, save_state, state_as_json
from src.vision import get_vision_adapter


load_dotenv()
st.set_page_config(page_title="知向教育 · 个性化 AI 解题", page_icon="✦", layout="wide")


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


def bootstrap() -> None:
    if "bootstrapped" in st.session_state:
        return
    saved = load_state()
    st.session_state.profile = saved["profile"]
    st.session_state.question = saved["question"]
    st.session_state.history = saved["history"]
    st.session_state.page = 1 if not saved["profile"] else 2
    st.session_state.ocr_text = ""
    st.session_state.vision_notice = ""
    st.session_state.bootstrapped = True


def persist() -> None:
    save_state(
        {
            "profile": st.session_state.profile,
            "question": st.session_state.question,
            "history": st.session_state.history,
        }
    )


def go(page: int) -> None:
    st.session_state.page = page


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
          <span class="eyebrow">✦ 价值观驱动的个性化学习</span>
          <h1>知向教育 AI 解题</h1>
          <p>先理解你偏好的学习方式，再理解题目。让每一次讲解不仅正确，也更适合你。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    page = st.session_state.page
    cards = [
        (1, "▣", "完成价值问卷", "10 道核心题生成 Schwartz 十维 Value Vector"),
        (2, "⌁", "上传或录入题目", "支持图片适配层与文字输入，提交前可核对题目"),
        (3, "◯", "获得个性化讲解", "知识点、分步推导、最终答案与连续追问"),
    ]
    html = '<div class="steps">'
    for number, icon, title, desc in cards:
        active = " active" if number == page else ""
        html += f'<div class="step-card{active}"><div class="step-top"><span class="step-icon">{icon}</span>Step {number}</div><h3>{title}</h3><p>{desc}</p></div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_profile_sidebar() -> None:
    with st.sidebar:
        st.markdown("### 我的学习画像")
        if not st.session_state.profile:
            st.caption("完成 10 道核心题后生成")
        else:
            vector = normalize_answers(st.session_state.profile)
            st.caption(f"当前核心偏好：{profile_summary(vector)}")
            for key, value in vector.items():
                width = int(value / 5 * 100)
                st.markdown(
                    f'<div class="value-row"><span>{VALUE_NAMES[key]}</span><div class="value-track"><div class="value-fill" style="width:{width}%"></div></div><b>{value:.1f}</b></div>',
                    unsafe_allow_html=True,
                )
            if st.button("重新测量", use_container_width=True):
                go(1)
                st.rerun()

        st.divider()
        st.markdown("### 数据与历史")
        state = {
            "profile": st.session_state.profile,
            "question": st.session_state.question,
            "history": st.session_state.history,
        }
        st.download_button(
            "导出画像与对话",
            data=state_as_json(state),
            file_name=f"zhixiang-learning-{datetime.now():%Y%m%d-%H%M}.json",
            mime="application/json",
            use_container_width=True,
        )
        imported = st.file_uploader("导入历史", type=["json"], label_visibility="collapsed")
        if imported is not None and st.button("确认导入", use_container_width=True):
            try:
                payload = json.loads(imported.getvalue().decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("根节点不是对象")
                st.session_state.profile = normalize_answers(payload.get("profile", {}))
                st.session_state.question = str(payload.get("question", ""))
                history = payload.get("history", [])
                st.session_state.history = history if isinstance(history, list) else []
                persist()
                st.success("已导入")
                st.rerun()
            except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                st.error(f"无法导入：{exc}")
        if st.button("清空本次题目与对话", use_container_width=True):
            st.session_state.question = ""
            st.session_state.history = []
            persist()
            st.rerun()
        st.caption("MVP 默认保存在本机 `data/state.json`；上传的原图不会被持久化。")


def render_questionnaire() -> None:
    st.markdown('<div class="section-card"><h2>先了解你的学习偏好</h2><p>请选择每句话与你的符合程度。这里测量的是讲解偏好，不是能力、成绩或人格好坏。</p></div>', unsafe_allow_html=True)
    with st.form("value_questionnaire"):
        answers: dict[str, int] = {}
        for index, q in enumerate(QUESTIONS, 1):
            st.markdown(f"**{index:02d}　{q.prompt}**")
            st.caption(f"{q.name} · {q.english}｜{q.hint}")
            default = int(st.session_state.profile.get(q.key, 3)) if st.session_state.profile else 3
            answers[q.key] = st.radio(
                q.prompt,
                options=[1, 2, 3, 4, 5],
                index=default - 1,
                format_func=lambda value: {1: "很不符合", 2: "不太符合", 3: "一般", 4: "比较符合", 5: "很符合"}[value],
                horizontal=True,
                label_visibility="collapsed",
                key=f"question_{q.key}",
            )
            if index < len(QUESTIONS):
                st.divider()
        submitted = st.form_submit_button("生成我的学习画像 →", use_container_width=True)
    if submitted:
        st.session_state.profile = normalize_answers(answers)
        persist()
        go(2)
        st.rerun()


def render_profile_summary() -> None:
    vector = normalize_answers(st.session_state.profile)
    top = top_values(vector)
    chips = "".join(f'<span class="profile-chip">{VALUE_NAMES[key]} {score:.1f}</span>' for key, score in top)
    strategies = "".join(f'<div class="strategy">{item}</div>' for item in build_learning_strategies(vector))
    st.markdown(
        f'<div class="section-card"><h2>你的个性化讲解策略</h2><p>当前最突出的三个价值偏好</p>{chips}<div style="height:.65rem"></div>{strategies}</div>',
        unsafe_allow_html=True,
    )


def render_question_input() -> None:
    render_profile_summary()
    st.markdown('<div class="section-card"><h2>录入你的题目</h2><p>上传题目截图，或直接输入文字。图片识别结果可以在提交前修改。</p></div>', unsafe_allow_html=True)

    left, right = st.columns([1, 1.2], gap="large")
    with left:
        uploaded = st.file_uploader("上传题目图片", type=["png", "jpg", "jpeg", "webp"])
        if uploaded is not None:
            st.image(uploaded, caption="题目预览", use_container_width=True)
            if st.button("识别图片中的题目", use_container_width=True):
                try:
                    result = get_vision_adapter().extract(uploaded.getvalue(), uploaded.type or "image/jpeg")
                    st.session_state.ocr_text = result.text
                    st.session_state.vision_notice = result.notice
                except Exception as exc:  # provider/network errors should be visible in the demo UI
                    st.session_state.vision_notice = f"图片识别失败：{exc}。请手动输入题目文字。"
            if st.session_state.vision_notice:
                if st.session_state.ocr_text:
                    st.success(st.session_state.vision_notice)
                else:
                    st.info(st.session_state.vision_notice)
        else:
            st.info("默认仅保存识别后的文字，不保存原始图片。")

    with right:
        default_text = st.session_state.ocr_text or st.session_state.question
        question = st.text_area(
            "题目文字",
            value=default_text,
            height=245,
            placeholder="例如：已知二次函数 f(x)=x²-4x+3，求它的顶点坐标和零点。",
        )
        subject = st.selectbox("学科（用于辅助判断）", ["自动判断", "数学", "语文", "英语", "物理", "化学", "生物", "历史", "其他"])
        tutor = DeepSeekTutor()
        if tutor.configured:
            st.success(f"DeepSeek 已连接 · {tutor.model}")
        else:
            st.warning("尚未配置 API Key，将进入明确标注的离线演示模式。")
        if st.button("开始个性化解答 →", type="primary", use_container_width=True, disabled=not question.strip()):
            st.session_state.question = question.strip()
            st.session_state.subject = subject
            with st.spinner("正在判断知识点并组织适合你的讲解…"):
                try:
                    system_prompt = build_system_prompt(st.session_state.profile)
                    user_prompt = build_user_prompt(f"学科：{subject}\n{question.strip()}")
                    answer = tutor.answer(system_prompt, user_prompt)
                    mode = "deepseek"
                except DeepSeekConfigurationError:
                    answer = demo_answer(question)
                    mode = "demo"
                except Exception as exc:
                    st.error(f"DeepSeek 调用失败：{exc}")
                    return
            st.session_state.history = [
                {"role": "user", "content": question.strip(), "kind": "question"},
                {"role": "assistant", "content": answer, "mode": mode},
            ]
            persist()
            go(3)
            st.rerun()


def render_solution() -> None:
    tutor = DeepSeekTutor()
    top_left, top_right = st.columns([4, 1])
    with top_left:
        st.markdown('<div class="section-card"><h2>个性化解答</h2><p>按照你的学习画像组织知识点、推导顺序与反馈方式。</p></div>', unsafe_allow_html=True)
    with top_right:
        if st.button("换一道题", use_container_width=True):
            st.session_state.question = ""
            st.session_state.history = []
            st.session_state.ocr_text = ""
            persist()
            go(2)
            st.rerun()

    if st.session_state.history and st.session_state.history[1].get("mode") == "demo":
        st.warning("以下是离线演示结构，不包含真实题目答案。配置 DeepSeek API Key 后可生成完整解答。")

    for item in st.session_state.history:
        role = item.get("role", "assistant")
        with st.chat_message(role, avatar="🎓" if role == "assistant" else "🧑‍💻"):
            st.markdown(normalize_markdown_math(item.get("content", "")))

    with st.form("follow_up_form", clear_on_submit=True):
        follow_up = st.text_input("继续追问", placeholder="例如：第 2 步为什么要这样变形？还有别的方法吗？")
        ask = st.form_submit_button("发送追问", use_container_width=True)
    if ask and follow_up.strip():
        with st.spinner("正在结合原题和你的学习画像继续讲解…"):
            try:
                if tutor.configured:
                    answer = tutor.answer(
                        build_system_prompt(st.session_state.profile),
                        build_user_prompt(st.session_state.question, follow_up.strip()),
                        history=st.session_state.history,
                    )
                    mode = "deepseek"
                else:
                    answer = demo_answer(follow_up)
                    mode = "demo"
            except Exception as exc:
                st.error(f"追问失败：{exc}")
                return
        st.session_state.history.extend(
            [
                {"role": "user", "content": follow_up.strip(), "kind": "follow_up"},
                {"role": "assistant", "content": answer, "mode": mode},
            ]
        )
        persist()
        st.rerun()

    with st.expander("查看本次结构化个性化 Prompt"):
        st.caption("用于研究展示与调试；实际产品可默认隐藏。")
        st.code(build_system_prompt(st.session_state.profile), language="markdown")


bootstrap()
render_profile_sidebar()
render_hero()

if st.session_state.page == 1:
    render_questionnaire()
elif st.session_state.page == 2:
    if not st.session_state.profile:
        go(1)
        st.rerun()
    render_question_input()
else:
    if not st.session_state.history:
        go(2)
        st.rerun()
    render_solution()

st.markdown(
    '<p class="small-note" style="text-align:center;margin-top:2.5rem">价值画像只用于调整教学表达，不用于预测能力、替代教师判断或自动评分。</p>',
    unsafe_allow_html=True,
)
