from __future__ import annotations

import os
from collections.abc import Iterable

from openai import OpenAI


class DeepSeekConfigurationError(RuntimeError):
    pass


class DeepSeekTutor:
    def __init__(self) -> None:
        self.api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        self.base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
        self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash").strip()

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def answer(
        self,
        system_prompt: str,
        user_prompt: str,
        history: Iterable[dict[str, str]] = (),
    ) -> str:
        if not self.configured:
            raise DeepSeekConfigurationError("未检测到 DEEPSEEK_API_KEY。")

        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        for item in list(history)[-8:]:
            if item.get("role") in {"user", "assistant"} and item.get("content"):
                messages.append({"role": item["role"], "content": item["content"]})
        messages.append({"role": "user", "content": user_prompt})

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.35,
            max_tokens=6000,
            extra_body={"thinking": {"type": "disabled"}},
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("DeepSeek 返回了空内容，请稍后重试。")
        return content


def demo_answer(question: str) -> str:
    short = question.strip().replace("\n", " ")[:80] or "（尚未输入题目）"
    return f"""## 题目理解

当前题目是：**{short}**。这是离线演示回答，用于体验界面；配置 DeepSeek API Key 后会生成真实解答。

## 知识点

- 识别已知条件与所求目标
- 选择适用的定义、公式或推理规则

## 解题思路

先整理条件，再建立条件与目标之间的关系，最后代入或推导并检查结果。

## 分步讲解

1. 圈出题目中的关键量和限制条件。
2. 写出与这些条件对应的核心关系。
3. 按顺序推导，确保每一步都有依据。
4. 将结果代回原题，检查单位、符号或逻辑是否合理。

## 最终答案

**离线模式不生成具体答案。** 请配置 `DEEPSEEK_API_KEY` 后重新提交，避免演示内容误导学习。

## 举一反三

如果改变题目中的一个条件，你认为原来的解法哪一步需要先调整？"""

