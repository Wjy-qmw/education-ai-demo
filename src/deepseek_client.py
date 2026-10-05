import os, time
from openai import OpenAI


class DeepSeekConfigurationError(RuntimeError):
    pass


def setting(name, default=""):
    value = os.getenv(name)
    if value is not None:
        return value
    try:
        import streamlit as st

        return str(st.secrets.get(name, default))
    except (FileNotFoundError, KeyError):
        return default


class DeepSeekTutor:
    def __init__(self):
        self.api_key = setting("DEEPSEEK_API_KEY").strip()
        self.base_url = setting("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip(
            "/"
        )
        self.model = setting("DEEPSEEK_MODEL", "deepseek-v4-flash").strip()
        self.last_metrics = {}

    @property
    def configured(self):
        return bool(self.api_key)

    def answer(self, system_prompt, user_prompt, history=(), *, structured=False):
        if not self.configured:
            raise DeepSeekConfigurationError("服务器未配置模型接口。")
        messages = [{"role": "system", "content": system_prompt}]
        for item in list(history)[-8:]:
            if (
                isinstance(item, dict)
                and item.get("role") in {"user", "assistant"}
                and isinstance(item.get("content"), str)
                and item["content"]
            ):
                messages.append({"role": item["role"], "content": item["content"]})
        messages.append({"role": "user", "content": user_prompt})
        client = OpenAI(
            api_key=self.api_key, base_url=self.base_url, timeout=45, max_retries=1
        )
        args = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 4000,
            "extra_body": {"thinking": {"type": "disabled"}},
        }
        if structured:
            args["response_format"] = {"type": "json_object"}
        started = time.perf_counter()
        response = client.chat.completions.create(**args)
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("模型返回空内容，未保存本轮状态。")
        self.last_metrics = {
            "elapsed_s": time.perf_counter() - started,
            "model": self.model,
            "usage": response.usage.model_dump() if response.usage else None,
        }
        return content


def demo_answer(question):
    return "本地模式请使用单变量分式；其他题目需要服务器配置模型接口。"
