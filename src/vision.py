from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from typing import Protocol

from openai import OpenAI


@dataclass(frozen=True)
class VisionResult:
    text: str
    provider: str
    notice: str


class VisionAdapter(Protocol):
    def extract(self, image_bytes: bytes, mime_type: str) -> VisionResult: ...


class DisabledVisionAdapter:
    def extract(self, image_bytes: bytes, mime_type: str) -> VisionResult:
        del image_bytes, mime_type
        return VisionResult(
            text="",
            provider="disabled",
            notice="图片已上传，但当前未启用图片识别。请在下方补充题目文字，或配置 VISION_PROVIDER=deepseek_vision。",
        )


class DeepSeekVisionAdapter:
    def __init__(self) -> None:
        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("图片识别需要 DEEPSEEK_API_KEY。")
        self.client = OpenAI(
            api_key=api_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/"),
        )
        self.model = os.getenv("DEEPSEEK_VISION_MODEL", "deepseek-v4-flash")

    def extract(self, image_bytes: bytes, mime_type: str) -> VisionResult:
        encoded = base64.b64encode(image_bytes).decode("ascii")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "请只转写图片中的完整题目文字、公式、选项和图形条件。看不清的部分标记为[无法识别]，不要解题。",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
                        },
                    ],
                }
            ],
            temperature=0,
            max_tokens=1800,
        )
        text = response.choices[0].message.content or ""
        return VisionResult(text=text, provider="deepseek_vision", notice="已由多模态适配层完成题目转写，请核对后再解答。")


def get_vision_adapter() -> VisionAdapter:
    provider = os.getenv("VISION_PROVIDER", "disabled").strip().lower()
    if provider == "deepseek_vision":
        return DeepSeekVisionAdapter()
    return DisabledVisionAdapter()

