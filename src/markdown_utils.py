from __future__ import annotations

import re


_CODE_SPAN_OR_BLOCK = re.compile(r"(```[\s\S]*?```|`[^`\n]*`)")


def normalize_markdown_math(markdown: str) -> str:
    r"""Convert common model LaTeX delimiters to Streamlit-supported Markdown.

    Streamlit renders ``$...$`` and ``$$...$$``. Models often emit
    ``\(...\)`` and ``\[...\]`` instead. Code blocks and inline code are left
    untouched so examples containing literal delimiters are not corrupted.
    """
    parts = _CODE_SPAN_OR_BLOCK.split(markdown)
    for index in range(0, len(parts), 2):
        parts[index] = (
            parts[index]
            .replace(r"\[", "$$")
            .replace(r"\]", "$$")
            .replace(r"\(", "$")
            .replace(r"\)", "$")
        )
    return "".join(parts)
