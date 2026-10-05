"""Local printed-text OCR; user confirmation is mandatory before math checks."""

import io, shutil, subprocess, tempfile
from dataclasses import dataclass
from pathlib import Path
from PIL import Image


@dataclass(frozen=True)
class VisionResult:
    text: str
    provider: str
    notice: str


class TesseractVisionAdapter:
    def extract(self, image_bytes, mime_type="image/png"):
        if len(image_bytes) > 10_000_000:
            raise ValueError("图片超过10MB。")
        if not shutil.which("tesseract"):
            raise RuntimeError("服务器未安装文字识别组件，请手动录入。")
        with Image.open(io.BytesIO(image_bytes)) as im:
            if im.width * im.height > 8_000_000:
                raise ValueError("图片像素过大。")
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "image.png"
                im.convert("RGB").save(path)
                result = subprocess.run(
                    ["tesseract", str(path), "stdout", "-l", "eng", "--psm", "6"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=True,
                )
        return VisionResult(
            result.stdout.strip(),
            "tesseract",
            "识别完成。公式可能转写错误，请逐字核对并勾选确认。",
        )


def get_vision_adapter():
    return TesseractVisionAdapter()
