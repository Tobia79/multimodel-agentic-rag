"""Quick PaddleOCR smoke test — generates a test image and runs OCR."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Windows CPU: avoid Paddle 3.3 + oneDNN PIR crash (see PaddleOCR #17539)
os.environ.setdefault("FLAGS_enable_pir_api", "0")
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

from PIL import Image, ImageDraw, ImageFont


def create_test_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (640, 200), color="white")
    draw = ImageDraw.Draw(img)
    lines = [
        "PaddleOCR Test 2026",
        "区块链 人工智能 RAG",
        "Hello World OCR",
    ]
    y = 30
    for line in lines:
        draw.text((40, y), line, fill="black")
        y += 50
    img.save(path)
    print(f"Created test image: {path}")


def run_ocr(image_path: Path) -> list[str]:
    from paddleocr import PaddleOCR

    ocr = PaddleOCR(use_textline_orientation=True, lang="ch", enable_mkldnn=False)
    result = ocr.predict(str(image_path))

    texts: list[str] = []
    if not result:
        return texts

    for item in result:
        if isinstance(item, dict):
            rec_texts = item.get("rec_texts") or item.get("text") or []
            if isinstance(rec_texts, list):
                texts.extend(str(t) for t in rec_texts if t)
            elif rec_texts:
                texts.append(str(rec_texts))
        elif isinstance(item, (list, tuple)):
            for line in item:
                if isinstance(line, (list, tuple)) and len(line) >= 2:
                    text_part = line[1]
                    if isinstance(text_part, (list, tuple)) and text_part:
                        texts.append(str(text_part[0]))
                    elif isinstance(text_part, str):
                        texts.append(text_part)
    return texts


def main() -> int:
    image_path = Path(__file__).resolve().parent / "fixtures" / "ocr_test.png"
    create_test_image(image_path)

    print("Running PaddleOCR (first run may download models)...")
    texts = run_ocr(image_path)

    print("\n--- OCR Result ---")
    if texts:
        for i, t in enumerate(texts, 1):
            print(f"  {i}. {t}")
        print("\nPaddleOCR test PASSED")
        return 0

    print("No text recognized.")
    print("PaddleOCR test FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
