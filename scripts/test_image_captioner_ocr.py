"""Unit-style smoke test for OCR → VLM ImageCaptioner pipeline (OCR-only path)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# project/ on path
PROJECT_DIR = Path(__file__).resolve().parents[1] / "project"
sys.path.insert(0, str(PROJECT_DIR))

os.environ.setdefault("OCR_ENABLED", "true")
os.environ.setdefault("OCR_PROVIDER", "paddle")
os.environ.setdefault("IMAGE_UNDERSTANDING_MODE", "ocr_only")
os.environ.setdefault("VISION_LLM_ENABLED", "false")

import config  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from ingestion.transform.image_captioner import ImageCaptioner  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402


def main() -> int:
    fixture = Path(__file__).resolve().parent / "fixtures" / "ocr_test.png"
    fixture.parent.mkdir(parents=True, exist_ok=True)
    if not fixture.exists():
        img = Image.new("RGB", (640, 200), color="white")
        draw = ImageDraw.Draw(img)
        for i, line in enumerate(["PaddleOCR Pipeline Test", "Hello World OCR"]):
            draw.text((40, 30 + i * 50), line, fill="black")
        img.save(fixture)

    doc = Document(
        page_content=f"See diagram below.\n[IMAGE: test_img_1]\nEnd.",
        metadata={
            "images": [{"id": "test_img_1", "path": str(fixture.resolve())}],
        },
    )

    captioner = ImageCaptioner()
    result = captioner.transform([doc])[0]
    text = result.page_content
    captions = (result.metadata or {}).get("image_captions", [])

    print("Mode:", config.IMAGE_UNDERSTANDING_MODE)
    print("Output text:\n", text)
    print("Metadata:", captions)

    if "(OCR Text:" in text and captions:
        print("\nImageCaptioner OCR pipeline PASSED")
        return 0

    print("\nImageCaptioner OCR pipeline FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
