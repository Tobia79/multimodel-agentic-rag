"""Smoke test: synthetic scanned PDF → Load stage page OCR."""

from __future__ import annotations

import io
import os
import sys
import tempfile
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1] / "project"
sys.path.insert(0, str(PROJECT_DIR))

os.environ.setdefault("OCR_ENABLED", "true")
os.environ.setdefault("OCR_PROVIDER", "paddle")
os.environ.setdefault("IMAGE_UNDERSTANDING_MODE", "ocr_only")
os.environ.setdefault("VISION_LLM_ENABLED", "false")
os.environ.setdefault("INGESTION_PDF_SCAN_OCR", "true")
os.environ.setdefault("INGESTION_PDF_SCAN_MODE", "auto")

import pymupdf  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from ingestion.loaders.document_loader import UniversalDocumentLoader  # noqa: E402
from ingestion.loaders.scanned_pdf import detect_scanned_pdf  # noqa: E402


def create_scanned_like_pdf(path: Path) -> None:
    """PDF with image-only pages (no text layer)."""
    img = Image.new("RGB", (600, 300), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((40, 40), "Scanned PDF Page Test", fill="black")
    draw.text((40, 100), "Line two for OCR verification", fill="black")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")

    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_image(page.rect, stream=buffer.getvalue())
    doc.save(path)
    doc.close()


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "scan_test.pdf"
        create_scanned_like_pdf(pdf_path)

        doc = pymupdf.open(pdf_path)
        is_scanned = detect_scanned_pdf(doc, "")
        doc.close()
        print("detect_scanned_pdf:", is_scanned)
        if not is_scanned:
            print("FAILED: PDF not detected as scanned")
            return 1

        loader = UniversalDocumentLoader(
            markdown_dir=str(Path(tmp) / "md"),
            images_dir=str(Path(tmp) / "images"),
        )
        loaded = loader.load(pdf_path)
        print("pdf_processing:", loaded.metadata.get("pdf_processing"))
        print("image_count:", len(loaded.metadata.get("images", [])))
        print("text preview:\n", loaded.text[:500])

        if loaded.metadata.get("pdf_processing") != "scanned_page_ocr_vlm":
            print("FAILED: expected scanned_page_ocr_vlm processing")
            return 1
        if "Scanned PDF" not in loaded.text and "OCR" not in loaded.text.upper():
            print("FAILED: OCR text not found in loaded markdown")
            return 1
        if "[IMAGE:" not in loaded.text:
            print("FAILED: page image placeholder missing")
            return 1

        print("\nScanned PDF load pipeline PASSED")
        return 0


if __name__ == "__main__":
    sys.exit(main())
