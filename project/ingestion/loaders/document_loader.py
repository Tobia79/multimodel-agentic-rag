"""Unified document loader for PDF, Markdown, and Word formats."""

from __future__ import annotations

import hashlib
import io
import logging
import re
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import config
from ingestion.loaders.doc_converter import convert_doc_to_docx

logger = logging.getLogger(__name__)

_DOCX_MEDIA_IMAGE_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp", ".emf", ".wmf",
}

_MARKDOWN_IMAGE_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp", ".svg",
}

_MARKDOWN_IMAGE_PATTERN = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

_PDF_PAGE_SEPARATOR_RE = re.compile(
    r"\n--- end of page\.page_number=(\d+) ---\n?",
    re.IGNORECASE,
)

_DOCX_W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_DOCX_R_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_DOCX_A_NS = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
_DOCX_V_NS = "{urn:schemas-microsoft-com:vml}"
_DOCX_IMAGE_REL_TYPES = {
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image",
    "http://schemas.microsoft.com/office/2007/relationships/hdphoto",
}

try:
    import pymupdf
    import pymupdf4llm

    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from markitdown import MarkItDown

    MARKITDOWN_AVAILABLE = True
except ImportError:
    MARKITDOWN_AVAILABLE = False

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


@dataclass
class LoadedDocument:
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    md_path: Optional[Path] = None
    file_hash: str = ""


class UniversalDocumentLoader:
    SUPPORTED_SUFFIXES = {".pdf", ".md", ".markdown", ".docx", ".doc"}

    def __init__(
        self,
        markdown_dir: Optional[str] = None,
        images_dir: Optional[str] = None,
        extract_images: bool = True,
    ):
        self.markdown_dir = Path(markdown_dir or config.MARKDOWN_DIR)
        self.images_dir = Path(images_dir or config.INGESTION_IMAGES_DIR)
        self.converted_docx_dir = Path(config.INGESTION_CONVERTED_DOCX_DIR)
        self.extract_images = extract_images
        self.markdown_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.converted_docx_dir.mkdir(parents=True, exist_ok=True)
        self._markitdown = MarkItDown() if MARKITDOWN_AVAILABLE else None

    @staticmethod
    def compute_file_hash(file_path: str | Path) -> str:
        digest = hashlib.sha256()
        with open(file_path, "rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def load(self, file_path: str | Path, file_hash: Optional[str] = None) -> LoadedDocument:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported file type: {suffix}")

        if file_hash is None:
            file_hash = self.compute_file_hash(path)
        metadata: Dict[str, Any] = {
            "source_path": str(path.resolve()),
            "doc_hash": file_hash,
            "doc_type": suffix.lstrip("."),
        }

        if suffix in {".md", ".markdown"}:
            text, md_path = self._load_markdown(path, file_hash, metadata)
        elif suffix == ".pdf":
            text = self._load_pdf(path, file_hash, metadata)
            md_path = self._persist_markdown(path.stem, text)
        else:
            text = self._load_office(path, file_hash, metadata)
            md_path = self._persist_markdown(path.stem, text)

        title = self._extract_title(text)
        if title:
            metadata["title"] = title

        return LoadedDocument(
            text=text,
            metadata=metadata,
            md_path=md_path,
            file_hash=file_hash,
        )

    def _load_markdown(
        self,
        path: Path,
        file_hash: str,
        metadata: Dict[str, Any],
    ) -> tuple[str, Path]:
        text = path.read_text(encoding="utf-8")
        if self.extract_images:
            try:
                text, images = self._extract_markdown_images(path, file_hash, text)
                if images:
                    metadata["images"] = images
            except Exception as exc:
                logger.warning("Markdown image extraction failed for %s: %s", path, exc)
        md_path = self._persist_markdown(path.stem, text, overwrite=True)
        return text, md_path

    @staticmethod
    def _is_remote_image_ref(image_ref: str) -> bool:
        ref = image_ref.strip().strip('"').strip("'")
        lowered = ref.lower()
        return lowered.startswith(("http://", "https://", "data:", "//"))

    @staticmethod
    def _normalize_image_extension(path: Path) -> str:
        ext = path.suffix.lstrip(".").lower()
        if ext == "jpeg":
            return "jpg"
        return ext or "png"

    def _store_image(
        self,
        *,
        doc_hash: str,
        image_index: int,
        image_bytes: bytes,
        image_ext: str,
        page: int,
        id_suffix: str,
        source_ref: Optional[str] = None,
    ) -> Dict[str, Any]:
        image_dir = self._image_output_dir(doc_hash)
        image_id = f"{doc_hash[:8]}_{id_suffix}_{image_index}"
        image_path = image_dir / f"{image_id}.{image_ext}"
        image_path.write_bytes(image_bytes)
        width, height = self._image_dimensions(image_bytes)
        meta: Dict[str, Any] = {
            "id": image_id,
            "path": str(image_path.resolve()),
            "page": page,
            "position": {"width": width, "height": height, "page": page},
        }
        if source_ref:
            meta["source_ref"] = source_ref
        return meta

    def _resolve_markdown_image_path(self, source_path: Path, image_ref: str) -> Optional[Path]:
        ref = image_ref.strip().strip('"').strip("'")
        if not ref or self._is_remote_image_ref(ref):
            return None

        candidate = Path(ref)
        if candidate.is_absolute():
            resolved = candidate
        else:
            resolved = (source_path.parent / candidate).resolve()

        if not resolved.is_file():
            return None
        if resolved.suffix.lower() not in _MARKDOWN_IMAGE_SUFFIXES:
            logger.warning("Unsupported markdown image type: %s", resolved)
            return None
        return resolved

    def _extract_markdown_images(
        self,
        source_path: Path,
        doc_hash: str,
        base_text: str,
    ) -> tuple[str, List[Dict[str, Any]]]:
        images_metadata: List[Dict[str, Any]] = []
        seen_paths: Dict[str, str] = {}

        def replace_match(match: re.Match[str]) -> str:
            image_ref = match.group(2)
            if self._is_remote_image_ref(image_ref):
                logger.info(
                    "Skipping remote markdown image in %s: %s",
                    source_path.name,
                    image_ref.strip(),
                )
                return match.group(0)

            resolved = self._resolve_markdown_image_path(source_path, image_ref)
            if resolved is None:
                logger.warning(
                    "Markdown image not found for %s: %s",
                    source_path.name,
                    image_ref.strip(),
                )
                return match.group(0)

            cache_key = str(resolved)
            if cache_key in seen_paths:
                return f"[IMAGE: {seen_paths[cache_key]}]"

            try:
                image_bytes = resolved.read_bytes()
                image_ext = self._normalize_image_extension(resolved)
                image_index = len(images_metadata) + 1
                meta = self._store_image(
                    doc_hash=doc_hash,
                    image_index=image_index,
                    image_bytes=image_bytes,
                    image_ext=image_ext,
                    page=image_index,
                    id_suffix="md",
                    source_ref=image_ref.strip(),
                )
                images_metadata.append(meta)
                seen_paths[cache_key] = meta["id"]
                return f"[IMAGE: {meta['id']}]"
            except Exception as exc:
                logger.warning(
                    "Failed to import markdown image %s from %s: %s",
                    image_ref.strip(),
                    source_path.name,
                    exc,
                )
                return match.group(0)

        modified_text = _MARKDOWN_IMAGE_PATTERN.sub(replace_match, base_text)
        return modified_text, images_metadata

    def _persist_markdown(self, stem: str, text: str, overwrite: bool = False) -> Path:
        md_path = self.markdown_dir / f"{stem}.md"
        if overwrite or not md_path.exists():
            cleaned = text.encode("utf-8", errors="surrogatepass").decode("utf-8", errors="ignore")
            md_path.write_text(cleaned, encoding="utf-8")
        return md_path

    def _load_pdf(self, path: Path, file_hash: str, metadata: Dict[str, Any]) -> str:
        if not PYMUPDF_AVAILABLE:
            raise RuntimeError("pymupdf4llm is required for PDF loading")

        doc = pymupdf.open(path)
        text = pymupdf4llm.to_markdown(
            doc,
            header=False,
            footer=False,
            page_separators=True,
            ignore_images=True,
            write_images=False,
            image_path=None,
        )

        from ingestion.loaders.scanned_pdf import (
            ScannedPdfProcessor,
            detect_scanned_pdf,
            is_scanned_pdf_processing_enabled,
        )

        scanned_handled = False
        if is_scanned_pdf_processing_enabled() and detect_scanned_pdf(doc, text):
            try:
                processor = ScannedPdfProcessor(
                    doc_hash=file_hash,
                    store_image=self._store_image,
                )
                text, images = processor.process(doc)
                if images:
                    metadata["images"] = images
                metadata["pdf_processing"] = "scanned_page_ocr_vlm"
                scanned_handled = True
                logger.info(
                    "Processed scanned PDF %s: %d pages, %d page images",
                    path.name,
                    len(doc),
                    len(images),
                )
            except Exception as exc:
                logger.warning(
                    "Scanned PDF processing failed for %s, falling back to standard load: %s",
                    path,
                    exc,
                )
                metadata["pdf_processing"] = "scanned_fallback"

        if not scanned_handled and self.extract_images:
            try:
                text, images = self._extract_pdf_images(doc, file_hash, text)
                if images:
                    metadata["images"] = images
            except Exception as exc:
                logger.warning("Image extraction failed for %s: %s", path, exc)

        doc.close()
        return text

    def _image_output_dir(self, doc_hash: str) -> Path:
        image_dir = self.images_dir / doc_hash[:16]
        image_dir.mkdir(parents=True, exist_ok=True)
        return image_dir

    @staticmethod
    def _image_dimensions(image_bytes: bytes) -> tuple[int, int]:
        if not PIL_AVAILABLE:
            return 0, 0
        try:
            img = Image.open(io.BytesIO(image_bytes))
            return img.size
        except Exception:
            return 0, 0

    @staticmethod
    def _split_pdf_markdown_pages(text: str) -> Dict[int, str]:
        matches = list(_PDF_PAGE_SEPARATOR_RE.finditer(text))
        if not matches:
            return {1: text}

        pages: Dict[int, str] = {}
        start = 0
        for match in matches:
            page_num = int(match.group(1))
            pages[page_num] = text[start:match.start()]
            start = match.end()
        pages[int(matches[-1].group(1)) + 1] = text[start:]
        return pages

    @staticmethod
    def _join_pdf_markdown_pages(pages: Dict[int, str]) -> str:
        if not pages:
            return ""
        ordered = sorted(pages)
        chunks: List[str] = []
        for index, page_num in enumerate(ordered):
            chunks.append(pages[page_num].rstrip("\n"))
            if index < len(ordered) - 1:
                chunks.append(f"\n--- end of page.page_number={page_num} ---\n")
        return "".join(chunks)

    @staticmethod
    def _page_text_blocks(page) -> List[tuple[float, float, str]]:
        blocks: List[tuple[float, float, str]] = []
        for block in page.get_text("blocks") or []:
            if len(block) < 7 or block[6] != 0:
                continue
            text = (block[4] or "").strip()
            if not text:
                continue
            blocks.append((float(block[1]), float(block[3]), text))
        return blocks

    @staticmethod
    def _anchor_text_above(
        text_blocks: List[tuple[float, float, str]],
        image_y0: float,
        margin: float = 8.0,
    ) -> Optional[str]:
        best_text: Optional[str] = None
        best_y1 = -1.0
        for _y0, y1, text in text_blocks:
            if y1 <= image_y0 + margin and y1 > best_y1:
                best_text = text
                best_y1 = y1
        if not best_text:
            return None
        lines = [line.strip() for line in best_text.splitlines() if line.strip()]
        return lines[-1] if lines else None

    @staticmethod
    def _anchor_match_candidates(anchor: str) -> List[str]:
        candidates = [anchor]
        compact = " ".join(anchor.split())
        if compact and compact not in candidates:
            candidates.append(compact)
        if len(anchor) > 120:
            tail = anchor[-120:]
            candidates.append(tail)
            candidates.append(" ".join(tail.split()))
        return candidates

    @staticmethod
    def _find_anchor_position(
        text: str,
        anchor: str,
        start: int = 0,
    ) -> Optional[Tuple[int, int]]:
        if not anchor:
            return None
        for candidate in UniversalDocumentLoader._anchor_match_candidates(anchor):
            if not candidate:
                continue
            index = text.find(candidate, start)
            if index >= 0:
                return index, len(candidate)
        return None

    @staticmethod
    def _insert_placeholder_after_anchor(
        text: str,
        anchor: Optional[str],
        placeholder: str,
    ) -> str:
        line = placeholder if placeholder.endswith("\n") else f"{placeholder}\n"
        if not anchor:
            trimmed = text.rstrip()
            return f"{trimmed}\n{line}" if trimmed else line

        match = UniversalDocumentLoader._find_anchor_position(text, anchor)
        if match is not None:
            index, length = match
            insert_at = index + length
            return text[:insert_at] + f"\n{line}" + text[insert_at:]

        trimmed = text.rstrip()
        return f"{trimmed}\n{line}" if trimmed else line

    @staticmethod
    def _insert_placeholders_sequential(
        text: str,
        placements: List[Tuple[str, str]],
    ) -> str:
        """Insert placeholders in document order without disturbing later anchors."""
        if not placements:
            return text

        inserts: List[Tuple[int, str]] = []
        cursor = 0
        trimmed_end = len(text.rstrip())

        for anchor, placeholder in placements:
            line = placeholder if placeholder.endswith("\n") else f"{placeholder}\n"
            match = None
            if anchor:
                match = UniversalDocumentLoader._find_anchor_position(text, anchor, cursor)
            if match is None:
                inserts.append((trimmed_end, line))
                continue

            index, length = match
            insert_at = index + length
            inserts.append((insert_at, line))
            cursor = insert_at

        modified = text
        for insert_at, line in sorted(inserts, key=lambda item: item[0], reverse=True):
            modified = modified[:insert_at] + f"\n{line}" + modified[insert_at:]
        return modified

    def _extract_pdf_images(
        self,
        doc,
        doc_hash: str,
        base_text: str,
    ) -> tuple[str, List[Dict[str, Any]]]:
        images_metadata: List[Dict[str, Any]] = []
        pages = self._split_pdf_markdown_pages(base_text)
        xref_meta: Dict[int, Dict[str, Any]] = {}
        next_image_index = 1

        for page_num in range(len(doc)):
            page = doc[page_num]
            page_key = page_num + 1
            page_text = pages.get(page_key, "")
            text_blocks = self._page_text_blocks(page)
            placements: List[tuple[float, str]] = []

            for img_index, img_info in enumerate(page.get_images(full=True)):
                xref = img_info[0]
                try:
                    rects = list(page.get_image_rects(xref))
                except Exception:
                    rects = []

                if xref not in xref_meta:
                    try:
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                    except Exception as exc:
                        logger.warning(
                            "Failed to extract PDF image xref %s on page %s: %s",
                            xref,
                            page_key,
                            exc,
                        )
                        continue

                    meta = self._store_image(
                        doc_hash=doc_hash,
                        image_index=next_image_index,
                        image_bytes=image_bytes,
                        image_ext=image_ext,
                        page=page_key,
                        id_suffix=str(page_key),
                    )
                    xref_meta[xref] = meta
                    images_metadata.append(meta)
                    next_image_index += 1

                placeholder = f"[IMAGE: {xref_meta[xref]['id']}]"
                if rects:
                    for rect in rects:
                        placements.append((float(rect.y0), placeholder))
                else:
                    placements.append((float("inf"), placeholder))

            placements.sort(key=lambda item: item[0])
            for _y0, placeholder in reversed(placements):
                anchor = None
                if _y0 != float("inf"):
                    anchor = self._anchor_text_above(text_blocks, _y0)
                page_text = self._insert_placeholder_after_anchor(
                    page_text,
                    anchor,
                    placeholder,
                )

            pages[page_key] = page_text

        return self._join_pdf_markdown_pages(pages), images_metadata

    @staticmethod
    def _resolve_docx_rel_target(target: str) -> str:
        normalized = target.replace("\\", "/").lstrip("/")
        if normalized.startswith("word/"):
            return normalized
        return f"word/{normalized}"

    @classmethod
    def _load_docx_image_relationships(cls, archive: zipfile.ZipFile) -> Dict[str, str]:
        rels_path = "word/_rels/document.xml.rels"
        if rels_path not in archive.namelist():
            return {}

        root = ET.fromstring(archive.read(rels_path))
        relationships: Dict[str, str] = {}
        for rel in root:
            rel_type = rel.attrib.get("Type", "")
            rel_id = rel.attrib.get("Id", "")
            target = rel.attrib.get("Target", "")
            if not rel_id or not target:
                continue
            if rel_type not in _DOCX_IMAGE_REL_TYPES:
                continue
            media_path = cls._resolve_docx_rel_target(target)
            if Path(media_path).suffix.lower() in _DOCX_MEDIA_IMAGE_SUFFIXES:
                relationships[rel_id] = media_path
        return relationships

    @staticmethod
    def _extract_docx_paragraph_text(paragraph: ET.Element) -> str:
        parts: List[str] = []
        for elem in paragraph.iter():
            if elem.tag == f"{_DOCX_W_NS}t":
                if elem.text:
                    parts.append(elem.text)
            elif elem.tag == f"{_DOCX_W_NS}tab":
                parts.append("\t")
            elif elem.tag == f"{_DOCX_W_NS}br":
                parts.append("\n")
        return "".join(parts).strip()

    @staticmethod
    def _docx_paragraph_anchor_text(paragraph_text: str) -> Optional[str]:
        if not paragraph_text:
            return None
        lines = [line.strip() for line in paragraph_text.splitlines() if line.strip()]
        return lines[-1] if lines else paragraph_text

    @staticmethod
    def _find_docx_image_rel_ids(paragraph: ET.Element) -> List[str]:
        rel_ids: List[str] = []
        seen: set[str] = set()
        for blip in paragraph.iter(f"{_DOCX_A_NS}blip"):
            rel_id = blip.attrib.get(f"{_DOCX_R_NS}embed")
            if rel_id and rel_id not in seen:
                seen.add(rel_id)
                rel_ids.append(rel_id)
        for imagedata in paragraph.iter(f"{_DOCX_V_NS}imagedata"):
            rel_id = imagedata.attrib.get(f"{_DOCX_R_NS}id")
            if rel_id and rel_id not in seen:
                seen.add(rel_id)
                rel_ids.append(rel_id)
        return rel_ids

    def _walk_docx_body_blocks(self, element: ET.Element) -> List[Tuple[str, List[str]]]:
        blocks: List[Tuple[str, List[str]]] = []
        tag = element.tag

        if tag == f"{_DOCX_W_NS}p":
            paragraph_text = self._extract_docx_paragraph_text(element)
            rel_ids = self._find_docx_image_rel_ids(element)
            if paragraph_text or rel_ids:
                blocks.append((paragraph_text, rel_ids))
            return blocks

        if tag == f"{_DOCX_W_NS}tbl":
            for row in element.findall(f".//{_DOCX_W_NS}tr"):
                for cell in row.findall(f"{_DOCX_W_NS}tc"):
                    for child in cell:
                        blocks.extend(self._walk_docx_body_blocks(child))
            return blocks

        if tag == f"{_DOCX_W_NS}sdt":
            content = element.find(f"{_DOCX_W_NS}sdtContent")
            if content is not None:
                for child in content:
                    blocks.extend(self._walk_docx_body_blocks(child))
            return blocks

        for child in element:
            blocks.extend(self._walk_docx_body_blocks(child))
        return blocks

    @classmethod
    def _collect_docx_image_placements(
        cls,
        archive: zipfile.ZipFile,
    ) -> List[Tuple[str, str, str]]:
        """Return ordered (anchor_text, media_path, rel_id) tuples from document body."""
        document_path = "word/document.xml"
        if document_path not in archive.namelist():
            return []

        relationships = cls._load_docx_image_relationships(archive)
        if not relationships:
            return []

        root = ET.fromstring(archive.read(document_path))
        body = root.find(f"{_DOCX_W_NS}body")
        if body is None:
            return []

        loader = cls.__new__(cls)
        paragraph_blocks = loader._walk_docx_body_blocks(body)
        placements: List[Tuple[str, str, str]] = []
        previous_anchor = ""

        for paragraph_text, rel_ids in paragraph_blocks:
            anchor = cls._docx_paragraph_anchor_text(paragraph_text) or previous_anchor
            if paragraph_text:
                paragraph_anchor = cls._docx_paragraph_anchor_text(paragraph_text)
                if paragraph_anchor:
                    previous_anchor = paragraph_anchor

            for rel_id in rel_ids:
                media_path = relationships.get(rel_id)
                if not media_path:
                    continue
                placements.append((anchor, media_path, rel_id))

        return placements

    def _extract_docx_images(
        self,
        path: Path,
        doc_hash: str,
        base_text: str,
    ) -> tuple[str, List[Dict[str, Any]]]:
        images_metadata: List[Dict[str, Any]] = []
        modified_text = base_text
        inline_placements: List[Tuple[str, str]] = []
        media_meta: Dict[str, Dict[str, Any]] = {}
        next_image_index = 1

        with zipfile.ZipFile(path) as archive:
            docx_placements = self._collect_docx_image_placements(archive)
            if not docx_placements:
                return modified_text, images_metadata

            for anchor, media_path, _rel_id in docx_placements:
                if media_path not in media_meta:
                    if media_path not in archive.namelist():
                        logger.warning(
                            "DOCX image path missing in archive %s: %s",
                            path.name,
                            media_path,
                        )
                        continue
                    try:
                        image_bytes = archive.read(media_path)
                        image_ext = Path(media_path).suffix.lstrip(".").lower() or "png"
                        if image_ext == "jpeg":
                            image_ext = "jpg"
                        meta = self._store_image(
                            doc_hash=doc_hash,
                            image_index=next_image_index,
                            image_bytes=image_bytes,
                            image_ext=image_ext,
                            page=next_image_index,
                            id_suffix="docx",
                            source_ref=media_path,
                        )
                    except Exception as exc:
                        logger.warning(
                            "Failed to extract DOCX image %s from %s: %s",
                            media_path,
                            path,
                            exc,
                        )
                        continue

                    media_meta[media_path] = meta
                    images_metadata.append(meta)
                    next_image_index += 1

                placeholder = f"[IMAGE: {media_meta[media_path]['id']}]"
                inline_placements.append((anchor, placeholder))

        if inline_placements:
            modified_text = self._insert_placeholders_sequential(
                modified_text,
                inline_placements,
            )

        return modified_text, images_metadata

    def _resolve_docx_source_for_office(
        self,
        path: Path,
        file_hash: str,
        metadata: Dict[str, Any],
    ) -> Path:
        if path.suffix.lower() != ".doc":
            return path
        if not self.extract_images or not config.INGESTION_DOC_CONVERT_TO_DOCX:
            return path

        output_dir = self.converted_docx_dir / file_hash[:16]
        converted = convert_doc_to_docx(path, output_dir)
        if converted is None:
            metadata["doc_converted_to_docx"] = False
            return path

        metadata["doc_converted_to_docx"] = True
        metadata["converted_docx_path"] = str(converted.resolve())
        return converted

    def _load_office(self, path: Path, file_hash: str, metadata: Dict[str, Any]) -> str:
        if not MARKITDOWN_AVAILABLE:
            raise RuntimeError(
                "markitdown is required for Word documents. Install with: pip install markitdown"
            )

        office_path = self._resolve_docx_source_for_office(path, file_hash, metadata)
        result = self._markitdown.convert(str(office_path))
        text = result.text_content if hasattr(result, "text_content") else str(result)

        if self.extract_images and office_path.suffix.lower() == ".docx":
            try:
                text, images = self._extract_docx_images(office_path, file_hash, text)
                if images:
                    metadata["images"] = images
            except Exception as exc:
                logger.warning("DOCX image extraction failed for %s: %s", office_path, exc)
        elif self.extract_images and path.suffix.lower() == ".doc":
            logger.info(
                "Legacy .doc %s was processed as text only because DOCX conversion was unavailable",
                path.name,
            )

        return text

    @staticmethod
    def _extract_title(text: str) -> Optional[str]:
        for line in text.splitlines()[:20]:
            stripped = line.strip()
            if stripped.startswith("# "):
                return stripped[2:].strip()
        for line in text.splitlines()[:10]:
            stripped = line.strip()
            if stripped:
                return stripped
        return None

    def copy_markdown_file(
        self, source_path: Path, file_hash: Optional[str] = None
    ) -> LoadedDocument:
        if file_hash is None:
            file_hash = self.compute_file_hash(source_path)
        metadata: Dict[str, Any] = {
            "source_path": str(source_path.resolve()),
            "doc_hash": file_hash,
            "doc_type": "md",
        }
        text, md_path = self._load_markdown(source_path, file_hash, metadata)
        title = self._extract_title(text)
        if title:
            metadata["title"] = title
        return LoadedDocument(
            text=text,
            metadata=metadata,
            md_path=md_path,
            file_hash=file_hash,
        )
