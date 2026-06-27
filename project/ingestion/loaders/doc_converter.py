"""Convert legacy .doc files to .docx for image extraction."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

_WORD_DOCX_FORMAT = 16  # wdFormatXMLDocument


def _libreoffice_candidates() -> List[str]:
    candidates = ["soffice", "libreoffice"]
    if sys.platform == "win32":
        program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
        program_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
        for root in (program_files, program_files_x86):
            candidates.append(str(Path(root) / "LibreOffice" / "program" / "soffice.exe"))
    return candidates


def _convert_with_libreoffice(doc_path: Path, output_dir: Path) -> Optional[Path]:
    expected = output_dir / f"{doc_path.stem}.docx"
    for command in _libreoffice_candidates():
        if Path(command).suffix and not Path(command).exists():
            continue
        try:
            completed = subprocess.run(
                [
                    command,
                    "--headless",
                    "--convert-to",
                    "docx",
                    "--outdir",
                    str(output_dir.resolve()),
                    str(doc_path.resolve()),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=180,
            )
            if completed.returncode != 0:
                logger.debug(
                    "LibreOffice conversion failed via %s for %s: %s",
                    command,
                    doc_path.name,
                    (completed.stderr or completed.stdout or "").strip(),
                )
                continue
            if expected.exists():
                return expected
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
            logger.debug("LibreOffice unavailable via %s: %s", command, exc)
    return None


def _convert_with_ms_word(doc_path: Path, output_path: Path) -> Optional[Path]:
    if sys.platform != "win32":
        return None
    try:
        import win32com.client  # type: ignore
    except ImportError:
        logger.debug("pywin32 is not installed; skipping Microsoft Word conversion")
        return None

    word = None
    document = None
    try:
        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False
        document = word.Documents.Open(str(doc_path.resolve()))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        document.SaveAs(str(output_path.resolve()), FileFormat=_WORD_DOCX_FORMAT)
        if output_path.exists():
            return output_path
    except Exception as exc:
        logger.warning("Microsoft Word conversion failed for %s: %s", doc_path.name, exc)
    finally:
        if document is not None:
            try:
                document.Close(False)
            except Exception:
                pass
        if word is not None:
            try:
                word.Quit()
            except Exception:
                pass
    return None


def convert_doc_to_docx(doc_path: Path, output_dir: Path) -> Optional[Path]:
    """Return path to converted .docx, or None if conversion is unavailable."""
    doc_path = Path(doc_path)
    if doc_path.suffix.lower() != ".doc":
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    expected = output_dir / f"{doc_path.stem}.docx"

    converted = _convert_with_libreoffice(doc_path, output_dir)
    if converted is not None:
        logger.info("Converted %s to DOCX via LibreOffice", doc_path.name)
        return converted

    converted = _convert_with_ms_word(doc_path, expected)
    if converted is not None:
        logger.info("Converted %s to DOCX via Microsoft Word", doc_path.name)
        return converted

    logger.warning(
        "Could not convert %s to DOCX. Install LibreOffice or Microsoft Word, "
        "or manually save the file as .docx before upload.",
        doc_path.name,
    )
    return None
