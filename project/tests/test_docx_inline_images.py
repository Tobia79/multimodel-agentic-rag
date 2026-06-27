"""Smoke test for DOCX inline image placeholder placement."""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

from ingestion.loaders.document_loader import UniversalDocumentLoader

_PNG_1X1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000a49444154789c63000100000500010d0a2db40000000049454e44ae426082"
)

_CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Default Extension="png" ContentType="image/png"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"""

_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""

_DOC_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId5" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/image1.png"/>
</Relationships>"""

_DOCUMENT = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
 xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
 xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
 xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">
<w:body>
<w:p><w:r><w:t>Introduction paragraph before the figure.</w:t></w:r></w:p>
<w:p><w:r><w:t>Figure 1 shows the architecture.</w:t></w:r>
<w:r><w:drawing><wp:inline><a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:pic><pic:blipFill><a:blip r:embed="rId5"/></pic:blipFill></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r>
</w:p>
<w:p><w:r><w:t>Conclusion after the figure.</w:t></w:r></w:p>
</w:body>
</w:document>"""


def _create_sample_docx(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", _CONTENT_TYPES)
        archive.writestr("_rels/.rels", _RELS)
        archive.writestr("word/_rels/document.xml.rels", _DOC_RELS)
        archive.writestr("word/document.xml", _DOCUMENT)
        archive.writestr("word/media/image1.png", _PNG_1X1)


def test_docx_inline_image_placeholder_near_anchor() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        docx_path = Path(tmp_dir) / "sample.docx"
        _create_sample_docx(docx_path)

        base_text = (
            "Introduction paragraph before the figure.\n\n"
            "Figure 1 shows the architecture.\n\n"
            "Conclusion after the figure."
        )
        loader = UniversalDocumentLoader(images_dir=Path(tmp_dir) / "images")
        text, images = loader._extract_docx_images(
            docx_path,
            "deadbeef" * 8,
            base_text,
        )

        assert len(images) == 1
        assert "[IMAGE:" in text
        figure_pos = text.find("Figure 1")
        image_pos = text.find("[IMAGE:")
        conclusion_pos = text.find("Conclusion")

        assert figure_pos >= 0
        assert image_pos > figure_pos
        assert image_pos < conclusion_pos


if __name__ == "__main__":
    test_docx_inline_image_placeholder_near_anchor()
    print("ok")
