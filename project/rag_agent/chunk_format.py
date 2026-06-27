"""Format retrieved chunks for agent tool responses (presentation layer only)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def build_section_hierarchy(metadata: Optional[Dict[str, Any]]) -> str:
    """Build 'H1 > H2 > H3' section path from chunk metadata."""
    if not metadata:
        return ""

    parts: List[str] = []
    for key in ("H1", "H2", "H3"):
        raw = metadata.get(key)
        if not raw:
            continue
        for segment in str(raw).split(" -> "):
            segment = segment.strip()
            if segment and (not parts or parts[-1] != segment):
                parts.append(segment)

    return " > ".join(parts)


def _format_tags(tags: Any) -> str:
    if not tags:
        return ""
    if isinstance(tags, list):
        return ", ".join(str(tag).strip() for tag in tags if str(tag).strip())
    return str(tags).strip()


def format_parent_chunk_for_agent(
    parent_id: str,
    content: str,
    metadata: Optional[Dict[str, Any]],
    *,
    include_trace_fields: bool = True,
) -> str:
    """Format a parent chunk with structured metadata prefix for LLM consumption."""
    meta = metadata or {}
    body = (content or "").strip()

    lines: List[str] = []
    if include_trace_fields:
        lines.append(f"Parent ID: {meta.get('parent_id') or parent_id}")
        lines.append(f"File Name: {meta.get('source', 'unknown')}")

    section = build_section_hierarchy(meta)
    if section:
        lines.append(f"Section: {section}")

    title = str(meta.get("title") or "").strip()
    if title:
        lines.append(f"Title: {title}")

    summary = str(meta.get("summary") or "").strip()
    if summary:
        lines.append(f"Summary: {summary}")

    tag_str = _format_tags(meta.get("tags"))
    if tag_str:
        lines.append(f"Tags: {tag_str}")

    lines.append("---")
    lines.append("Content:")
    lines.append(body)
    return "\n".join(lines)
