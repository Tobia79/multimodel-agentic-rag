"""Detect bibliography / reference sections from header metadata."""

from __future__ import annotations

import re
from typing import Any, Dict

REFERENCE_HEADER_PATTERN = re.compile(
    r"(?i)"
    r"(?:"
    r"参考文献|参考书目|引用文献|文献目录|附录|"
    r"references?|bibliograph(?:y|ies)|works?\s+cited|citations?"
    r")",
)

_HEADER_KEYS = ("H1", "H2", "H3")


def is_reference_section(metadata: Dict[str, Any] | None) -> bool:
    if not metadata:
        return False

    for key in _HEADER_KEYS:
        value = metadata.get(key)
        if not value:
            continue
        if REFERENCE_HEADER_PATTERN.search(str(value)):
            return True
    return False
