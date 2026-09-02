from __future__ import annotations

"""Typed content blocks for lab transcripts.

Classification uses the declared ``type`` (or a string body). Hidden
reasoning text is never read to decide a block's kind.
"""

import copy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Tuple

BLOCK_TEXT = "text"
BLOCK_THINKING = "thinking"
BLOCK_REDACTED_THINKING = "redacted_thinking"
BLOCK_TOOL_USE = "tool_use"
BLOCK_UNKNOWN = "unknown"

BOUND_BLOCK_TYPES = frozenset({BLOCK_THINKING, BLOCK_REDACTED_THINKING})

_DECLARED_TYPES = frozenset({
    BLOCK_TEXT,
    BLOCK_THINKING,
    BLOCK_REDACTED_THINKING,
    BLOCK_TOOL_USE,
})


def _frozen_fields(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({
        key: copy.deepcopy(value) for key, value in payload.items()
    })


@dataclass(frozen=True)
class ContentBlock:
    type: str
    provenance: str
    fields: Mapping[str, Any]


def parse_content(content: Any) -> Tuple[ContentBlock, ...]:
    """Turn message content into typed blocks. Unknown shapes fail closed."""
    if content is None:
        return ()
    if isinstance(content, str):
        return (
            ContentBlock(
                BLOCK_TEXT,
                "string",
                _frozen_fields({"text": content}),
            ),
        )
    if isinstance(content, (list, tuple)):
        return tuple(_parse_item(item) for item in content)
    return (
        ContentBlock(
            BLOCK_UNKNOWN,
            "unknown",
            _frozen_fields({"raw": repr(content)}),
        ),
    )


def _parse_item(item: Any) -> ContentBlock:
    if isinstance(item, str):
        return ContentBlock(
            BLOCK_TEXT,
            "string",
            _frozen_fields({"text": item}),
        )
    if not isinstance(item, dict):
        return ContentBlock(
            BLOCK_UNKNOWN,
            "unknown",
            _frozen_fields({"raw": repr(item)}),
        )
    declared = item.get("type")
    if declared in _DECLARED_TYPES:
        fields = {key: value for key, value in item.items() if key != "type"}
        return ContentBlock(str(declared), "typed", _frozen_fields(fields))
    return ContentBlock(
        BLOCK_UNKNOWN,
        "typed" if declared else "unknown",
        _frozen_fields(item),
    )
