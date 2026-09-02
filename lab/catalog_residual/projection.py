from __future__ import annotations

"""Maximal protocol-valid projection Pi_tilde.

After a client-side prefix rewrite, thinking blocks are bound to the
old conversation and must not be replayed. This projection drops
``thinking`` and ``redacted_thinking`` from carried assistant turns,
keeps text and protocol-valid ``tool_use``, and refuses unknown
provider state. It does not read hidden reasoning text.
"""

import copy
from collections import Counter
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence, Tuple

from catalog_residual.blocks import (
    BLOCK_TEXT,
    BLOCK_TOOL_USE,
    BLOCK_UNKNOWN,
    BOUND_BLOCK_TYPES,
    ContentBlock,
    parse_content,
)


@dataclass(frozen=True)
class ProjectionResult:
    messages: Tuple[dict, ...]
    blocks_dropped: int
    dropped_types: Mapping[str, int]
    bound_thinking_after_prefix: bool
    tool_round_valid: bool
    tool_round_resolved: bool
    wire_protocol_valid: bool
    projection_complete: bool
    projection_approved: bool
    projected_request_valid: bool


def project_transcript(messages: Optional[Sequence[Any]]) -> ProjectionResult:
    if not isinstance(messages, (list, tuple)) or not messages:
        return _closed_empty()

    dropped: Counter[str] = Counter()
    unknown_seen = False
    projected: list[dict] = []

    for raw in messages:
        if not isinstance(raw, dict):
            unknown_seen = True
            continue
        role = str(raw.get("role") or "")
        message = {
            key: copy.deepcopy(value)
            for key, value in raw.items()
            if key != "content"
        }
        content = raw.get("content")
        if role != "assistant":
            message["content"] = copy.deepcopy(content)
            projected.append(message)
            continue
        if isinstance(content, str):
            message["content"] = content
            projected.append(message)
            continue

        kept: list[dict] = []
        for block in parse_content(content):
            wire, drop_type, unknown = _project_assistant_block(block)
            if drop_type:
                dropped[drop_type] += 1
            if unknown:
                unknown_seen = True
            if wire is not None:
                kept.append(wire)

        if kept:
            message["content"] = kept
        elif message.get("tool_calls"):
            message["content"] = ""
        else:
            continue
        projected.append(message)

    bound = _bound_thinking_present(projected)
    tool_round_valid, tool_round_resolved = _tool_round_flags(projected)
    dropped_types = MappingProxyType({key: int(value) for key, value in dropped.items()})
    wire_protocol_valid = (
        (not bound)
        and tool_round_resolved
        and (not _unknown_remaining(projected))
    )
    projection_complete = not unknown_seen
    projection_approved = bool(wire_protocol_valid and projection_complete)
    return ProjectionResult(
        messages=tuple(projected),
        blocks_dropped=int(sum(dropped.values())),
        dropped_types=dropped_types,
        bound_thinking_after_prefix=bound,
        tool_round_valid=tool_round_valid,
        tool_round_resolved=tool_round_resolved,
        wire_protocol_valid=wire_protocol_valid,
        projection_complete=projection_complete,
        projection_approved=projection_approved,
        projected_request_valid=projection_approved,
    )


def main(argv: Optional[list[str]] = None) -> int:
    from catalog_residual.projection_lab import main as lab_main

    return lab_main(argv)


def _closed_empty() -> ProjectionResult:
    return ProjectionResult(
        messages=(),
        blocks_dropped=0,
        dropped_types=MappingProxyType({}),
        bound_thinking_after_prefix=False,
        tool_round_valid=False,
        tool_round_resolved=False,
        wire_protocol_valid=False,
        projection_complete=False,
        projection_approved=False,
        projected_request_valid=False,
    )


def _project_assistant_block(
    block: ContentBlock,
) -> Tuple[Optional[dict], str, bool]:
    if block.type in BOUND_BLOCK_TYPES:
        return None, block.type, False
    if block.type == BLOCK_UNKNOWN:
        return None, BLOCK_UNKNOWN, True
    if block.type == BLOCK_TEXT:
        return (
            {"type": BLOCK_TEXT, "text": str(block.fields.get("text") or "")},
            "",
            False,
        )
    if block.type == BLOCK_TOOL_USE:
        tool_id = str(block.fields.get("id") or "").strip()
        name = str(block.fields.get("name") or "").strip()
        if not tool_id or not name:
            return None, BLOCK_TOOL_USE, True
        wire = {
            "type": BLOCK_TOOL_USE,
            "id": tool_id,
            "name": name,
            "input": copy.deepcopy(block.fields.get("input") or {}),
        }
        return wire, "", False
    return None, block.type or BLOCK_UNKNOWN, True


def _bound_thinking_present(messages: Sequence[dict]) -> bool:
    for message in messages:
        if str(message.get("role") or "") != "assistant":
            continue
        for block in parse_content(message.get("content")):
            if block.type in BOUND_BLOCK_TYPES:
                return True
    return False


def _unknown_remaining(messages: Sequence[dict]) -> bool:
    for message in messages:
        if str(message.get("role") or "") != "assistant":
            continue
        for block in parse_content(message.get("content")):
            if block.type == BLOCK_UNKNOWN:
                return True
    return False


def _assistant_tool_ids(message: Mapping[str, Any]) -> list[str]:
    ids: list[str] = []
    for block in parse_content(message.get("content")):
        if block.type != BLOCK_TOOL_USE:
            continue
        tool_id = str(block.fields.get("id") or "").strip()
        if tool_id:
            ids.append(tool_id)
    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        tool_id = str(call.get("id") or "").strip()
        if tool_id:
            ids.append(tool_id)
    return ids


def _tool_round_flags(messages: Sequence[dict]) -> Tuple[bool, bool]:
    pending: list[str] = []
    intact = True
    for message in messages:
        if not isinstance(message, dict):
            intact = False
            continue
        role = str(message.get("role") or "")
        if role == "assistant":
            if pending:
                intact = False
            pending = _assistant_tool_ids(message)
        elif role == "tool":
            tool_id = str(message.get("tool_call_id") or "").strip()
            if not pending or tool_id not in pending:
                intact = False
            elif tool_id:
                pending = [item for item in pending if item != tool_id]
        elif role in ("user", "system") and pending:
            intact = False
            pending = []
    return intact, bool(intact and not pending)


if __name__ == "__main__":
    raise SystemExit(main())
