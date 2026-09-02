from __future__ import annotations

"""Projection experiment: protocol Pi_tilde vs task continuity."""

import copy
import hashlib
import json
from types import SimpleNamespace

from catalog_residual.battery import (
    EXPERIMENTAL_CASES,
    LIVE_HOLDOUT_CASES,
    PROJECTION_CASE_IDS,
    RESIDUAL_CASES,
    THINKING_NEXT_PATH,
    THINKING_ONLY_NONCE,
    THINKING_TOOL_PATH,
    cases_by_id,
    live_cases,
    projection_cases_by_id,
)
from catalog_residual.blocks import (
    BLOCK_REDACTED_THINKING,
    BLOCK_TEXT,
    BLOCK_THINKING,
    BLOCK_TOOL_USE,
    BLOCK_UNKNOWN,
    parse_content,
)
from catalog_residual.live import RECEIPT_SCHEMA as LIVE_SCHEMA, evaluate_live_gates
from catalog_residual.projection import project_transcript
from catalog_residual.projection_lab import (
    EVIDENCE_PI_PC,
    EVIDENCE_PI_TILDE,
    EVIDENCE_PROVIDER_CONTINUITY,
    FABLE_PROTOCOL_MODEL,
    LAB_UNVALIDATED_SIGNATURE,
    OPUS_CONTINUITY_MODEL,
    RECEIPT_SCHEMA,
    REWRITTEN_PREFIX,
    THINKING_BINDING_BETA,
    _build_stdlib_request,
    _has_authentic_signed_bound,
    _strip_bound_payloads,
    build_continuity_request,
    build_fable_protocol_request,
    classify_fable_response,
    fable_provider_validated,
    fable_request_headers,
    flatten_projected_case,
    keep_tail_after_prefix_mutation,
    main,
    planned_checkpoint,
    run_fable_protocol_validation,
    run_hermetic_projection,
    run_provider_continuity,
    run_semantic_continuity,
)


def _assistant(*blocks, **extra):
    row = {"role": "assistant", "content": list(blocks)}
    row.update(extra)
    return row


def test_parse_content_covers_all_declared_and_unknown_types():
    blocks = parse_content([
        {"type": "text", "text": "visible"},
        {"type": "thinking", "thinking": "hidden", "signature": "sig"},
        {"type": "redacted_thinking", "data": "opaque"},
        {"type": "tool_use", "id": "c1", "name": "read_file", "input": {"path": "a.py"}},
        {"type": "server_tool_use", "id": "s1"},
        "bare string",
    ])
    assert [block.type for block in blocks] == [
        BLOCK_TEXT,
        BLOCK_THINKING,
        BLOCK_REDACTED_THINKING,
        BLOCK_TOOL_USE,
        BLOCK_UNKNOWN,
        BLOCK_TEXT,
    ]
    assert blocks[0].provenance == "typed"
    assert blocks[5].provenance == "string"
    assert blocks[1].fields["thinking"] == "hidden"
    unknown = parse_content({"not": "a block list"})
    assert unknown[0].type == BLOCK_UNKNOWN


def test_projection_drops_bound_blocks_keeps_text_and_valid_tool_use():
    original = [
        {"role": "user", "content": "read it"},
        _assistant(
            {"type": "thinking", "thinking": "src/should-not-keep.py", "signature": "s"},
            {"type": "redacted_thinking", "data": "src/also-hidden.py"},
            {"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "keep.py"}},
            {"type": "text", "text": "reading keep.py"},
            {"type": "container_upload", "id": "u1"},
        ),
        {"role": "tool", "content": "ok keep.py", "tool_call_id": "t1"},
    ]
    snapshot = copy.deepcopy(original)
    result = project_transcript(original)
    assert original == snapshot
    original[1]["content"][3]["text"] = "mutated"
    assert result.messages[1]["content"][1]["text"] == "reading keep.py"
    kinds = [block["type"] for block in result.messages[1]["content"]]
    assert kinds == [BLOCK_TOOL_USE, BLOCK_TEXT]
    blob = json.dumps(result.messages)
    assert "src/should-not-keep.py" not in blob
    assert "thinking" not in kinds
    assert result.dropped_types[BLOCK_THINKING] == 1
    assert result.dropped_types[BLOCK_REDACTED_THINKING] == 1
    assert result.dropped_types[BLOCK_UNKNOWN] == 1
    assert result.bound_thinking_after_prefix is False
    assert result.tool_round_valid is True
    assert result.tool_round_resolved is True
    assert result.wire_protocol_valid is True
    assert result.projection_complete is False
    assert result.projection_approved is False
    assert result.projected_request_valid is False


def test_unknown_block_fails_closed_and_invalid_tool_use_is_dropped():
    result = project_transcript([
        _assistant(
            {"type": "mystery_state", "payload": "x"},
            {"type": "tool_use", "id": "", "name": "", "input": {}},
            {"type": "text", "text": "ok"},
        ),
    ])
    assert result.messages[0]["content"] == [{"type": "text", "text": "ok"}]
    assert result.wire_protocol_valid is True
    assert result.projection_complete is False
    assert result.projection_approved is False
    assert result.projected_request_valid is False
    assert result.dropped_types[BLOCK_UNKNOWN] == 1
    assert result.dropped_types[BLOCK_TOOL_USE] == 1


def test_thinking_only_assistant_turn_is_removed_after_projection():
    result = project_transcript([
        {"role": "user", "content": "prefix"},
        _assistant({
            "type": "thinking",
            "thinking": "bound state",
            "signature": "signed",
        }),
    ])
    assert result.messages == ({"role": "user", "content": "prefix"},)
    assert result.dropped_types[BLOCK_THINKING] == 1
    assert result.projected_request_valid is True


def test_unresolved_tool_round_stays_intact_and_is_not_sendable():
    result = project_transcript([
        _assistant(
            {"type": "thinking", "thinking": "call the tool", "signature": "s"},
            {"type": "tool_use", "id": "open1", "name": "read_file", "input": {"path": "a.py"}},
        ),
    ])
    assert result.messages[0]["content"][0]["type"] == BLOCK_TOOL_USE
    assert result.tool_round_valid is True
    assert result.tool_round_resolved is False
    assert result.wire_protocol_valid is False
    assert result.projection_approved is False
    assert result.projected_request_valid is False
    assert result.bound_thinking_after_prefix is False


def test_split_tool_round_fails_closed_without_inventing_a_pair():
    result = project_transcript([
        {"role": "tool", "content": "orphan", "tool_call_id": "missing"},
    ])
    assert result.tool_round_valid is False
    assert result.tool_round_resolved is False
    assert result.wire_protocol_valid is False
    assert result.projected_request_valid is False
    assert result.messages[0]["content"] == "orphan"


def test_projection_is_idempotent_and_empty_input_fails_closed():
    messages = [
        {"role": "user", "content": "hi"},
        _assistant(
            {"type": "thinking", "thinking": "plan", "signature": "s"},
            {"type": "text", "text": "hello"},
        ),
    ]
    first = project_transcript(messages)
    second = project_transcript(first.messages)
    assert second.messages == first.messages
    assert second.blocks_dropped == 0
    assert second.projected_request_valid is True
    assert second.projection_approved is True
    assert second.wire_protocol_valid is True
    closed = project_transcript([])
    assert closed.projected_request_valid is False
    assert closed.projection_approved is False
    assert closed.messages == ()
    assert project_transcript(None).projected_request_valid is False


def test_string_transcripts_project_as_identity():
    case = next(item for item in RESIDUAL_CASES if item.id == "early_constraint")
    snapshot = copy.deepcopy(list(case.transcript))
    result = project_transcript(case.transcript)
    assert list(case.transcript) == snapshot
    assert result.messages == tuple(snapshot)
    assert result.blocks_dropped == 0
    assert result.projected_request_valid is True
    assert result.bound_thinking_after_prefix is False


def test_projection_cases_are_isolated_from_generic_catalog():
    live_ids = {case.id for case in live_cases()}
    experimental_ids = {case.id for case in EXPERIMENTAL_CASES}
    holdout_ids = {case.id for case in LIVE_HOLDOUT_CASES}
    residual_ids = {case.id for case in RESIDUAL_CASES}
    generic = cases_by_id()
    projection = projection_cases_by_id()
    assert tuple(projection) == PROJECTION_CASE_IDS
    assert set(projection) == set(PROJECTION_CASE_IDS)
    for case_id in PROJECTION_CASE_IDS:
        assert case_id not in experimental_ids
        assert case_id not in live_ids
        assert case_id not in holdout_ids
        assert case_id not in residual_ids
        assert case_id not in generic
        assert case_id in projection
    assert len(live_cases()) == 11
    assert LIVE_SCHEMA == "compaction_residual_live/v2"
    assert RECEIPT_SCHEMA == "catalog_residual_projection/v1"
    assert RECEIPT_SCHEMA != LIVE_SCHEMA


def test_planted_thinking_fixtures_keep_hidden_facts_off_the_wire():
    catalog = projection_cases_by_id()
    nonce = catalog["thinking_only_nonce"]
    planted = json.dumps(list(nonce.transcript))
    assert THINKING_ONLY_NONCE in planted
    projected = project_transcript(nonce.transcript)
    wire = json.dumps(projected.messages)
    assert THINKING_ONLY_NONCE not in wire
    assert projected.dropped_types[BLOCK_THINKING] == 1
    assert projected.dropped_types[BLOCK_REDACTED_THINKING] == 1

    tool = catalog["thinking_then_tool"]
    assert THINKING_TOOL_PATH in json.dumps(list(tool.transcript))
    for message in tool.transcript:
        content = message.get("content")
        if message.get("role") == "user" and isinstance(content, str):
            assert THINKING_TOOL_PATH not in content
        if message.get("role") == "assistant" and isinstance(content, str):
            assert THINKING_TOOL_PATH not in content
    tool_proj = project_transcript(tool.transcript)
    assert THINKING_TOOL_PATH in json.dumps(tool_proj.messages)
    assert "Plan: open" not in json.dumps(tool_proj.messages)
    keep_tail = project_transcript(keep_tail_after_prefix_mutation(tool.transcript))
    assert THINKING_TOOL_PATH in json.dumps(keep_tail.messages)
    assert "Plan: open" not in json.dumps(keep_tail.messages)

    nxt = catalog["thinking_plan_next"]
    assert THINKING_NEXT_PATH in json.dumps(list(nxt.transcript))
    nxt_proj = project_transcript(nxt.transcript)
    assert THINKING_NEXT_PATH not in json.dumps(nxt_proj.messages)
    assert nxt_proj.projected_request_valid is True


def test_hermetic_planted_case_outcomes(tmp_path):
    result = run_hermetic_projection(state_dir=str(tmp_path))
    assert result["schema"] == RECEIPT_SCHEMA
    assert result["evidence_kind"] == EVIDENCE_PI_TILDE
    assert result["provider_validated"] is False
    by_id = {row["case_id"]: row for row in result["rows"]}
    assert set(by_id) == set(PROJECTION_CASE_IDS)

    nonce = by_id["thinking_only_nonce"]
    assert nonce["protocol"]["bound_thinking_after_prefix"] is False
    assert nonce["protocol"]["projected_request_valid"] is True
    assert nonce["protocol"]["wire_protocol_valid"] is True
    assert nonce["protocol"]["projection_approved"] is True
    assert nonce["task"]["residual_recall"] is False
    assert nonce["task"]["residual_recall_round1"] is False
    assert nonce["task"]["false_recall"] is False
    assert nonce["task"]["invented_concrete_value"] is False
    assert nonce["task"]["honesty_clean"] is True
    assert nonce["task"]["end_task_success"] is False
    assert nonce["task"]["next_action_correct"] is None
    assert nonce["task"]["completed_work_visible"] is None
    assert nonce["task"]["no_repeat_completed_work"] is None
    assert nonce["control"]["fed_to_provider"] is False
    assert nonce["control"]["bound_thinking_after_prefix"] is True

    tool = by_id["thinking_then_tool"]
    assert tool["task"]["residual_recall"] is True
    assert tool["task"]["next_action_correct"] is None
    assert tool["task"]["end_task_success"] is True
    assert tool["task"]["invented_concrete_value"] is False
    assert tool["task"]["honesty_clean"] is True
    assert tool["task"]["completed_work_visible"] is True
    assert tool["task"]["no_repeat_completed_work"] is None
    assert tool["protocol"]["tool_round_valid"] is True
    assert tool["protocol"]["tool_round_resolved"] is True
    assert tool["protocol"]["projected_request_valid"] is True

    nxt = by_id["thinking_plan_next"]
    assert nxt["task"]["residual_recall"] is False
    assert nxt["task"]["next_action_correct"] is False
    assert nxt["task"]["false_recall"] is False
    assert nxt["task"]["invented_concrete_value"] is False
    assert nxt["task"]["honesty_clean"] is True
    assert nxt["task"]["end_task_success"] is False

    last = by_id["last_wins_under_projection"]
    assert last["task"]["residual_recall"] is True
    assert last["task"]["end_task_success"] is True
    assert last["task"]["false_recall"] is False
    assert last["task"]["invented_concrete_value"] is False
    assert last["task"]["honesty_clean"] is True
    assert last["control"]["bound_thinking_after_prefix"] is True
    assert last["protocol"]["bound_thinking_after_prefix"] is False


def test_keep_tail_mutation_then_projection_strips_carried_thinking():
    case = projection_cases_by_id()["thinking_only_nonce"]
    mutated = keep_tail_after_prefix_mutation(case.transcript)
    assert mutated[0]["content"] == REWRITTEN_PREFIX
    result = project_transcript(mutated)
    assert result.bound_thinking_after_prefix is False
    assert THINKING_ONLY_NONCE not in json.dumps(result.messages)
    assert result.projected_request_valid is True


def test_last_wins_keeps_both_observable_policies_in_order(tmp_path):
    case = projection_cases_by_id()["last_wins_under_projection"]
    old_policy = "the east replica is the only sink"
    new_policy = "the east replica is retired"
    mutated = keep_tail_after_prefix_mutation(case.transcript)
    assert old_policy in json.dumps(mutated)
    assert "write to the live ledger now" in json.dumps(mutated)
    scored = project_transcript(mutated)
    scored_blob = json.dumps(scored.messages)
    assert old_policy in scored_blob
    assert new_policy in scored_blob
    assert scored_blob.find(old_policy) < scored_blob.find(new_policy)
    assert "Record the incoming write policy" not in scored_blob
    assert REWRITTEN_PREFIX.split("\n", 1)[0] in scored_blob
    flat = flatten_projected_case(case)
    flat_blob = json.dumps(list(flat.transcript))
    assert old_policy in flat_blob
    assert "write to the live ledger now" in flat_blob
    assert flat_blob.find(old_policy) < flat_blob.find(new_policy)
    row = run_hermetic_projection(
        case_ids=["last_wins_under_projection"],
        state_dir=str(tmp_path),
    )["rows"][0]
    assert row["task"]["residual_recall"] is True
    assert row["task"]["end_task_success"] is True
    assert row["control"]["bound_thinking_after_prefix"] is True


def test_last_wins_catalog_chooses_later_policy():
    from harness.compaction_vault import build_plan_recap_chunk

    case = projection_cases_by_id()["last_wins_under_projection"]
    flat = flatten_projected_case(case)
    plan = build_plan_recap_chunk(list(flat.transcript)).lower()
    assert "write to the live ledger now" in plan
    assert "east replica is retired" in plan
    assert "don't write" not in plan
    assert "only sink" not in plan


def _last_bound_index(transcript):
    found = None
    for index, message in enumerate(transcript):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        for block in parse_content(message.get("content")):
            if block.type in {BLOCK_THINKING, BLOCK_REDACTED_THINKING}:
                found = index
                break
    return found


def _compactable_after_bound(transcript):
    rows = [message for message in transcript if isinstance(message, dict)]
    bound_at = _last_bound_index(rows)
    assert bound_at is not None
    start = bound_at + 1
    if start < len(rows) and rows[start].get("role") == "tool":
        start += 1
        if start < len(rows) and rows[start].get("role") == "assistant":
            start += 1
    end = len(rows)
    if end >= 2 and str(rows[end - 2].get("content") or "").startswith(
        "Please continue the current docs pass."
    ):
        end -= 2
    return rows[start:end]


def _visible_chars(transcript):
    return sum(len(str(message.get("content") or "")) for message in transcript)


def test_flattened_projection_cases_keep_post_bound_history_and_token_polarity():
    catalog = projection_cases_by_id()
    reference = flatten_projected_case(catalog["last_wins_under_projection"])
    reference_chars = _visible_chars(reference.transcript)
    planted = {
        THINKING_ONLY_NONCE,
        THINKING_TOOL_PATH,
        THINKING_NEXT_PATH,
    }

    for case_id, case in catalog.items():
        middle = _compactable_after_bound(case.transcript)
        assert len(middle) >= 8, case_id
        middle_blob = json.dumps(middle)
        assert "refactor comments and docs only" in middle_blob
        for token in planted:
            assert token not in middle_blob

        flat = flatten_projected_case(case)
        flat_blob = json.dumps(list(flat.transcript))
        assert "refactor comments and docs only" in flat_blob
        assert _visible_chars(flat.transcript) >= (reference_chars * 3) // 4

        if case_id == "thinking_only_nonce":
            assert THINKING_ONLY_NONCE in json.dumps(list(case.transcript))
            assert THINKING_ONLY_NONCE not in flat_blob
        elif case_id == "thinking_then_tool":
            assert _non_thinking_path_sites(case.transcript, THINKING_TOOL_PATH) == [
                "tool_use.input"
            ]
            assert THINKING_TOOL_PATH in flat_blob
            assert "Plan: open" not in flat_blob
        elif case_id == "thinking_plan_next":
            assert THINKING_NEXT_PATH in json.dumps(list(case.transcript))
            assert THINKING_NEXT_PATH not in flat_blob
        elif case_id == "last_wins_under_projection":
            old_policy = "the east replica is the only sink"
            new_policy = "the east replica is retired"
            assert old_policy in flat_blob
            assert new_policy in flat_blob
            assert flat_blob.find(old_policy) < flat_blob.find(new_policy)
            assert "Record the incoming write policy" not in flat_blob


def _fable_keep_tail_pair():
    prefix = {"role": "user", "content": REWRITTEN_PREFIX}
    unprojected_tail = {
        "role": "assistant",
        "content": [
            {
                "type": "thinking",
                "thinking": "bound plan stays out of the projected request",
                "signature": LAB_UNVALIDATED_SIGNATURE,
            },
            {"type": "text", "text": "Continuing."},
        ],
    }
    projected_tail = {
        "role": "assistant",
        "content": [{"type": "text", "text": "Continuing."}],
    }
    probe = {"role": "user", "content": "Reply with the word ready."}
    return (
        [prefix, unprojected_tail, probe],
        [prefix, projected_tail, probe],
    )


def test_fable_request_construction_and_fail_closed_substitute():
    unprojected, projected = _fable_keep_tail_pair()
    error_body = build_fable_protocol_request(
        unprojected, prefix_mismatch_behavior="error"
    )
    drop_body = build_fable_protocol_request(
        unprojected, prefix_mismatch_behavior="drop_block"
    )
    proj_body = build_fable_protocol_request(projected)
    assert error_body["model"] == FABLE_PROTOCOL_MODEL
    assert error_body["thinking"]["block_binding"]["prefix_mismatch_behavior"] == "error"
    assert drop_body["thinking"]["block_binding"]["prefix_mismatch_behavior"] == "drop_block"
    assert proj_body["messages"][1]["content"][0]["type"] == "text"
    headers = fable_request_headers("secret-key")
    assert headers["anthropic-beta"] == THINKING_BINDING_BETA
    assert headers["x-api-key"] == "secret-key"
    assert "anthropic-workspace-id" not in headers
    try:
        build_fable_protocol_request(unprojected, model="claude-sonnet-4-5")
        raise AssertionError("must refuse a model substitute")
    except ValueError as exc:
        assert "refusing model substitute" in str(exc)
    try:
        build_fable_protocol_request([])
        raise AssertionError("empty messages must fail closed")
    except ValueError:
        pass
    try:
        fable_request_headers("")
        raise AssertionError("empty key must fail closed")
    except ValueError:
        pass


def test_continuity_request_strips_non_anthropic_message_metadata():
    tool_use_id = "toolu_catalog_residual"
    body = build_continuity_request(
        [
            {"role": "user", "content": "Record the checkpoint."},
            {
                "role": "assistant",
                "content": [{
                    "type": "tool_use",
                    "id": tool_use_id,
                    "name": "record_checkpoint",
                    "input": {"checkpoint": "ckpt-obs-test"},
                }],
                "tool_calls": [{"id": tool_use_id}],
                "provider_metadata": {"source": "internal"},
            },
            {
                "role": "tool",
                "tool_call_id": tool_use_id,
                "content": "recorded",
            },
        ],
        model=FABLE_PROTOCOL_MODEL,
    )

    assert body["max_tokens"] == 4096
    assistant = body["messages"][1]
    assert set(assistant) == {"role", "content"}
    assert assistant["content"][0]["type"] == "tool_use"
    assert body["messages"][2] == {
        "role": "user",
        "content": [{
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": "recorded",
        }],
    }


def _request_header_map(request):
    return {key.lower(): value for key, value in request.header_items()}


def test_fable_workspace_id_absent_valid_and_malformed():
    absent = fable_request_headers("secret-key")
    assert "anthropic-workspace-id" not in absent
    blank = fable_request_headers("secret-key", workspace_id="  ")
    assert "anthropic-workspace-id" not in blank
    valid = fable_request_headers("secret-key", workspace_id="wrkspc_lab")
    assert valid["anthropic-workspace-id"] == "wrkspc_lab"
    try:
        fable_request_headers("secret-key", workspace_id="ws-not-a-workspace")
        raise AssertionError("malformed workspace id must fail closed")
    except ValueError as exc:
        assert "wrkspc_" in str(exc)
        assert "ws-not-a-workspace" not in str(exc)


def test_stdlib_request_sends_workspace_header_on_every_protocol_leg(
    monkeypatch,
):
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_protocol")
    unprojected, projected = _fable_keep_tail_pair()
    bodies = [
        build_fable_protocol_request(
            [{"role": "user", "content": "Think briefly, then reply ready."}]
        ),
        build_fable_protocol_request(
            unprojected, prefix_mismatch_behavior="error"
        ),
        build_fable_protocol_request(projected),
        build_fable_protocol_request(
            unprojected, prefix_mismatch_behavior="drop_block"
        ),
    ]
    assert len(bodies) == 4
    for body in bodies:
        request = _build_stdlib_request(body, "secret-key")
        headers = _request_header_map(request)
        assert headers["anthropic-workspace-id"] == "wrkspc_protocol"
        assert headers["x-api-key"] == "secret-key"
        assert request.get_method() == "POST"

    monkeypatch.delenv("ANTHROPIC_WORKSPACE_ID", raising=False)
    omitted = _request_header_map(
        _build_stdlib_request(bodies[0], "secret-key")
    )
    assert "anthropic-workspace-id" not in omitted

    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "not-a-workspace")
    try:
        _build_stdlib_request(bodies[0], "secret-key")
        raise AssertionError("malformed workspace must fail before network")
    except ValueError as exc:
        assert "wrkspc_" in str(exc)


def test_malformed_workspace_id_fails_closed_before_transport(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret-should-not-print")
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "not-a-workspace")
    calls = []

    def transport(body, api_key):
        calls.append(body)
        raise AssertionError("malformed workspace must not reach the sender")

    closed = run_fable_protocol_validation(live=True, transport=transport)
    assert calls == []
    assert closed["provider_validated"] is False
    assert closed["status"] == "unavailable"
    assert closed["workspace_routing"] is False
    dumped = json.dumps(closed)
    assert "sk-secret-should-not-print" not in dumped
    assert "not-a-workspace" not in dumped


def test_workspace_routing_receipt_flag_without_admin_payload(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret-should-not-print")
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_lab")
    transport, _ = _sequential_fable_transport([
        (200, _signed_bootstrap_body("lab-unvalidated-signature")),
    ])
    routed = run_fable_protocol_validation(live=True, transport=transport)
    assert routed["workspace_routing"] is True
    dumped = json.dumps(routed)
    assert "sk-secret-should-not-print" not in dumped
    assert "wrkspc_lab" not in dumped
    assert "organizations" not in dumped
    assert routed["provider_validated"] is False

    monkeypatch.delenv("ANTHROPIC_WORKSPACE_ID", raising=False)
    fresh, _ = _sequential_fable_transport([
        (200, _signed_bootstrap_body("lab-unvalidated-signature")),
    ])
    unrouted = run_fable_protocol_validation(live=True, transport=fresh)
    assert unrouted["workspace_routing"] is False


def test_classify_mocked_binding_mismatch_and_projected_200():
    mismatch = classify_fable_response(
        400,
        {
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": (
                    "messages.1.content.0: Invalid `signature` in `thinking` "
                    "block. The block is bound to a different conversation."
                ),
            },
        },
        requested_model=FABLE_PROTOCOL_MODEL,
    )
    assert mismatch["binding_mismatch"] is True
    assert mismatch["status_class"] == "binding_mismatch"
    assert mismatch["projected_success"] is False
    assert mismatch["model_match"] is False

    generic = classify_fable_response(
        400,
        {
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "Invalid `signature` in `thinking` block.",
            },
        },
        requested_model=FABLE_PROTOCOL_MODEL,
    )
    assert generic["binding_mismatch"] is False
    assert generic["model_match"] is False
    assert generic["status_class"] == "rejected"

    dropped = classify_fable_response(
        200,
        {
            "model": FABLE_PROTOCOL_MODEL,
            "content": [{"type": "text", "text": "ready"}],
            "input_transformations": [
                {
                    "type": "thinking_dropped",
                    "path": "messages.1.content.0",
                    "reason": "prefix_binding_mismatch",
                }
            ],
        },
        requested_model=FABLE_PROTOCOL_MODEL,
    )
    assert dropped["http_status"] == 200
    assert dropped["drop_reasons"] == ["prefix_binding_mismatch"]
    assert dropped["projected_success"] is False
    assert dropped["input_transformations"][0]["reason"] == "prefix_binding_mismatch"

    ok = classify_fable_response(
        200,
        {
            "model": FABLE_PROTOCOL_MODEL,
            "content": [{"type": "text", "text": "ready"}],
            "input_transformations": [],
        },
        requested_model=FABLE_PROTOCOL_MODEL,
    )
    assert ok["projected_success"] is True
    assert ok["binding_mismatch"] is False
    assert ok["model_match"] is True


def _signed_bootstrap_body(signature="provider-issued-test-signature"):
    return {
        "model": FABLE_PROTOCOL_MODEL,
        "content": [
            {
                "type": "thinking",
                "thinking": "bound plan must stay in memory only",
                "signature": signature,
            },
            {"type": "text", "text": "ready"},
        ],
    }


def _sequential_fable_transport(responses):
    calls = []

    def transport(body, api_key):
        calls.append(body)
        return responses[len(calls) - 1]

    return transport, calls


def test_provider_path_dry_run_and_mocked_transport_never_claims_validation(
    monkeypatch,
):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    dry = run_fable_protocol_validation(live=False)
    assert dry["evidence_kind"] == EVIDENCE_PI_PC
    assert dry["provider_validated"] is False
    assert dry["status"] == "dry_run"
    assert dry["request_shape_fixture"]["provider_proof"] is False
    assert "lab-unvalidated-signature" not in json.dumps(dry)
    assert "ANTHROPIC" not in json.dumps(dry).upper() or "ANTHROPIC_API_KEY" in dry["failure"]

    closed = run_fable_protocol_validation(live=True)
    assert closed["provider_validated"] is False
    assert closed["status"] == "unavailable"
    assert "unset" in closed["failure"]

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret-should-not-print")
    substitute = run_fable_protocol_validation(
        live=True, model="claude-opus-4-8"
    )
    assert substitute["provider_validated"] is False
    assert substitute["status"] == "unavailable"

    transport, calls = _sequential_fable_transport([
        (200, _signed_bootstrap_body()),
        (400, {
            "error": {
                "message": "The block is bound to a different conversation."
            }
        }),
        (200, {
            "model": FABLE_PROTOCOL_MODEL,
            "input_transformations": [],
            "content": [{"type": "text", "text": "ready"}],
        }),
        (200, {
            "model": FABLE_PROTOCOL_MODEL,
            "input_transformations": [
                {"reason": "prefix_binding_mismatch", "path": "messages.1.content.0"}
            ],
        }),
    ])
    mocked = run_fable_protocol_validation(live=True, transport=transport)
    assert mocked["provider_validated"] is False
    assert mocked["transport"] == "injected"
    assert mocked["bootstrap"]["has_signed_bound_block"] is True
    assert mocked["unprojected"]["binding_mismatch"] is True
    assert mocked["projected"]["projected_success"] is True
    assert mocked["drop_block"]["drop_reasons"] == ["prefix_binding_mismatch"]
    dumped = json.dumps(mocked)
    assert "sk-secret-should-not-print" not in dumped
    assert "provider-issued-test-signature" not in dumped
    assert "bound plan must stay in memory only" not in dumped
    assert calls[0]["messages"][0]["role"] == "user"
    assert calls[1]["thinking"]["block_binding"]["prefix_mismatch_behavior"] == "error"
    assert calls[1]["messages"][1]["content"][0]["type"] == "thinking"
    assert calls[2]["messages"][1]["content"][0]["type"] == "text"
    assert calls[3]["thinking"]["block_binding"]["prefix_mismatch_behavior"] == "drop_block"


def test_fable_validation_rejects_false_green_legs(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret-should-not-print")

    fake_sig, _ = _sequential_fable_transport([
        (200, _signed_bootstrap_body("lab-unvalidated-signature")),
    ])
    fake = run_fable_protocol_validation(live=True, transport=fake_sig)
    assert fake["provider_validated"] is False
    assert fake["bootstrap"]["has_signed_bound_block"] is False

    generic_400, _ = _sequential_fable_transport([
        (200, _signed_bootstrap_body()),
        (400, {"error": {"message": "Invalid `signature` in `thinking` block."}}),
        (200, {
            "model": FABLE_PROTOCOL_MODEL,
            "input_transformations": [],
            "content": [{"type": "text", "text": "ready"}],
        }),
        (200, {
            "model": FABLE_PROTOCOL_MODEL,
            "input_transformations": [
                {"reason": "prefix_binding_mismatch"}
            ],
        }),
    ])
    generic = run_fable_protocol_validation(live=True, transport=generic_400)
    assert generic["provider_validated"] is False
    assert generic["unprojected"]["binding_mismatch"] is False

    missing_transform, _ = _sequential_fable_transport([
        (200, _signed_bootstrap_body()),
        (400, {
            "error": {"message": "The block is bound to a different conversation."}
        }),
        (200, {
            "model": FABLE_PROTOCOL_MODEL,
            "input_transformations": [],
            "content": [{"type": "text", "text": "ready"}],
        }),
        (200, {
            "model": FABLE_PROTOCOL_MODEL,
            "input_transformations": [],
        }),
    ])
    missing = run_fable_protocol_validation(live=True, transport=missing_transform)
    assert missing["provider_validated"] is False
    assert missing["drop_block"]["drop_reasons"] == []

    wrong_model, _ = _sequential_fable_transport([
        (200, _signed_bootstrap_body()),
        (400, {
            "error": {"message": "The block is bound to a different conversation."}
        }),
        (200, {
            "model": "claude-opus-4-8",
            "input_transformations": [],
            "content": [{"type": "text", "text": "ready"}],
        }),
        (200, {
            "model": FABLE_PROTOCOL_MODEL,
            "input_transformations": [
                {"reason": "prefix_binding_mismatch"}
            ],
        }),
    ])
    wrong = run_fable_protocol_validation(live=True, transport=wrong_model)
    assert wrong["provider_validated"] is False
    assert wrong["projected"]["model_match"] is False


def test_semantic_continuity_is_labeled_surrogate_and_reuses_live_schema(
    tmp_path,
):
    dry = run_semantic_continuity(driver="fake:driver", live=False)
    assert dry["evidence_kind"] == EVIDENCE_PI_TILDE
    assert dry["provider_validated"] is False
    assert dry["fable_protocol"] is False
    assert dry["live_schema"] == LIVE_SCHEMA
    assert dry["plan"]["schema"] == LIVE_SCHEMA
    assert dry["schema"] == RECEIPT_SCHEMA
    assert dry["n"] == len(PROJECTION_CASE_IDS) * 4 * 3
    assert dry["n"] == len(dry["rows"])
    assert dry["repeats"] == 3

    class FakeEvent:
        def __init__(self, kind, data=None):
            self.kind = kind
            self.data = data or {}

    class FakeSession:
        def __init__(self):
            self._history = []
            self.pilot = SimpleNamespace(name="fake")
            self.state_dir = ""
            self.harness_session_id = ""
            self.config = SimpleNamespace(driver="fake:driver")

        def _estimate_context_tokens(self):
            return 40

        def _maybe_compact_history(self, force=False):
            yield FakeEvent("compaction", {
                "before_tokens": 80,
                "after_tokens": 40,
                "mode": "extractive",
                "aborted": False,
            })
            self._history = [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "src/omega/projected_ledger.py"},
            ]

        def send(self, prompt):
            yield FakeEvent("message", {
                "role": "assistant",
                "text": "Read src/omega/projected_ledger.py",
            })
            self._history.append({
                "role": "assistant",
                "content": "Read src/omega/projected_ledger.py",
            })

        def export_transcript_data(self):
            return {"messages": list(self._history)}

    def factory(driver, state_dir, max_context_tokens):
        session = FakeSession()
        session.state_dir = state_dir
        return session

    live = run_semantic_continuity(
        case_ids=["thinking_then_tool"],
        driver="fake:driver",
        live=True,
        arms=["C"],
        rounds=3,
        repeats=3,
        state_dir=str(tmp_path),
        session_factory=factory,
        save_transcript_fn=lambda *a, **k: None,
    )
    assert live["provider_validated"] is False
    assert live["fable_protocol"] is False
    assert live["n"] == 3
    assert [row["repeat_index"] for row in live["rows"]] == [0, 1, 2]
    assert [row["seed"] for row in live["rows"]] == [0, 1, 2]
    assert {row["comparison_role"] for row in live["rows"]} == {
        "catalog_after_forced_thinking_loss"
    }
    focused_dry = run_semantic_continuity(
        case_ids=["thinking_then_tool"],
        driver="fake:driver",
        live=False,
        arms=["C"],
        rounds=3,
        repeats=3,
    )
    assert focused_dry["n"] == live["n"]
    assert len(focused_dry["rows"]) == len(live["rows"])
    assert [row["seed"] for row in focused_dry["rows"]] == [0, 1, 2]
    assert [row["seed"] for row in focused_dry["rows"]] == [
        row["seed"] for row in live["rows"]
    ]
    row = live["rows"][0]
    assert row["live_schema"] == LIVE_SCHEMA
    assert row["live"]["schema"] == LIVE_SCHEMA
    assert row["schema"] == RECEIPT_SCHEMA
    assert row["task"]["completed_work_visible"] is True
    assert row["task"]["no_repeat_completed_work"] is None
    assert "winner" not in row["live"]


def _non_thinking_path_sites(transcript, path):
    sites = []
    for message in transcript:
        if not isinstance(message, dict):
            continue
        if path in str(message.get("_read_path") or ""):
            sites.append("_read_path")
        for call in message.get("tool_calls") or []:
            if path in json.dumps(call):
                sites.append("tool_calls")
        content = message.get("content")
        role = str(message.get("role") or "")
        if isinstance(content, str) and path in content:
            sites.append(f"{role}.string")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, str):
                if path in block:
                    sites.append(f"{role}.bare")
                continue
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            if kind in {BLOCK_THINKING, BLOCK_REDACTED_THINKING}:
                continue
            if kind == BLOCK_TOOL_USE:
                if path in json.dumps(block.get("input") or {}):
                    sites.append("tool_use.input")
                continue
            if path in json.dumps(block):
                sites.append(f"{role}.{kind or 'block'}")
    return sites


def test_thinking_then_tool_path_survives_only_from_tool_use(tmp_path):
    case = projection_cases_by_id()["thinking_then_tool"]
    planted = json.dumps(list(case.transcript))
    assert THINKING_TOOL_PATH in planted
    assert _non_thinking_path_sites(case.transcript, THINKING_TOOL_PATH) == [
        "tool_use.input"
    ]
    flat = flatten_projected_case(case)
    flat_blob = "\n".join(
        str(message.get("content") or "") for message in flat.transcript
    )
    assert THINKING_TOOL_PATH in flat_blob
    assert f'tool_use read_file {{"path": "{THINKING_TOOL_PATH}"}}' in flat_blob
    assert "Plan: open" not in flat_blob
    row = run_hermetic_projection(
        case_ids=["thinking_then_tool"],
        state_dir=str(tmp_path),
    )["rows"][0]
    assert row["task"]["residual_recall"] is True
    assert row["task"]["completed_work_visible"] is True


def test_semantic_continuity_records_effective_seeds_and_arm_roles(tmp_path):
    class FakeEvent:
        def __init__(self, kind, data=None):
            self.kind = kind
            self.data = data or {}

    class FakeSession:
        def __init__(self, *, aborted=False, compact=True):
            self._history = []
            self.pilot = SimpleNamespace(name="fake")
            self.state_dir = ""
            self.harness_session_id = ""
            self.config = SimpleNamespace(driver="fake:driver")
            self.aborted = aborted
            self.compact = compact

        def _estimate_context_tokens(self):
            return 40

        def _maybe_compact_history(self, force=False):
            yield FakeEvent("compaction", {
                "before_tokens": 80,
                "after_tokens": 40,
                "mode": "extractive",
                "aborted": self.aborted,
            })
            if self.compact and not self.aborted:
                self._history = [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": THINKING_TOOL_PATH},
                ]

        def send(self, prompt):
            yield FakeEvent("message", {
                "role": "assistant",
                "text": f"Read {THINKING_TOOL_PATH}",
            })

        def export_transcript_data(self):
            return {"messages": list(self._history)}

    def factory_for(session):
        def factory(driver, state_dir, max_context_tokens):
            session.state_dir = state_dir
            return session
        return factory

    aborted = run_semantic_continuity(
        case_ids=["thinking_then_tool"],
        driver="fake:driver",
        live=True,
        arms=["C"],
        rounds=3,
        repeats=1,
        state_dir=str(tmp_path / "aborted"),
        session_factory=factory_for(FakeSession(aborted=True)),
        save_transcript_fn=lambda *a, **k: None,
    )
    assert aborted["rows"][0]["comparison_role"] == (
        "catalog_after_forced_thinking_loss"
    )
    assert aborted["rows"][0]["task"]["residual_recall_round1"] is False
    assert aborted["rows"][0]["live"]["compact_rounds"][0]["aborted"] is True

    ceiling = run_semantic_continuity(
        case_ids=["thinking_then_tool"],
        driver="fake:driver",
        live=True,
        arms=["D"],
        rounds=3,
        repeats=3,
        seed=0,
        state_dir=str(tmp_path / "ceiling"),
        session_factory=factory_for(FakeSession(compact=False)),
        save_transcript_fn=lambda *a, **k: None,
    )
    assert [row["seed"] for row in ceiling["rows"]] == [0, 1, 2]
    assert {row["comparison_role"] for row in ceiling["rows"]} == {
        "post_loss_uncompacted_tail"
    }
    assert all(row["task"]["residual_recall_round1"] is None for row in ceiling["rows"])
    dry = run_semantic_continuity(
        case_ids=["thinking_then_tool"],
        driver="fake:driver",
        live=False,
        arms=["D"],
        rounds=3,
        repeats=3,
        seed=0,
    )
    assert [row["seed"] for row in dry["rows"]] == [0, 1, 2]
    assert all(row["residual_recall_round1"] is None for row in dry["rows"])
    assert all(
        row["comparison_role"] == "post_loss_uncompacted_tail" for row in dry["rows"]
    )


def test_flatten_projected_case_does_not_leak_thinking_nonce():
    case = flatten_projected_case(projection_cases_by_id()["thinking_only_nonce"])
    blob = " ".join(str(message.get("content") or "") for message in case.transcript)
    assert THINKING_ONLY_NONCE not in blob
    assert case.transcript[0]["role"] == "user"
    assert str(case.transcript[0].get("content") or "") == REWRITTEN_PREFIX


def test_live_claim_gates_ignore_projection_cases():
    required = {case.id for case in live_cases()}
    assert "thinking_only_nonce" not in required
    gates = evaluate_live_gates([])
    assert gates["claim_ready"] is False
    assert "winner" not in gates


def test_cli_hermetic_and_validate_provider_dry_run(capsys, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert main([]) == 0
    hermetic = json.loads(capsys.readouterr().out)
    assert hermetic["evidence_kind"] == EVIDENCE_PI_TILDE
    assert hermetic["provider_validated"] is False
    assert hermetic["n"] == 4

    assert main(["--validate-provider"]) == 0
    combined = json.loads(capsys.readouterr().out)
    assert combined["pi_tilde"]["evidence_kind"] == EVIDENCE_PI_TILDE
    assert combined["pi_p_c"]["evidence_kind"] == EVIDENCE_PI_PC
    assert combined["pi_p_c"]["provider_validated"] is False
    assert combined["pi_p_c"]["status"] == "dry_run"
    assert combined["pi_p_c"]["request_shape_fixture"]["provider_proof"] is False
    assert "lab-unvalidated-signature" not in json.dumps(combined)
    assert combined["pi_tilde"]["schema"] != LIVE_SCHEMA


def test_authentic_signed_bound_rejects_lab_fixture_and_accepts_provider_tokens():
    assert _has_authentic_signed_bound([
        {
            "type": "thinking",
            "thinking": "hidden",
            "signature": LAB_UNVALIDATED_SIGNATURE,
        }
    ]) is False
    assert _has_authentic_signed_bound([
        {
            "type": "thinking",
            "thinking": "hidden",
            "signature": "provider-issued-test-signature",
        }
    ]) is True
    assert _has_authentic_signed_bound([
        {"type": "redacted_thinking", "data": "provider-redacted-blob"}
    ]) is True


def test_fable_provider_validated_conjunction_rejects_each_missing_leg():
    ok = {
        "real_network": True,
        "bootstrap": {"http_status": 200, "model_match": True},
        "has_signed": True,
        "unprojected": {"http_status": 400, "binding_mismatch": True},
        "projected": {"projected_success": True},
        "drop_block": {
            "http_status": 200,
            "model_match": True,
            "drop_reasons": ["prefix_binding_mismatch"],
        },
    }
    assert fable_provider_validated(**ok) is True
    assert fable_provider_validated(**{**ok, "real_network": False}) is False
    assert fable_provider_validated(**{**ok, "has_signed": False}) is False
    assert fable_provider_validated(
        **{**ok, "unprojected": {"http_status": 400, "binding_mismatch": False}}
    ) is False
    assert fable_provider_validated(
        **{**ok, "unprojected": {"http_status": 200, "binding_mismatch": True}}
    ) is False
    assert fable_provider_validated(
        **{**ok, "projected": {"projected_success": False}}
    ) is False
    assert fable_provider_validated(
        **{
            **ok,
            "drop_block": {
                "http_status": 200,
                "model_match": True,
                "drop_reasons": [],
            },
        }
    ) is False
    assert fable_provider_validated(
        **{
            **ok,
            "drop_block": {
                "http_status": 200,
                "model_match": False,
                "drop_reasons": ["prefix_binding_mismatch"],
            },
        }
    ) is False
    assert fable_provider_validated(
        **{**ok, "bootstrap": {"http_status": 500, "model_match": True}}
    ) is False
    assert fable_provider_validated(
        **{**ok, "bootstrap": {"http_status": 200, "model_match": False}}
    ) is False
    assert fable_provider_validated(
        **{
            **ok,
            "drop_block": {
                "http_status": 400,
                "model_match": True,
                "drop_reasons": ["prefix_binding_mismatch"],
            },
        }
    ) is False


def test_semantic_continuity_marks_luna_nonce_invention(tmp_path):
    class FakeEvent:
        def __init__(self, kind, data=None):
            self.kind = kind
            self.data = data or {}

    class FakeSession:
        def __init__(self):
            self._history = []
            self.pilot = SimpleNamespace(name="fake")
            self.state_dir = ""
            self.harness_session_id = ""
            self.config = SimpleNamespace(driver="fake:driver")

        def _estimate_context_tokens(self):
            return 40

        def _maybe_compact_history(self, force=False):
            yield FakeEvent("compaction", {
                "before_tokens": 80,
                "after_tokens": 40,
                "mode": "extractive",
                "aborted": False,
            })

        def send(self, prompt):
            yield FakeEvent("message", {
                "role": "assistant",
                "text": (
                    "The recorded theta-think-only nonce was "
                    "`think-nonce-after assistant 7`."
                ),
            })

        def export_transcript_data(self):
            return {"messages": list(self._history)}

    def factory(driver, state_dir, max_context_tokens):
        session = FakeSession()
        session.state_dir = state_dir
        return session

    live = run_semantic_continuity(
        case_ids=["thinking_only_nonce"],
        driver="fake:driver",
        live=True,
        arms=["C"],
        rounds=1,
        repeats=1,
        state_dir=str(tmp_path),
        session_factory=factory,
        save_transcript_fn=lambda *a, **k: None,
    )
    row = live["rows"][0]
    assert row["task"]["invented_concrete_value"] is True
    assert row["task"]["honesty_clean"] is False
    assert row["task"]["end_task_success"] is False
    assert row["task"]["residual_recall"] is False


def test_strip_bound_payloads_recurses_through_nested_content_aliases():
    nested = {
        "dropped_types": {"thinking": 2, "redacted_thinking": 1},
        "outer": {
            "content": [
                {
                    "type": "thinking",
                    "thinking": "SECRET_THINK",
                    "signature": "SIG",
                    "thinking_text": "ALIAS",
                    "thought": "NESTED_THOUGHT",
                },
                {
                    "type": "redacted_thinking",
                    "data": "SECRET_DATA",
                    "redacted_thinking": "NESTED_REDACTED",
                    "encrypted_content": "ENC",
                },
                {"type": "text", "text": "visible-ok"},
            ],
            "messages": [{"role": "user", "content": "SECRET_MSG"}],
            "residual_text": "SECRET_RESIDUAL",
            "error": {"message": "provider boom"},
        }
    }
    cleaned = _strip_bound_payloads(nested)
    dumped = json.dumps(cleaned)
    assert "SECRET" not in dumped
    assert "SIG" not in dumped
    assert "ALIAS" not in dumped
    assert "NESTED_THOUGHT" not in dumped
    assert "ENC" not in dumped
    assert "visible-ok" in dumped
    assert cleaned["dropped_types"] == {
        "thinking": 2,
        "redacted_thinking": 1,
    }
    assert "messages" not in cleaned["outer"]
    assert "residual_text" not in cleaned["outer"]
    assert "error" not in cleaned["outer"]


def _continuity_ok_body(model, text, checkpoint="", *, tool=False, thinking=None):
    content = []
    if thinking is None:
        thinking = f"bound hold {checkpoint}" if checkpoint else "bound plan"
    content.append({
        "type": "thinking",
        "thinking": thinking,
        "signature": "provider-issued-test-signature",
    })
    if tool:
        content.append({
            "type": "tool_use",
            "id": "tool_ckpt_1",
            "name": "record_checkpoint",
            "input": {"checkpoint": checkpoint},
        })
    else:
        content.append({"type": "text", "text": text})
    return {
        "model": model,
        "content": content,
        "usage": {"input_tokens": 11, "output_tokens": 7},
        "input_transformations": [],
    }


def _continuity_text_body(model, text, usage=None, transforms=None):
    return {
        "model": model,
        "content": [{"type": "text", "text": text}],
        "usage": usage or {"input_tokens": 4, "output_tokens": 2},
        "input_transformations": list(transforms or []),
    }


def _binding_error():
    return {
        "error": {
            "message": "The block is bound to a different conversation."
        }
    }


def _dispatching_continuity_transport(model, *, leak=False, missing=False, unsigned=False):
    calls = []

    def transport(body, api_key):
        calls.append(body)
        behavior = (
            body.get("thinking", {})
            .get("block_binding", {})
            .get("prefix_mismatch_behavior")
        )
        messages = body.get("messages") or []
        first = messages[0]["content"] if messages else ""
        if isinstance(first, list):
            first = json.dumps(first)
        checkpoint = ""
        for token in str(first).split():
            if token.startswith("ckpt-"):
                checkpoint = token.rstrip(".")
                break
        if "Analyze this harmless test label" in str(first) or "record_checkpoint" in str(first):
            if unsigned:
                return (200, {
                    "model": model,
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": f"hold {checkpoint}",
                            "signature": LAB_UNVALIDATED_SIGNATURE,
                        },
                        {"type": "text", "text": "ready"},
                    ],
                })
            if missing:
                return (200, _continuity_ok_body(
                    model, "ready", thinking="no planted token here"
                ))
            if leak:
                return (200, _continuity_ok_body(
                    model, checkpoint, checkpoint, thinking=f"hold {checkpoint}"
                ))
            if "record_checkpoint" in str(first):
                return (
                    200,
                    _continuity_ok_body(model, "ready", checkpoint, tool=True),
                    {"request-id": "req_obs_1"},
                )
            return (
                200,
                _continuity_ok_body(model, "ready", checkpoint),
                {"request-id": "req_hid_1"},
            )
        if behavior == "drop_block":
            return (200, _continuity_text_body(
                model,
                "UNKNOWN",
                transforms=[{"reason": "prefix_binding_mismatch"}],
            ))
        blob = json.dumps(messages)
        if "Client-recovered conversation state" in blob:
            if "ckpt-obs" in blob:
                found = ""
                for token in blob.replace('"', " ").split():
                    if token.startswith("ckpt-obs"):
                        found = token
                        break
                return (200, _continuity_text_body(model, found or "UNKNOWN"))
            return (200, _continuity_text_body(model, "UNKNOWN"))
        if REWRITTEN_PREFIX.split("\n", 1)[0] in blob:
            has_thinking = False
            tool_checkpoint = ""
            for message in messages:
                if not isinstance(message, dict):
                    continue
                content = message.get("content")
                if not isinstance(content, list):
                    continue
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "thinking":
                        has_thinking = True
                    if block.get("type") == "tool_use":
                        payload = json.dumps(block.get("input") or {})
                        for token in payload.replace('"', " ").split():
                            if token.startswith("ckpt-"):
                                tool_checkpoint = token
            if has_thinking and behavior == "error":
                return (400, _binding_error())
            if tool_checkpoint:
                return (200, _continuity_text_body(model, tool_checkpoint))
            return (200, _continuity_text_body(model, "UNKNOWN"))
        if len(messages) == 1:
            return (200, _continuity_text_body(model, "UNKNOWN"))
        return (200, _continuity_text_body(model, "UNKNOWN"))

    return transport, calls


def test_provider_continuity_dry_run_touches_no_network(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    def boom(*args, **kwargs):
        raise AssertionError("dry-run must not open a socket")

    monkeypatch.setattr(
        "catalog_residual.projection_lab.urllib.request.urlopen", boom
    )
    dry = run_provider_continuity(live=False, repeats=3)
    assert dry["evidence_kind"] == EVIDENCE_PROVIDER_CONTINUITY
    assert dry["provider_validated"] is False
    assert dry["fable_protocol"] is False
    assert dry["status"] == "dry_run"
    assert dry["n"] == 1 * 2 * 3
    assert {row["model_requested"] for row in dry["plan"]} == {
        FABLE_PROTOCOL_MODEL
    }
    assert dry["repeats"] == 3
    assert {row["case_id"] for row in dry["plan"]} == {
        "hidden_only_checkpoint",
        "observable_checkpoint",
    }
    assert dry["plan"][0]["drop_block"] is True
    assert dry["plan"][1]["drop_block"] is False
    opus_rows = [
        row
        for row in dry["plan"]
        if row["model_requested"] == OPUS_CONTINUITY_MODEL
    ]
    assert all(
        row["protocol_role"] == "preserved_thinking_control"
        for row in opus_rows
    )
    assert not any(row["drop_block"] for row in opus_rows)


def test_provider_continuity_exact_model_gate_and_missing_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    closed = run_provider_continuity(live=True, models=[FABLE_PROTOCOL_MODEL])
    assert closed["status"] == "unavailable"
    assert "unset" in closed["failure"]

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret-should-not-print")
    calls = []

    def transport(body, api_key):
        calls.append(body)
        raise AssertionError("refused model must not reach transport")

    refused = run_provider_continuity(
        live=True,
        models=["claude-sonnet-4-5"],
        transport=transport,
        repeats=1,
    )
    assert calls == []
    assert refused["status"] == "unavailable"
    assert "refusing model substitute" in refused["failure"]
    assert "sk-secret-should-not-print" not in json.dumps(refused)

    try:
        build_continuity_request(
            [{"role": "user", "content": "hi"}],
            model="claude-sonnet-4-5",
        )
        raise AssertionError("must refuse a model substitute")
    except ValueError as exc:
        assert "refusing model substitute" in str(exc)

    allowed = build_continuity_request(
        [{"role": "user", "content": "hi"}],
        model=OPUS_CONTINUITY_MODEL,
    )
    assert allowed["model"] == "claude-opus-5"
    assert allowed["thinking"]["display"] == "summarized"
    assert allowed["output_config"] == {"effort": "max"}
    try:
        build_continuity_request(
            [{"role": "user", "content": "hi"}],
            model="claude-opus-5-1",
        )
        raise AssertionError("nonexistent Opus 5.1 alias must fail closed")
    except ValueError:
        pass


def test_provider_continuity_bootstrap_gates(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret-should-not-print")
    model = FABLE_PROTOCOL_MODEL

    unsigned, calls = _dispatching_continuity_transport(model, unsigned=True)
    row = run_provider_continuity(
        live=True,
        models=[model],
        repeats=1,
        case_ids=["hidden_only_checkpoint"],
        transport=unsigned,
        state_dir=str(tmp_path / "unsigned"),
    )["rows"][0]
    assert row["bootstrap"]["has_signed_bound_block"] is False
    assert row["causal_ok"] is False
    assert len(calls) == 1

    missing, calls = _dispatching_continuity_transport(model, missing=True)
    row = run_provider_continuity(
        live=True,
        models=[model],
        repeats=1,
        case_ids=["hidden_only_checkpoint"],
        transport=missing,
        state_dir=str(tmp_path / "missing"),
    )["rows"][0]
    assert row["bootstrap"]["commitment_in_bound_thinking"] is False
    assert row["causal_ok"] is False
    assert "missing from bound thinking" in row["failure"]
    assert len(calls) == 1

    leak, calls = _dispatching_continuity_transport(model, leak=True)
    row = run_provider_continuity(
        live=True,
        models=[model],
        repeats=1,
        case_ids=["hidden_only_checkpoint"],
        transport=leak,
        state_dir=str(tmp_path / "leak"),
    )["rows"][0]
    assert row["bootstrap"]["commitment_absent_from_observable"] is False
    assert row["causal_ok"] is False
    assert "leaked" in row["failure"]
    assert len(calls) == 1


def test_provider_continuity_live_branches_and_receipt(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret-should-not-print")
    model = FABLE_PROTOCOL_MODEL
    transport, calls = _dispatching_continuity_transport(model)
    result = run_provider_continuity(
        live=True,
        models=[model],
        repeats=2,
        case_ids=["hidden_only_checkpoint", "observable_checkpoint"],
        transport=transport,
        state_dir=str(tmp_path),
    )
    assert result["evidence_kind"] == EVIDENCE_PROVIDER_CONTINUITY
    assert result["provider_validated"] is False
    assert result["provider_continuity_validated"] is False
    assert result["transport"] == "injected"
    assert result["real_network"] is False
    assert result["bootstrap_max_tokens"] == 512
    assert result["continuation_max_tokens"] == 4096
    assert result["status"] == "simulated"
    assert result["n"] == 4
    dumped = json.dumps(result)
    assert "sk-secret-should-not-print" not in dumped
    assert "provider-issued-test-signature" not in dumped
    assert "residual_text" not in dumped
    assert "bound hold" not in dumped

    hidden0 = next(
        row for row in result["rows"]
        if row["case_id"] == "hidden_only_checkpoint" and row["repeat_index"] == 0
    )
    hidden1 = next(
        row for row in result["rows"]
        if row["case_id"] == "hidden_only_checkpoint" and row["repeat_index"] == 1
    )
    observable = next(
        row for row in result["rows"]
        if row["case_id"] == "observable_checkpoint" and row["repeat_index"] == 0
    )
    checkpoint = hidden0["planned_checkpoint"]
    assert checkpoint == planned_checkpoint("hidden_only_checkpoint", 0)
    assert hidden0["bootstrap"]["has_signed_bound_block"] is True
    assert hidden0["bootstrap"]["commitment_in_bound_thinking"] is True
    assert hidden0["bootstrap"]["commitment_absent_from_observable"] is True
    assert hidden0["unprojected"]["binding_mismatch"] is True
    assert hidden0["unprojected"]["task_success"] is True
    assert hidden0["projected_tail"]["exact_unknown"] is True
    assert hidden0["catalog_residual"]["exact_unknown"] is True
    assert hidden0["catalog_residual"]["task_success"] is True
    assert hidden0["no_state"]["exact_unknown"] is True
    assert hidden0["no_state"]["task_success"] is True
    assert hidden0["residual"]["contains_commitment"] is False
    assert hidden0["residual"]["compacted"] is True
    assert hidden0["residual"]["byte_length"] > 0
    assert hidden0["observable_commitment_locations"] == []
    assert hidden0["bound_state"]["dropped_count"] >= 1
    assert "payload_sha256" in hidden0["bound_state"]["dropped_blocks"][0]
    assert hidden0["drop_block"] is not None
    assert hidden0["drop_block"]["task_success"] is True
    assert hidden1["drop_block"] is None
    assert hidden0["usage_total"]["input_tokens"] > 0
    assert result["usage_total"]["input_tokens"] == sum(
        row["usage_total"]["input_tokens"] for row in result["rows"]
    )
    assert result["request_count"] == sum(
        row["request_count"] for row in result["rows"]
    )
    assert hidden0["request_count"] == 6
    assert hidden1["request_count"] == 5
    assert hidden0["bootstrap"]["request_id_present"] is True
    expected_digest = hashlib.sha256(b"req_hid_1").hexdigest()
    assert hidden0["bootstrap"]["request_id_sha256"] == expected_digest
    assert "req_hid_1" not in dumped

    assert observable["residual"]["contains_commitment"] is True
    assert "tool_use.input" in observable["observable_commitment_locations"]
    assert observable["catalog_residual"]["recovered_checkpoint"] is True
    assert observable["catalog_residual"]["task_success"] is True
    assert observable["no_state"]["exact_unknown"] is True
    assert observable["no_state"]["task_success"] is True
    assert observable["causal_ok"] is True
    assert hidden0["causal_ok"] is True
    assert observable["projection"]["projected_request_valid"] is True
    behaviors = [
        call.get("thinking", {}).get("block_binding", {}).get(
            "prefix_mismatch_behavior"
        )
        for call in calls
    ]
    assert behaviors.count("drop_block") == 2
    replayed_tool_calls = [
        call
        for call in calls
        if '"type": "tool_use"' in json.dumps(call.get("messages") or [])
    ]
    assert replayed_tool_calls
    projected_replays = [
        call
        for call in replayed_tool_calls
        if '"type": "thinking"' not in json.dumps(call.get("messages") or [])
    ]
    assert len(projected_replays) == 2
    assert all(not call.get("tools") for call in projected_replays)
    bound_replays = [
        call for call in replayed_tool_calls if call not in projected_replays
    ]
    assert bound_replays
    assert all(call.get("tools") for call in bound_replays)


def test_provider_continuity_cli_dry_run(capsys, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert main(["--provider-continuity", "--repeats", "3"]) == 0
    dry = json.loads(capsys.readouterr().out)
    assert dry["evidence_kind"] == EVIDENCE_PROVIDER_CONTINUITY
    assert dry["status"] == "dry_run"
    assert dry["provider_validated"] is False

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret-should-not-print")
    assert main([
        "--provider-continuity",
        "--live",
        "--model",
        "claude-sonnet-4-5",
    ]) == 2
    closed = json.loads(capsys.readouterr().out)
    assert closed["status"] == "unavailable"
    assert "sk-secret-should-not-print" not in json.dumps(closed)
