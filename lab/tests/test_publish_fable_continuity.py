from __future__ import annotations

import json

import pytest

from catalog_residual.publish_fable_continuity import (
    FABLE_MODEL,
    build_public_metrics,
    build_public_receipt,
    build_run_ledger,
    publish,
)


def _outcome(
    *,
    recovered_checkpoint: bool = False,
    exact_unknown: bool = False,
    http_status: int = 200,
    binding_mismatch: bool = False,
) -> dict:
    return {
        "http_status": http_status,
        "binding_mismatch": binding_mismatch,
        "request_id_present": True,
        "stop_reason": "end_turn" if http_status == 200 else "",
        "task_success": True,
        "recovered_checkpoint": recovered_checkpoint,
        "exact_unknown": exact_unknown,
        "honesty_clean": True,
    }


def _trial(case_id: str, repeat_index: int) -> dict:
    hidden = case_id == "hidden_only_checkpoint"
    return {
        "case_id": case_id,
        "repeat_index": repeat_index,
        "causal_ok": True,
        "bootstrap": {
            "http_status": 200,
            "model_match": True,
            "has_signed_bound_block": True,
            "commitment_in_bound_thinking": True,
            "commitment_absent_from_observable": hidden,
            "required_tool_use_present": not hidden,
            "request_id_present": True,
            "stop_reason": "end_turn",
            "task_success": True,
            "binding_mismatch": False,
            "recovered_checkpoint": False,
            "exact_unknown": False,
            "honesty_clean": True,
        },
        "bound_state": {
            "dropped_count": 1,
            "dropped_bytes": 128,
        },
        "residual": {
            "byte_length": 700,
            "contains_commitment": not hidden,
        },
        "unprojected": _outcome(
            http_status=400,
            binding_mismatch=True,
        ),
        "projected_tail": _outcome(
            recovered_checkpoint=not hidden,
            exact_unknown=hidden,
        ),
        "catalog_residual": _outcome(
            recovered_checkpoint=not hidden,
            exact_unknown=hidden,
        ),
        "no_state": _outcome(exact_unknown=True),
        "drop_block": (
            _outcome(
                recovered_checkpoint=not hidden,
                exact_unknown=hidden,
                binding_mismatch=True,
            )
            if repeat_index == 0
            else None
        ),
        "request_count": 6 if repeat_index == 0 else 5,
        "usage_total": {
            "input_tokens": 100,
            "output_tokens": 50,
        },
    }


def _receipt() -> dict:
    rows = [
        _trial(case_id, repeat_index)
        for case_id in (
            "hidden_only_checkpoint",
            "observable_checkpoint",
        )
        for repeat_index in range(10)
    ]
    return {
        "models": [FABLE_MODEL],
        "real_network": True,
        "provider_continuity_validated": True,
        "transport": "stdlib",
        "workspace_routing": True,
        "bootstrap_max_tokens": 512,
        "continuation_max_tokens": 4096,
        "repeats": 10,
        "request_count": 102,
        "usage_total": {
            "input_tokens": 2000,
            "output_tokens": 1000,
        },
        "rows": rows,
    }


def test_build_public_metrics_requires_and_summarizes_confirmatory_run():
    metrics = build_public_metrics(
        _receipt(),
        receipt_sha256="a" * 64,
        run_date="2026-09-02",
    )

    assert metrics["planned_trials"] == 20
    assert metrics["qualified_trials"] == 20
    assert metrics["causal_passes"] == 20
    assert metrics["request_count"] == 102
    assert metrics["request_ids_present"] == 102
    assert metrics["pricing"]["estimated_cost_usd"] == 0.07
    assert metrics["bootstrap_max_output_tokens"] == 512
    assert metrics["continuation_max_output_tokens"] == 4096
    hidden, observable = metrics["cases"]
    assert hidden["catalog_residual_successes"] == 10
    assert hidden["residual_state_expected"] == "absent"
    assert observable["catalog_residual_successes"] == 10
    assert observable["residual_state_expected"] == "present"


def test_build_public_metrics_reports_failed_bootstrap_without_retrying():
    receipt = _receipt()
    failed_setup = receipt["rows"][0]
    failed_setup["bootstrap"]["commitment_absent_from_observable"] = False
    failed_setup["causal_ok"] = False
    failed_setup["failure"] = "hidden commitment leaked"
    for arm in (
        "unprojected",
        "projected_tail",
        "catalog_residual",
        "no_state",
        "drop_block",
    ):
        failed_setup[arm] = None
    failed_setup["request_count"] = 1
    receipt["request_count"] = 97
    receipt["provider_continuity_validated"] = False

    metrics = build_public_metrics(
        receipt,
        receipt_sha256="b" * 64,
        run_date="2026-09-02",
    )

    assert metrics["planned_trials"] == 20
    assert metrics["qualified_trials"] == 19
    assert metrics["bootstrap_failures"] == 1
    assert metrics["qualified_causal_passes"] == 19
    assert metrics["trials_detail"][0]["bootstrap_qualified"] is False


def test_build_public_metrics_keeps_failed_qualified_and_rejects_private_receipts():
    failed = _receipt()
    failed["rows"][1]["causal_ok"] = False
    metrics = build_public_metrics(
        failed,
        receipt_sha256="c" * 64,
        run_date="2026-09-02",
    )
    assert metrics["qualified_trials"] == 20
    assert metrics["qualified_causal_passes"] == 19
    assert metrics["trials_detail"][1]["causal_ok"] is False

    private = _receipt()
    private["signature"] = "provider-secret"
    with pytest.raises(ValueError, match="private keys"):
        build_public_metrics(
            private,
            receipt_sha256="d" * 64,
            run_date="2026-09-02",
        )

    safe_aggregate = _receipt()
    safe_aggregate["rows"][0]["projection"] = {
        "dropped_types": {"thinking": 1},
    }
    build_public_metrics(
        safe_aggregate,
        receipt_sha256="e" * 64,
        run_date="2026-09-02",
    )


def test_publish_writes_reconciled_receipt_and_metrics(tmp_path):
    receipt_path = tmp_path / "receipt.json"
    source = _receipt()
    source["unknown_private_blob"] = "must-not-be-published"
    receipt_path.write_text(json.dumps(source), encoding="utf-8")

    receipt_output, metrics_output, ledger_output = publish(
        receipt_path,
        tmp_path / "public",
        run_date="2026-09-02",
    )

    public_receipt = json.loads(receipt_output.read_text())
    assert public_receipt["request_count"] == 102
    assert "unknown_private_blob" not in public_receipt
    assert json.loads(metrics_output.read_text())["causal_passes"] == 20
    assert json.loads(ledger_output.read_text())["attempts"][0][
        "selected_confirmatory_receipt"
    ] is True


def test_public_receipt_is_allowlisted_and_ledger_retains_failed_attempt(tmp_path):
    source = _receipt()
    source["rows"][0]["bootstrap"]["unknown_provider_field"] = "private"
    public = build_public_receipt(source)
    assert "unknown_provider_field" not in public["rows"][0]["bootstrap"]

    failed_path = tmp_path / "failed.json"
    failed = _receipt()
    failed["status"] = "error"
    failed["failure"] = "[Errno 54] Connection reset by peer"
    failed["rows"] = failed["rows"][:4]
    failed_path.write_text(json.dumps(failed), encoding="utf-8")
    selected_path = tmp_path / "selected.json"
    selected_path.write_text(json.dumps(_receipt()), encoding="utf-8")
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps({
        "pi_p_c": {
            "evidence_kind": "pi_p_c",
            "model": FABLE_MODEL,
            "provider_validated": True,
            "status": "ok",
            "bootstrap": {"http_status": 200},
            "unprojected": {"http_status": 400},
            "projected": {"http_status": 200},
            "drop_block": {"http_status": 200},
        },
    }), encoding="utf-8")

    ledger = build_run_ledger(
        [protocol_path, failed_path, selected_path],
        selected_receipt=selected_path,
        run_date="2026-09-02",
    )

    protocol = ledger["attempts"][0]
    assert protocol["outcome"] == "ok"
    assert protocol["completed_requests"] == 4
    assert protocol["causal_passes"] == 1
    assert protocol["usage_reported"] is False
    assert protocol["estimated_cost_usd"] is None
    assert ledger["attempts"][1]["outcome"] == "transport_reset"
    assert ledger["attempts"][1]["selected_confirmatory_receipt"] is False
    assert ledger["attempts"][2]["selected_confirmatory_receipt"] is True
    assert ledger["reported_usage_estimated_cost_usd"] > 0
