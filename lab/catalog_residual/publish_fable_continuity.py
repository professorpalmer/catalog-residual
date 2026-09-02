from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Optional

FABLE_MODEL = "claude-fable-5-1"
PUBLIC_SCHEMA = "catalog_residual_fable_continuity/v1"
INPUT_PRICE_PER_MILLION = 10.0
OUTPUT_PRICE_PER_MILLION = 50.0
FORBIDDEN_PUBLIC_KEYS = frozenset({
    "api_key",
    "anthropic-workspace-id",
    "data",
    "encrypted_content",
    "error",
    "messages",
    "reasoning",
    "redacted_thinking",
    "request_body",
    "residual_text",
    "signature",
    "thinking",
    "thinking_signature",
    "thinking_text",
    "thought",
    "workspace_id",
    "x-api-key",
})
PUBLIC_RECEIPT_NAME = "fable-continuity-receipt.json"
PUBLIC_METRICS_NAME = "fable-continuity-metrics.json"
PUBLIC_LEDGER_NAME = "fable-run-ledger.json"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _forbidden_keys(value: Any, parent_key: str = "") -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            normalized_key = str(key).lower()
            if (
                parent_key != "dropped_types"
                and normalized_key in FORBIDDEN_PUBLIC_KEYS
            ):
                found.add(str(key))
            found.update(_forbidden_keys(item, parent_key=normalized_key))
    elif isinstance(value, list):
        for item in value:
            found.update(_forbidden_keys(item, parent_key=parent_key))
    return found


def _outcomes(row: dict[str, Any]) -> Iterable[dict[str, Any]]:
    yield row["bootstrap"]
    for arm in (
        "unprojected",
        "projected_tail",
        "catalog_residual",
        "no_state",
        "drop_block",
    ):
        outcome = row.get(arm)
        if isinstance(outcome, dict) and "http_status" in outcome:
            yield outcome


def _bootstrap_qualified(row: dict[str, Any]) -> bool:
    bootstrap = row["bootstrap"]
    qualified = bool(
        bootstrap.get("http_status") == 200
        and bootstrap.get("model_match")
        and bootstrap.get("has_signed_bound_block")
        and bootstrap.get("commitment_in_bound_thinking")
    )
    if row.get("case_id") == "hidden_only_checkpoint":
        return qualified and bool(
            bootstrap.get("commitment_absent_from_observable")
        )
    return qualified and bool(bootstrap.get("required_tool_use_present"))


def _case_metrics(
    rows: list[dict[str, Any]],
    case_id: str,
) -> dict[str, Any]:
    selected = [row for row in rows if row.get("case_id") == case_id]
    _require(selected, f"receipt contains no {case_id} rows")

    hidden = case_id == "hidden_only_checkpoint"
    qualified = [row for row in selected if _bootstrap_qualified(row)]
    return {
        "case_id": case_id,
        "planned_trials": len(selected),
        "qualified_trials": len(qualified),
        "bootstrap_failures": len(selected) - len(qualified),
        "causal_passes": sum(bool(row.get("causal_ok")) for row in selected),
        "authentic_signed_bootstraps": sum(
            bool(row["bootstrap"].get("has_signed_bound_block"))
            for row in selected
        ),
        "expected_binding_rejections": sum(
            (row.get("unprojected") or {}).get("http_status") == 400
            and bool((row.get("unprojected") or {}).get("binding_mismatch"))
            for row in selected
        ),
        "projected_tail_successes": sum(
            bool((row.get("projected_tail") or {}).get("task_success"))
            for row in selected
        ),
        "catalog_residual_successes": sum(
            bool((row.get("catalog_residual") or {}).get("task_success"))
            for row in selected
        ),
        "projected_tail_refusals": sum(
            (row.get("projected_tail") or {}).get("stop_reason") == "refusal"
            for row in selected
        ),
        "catalog_residual_refusals": sum(
            (row.get("catalog_residual") or {}).get("stop_reason") == "refusal"
            for row in selected
        ),
        "invented_checkpoint_outcomes": sum(
            bool((row.get(arm) or {}).get("invented_concrete_value"))
            for row in selected
            for arm in ("projected_tail", "catalog_residual", "no_state")
        ),
        "no_state_unknowns": sum(
            bool((row.get("no_state") or {}).get("exact_unknown"))
            for row in selected
        ),
        "honesty_clean_trials": sum(
            bool((row.get("projected_tail") or {}).get("honesty_clean"))
            and bool((row.get("catalog_residual") or {}).get("honesty_clean"))
            and bool((row.get("no_state") or {}).get("honesty_clean"))
            for row in selected
        ),
        "residual_state_expected": (
            "absent" if hidden else "present"
        ),
        "residual_state_matches": sum(
            bool(row["residual"].get("contains_commitment")) is (not hidden)
            for row in qualified
        ),
    }


def _public_outcome(outcome: Any) -> Optional[dict[str, Any]]:
    if not isinstance(outcome, dict) or "http_status" not in outcome:
        return None
    return {
        "http_status": outcome["http_status"],
        "stop_reason": outcome["stop_reason"],
        "task_success": bool(outcome["task_success"]),
        "binding_mismatch": bool(outcome["binding_mismatch"]),
        "recovered_checkpoint": bool(outcome["recovered_checkpoint"]),
        "exact_unknown": bool(outcome["exact_unknown"]),
    }


def _public_trial(row: dict[str, Any]) -> dict[str, Any]:
    bootstrap_qualified = _bootstrap_qualified(row)
    return {
        "case_id": row["case_id"],
        "repeat_index": row["repeat_index"],
        "bootstrap_qualified": bootstrap_qualified,
        "causal_ok": bool(row["causal_ok"]),
        "failure": str(row.get("failure") or ""),
        "bound_block_count": row["bound_state"]["dropped_count"],
        "bound_payload_bytes": row["bound_state"]["dropped_bytes"],
        "residual_bytes": int(row["residual"].get("byte_length") or 0),
        "residual_contains_checkpoint": bool(
            row["residual"].get("contains_commitment")
        ),
        "unprojected": _public_outcome(row.get("unprojected")),
        "projected_tail": _public_outcome(row.get("projected_tail")),
        "catalog_residual": _public_outcome(row.get("catalog_residual")),
        "no_state": _public_outcome(row.get("no_state")),
        "drop_block": _public_outcome(row.get("drop_block")),
        "request_count": row["request_count"],
        "usage": row["usage_total"],
    }


def _selected_fields(source: Any, names: Iterable[str]) -> dict[str, Any]:
    if not isinstance(source, dict):
        return {}
    return {
        name: source[name]
        for name in names
        if name in source
    }


def _public_provider_outcome(outcome: Any) -> Optional[dict[str, Any]]:
    if not isinstance(outcome, dict) or "http_status" not in outcome:
        return None
    return _selected_fields(outcome, (
        "answer_excerpt",
        "arm",
        "binding_mismatch",
        "elapsed_ms",
        "exact_unknown",
        "honesty_clean",
        "http_status",
        "invented_concrete_value",
        "model_match",
        "recovered_checkpoint",
        "request_id_present",
        "request_id_sha256",
        "served_model",
        "status_class",
        "stop_reason",
        "task_success",
        "usage",
    ))


def build_public_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    public_rows = []
    for row in receipt["rows"]:
        public_rows.append({
            **_selected_fields(row, (
                "causal_ok",
                "case_id",
                "elapsed_ms_total",
                "evidence_kind",
                "fable_protocol",
                "failure",
                "model_requested",
                "model_served",
                "observable_commitment_locations",
                "planned_checkpoint",
                "protocol_role",
                "provider_validated",
                "repeat_index",
                "request_count",
                "schema",
                "status",
                "usage_total",
            )),
            "bootstrap": _selected_fields(row.get("bootstrap"), (
                "commitment_absent_from_observable",
                "commitment_in_bound_thinking",
                "elapsed_ms",
                "has_signed_bound_block",
                "http_completed",
                "http_status",
                "model_match",
                "observable_commitment_locations",
                "request_id_present",
                "request_id_sha256",
                "required_tool_use_present",
                "served_model",
                "status_class",
                "stop_reason",
                "usage",
            )),
            "bound_state": {
                **_selected_fields(row.get("bound_state"), (
                    "authentic_signed",
                    "commitment_in_bound",
                    "dropped_bytes",
                    "dropped_count",
                )),
                "dropped_blocks": [
                    _selected_fields(block, (
                        "payload_bytes",
                        "payload_sha256",
                        "type",
                    ))
                    for block in (row.get("bound_state") or {}).get(
                        "dropped_blocks", []
                    )
                ],
            },
            "projection": _selected_fields(row.get("projection"), (
                "blocks_dropped",
                "bound_thinking_after_prefix",
                "dropped_types",
                "projected_request_valid",
                "projection_approved",
                "projection_complete",
                "tool_round_resolved",
                "tool_round_valid",
                "wire_protocol_valid",
            )),
            "residual": _selected_fields(row.get("residual"), (
                "byte_length",
                "compacted",
                "contains_commitment",
                "digest",
            )),
            "unprojected": _public_provider_outcome(row.get("unprojected")),
            "projected_tail": _public_provider_outcome(
                row.get("projected_tail")
            ),
            "catalog_residual": _public_provider_outcome(
                row.get("catalog_residual")
            ),
            "no_state": _public_provider_outcome(row.get("no_state")),
            "drop_block": _public_provider_outcome(row.get("drop_block")),
        })

    return {
        **_selected_fields(receipt, (
            "dry_run",
            "bootstrap_max_tokens",
            "continuation_max_tokens",
            "evidence_kind",
            "fable_protocol",
            "failure",
            "mode",
            "models",
            "n",
            "protocol",
            "provider_continuity_validated",
            "provider_validated",
            "real_network",
            "repeats",
            "request_count",
            "schema",
            "status",
            "transport",
            "usage_total",
            "workspace_routing",
        )),
        "rows": public_rows,
    }


def build_public_metrics(
    receipt: dict[str, Any],
    *,
    receipt_sha256: str,
    run_date: str,
) -> dict[str, Any]:
    forbidden = _forbidden_keys(receipt)
    _require(not forbidden, "receipt contains private keys: " + ", ".join(
        sorted(forbidden)
    ))
    rows = receipt.get("rows")
    _require(isinstance(rows, list) and rows, "receipt has no trial rows")
    repeats = int(receipt.get("repeats") or 0)
    _require(receipt.get("models") == [FABLE_MODEL], "model is not exact Fable 5.1")
    _require(receipt.get("real_network") is True, "receipt is not a real network run")
    _require(repeats >= 10, "confirmatory receipt requires at least 10 repeats")
    _require(len(rows) == repeats * 2, "receipt trial count does not match design")
    bootstrap_max_tokens = int(receipt.get("bootstrap_max_tokens") or 0)
    continuation_max_tokens = int(
        receipt.get("continuation_max_tokens") or 0
    )
    _require(bootstrap_max_tokens > 0, "bootstrap token budget is missing")
    _require(continuation_max_tokens > 0, "continuation token budget is missing")
    qualified_rows = [row for row in rows if _bootstrap_qualified(row)]
    _require(
        qualified_rows,
        "receipt contains no manipulation-qualified trials",
    )
    _require(
        all(
            outcome.get("stop_reason") != "max_tokens"
            for row in rows
            for outcome in _outcomes(row)
        ),
        "one or more provider outcomes hit max_tokens",
    )

    outcomes = [outcome for row in rows for outcome in _outcomes(row)]
    _require(
        all(bool(outcome.get("request_id_present")) for outcome in outcomes),
        "one or more provider request IDs are missing",
    )
    _require(
        len(outcomes) == int(receipt.get("request_count") or 0),
        "provider request count does not reconcile",
    )
    usage = receipt["usage_total"]
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    estimated_cost = (
        input_tokens * INPUT_PRICE_PER_MILLION
        + output_tokens * OUTPUT_PRICE_PER_MILLION
    ) / 1_000_000

    case_metrics = [
        _case_metrics(rows, "hidden_only_checkpoint"),
        _case_metrics(rows, "observable_checkpoint"),
    ]
    hidden_metrics, observable_metrics = case_metrics
    return {
        "schema": PUBLIC_SCHEMA,
        "run_date": run_date,
        "receipt_sha256": receipt_sha256,
        "model": FABLE_MODEL,
        "transport": receipt["transport"],
        "workspace_routing": bool(receipt["workspace_routing"]),
        "thinking_binding_beta": "thinking-binding-controls-2026-08-01",
        "prefix_mismatch_behaviors": ["error", "drop_block"],
        "bootstrap_max_output_tokens": bootstrap_max_tokens,
        "continuation_max_output_tokens": continuation_max_tokens,
        "repeats_per_case": repeats,
        "planned_trials": len(rows),
        "qualified_trials": len(qualified_rows),
        "bootstrap_failures": len(rows) - len(qualified_rows),
        "causal_passes": sum(bool(row["causal_ok"]) for row in rows),
        "qualified_causal_passes": sum(
            bool(row["causal_ok"]) for row in qualified_rows
        ),
        "provider_continuity_validated": bool(
            receipt.get("provider_continuity_validated")
        ),
        "request_count": int(receipt["request_count"]),
        "request_ids_present": len(outcomes),
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_input_tokens": int(
                usage.get("cache_read_input_tokens") or 0
            ),
            "cache_creation_input_tokens": int(
                usage.get("cache_creation_input_tokens") or 0
            ),
        },
        "pricing": {
            "input_usd_per_million": INPUT_PRICE_PER_MILLION,
            "output_usd_per_million": OUTPUT_PRICE_PER_MILLION,
            "estimated_cost_usd": round(estimated_cost, 4),
        },
        "cases": case_metrics,
        "trials_detail": [_public_trial(row) for row in rows],
        "claim_boundary": {
            "supported": (
                "After bound thinking was removed, Catalog residual produced "
                f"the predicted target result in "
                f"{hidden_metrics['catalog_residual_successes']}/"
                f"{hidden_metrics['qualified_trials']} hidden-only and "
                f"{observable_metrics['catalog_residual_successes']}/"
                f"{observable_metrics['qualified_trials']} observable trials, "
                "with no invented checkpoint."
            ),
            "not_supported": (
                "Catalog Residual recovers discarded reasoning, recovers "
                "hidden-only state, or outperforms an intact projected tail."
            ),
        },
        "sources": {
            "anthropic_thinking": (
                "https://docs.anthropic.com/en/docs/about-claude/models/"
                "extended-thinking-models"
            ),
            "anthropic_release_notes": (
                "https://docs.anthropic.com/en/release-notes/api"
            ),
        },
    }


def _receipt_usage(receipt: dict[str, Any]) -> dict[str, int]:
    total = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }
    rows = receipt.get("rows") or []
    source = receipt.get("usage_total") or {}
    if not source and rows:
        for row in rows:
            for key in total:
                total[key] += int((row.get("usage_total") or {}).get(key) or 0)
        return total
    for key in total:
        total[key] = int(source.get(key) or 0)
    return total


def build_run_ledger(
    receipt_paths: Iterable[Path],
    *,
    selected_receipt: Path,
    run_date: str,
) -> dict[str, Any]:
    attempts = []
    selected_resolved = selected_receipt.resolve()
    for receipt_path in receipt_paths:
        receipt_bytes = receipt_path.read_bytes()
        receipt = json.loads(receipt_bytes)
        evidence_receipt = receipt.get("pi_p_c") or receipt
        rows = evidence_receipt.get("rows") or []
        failure = str(evidence_receipt.get("failure") or "")
        if "connection reset" in failure.lower():
            outcome = "transport_reset"
        elif evidence_receipt.get("status") in ("ok", "simulated"):
            outcome = str(evidence_receipt["status"])
        elif evidence_receipt.get("status") == "unvalidated":
            outcome = "causal_gate_not_met"
        else:
            outcome = str(evidence_receipt.get("status") or "unknown")
        usage = _receipt_usage(evidence_receipt)
        usage_reported = bool(
            evidence_receipt.get("usage_total")
            or any(row.get("usage_total") for row in rows)
        )
        estimated_cost = None
        if usage_reported:
            estimated_cost = round((
                usage["input_tokens"] * INPUT_PRICE_PER_MILLION
                + usage["output_tokens"] * OUTPUT_PRICE_PER_MILLION
            ) / 1_000_000, 4)
        protocol_legs = sum(
            isinstance(evidence_receipt.get(name), dict)
            and "http_status" in evidence_receipt[name]
            for name in (
                "bootstrap",
                "unprojected",
                "projected",
                "drop_block",
            )
        )
        planned_trials = int(evidence_receipt.get("n") or 0)
        causal_passes = sum(bool(row.get("causal_ok")) for row in rows)
        if evidence_receipt is not receipt:
            planned_trials = 1
            causal_passes = int(bool(
                evidence_receipt.get("provider_validated")
            ))
        qualified_trials = sum(_bootstrap_qualified(row) for row in rows)
        qualified_causal_passes = sum(
            bool(row.get("causal_ok"))
            for row in rows
            if _bootstrap_qualified(row)
        )
        if (
            outcome != "transport_reset"
            and rows
            and qualified_trials
            and qualified_causal_passes == qualified_trials
            and causal_passes < planned_trials
        ):
            outcome = "qualified_gate_passed_with_setup_failures"
        elif (
            outcome != "transport_reset"
            and rows
            and causal_passes < planned_trials
        ):
            outcome = "full_multi_arm_gate_not_met"
        attempts.append({
            "artifact": receipt_path.name,
            "artifact_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
            "selected_confirmatory_receipt": (
                receipt_path.resolve() == selected_resolved
            ),
            "evidence_kind": str(evidence_receipt.get("evidence_kind") or ""),
            "model": list(evidence_receipt.get("models") or [
                evidence_receipt.get("model")
                or evidence_receipt.get("model_requested")
            ]),
            "planned_trials": planned_trials,
            "rows_recorded": len(rows),
            "causal_passes": causal_passes,
            "qualified_trials": qualified_trials,
            "qualified_causal_passes": qualified_causal_passes,
            "completed_requests": max(
                int(evidence_receipt.get("request_count") or 0),
                sum(int(row.get("request_count") or 0) for row in rows),
                protocol_legs,
            ),
            "usage": usage,
            "usage_reported": usage_reported,
            "estimated_cost_usd": estimated_cost,
            "source_status": str(evidence_receipt.get("status") or ""),
            "outcome": outcome,
        })
    return {
        "schema": "catalog_residual_fable_run_ledger/v1",
        "run_date": run_date,
        "scope": "Self-reported local artifacts; not a provider billing export.",
        "reported_usage_estimated_cost_usd": round(sum(
            attempt["estimated_cost_usd"] or 0 for attempt in attempts
        ), 4),
        "cost_scope": (
            "Excludes artifacts without usage fields and the unreceipted event."
        ),
        "attempts": attempts,
        "unreceipted_events": [{
            "event": "Three-repeat budget pilot stopped before receipt write",
            "reason": (
                "Superseded when the continuation output allowance increased "
                "from 1,024 to 4,096 tokens."
            ),
        }],
    }


def publish(
    receipt_path: Path,
    output_directory: Path,
    *,
    run_date: str,
    history_receipts: Iterable[Path] = (),
) -> tuple[Path, Path, Path]:
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    public_receipt = build_public_receipt(receipt)
    public_receipt_text = json.dumps(
        public_receipt,
        indent=2,
        sort_keys=True,
    ) + "\n"
    public_receipt_sha256 = hashlib.sha256(
        public_receipt_text.encode("utf-8")
    ).hexdigest()
    metrics = build_public_metrics(
        receipt,
        receipt_sha256=public_receipt_sha256,
        run_date=run_date,
    )
    metrics["source_receipt_sha256"] = hashlib.sha256(receipt_bytes).hexdigest()
    history = list(history_receipts)
    if receipt_path not in history:
        history.append(receipt_path)
    ledger = build_run_ledger(
        history,
        selected_receipt=receipt_path,
        run_date=run_date,
    )

    output_directory.mkdir(parents=True, exist_ok=True)
    receipt_output = output_directory / PUBLIC_RECEIPT_NAME
    metrics_output = output_directory / PUBLIC_METRICS_NAME
    ledger_output = output_directory / PUBLIC_LEDGER_NAME
    receipt_output.write_text(
        public_receipt_text,
        encoding="utf-8",
    )
    metrics_output.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    ledger_output.write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt_output, metrics_output, ledger_output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("docs/data"),
    )
    parser.add_argument("--run-date", required=True)
    parser.add_argument(
        "--history-receipt",
        action="append",
        default=[],
        type=Path,
    )
    args = parser.parse_args()
    outputs = publish(
        args.receipt,
        args.output_directory,
        run_date=args.run_date,
        history_receipts=args.history_receipt,
    )
    print("\n".join(str(path) for path in outputs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
