from __future__ import annotations

"""Battery-driven vault contracts. Product unit tests stay in Marionette."""

from harness.compaction_archive import retain_archive_messages
from harness.compaction_residual import build_catalog_residual
from harness.compaction_vault import (
    _filler_like,
    build_plan_recap_chunk,
    build_turn_vault_section,
    index_elided_messages,
    is_recap_ask,
    retrieve_vault_chunks,
    retrieve_vault_result,
    vault_match_query,
)
from catalog_residual.battery import RESIDUAL_CASES, cases_by_id
from catalog_residual.live import score_end_task_text


def test_vault_recalls_battery_catalog_miss_probe(tmp_path):
    case = next(c for c in RESIDUAL_CASES if c.id == "catalog_miss_plain_fact")
    written = index_elided_messages(str(tmp_path), "sess-miss", list(case.transcript))
    assert written >= 1
    section = build_turn_vault_section(
        str(tmp_path),
        "sess-miss",
        case.probe_prompt,
    )
    lowered = section.lower()
    for token in case.must_contain:
        assert token.lower() in lowered


def test_vault_only_prose_selected_story_and_vault_hit(tmp_path):
    case = cases_by_id()["vault_only_prose_cutoff"]
    catalog = build_catalog_residual(list(case.transcript), char_budget=4000)
    assert "fourteenth of each month" in catalog.lower()
    written = index_elided_messages(str(tmp_path), "sess-prose", list(case.transcript))
    assert written >= 1
    section = build_turn_vault_section(str(tmp_path), "sess-prose", case.probe_prompt)
    assert "fourteenth of each month" in section.lower()


def test_vault_narrative_and_paraphrase_miss_lexical_twin_false_hits(tmp_path):
    narrative = cases_by_id()["vault_narrative_no_overlap"]
    paraphrase = cases_by_id()["vault_paraphrase_no_overlap"]
    twin = cases_by_id()["vault_false_retrieve_twin"]
    for case, sid in (
        (narrative, "sess-narr"),
        (paraphrase, "sess-para"),
        (twin, "sess-twin"),
    ):
        catalog = build_catalog_residual(list(case.transcript), char_budget=4000)
        lowered = catalog.lower()
        for token in case.must_contain:
            assert token.lower() in lowered
        index_elided_messages(str(tmp_path), sid, list(case.transcript))

    recap_q = vault_match_query(narrative.probe_prompt)
    assert "spare" not in recap_q.lower()
    assert is_recap_ask(narrative.probe_prompt) is True
    raw_fts = vault_match_query(narrative.probe_prompt)
    assert "Remind" in raw_fts or "decided" in raw_fts
    plan = build_plan_recap_chunk(list(narrative.transcript))
    assert "spare region" in plan.lower()
    narr = retrieve_vault_result(str(tmp_path), "sess-narr", narrative.probe_prompt)
    assert narr["route"] == "recap_plan"
    assert "spare region" in "\n".join(narr["hits"]).lower()
    assert "earlier facts" not in "\n".join(narr["hits"]).lower()

    para_q = vault_match_query(paraphrase.probe_prompt)
    assert "twenty" not in para_q.lower()
    assert "omega" not in para_q.lower()
    para = retrieve_vault_result(str(tmp_path), "sess-para", paraphrase.probe_prompt)
    assert para["route"] == "empty"
    assert para["hits"] == []

    twin_hits = retrieve_vault_chunks(str(tmp_path), "sess-twin", twin.probe_prompt)
    twin_blob = "\n".join(twin_hits).lower()
    assert "spare region" in twin_blob
    assert "primary region" not in twin_blob
    twin_section = build_turn_vault_section(
        str(tmp_path), "sess-twin", twin.probe_prompt
    )
    assert len(twin_section) >= 80

    assert score_end_task_text(paraphrase, "Invoices freeze on the 27th.")[
        "end_task_success"
    ] is True
    assert score_end_task_text(paraphrase, "Invoices freeze mid-month.")[
        "end_task_success"
    ] is False


def test_vault_peek_evicted_case_drops_archive_and_keeps_vault(tmp_path):
    import json

    case = cases_by_id()["vault_peek_evicted_cutoff"]
    assert case.hide_peek is False
    retained = retain_archive_messages(list(case.transcript))
    assert "fourteenth" not in json.dumps(retained).lower()
    written = index_elided_messages(str(tmp_path), "sess-evict-case", list(case.transcript))
    assert written >= 1
    section = build_turn_vault_section(
        str(tmp_path),
        "sess-evict-case",
        case.probe_prompt,
    )
    assert "fourteenth of each month" in section.lower()


def test_topic_last_wins_drops_superseded_obligation(tmp_path):
    reversal = cases_by_id()["unprefixed_reversal"]
    plan = build_plan_recap_chunk(list(reversal.transcript))
    low = plan.lower()
    assert "write to the live ledger now" in low
    assert "east replica is retired" in low
    assert "don't write" not in low
    assert "only sink" not in low
    assert "reversed." not in low

    twin = cases_by_id()["vault_false_retrieve_twin"]
    twin_plan = build_plan_recap_chunk(list(twin.transcript))
    assert "spare region" in twin_plan.lower()
    assert "primary region" not in twin_plan.lower()

    kept = cases_by_id()["unprefixed_obligation"]
    kept_plan = build_plan_recap_chunk(list(kept.transcript))
    assert "don't write" in kept_plan.lower() or "live ledger" in kept_plan.lower()

    index_elided_messages(str(tmp_path), "sess-rev", list(reversal.transcript))
    rev_hit = retrieve_vault_result(
        str(tmp_path), "sess-rev", reversal.probe_prompt
    )
    rev_blob = "\n".join(rev_hit["hits"]).lower()
    assert "write to the live ledger now" in rev_blob
    assert "don't write" not in rev_blob
    assert "only sink" not in rev_blob

    index_elided_messages(str(tmp_path), "sess-keep", list(kept.transcript))
    keep_hit = retrieve_vault_result(
        str(tmp_path), "sess-keep", kept.probe_prompt
    )
    keep_blob = "\n".join(keep_hit["hits"]).lower()
    assert "live ledger" in keep_blob


def test_vault_selector_default_worthy_contract(tmp_path):
    catalog = cases_by_id()
    leak = catalog["vault_selector_plausible_filler"]
    kept = catalog["vault_selector_docs_only_plan"]
    capped = catalog["vault_selector_cap_drops_late"]
    assistant = catalog["vault_selector_assistant_only"]
    wrong = catalog["vault_selector_miss_wrong_plan"]
    false_fire = catalog["vault_recap_false_fire"]

    leak_plan = build_plan_recap_chunk(list(leak.transcript))
    assert "spare region" in leak_plan.lower()
    assert "warmer" in leak_plan.lower()

    assert not _filler_like(
        "please keep this docs only: ship the canary to the spare "
        "region before Friday."
    )
    kept_plan = build_plan_recap_chunk(list(kept.transcript))
    assert "spare region" in kept_plan.lower()
    index_elided_messages(str(tmp_path), "sess-docs", list(kept.transcript))
    kept_hit = retrieve_vault_result(
        str(tmp_path), "sess-docs", kept.probe_prompt
    )
    assert kept_hit["route"] == "recap_plan"
    assert "spare region" in "\n".join(kept_hit["hits"]).lower()

    capped_plan = build_plan_recap_chunk(list(capped.transcript))
    assert "spare region" in capped_plan.lower()
    assert "primary region" not in capped_plan.lower()

    assistant_plan = build_plan_recap_chunk(list(assistant.transcript))
    assert "spare region" in assistant_plan.lower()

    index_elided_messages(str(tmp_path), "sess-wrong", list(wrong.transcript))
    wrong_hit = retrieve_vault_result(str(tmp_path), "sess-wrong", wrong.probe_prompt)
    assert wrong_hit["route"] == "empty"
    assert wrong_hit["hits"] == []

    assert is_recap_ask(false_fire.probe_prompt) is False
    index_elided_messages(str(tmp_path), "sess-fire", list(false_fire.transcript))
    fire_hit = retrieve_vault_result(
        str(tmp_path), "sess-fire", false_fire.probe_prompt
    )
    assert fire_hit["route"] != "recap_plan"
    assert "spare region" not in "\n".join(fire_hit["hits"]).lower()
