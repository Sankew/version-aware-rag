import json
from datetime import date
from types import SimpleNamespace

from fastapi.testclient import TestClient

import pytest
from google.genai import errors as gemini_errors
from google.genai import types as gemini_types

from eval.score import answered, attributed, bootstrap, kappa, paired_delta, retrieval, wilson
from eval.validate import check
from rag import api
from rag.answer import ABSTAIN, answer, citations, judge, prompt
from rag.ingest import ROOT, Section, chunk, manifest, parse
from rag.models import LLM, FakeEmbedder, LLMError, transient
from versioning import changelog, evaluate_contradictions, parse_as_of, sentences


def row(document_id: str, version: int) -> dict:
    return next(r for r in manifest() if r["document_id"] == document_id and r["version"] == version)


def test_repository_data_is_valid():
    assert check() == []


def test_parse_markdown_html_and_pdf_into_sections():
    v3 = {s.heading: s.text for s in parse(ROOT / row("employee-handbook", 3)["path"], "Employee Handbook")}
    assert "Remote work" not in v3 and "Where you work" in v3 and "Up to 10 unused days" in v3["Vacation"]
    faq = {s.heading: s.text for s in parse(ROOT / row("benefits-faq", 1)["path"], "Benefits FAQ")}
    assert "20 vacation days" in faq["Time off"] and "Intranet home" in faq["Benefits FAQ"]
    calendar = parse(ROOT / row("holiday-calendar-2026", 1)["path"], "2026 Holiday Calendar")
    assert "14 Sep | Founders' Day" in calendar[0].text
    pdf = parse(ROOT / row("tern-datasheet", 2)["path"], "Tern AMR Datasheet")
    assert [s.heading for s in pdf] == ["Tern AMR Datasheet"] and "600 kg" in pdf[0].text


def test_chunks_respect_budget_and_carry_overlap():
    text = "\n".join(f"Line {i} has six words here." for i in range(40))
    chunks = chunk({"document_id": "d", "version": 1, "title": "T"}, [Section("S", text)], budget=40, overlap=10)
    assert len(chunks) > 1
    assert all(len(c.content.split()) <= 40 for c in chunks)
    assert all(c.content.startswith("T, S\n") for c in chunks)
    first, second = chunks[0].content.splitlines(), chunks[1].content.splitlines()
    assert first[-1] == second[1]  # last line of a chunk opens the next one


def test_carried_overlap_never_pushes_a_chunk_over_budget_and_empty_sections_are_skipped():
    lines = ["one two three four five", " ".join("abcdefghijklmnopqrstuvwxyzabcd")]  # 5 and 30 tokens
    chunks = chunk({"document_id": "d", "version": 1, "title": "T"}, [Section("S", "\n".join(lines * 3)), Section("Empty", "")],
                   budget=36, overlap=10)
    assert all(len(c.content.split()) <= 36 for c in chunks) and len(chunks) > 1
    assert {c.section for c in chunks} == {"S"}


def test_html_keeps_text_after_the_last_block_and_rows_without_closing_tags(tmp_path):
    page = tmp_path / "p.html"
    page.write_text("<h1>Title</h1><p>First.</p>Trailing text with no closing tag", encoding="utf-8")
    assert parse(page, "T")[0].text == "First.\nTrailing text with no closing tag"
    page.write_text("<h1>Fees</h1><table><tr><td>A<td>1<tr><td>B<td>2</table><p>After.", encoding="utf-8")
    assert parse(page, "T")[0].text == "A | 1\nB | 2\nAfter."


def test_as_of_dates():
    today = date(2026, 10, 1)
    assert parse_as_of("How many vacation days do we get?") == today
    assert parse_as_of("How many days did we get in 2024?") == date(2024, 12, 31)
    assert parse_as_of("As of June 2025, how many office days?") == date(2025, 6, 30)
    assert parse_as_of("As of 20 February 2026, what was the carry-over?") == date(2026, 2, 20)
    assert parse_as_of("What changed on March 1, 2026?") == date(2026, 3, 1)
    assert parse_as_of("What will apply in 2027?") == today  # capped at the eval clock
    assert parse_as_of("As of 2026-01-15, who approved?") == date(2026, 1, 15)


def test_as_of_ignores_impossible_dates_and_numbers_that_are_not_years():
    today = date(2026, 10, 1)
    assert parse_as_of("As of 31 February 2026, what applied?") == date(2026, 2, 28)  # the month's last day
    assert parse_as_of("As of February 31, 2026?") == date(2026, 2, 28)
    assert parse_as_of("As of 2025-02-30?") == date(2025, 2, 28)
    assert parse_as_of("What applied on 2026-13-01?") == today
    assert parse_as_of("Who approves a purchase for 1500 dollars?") == today
    assert parse_as_of("Is a battery covered for 1200 charge cycles?") == today
    assert parse_as_of("May employees buy their old laptop?") == today


def test_changelog_reports_added_removed_and_changed_sections():
    log = changelog("employee-handbook")
    assert "**Removed** Remote work" in log and "**Added** Where you work" in log
    assert "- Full-time employees receive 20 paid vacation days" in log and "+ Full-time employees receive 25" in log


class FakeClient:
    def __init__(self, text: str, stop_reason: str = "end_turn") -> None:
        self.text, self.stop_reason, self.calls = text, stop_reason, []
        self.messages = self

    def create(self, **request):
        self.calls.append(request)
        return SimpleNamespace(stop_reason=self.stop_reason, model=request["model"],
                               content=[SimpleNamespace(type="text", text=self.text)],
                               usage=SimpleNamespace(input_tokens=100, output_tokens=20))


def test_llm_cache_and_request_shape(tmp_path):
    client = FakeClient(json.dumps({"correct": True, "reason": "same number"}))
    llm = LLM(client, tmp_path)
    first = judge(llm, "Q?", "25 days.", "You get 25 days [employee-handbook:Vacation].")
    second = judge(llm, "Q?", "25 days.", "You get 25 days [employee-handbook:Vacation].")
    assert first["correct"] and not first["cached"] and second["cached"] and len(client.calls) == 1
    request = client.calls[0]
    assert "temperature" not in request and request["output_config"]["effort"] == "high"
    assert request["output_config"]["format"]["type"] == "json_schema"
    assert [p.suffix for p in tmp_path.iterdir()] == [".json"]  # written atomically, no temp files left


@pytest.mark.parametrize("stop_reason", ["max_tokens", "refusal", "end_turn"])  # end_turn: the JSON is cut short
def test_incomplete_responses_raise_and_are_not_cached(tmp_path, stop_reason):
    client = FakeClient('{"correct": tr', stop_reason)
    llm = LLM(client, tmp_path)
    for _ in range(2):
        with pytest.raises(LLMError):
            judge(llm, "Q?", "25 days.", "25 days.")
    assert len(client.calls) == 2 and not list(tmp_path.iterdir())


def test_versioned_prompt_and_citations():
    excerpt = {"document_id": "employee-handbook", "section": "Vacation", "title": "Employee Handbook", "version": 2,
               "effective_from": date(2025, 3, 1), "effective_to": date(2026, 3, 1), "content": "25 days."}
    text = prompt("How many days?", [excerpt], as_of=date(2025, 6, 30))
    assert "version 2, in effect 2025-03-01 to 2026-03-01" in text and "As-of date: 2025-06-30" in text
    assert "version" not in prompt("How many days?", [excerpt])
    assert "<excerpt>\n[employee-handbook:Vacation]" in text and text.count("</excerpt>") == 1
    assert citations("25 days [employee-handbook:Vacation] and [benefits-faq:Time off] [employee-handbook:Vacation].") == [
        ("employee-handbook", "Vacation"), ("benefits-faq", "Time off")]


def test_a_refused_or_truncated_answer_is_scored_wrong_with_its_cost(tmp_path):
    q = {"question": "How many days?", "reference_answer": "25.", "sources": []}
    excerpt = {"document_id": "employee-handbook", "section": "Vacation", "content": "25 days."}
    row = answered(LLM(FakeClient("25", "max_tokens"), tmp_path), q, [excerpt], None)
    assert row["correct"] == 0.0 and row["generation_usd"] > 0 and "max_tokens" in row["answer_error"]


def test_answer_keeps_only_retrieved_citations_and_abstains_without_a_call_when_nothing_is_retrieved(tmp_path):
    client = FakeClient("25 days [employee-handbook:Vacation], see also [made-up:Page].")
    excerpt = {"document_id": "employee-handbook", "section": "Vacation", "content": "25 days."}
    result = answer(LLM(client, tmp_path), "How many days?", [excerpt])
    assert result["citations"] == [("employee-handbook", "Vacation")]
    assert result["unsupported_citations"] == [("made-up", "Page")]
    assert "[made-up:Page]" not in result["text"] and "[employee-handbook:Vacation]" in result["text"]
    empty = answer(LLM(client, tmp_path), "How many days?", [])
    assert empty["text"] == ABSTAIN and len(client.calls) == 1


def test_metrics():
    mean, lo, hi = bootstrap([1, 1, 0, 1, 0, 1, 1, 1])
    assert lo <= mean <= hi and mean == 0.75
    lo, hi = wilson(3, 3)
    assert 0.4 < lo < 0.5 and hi == 1.0  # not [1, 1] at n = 3
    sources = [{"document_id": "employee-handbook", "version": 3}]
    v2 = {"document_id": "employee-handbook", "version": 2, "effective_from": date(2025, 3, 1), "effective_to": date(2026, 3, 1)}
    v3 = {"document_id": "employee-handbook", "version": 3, "effective_from": date(2026, 3, 1), "effective_to": None}
    on = date(2026, 10, 1)
    assert retrieval([v2], sources, on) == {"hit": 0.0, "all_sources": 0.0, "top1_is_gold": 0.0, "stale_frac": 1.0}
    assert retrieval([v2, v3], sources, on) == {"hit": 1.0, "all_sources": 1.0, "top1_is_gold": 0.0, "stale_frac": 0.5}
    assert retrieval([], [], on) == {"hit": None, "all_sources": None, "top1_is_gold": None, "stale_frac": None}
    assert attributed("Per version 3, effective 1 March 2026, 25 days.", sources) == 1.0
    assert attributed("25 days (handbook v3).", sources) == 1.0
    assert attributed("25 days.", sources) == 0.0


def test_paired_delta_resamples_groups_and_kappa():
    a = [{"id": str(i), "group": f"g{i // 2}", "correct": 0.0} for i in range(10)]
    b = [{"id": str(i), "group": f"g{i // 2}", "correct": float(i % 2)} for i in range(10)]
    result = paired_delta(a, b, "correct")
    assert result["delta"] == 0.5 and result["groups"] == 5 and result["ci95"] == [0.5, 0.5]  # every group is +0.5
    assert kappa([(1, 1), (0, 0), (1, 0), (1, 1)]) == {"n": 4, "agreement": 0.75, "kappa": 0.5,
                                                        "judge_yes_human_no": 1, "judge_no_human_yes": 0}


def test_contradiction_evaluation_counts_pairs_by_section():
    def c(doc, section):
        return {"document_id": doc, "version": 1, "section": section}
    chunks = [c("faq", "Time off"), c("handbook", "Vacation"), c("contractors", "Time off")]
    labels = [{"a": c("faq", "Time off"), "b": c("handbook", "Vacation"), "label": "contradiction", "kind": "numeric"},
              {"a": c("contractors", "Time off"), "b": c("handbook", "Vacation"), "label": "consistent", "kind": "scope"}]
    result = evaluate_contradictions(labels, chunks, {(0, 1): 0.9, (1, 2): 0.8}, [(0, 1, 0.8), (1, 2, 0.7)])
    assert result["candidate_recall"] == 1.0 and result["nli_recall"] == 1.0 and result["end_to_end_recall"] == 1.0
    assert result["precision_on_labelled"] == 0.5 and result["false_positives_on_hard_negatives"] == 1


def test_nli_recall_counts_only_pairs_the_candidate_filter_surfaced():
    def c(doc, section):
        return {"document_id": doc, "version": 1, "section": section}
    chunks = [c("a", "x"), c("b", "x"), c("c", "x"), c("d", "x")]
    labels = [{"a": c("a", "x"), "b": c("b", "x"), "label": "contradiction", "kind": "numeric"},
              {"a": c("c", "x"), "b": c("d", "x"), "label": "contradiction", "kind": "policy"}]
    result = evaluate_contradictions(labels, chunks, {(0, 1): 0.9}, [(0, 1, 0.9)])  # c-d was never a candidate
    assert result["candidate_recall"] == 0.5 and result["nli_recall"] == 1.0 and result["end_to_end_recall"] == 0.5
    assert result["by_kind"]["contradiction/policy"] == {"labelled": 1, "surfaced": 0, "flagged": 0}


def test_sentences_skip_header_and_document_stamps():
    content = ("Handbook, Intro\nVersion 2 · Effective 1 March 2025 · Owner: People Operations\nYou get 25 days. Plan ahead.\n"
               "Published December 2025. 12 paid holidays.\nThe calendar is published each December.\n"
               "Applies to: full-time employees")
    assert sentences(content) == ["You get 25 days.", "12 paid holidays.", "The calendar is published each December.",
                                  "Applies to: full-time employees"]


def test_api_validates_answers_and_maps_model_failures(monkeypatch):
    chunk_row = {"document_id": "employee-handbook", "version": 3, "section": "Vacation", "title": "Employee Handbook",
                 "effective_from": date(2026, 3, 1), "effective_to": None, "content": "25 days.", "score": 0.9}
    searched = []

    def fake_search(svc, question, mode, as_of):
        searched.append(as_of)
        return [chunk_row]

    def fake_answer(llm, question, chunks, as_of):
        if "fail" in question:
            raise LLMError("max_tokens")
        return {"text": "25 days [employee-handbook:Vacation].", "citations": [("employee-handbook", "Vacation")]}
    monkeypatch.setattr(api, "search", fake_search)
    monkeypatch.setattr(api, "answer", fake_answer)
    api.app.dependency_overrides[api.services] = lambda: SimpleNamespace(embedder=FakeEmbedder(), llm="llm")
    try:
        client = TestClient(api.app)
        response = client.post("/ask", json={"question": "As of June 2025, how many days?"})
        assert response.status_code == 200
        body = response.json()
        assert body["as_of"] == "2025-06-30" and body["citations"][0]["section"] == "Vacation"
        assert body["sources"][0]["version"] == 3
        assert client.post("/ask", json={"question": "How many days?", "mode": "current"}).json()["as_of"] is None
        assert searched[-1] == date.today()  # the real clock, not the eval's
        assert client.post("/ask", json={"question": "please fail"}).status_code == 502
        assert client.post("/ask", json={"question": "   "}).status_code == 422
        assert client.post("/ask", json={"question": "a\x00b"}).status_code == 422
        assert client.post("/ask", json={"question": "x", "mode": "bogus"}).status_code == 422
    finally:
        api.app.dependency_overrides.clear()


def test_labelsheet_preserves_hand_marks(tmp_path, monkeypatch):
    import eval.score as scoring
    monkeypatch.setattr(scoring, "RESULTS", tmp_path)
    run = tmp_path / "p1_aware.csv"
    run.write_text("id,question,reference_answer,answer,correct\na,Q,25,25,1\n")
    sheet = tmp_path / "labels" / "p1_aware_sheet.csv"
    sheet.parent.mkdir()
    sheet.write_text("id,human_correct\na,1\n")
    with pytest.raises(SystemExit, match="already has hand marks"):
        scoring.labelsheet(run, 1, 0)


def test_real_model_client_requires_explicit_model_ids(monkeypatch):
    monkeypatch.delenv("GENERATOR_MODEL", raising=False)
    monkeypatch.delenv("JUDGE_MODEL", raising=False)
    with pytest.raises(ValueError, match="GENERATOR_MODEL"):
        LLM()


class FakeGemini:
    """Stands in for google.genai.Client: answers every call with `text` and the given finish reason."""

    def __init__(self, text: str, finish: str = "STOP") -> None:
        self.text, self.finish, self.calls = text, finish, []
        self.models = self

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        return SimpleNamespace(
            candidates=[SimpleNamespace(finish_reason=gemini_types.FinishReason[self.finish])], text=self.text,
            usage_metadata=SimpleNamespace(prompt_token_count=100, candidates_token_count=20, thoughts_token_count=30),
            model_version=f"{model}-001", prompt_feedback=None)


def test_gemini_models_get_a_json_schema_and_thinking_level_and_thinking_counts_as_output(tmp_path):
    gemini = FakeGemini(json.dumps({"correct": True, "reason": "same"}))
    schema = {"type": "object", "properties": {"correct": {"type": "boolean"}}}
    llm = LLM(cache_dir=tmp_path, gemini=gemini)
    result = llm.complete("gemini-test-pro", "Grade it.", "Q?", effort="high", max_tokens=16000, schema=schema)
    config = gemini.calls[0]["config"]
    assert config.system_instruction == "Grade it." and config.max_output_tokens == 16000
    assert config.thinking_config.thinking_level == gemini_types.ThinkingLevel.HIGH and config.temperature is None
    assert config.response_mime_type == "application/json" and config.response_json_schema == schema
    assert result["input_tokens"] == 100 and result["output_tokens"] == 50
    assert llm.complete("gemini-test-pro", "Grade it.", "Q?", effort="high", max_tokens=16000, schema=schema)["cached"]


@pytest.mark.parametrize("finish", ["MAX_TOKENS", "SAFETY"])
def test_gemini_incomplete_responses_raise_with_their_cost_and_are_not_cached(tmp_path, finish):
    llm = LLM(cache_dir=tmp_path, gemini=FakeGemini("partial", finish))
    with pytest.raises(LLMError) as raised:
        llm.complete("gemini-test-flash", "s", "p", effort="low", max_tokens=10)
    assert raised.value.output_tokens == 50 and not list(tmp_path.iterdir())


def test_gemini_rate_limits_are_transient_and_bad_requests_are_not():
    assert transient(gemini_errors.ClientError(429, {"error": {"message": "quota"}}))
    assert transient(gemini_errors.ServerError(503, {"error": {"message": "overloaded"}}))
    assert not transient(gemini_errors.ClientError(400, {"error": {"message": "unknown model"}}))
