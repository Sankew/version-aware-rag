# Version-aware RAG

[![Checks](https://github.com/Sankew/version-aware-rag/actions/workflows/checks.yml/badge.svg)](https://github.com/Sankew/version-aware-rag/actions/workflows/checks.yml)

Company documents change, and the old versions don't go away. A plain RAG system over a document store answers "how many vacation days do I get?" with whatever version it retrieves. Deleting old versions fixes that and breaks "how many did we get in 2024?". This project keeps every version, retrieves by the date a question is about, and says which version an answer came from. It also flags pages that still contradict current policy.

It is evaluated on a fictional company, Brackwater Robotics. Public corpora are already memorised by the model, which would hide whether retrieval found the right version.

```mermaid
flowchart LR
    Q[Question] --> D[As-of date<br/>from the question, else today]
    Q --> E[bge-small<br/>query embedding]
    D --> F[Versions in effect<br/>on that date]
    E --> S[Exact cosine search<br/>pgvector, top 5]
    F --> S
    S --> G[Configured generation model<br/>cites document:section,<br/>names the version]
    G --> A[Answer]
```

## What it does

- **Ingest never deletes.** Each document version is a row in `doc_versions` with its effective date range. Adding a version closes the previous one's range. Versions must arrive in order; v2 after v3 is rejected.
- **As-of retrieval.** `20 February 2026`, `June 2025` and `in 2024` in a question set the date (a month or a year means its last day). With no date, the date is today. Only 19xx and 20xx count as years, so "for 1200 charge cycles" stays a number. Retrieval filters on the effective date range, never on a `status` flag. Handbook v3 was uploaded on 10 February 2026 but takes effect on 1 March, so a status flag would give the wrong answer for anything asked in between.
- **Attribution.** Excerpts carry their version and effective dates, and answers name the version they used. Citations that don't match a retrieved excerpt are dropped and counted.
- **Change log.** `uv run python versioning.py changelog employee-handbook` aligns sections by heading and diffs them with `difflib`.
- **Contradictions.**
  - Chunks from different current documents with cosine similarity above a threshold are candidates.
  - For each candidate, the 3 most similar sentence pairs are checked. Every sentence is prefixed with its document title and section, so "contractors" or "full-time employees" survives the split. Document stamps (version lines, owners, "last updated") are dropped first, since they differ between any two pages.
  - `cross-encoder/nli-deberta-v3-base` scores each pair in both directions, and the higher score counts.
  - Each stage is scored on its own: candidate recall, then NLI recall on the pairs the filter surfaced, then end to end.

## Evaluation

The comparison runs the same questions through:

| Mode | Retrieves from | Fails on |
|---|---|---|
| `all` | every version | stale versions polluting current answers |
| `current` | the latest version of each document (the "delete old versions" approach) | historical questions |
| `aware` | versions in effect on the question's as-of date | misread dates |
| `labelled` | as `aware`, on the labelled as-of date instead of the parsed one (eval only) | — |
| `lexical` | Postgres full-text search, as-of versions | a baseline for wording overlap |

`aware` minus `labelled` is what date extraction costs; `labelled` minus `current` is what filtering buys.

**Data**

- **Corpus:** 62 files: 47 documents, 14 of them with two or three versions (15 superseded versions in all), in Markdown, HTML and PDF. It contains:
  - stale FAQ, wiki and onboarding pages that still state old policy;
  - a duplicate page;
  - tables;
  - upload dates that differ from effective dates.
- `eval/step0.jsonl`: 97 baseline questions (single-source, multi-source and unanswerable).
- `eval/p1.jsonl`: 67 version questions:
  - 33 whose answer changed between versions, asked about today. 3 of them contain numbers or month words that aren't dates ("for 1200 charge cycles", "May employees…"), and one is about a handbook section that was removed;
  - 24 asked about a past date, including the effective-date boundary and both upload-date traps;
  - 6 whose answer never changed;
  - 4 with no answer on their date: two before any document existed, one never documented, and one about a stipend that only appears after the date asked about.
- `eval/contradictions.jsonl`: 12 planted contradictions and 9 hard negatives (same topic, different scope, or the same facts stated twice).
- `eval/human/`: hand-written questions in the same format, scored as their own slice. Empty so far.

`uv run python -m eval.validate` checks every question's sources against the manifest. It also checks that every number in a reference answer appears in that question's source documents.

**Metrics**

- **Judge correctness.** Configured judging model compares each answer with the reference answer.
- **Retrieval hit, top-1 and all-source recall.** Counted on (document, version), not on document alone.
- **Stale fraction.** The share of retrieved chunks from versions that weren't in effect on the question's date.
- **Date extraction accuracy**, on dated and undated questions, with the misses listed.
- **Version attribution rate** and unsupported citations.
- **Generation cost**, from the token counts the API returns.

Proportions come with k/n and a 95% Wilson interval. Differences between modes are paired, with a 95% interval from resampling whole question groups (paraphrases of one fact move together). At 25 to 65 questions per slice, a 10-point gap can be noise.

**Checking the judge.** `eval.score labelsheet` samples 50 answers from a run without the judge's verdict. After they are labelled by hand, `eval.score agreement` reports agreement and Cohen's kappa per run. No judge number should be trusted before that.

## Results

**Pending.** Real numbers need `bge-small-en-v1.5`, the NLI model and an Anthropic API key, and haven't been run yet. CI runs the same pipeline with a hashing embedder and a number-mismatch stand-in for NLI. Those runs prove the plumbing; they are not results, and `results/*_fake*` is gitignored.

Two things are already visible from building it:

- **Full-text search finds the right version for 84 of 85 answerable baseline questions and every P1 question on the fake run.** The documents and questions were written by the same model and share wording. Hand-written questions in `eval/human/` (same format, `"author": "human"`) are the fix.
- **Date extraction can misfire.** It reads "Which office opened **in 2025**?" as a question about 2025, and hides the company-history page, which took effect in January 2026. Regex can't tell an event date from an as-of date.

## Run

Python 3.12, [uv](https://docs.astral.sh/uv/), Docker:

```bash
docker compose up -d db
uv sync --group models --group dev
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/rag ANTHROPIC_API_KEY=...
export GENERATOR_MODEL="<generation-model-id>" JUDGE_MODEL="<judging-model-id>"
export GENERATOR_INPUT_USD_PER_MILLION="<input-rate>" GENERATOR_OUTPUT_USD_PER_MILLION="<output-rate>"
uv run python -m rag.pipeline index
uv run python -m eval.score qa --set p1 --modes all current aware labelled
uv run python -m eval.score qa --set step0 --modes aware lexical
uv run python -m eval.score contradictions
uv run python -m eval.score labelsheet results/p1_aware_k5.csv   # then label it and run `eval.score agreement`
uv run uvicorn rag.api:app        # POST /ask {"question": "...", "mode": "aware"}
```

Tests rebuild their tables, so they use a separate database:

```bash
docker compose exec db createdb -U postgres rag_test
TEST_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/rag_test uv run pytest -q
```

The index records which embedder built it and a hash of the corpus and chunking code. Eval runs rebuild it when the corpus or code changes, but won't replace an index built with a different embedder (the API's, say) unless given `--reindex`. The API returns 503 if the index came from a different embedder. LLM calls are cached in `.cache/llm/` by a hash of the full request, so reruns cost nothing. Only complete responses are cached: a refusal or a response cut off at `max_tokens` is recorded as an error row and retried on the next run. Without models or a key: `uv run python -m eval.score qa --set p1 --retrieval-only --fake`.

Cost estimates use configurable input and output token rates; the defaults (0.10 and 0.50 per million) are illustrative. Set the selected model's current rates before reporting costs.

**API.** `as_of` in the response is the date retrieval filtered on, or null for `all` and `current`, which ignore it. Questions without a date use the real date, not the eval clock. Model failures return 502, an unreachable database 503.

## Limitations

- The corpus, questions and reference answers are synthetic fixtures; answers are graded automatically. This can reinforce shared errors. Agreement between the judge and about 50 hand-labelled answers should be reported before any number is trusted.
- Exact scan, no ANN index. Past ~100k chunks, add HNSW with `hnsw.iterative_scan` so filtered queries still return k rows.
- NLI is expected to be weak on numeric conflicts ("20 days" vs "25 days"). Scoring it on the planted numeric pairs is part of the point.
- The as-of date comes from a regular expression, not a model. It handles explicit dates and misreads event dates (see above), and a year-like amount after "for" or "by" ("for 2000 dollars") reads as a year.
- Wilson intervals treat questions as independent. Paraphrases of one fact aren't, so per-slice intervals are too narrow; the paired deltas resample groups and don't have this problem.
- The API and models run natively; Docker runs only Postgres.
