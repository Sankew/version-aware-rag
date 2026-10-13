# Version-aware RAG

[![Checks](https://github.com/Sankew/version-aware-rag/actions/workflows/checks.yml/badge.svg?branch=p1-version-aware)](https://github.com/Sankew/version-aware-rag/actions?query=branch%3Ap1-version-aware)

Company policies change. An answer about today's vacation allowance and an answer about the allowance in 2024 may need different versions of the same handbook. This project keeps document versions, retrieves the passages in effect on the date a question asks about, and returns their version and effective dates with the answer.

It also produces section-by-section change logs and checks for contradictions between documents that are still in effect. The corpus belongs to a fictional company, Brackwater Robotics, which reduces the chance of answering from prior knowledge. It does not prevent a model from guessing.

**Status:** the implementation, data validation and automated tests pass CI. Evaluation with real embeddings, NLI and generation models is pending. The deterministic substitutes used in CI check the pipeline; they do not establish answer quality or model performance.

## How it works

```mermaid
flowchart LR
    Q[Question] --> D[Parse the as-of date]
    Q --> E[Embed the question]
    D --> F[Filter versions by effective dates]
    E --> S[Exact cosine search in pgvector]
    F --> S
    S --> C[Top 5 passages with version metadata]
    C --> G[Generate an answer with citations]
    G --> A[Answer and sources]
```

- **Version storage:** each version has an effective date range. Adding a later version closes the previous range and preserves its contents. Versions must be inserted in order. The `index` command is a full rebuild from the corpus manifest and resets the project's tables.
- **Date selection:** the parser recognises explicit dates, months and years such as `20 February 2026`, `June 2025` and `in 2024`. A month or year means its last day, capped at the current date used by the caller. Without a match, it uses that date. Evaluation freezes this clock at **1 October 2026**; the API uses the actual day.
- **Retrieval:** `aware` filters by effective dates before searching. This handles cases where a document was uploaded before it took effect. The API also accepts an explicit `as_of` override.
- **Citations:** retrieved passages carry the document, section, version and effective dates. Citation labels that do not belong to a retrieved passage are removed from the answer and counted. This checks citation membership, not whether a passage supports every claim.
- **Change logs:** sections are aligned by heading, then compared with `difflib` to show additions, removals and edits.
- **Contradiction detection:** similar chunks from different documents in effect on the evaluation date become candidates. The three most similar sentence pairs are scored in both directions by `cross-encoder/nli-deberta-v3-base`. Titles and section headings preserve context; document stamps such as edit dates are excluded. The report separates candidate recall, NLI recall among surfaced pairs, and end-to-end recall.

## Setup

Use Python 3.12, uv and Docker. Run commands from the repository root.

```bash
uv sync --locked --group models --group dev
docker compose up -d --wait db
docker compose exec db createdb -U postgres version_rag
export DATABASE_URL="postgresql://postgres:postgres@localhost:5432/version_rag"
```

Create `version_rag` once. The Compose service binds to local port 5432. If another project already runs pgvector on that port, reuse that server and create this project's separate database there.

Real answer generation and judging use the Anthropic SDK, or the Gemini API for model IDs that start with `gemini` (set `GEMINI_API_KEY` instead; effort becomes Gemini's thinking level). Set an API key and actual model IDs that support the request's effort settings and structured JSON output. Replace the placeholders before running the evaluation:

```bash
export ANTHROPIC_API_KEY="your-api-key"
export GENERATOR_MODEL="your-generation-model-id"
export JUDGE_MODEL="your-judging-model-id"
uv run --locked --group models python -m rag.pipeline index
```

Local embedding and NLI models are downloaded on first use. Before reporting cost, set `GENERATOR_INPUT_USD_PER_MILLION` and `GENERATOR_OUTPUT_USD_PER_MILLION` to numeric rates for the chosen generation model. Their defaults, 0.10 and 0.50, are illustrative.

### Run the API and change log

```bash
uv run --locked --group models uvicorn rag.api:app
```

In another terminal, with the same environment configured:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"How many vacation days were available in 2024?","mode":"aware"}'
uv run --locked python versioning.py changelog employee-handbook
```

`POST /ask` accepts `question`, `mode` and an optional ISO date in `as_of`. The response includes the answer, citations, source versions and the date applied to retrieval. That date is null for `all` and `current`, which ignore it. Model failures return 502; an unavailable database or incompatible index returns 503.

## Evaluation

The same questions run through five retrieval modes:

| Mode | Documents searched | Purpose |
|---|---|---|
| `all` | Every version | Measure interference from old versions |
| `current` | The latest stored version of each document | Compare against keeping only the latest version |
| `aware` | Versions in effect on the parsed date | Evaluate date parsing and version filtering together |
| `labelled` | Versions in effect on the reference date | Isolate retrieval from date-parsing errors |
| `lexical` | Postgres full-text search over versions in effect on the parsed date | Compare with a lexical baseline |

Compare `aware` with `labelled` to assess the effect of date parsing. Compare `labelled` with `current` to assess date-based version selection without parsing errors.

### Data

The corpus contains **62 files covering 47 documents**, including 14 documents with multiple versions and 15 superseded versions. Markdown, HTML and PDF sources include tables, duplicate content, stale policy summaries and upload dates that differ from effective dates.

| Evaluation file | Contents |
|---|---|
| `eval/step0.jsonl` | 97 baseline questions about one source, multiple sources or undocumented facts |
| `eval/p1.jsonl` | 67 version questions: 33 about changed answers at the evaluation date, 24 historical, 6 unchanged and 4 unanswerable |
| `eval/contradictions.jsonl` | 12 planted contradictions and 9 consistent pairs with similar wording or different scope |
| [`eval/human/`](eval/human/README.md) | Optional independently written questions; none are included yet |

The validator checks the manifest, source references and whether numbers in reference answers appear in the listed source documents. These checks do not establish that every reference answer is correct.

```bash
uv run --locked python -m eval.validate
uv run --locked --group models python -m eval.score qa --set p1 --modes all current aware labelled
uv run --locked --group models python -m eval.score qa --set step0 --modes aware lexical
uv run --locked --group models python -m eval.score contradictions
```

### Metrics and review

Reports include judged answer correctness, retrieval hit rate, top-1 and all-source recall, stale-chunk fraction, date extraction accuracy, version attribution, unsupported citation counts and estimated generation cost. Retrieval is scored on **document and version together**.

Proportions include counts and 95% Wilson intervals. Differences between modes use paired bootstrap intervals that resample whole question groups, keeping related paraphrases together. Wilson intervals assume independent questions, so they can understate uncertainty when questions share a fact.

Check the judge by sampling answers without its verdict:

```bash
uv run --locked python -m eval.score labelsheet results/p1_aware_k5.csv
# Fill human_correct with 1 or 0 in the generated sheet, then:
uv run --locked python -m eval.score agreement results/labels/p1_aware_k5_sheet.csv
```

The sample contains up to 50 answers. Agreement and Cohen's kappa compare the completed sheet with the judge's verdicts. Sampling refuses to overwrite a sheet that already has review marks.

QA runs write per-mode CSV files and a summary JSON file in `results/`. Contradiction runs write a report and the flagged pairs. Fake runs have `_fake` in their filenames and are excluded from version control.

## Checks without model calls

These commands require pgvector but no API key or model downloads. Use a separate database because the first run builds an index with the hashing embedder:

```bash
uv sync --locked --group dev
docker compose exec db createdb -U postgres version_rag_smoke
DATABASE_URL="postgresql://postgres:postgres@localhost:5432/version_rag_smoke" \
  uv run --locked python -m eval.score qa --set p1 --modes all current aware lexical --retrieval-only --fake
DATABASE_URL="postgresql://postgres:postgres@localhost:5432/version_rag_smoke" \
  uv run --locked python -m eval.score contradictions --fake --threshold 0.5
```

For the full test suite, create a dedicated test database once. Database tests rebuild its tables:

```bash
docker compose exec db createdb -U postgres version_rag_test
TEST_DATABASE_URL="postgresql://postgres:postgres@localhost:5432/version_rag_test" \
  uv run --locked --group dev pytest -q
```

## Caching and limitations

- Evaluation rebuilds the index when the corpus or chunking fingerprint changes. Switching embedders requires a separate database or `--reindex`; the explicit `index` command always rebuilds.
- Complete model responses are cached in `.cache/llm/` by the full request. Matching requests reuse responses; new or failed requests can still incur charges. Reported cost estimates retain the original token counts and exclude judging, so they are not the bill for a rerun.
- Refused or truncated answers are scored as wrong. Failed judge calls and transient API errors become error rows, excluded from metrics and retried on the next run. Invalid credentials or model IDs stop the run.
- The corpus, questions and reference answers are synthetic. Shared wording can favour lexical retrieval, and shared mistakes can affect judging. Independently written questions and manual review are still needed.
- The date parser cannot reliably distinguish an event date from the date a policy should be evaluated on. For example, a question about an office opened in 2025 may incorrectly exclude a history page published in 2026.
- Exact vector search has no ANN index. Performance at larger document counts has not been measured.
- NLI may miss numeric disagreements or misread scope. The planted pairs test those cases, but real-model results remain pending.
