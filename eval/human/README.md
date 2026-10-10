# Independently written evaluation questions

This folder is reserved for questions written independently of the synthetic question sets. It currently contains no questions. The aim is to check how well retrieval handles wording that differs from the corpus and its existing questions.

## Add questions

Create `step0.jsonl` for baseline questions or `p1.jsonl` for version questions in this folder. Use one JSON object per line, following the schema in [`../step0.jsonl`](../step0.jsonl) or [`../p1.jsonl`](../p1.jsonl), and set `"origin": "human"`.

Every question needs these fields:

| Fields | What to include |
|---|---|
| `id`, `question`, `reference_answer` | A globally unique ID, the question and a checked reference answer |
| `sources` | A list of source objects with `document_id` and `version`; use an empty list for unanswerable questions |
| `type` | One of `single`, `multi`, `unanswerable`, `general` or `followup` |
| `history` | Conversation turns in the existing format, or an empty list |
| `as_of`, `answer_changed` | An ISO date and a boolean when applicable; otherwise null |
| `group`, `split` | A group identifier and either `train` or `test` |
| `origin` | `human` |

Check the reference against the exact source version. Keep paraphrases of the same fact in one group, and avoid copying wording from the existing question sets.

## Validate and score

From the repository root:

```bash
uv run --locked python -m eval.validate
```

The validator checks these files alongside the main question sets, including duplicate IDs, source references and numbers in reference answers. The QA evaluator adds each file to its matching set and reports these questions separately as the **`origin_human`** slice. The same model and database requirements described in the [main README](../../README.md) apply when generating and judging answers.
