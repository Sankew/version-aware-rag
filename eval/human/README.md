# Hand-written questions

Questions written by a person, not a model, in the same format as `eval/step0.jsonl` and `eval/p1.jsonl`,
with `"origin": "human"`. Put them in `step0.jsonl` or `p1.jsonl` here. `eval.validate` checks them, and
`eval.score qa` adds them to the matching set and reports them as the `author_human` slice.

The LLM-written questions share wording with the LLM-written corpus, which flatters lexical search. These
are the check on that.
