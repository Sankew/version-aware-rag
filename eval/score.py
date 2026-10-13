"""Score retrieval modes on a question set, the judge against hand labels, and the contradiction detector.

    uv run --group models python -m eval.score qa --set p1 --modes all current aware labelled
    uv run --group models python -m eval.score qa --set step0 --modes aware lexical
    uv run --group models python -m eval.score contradictions
    uv run python -m eval.score labelsheet results/p1_aware_k5.csv       # sample answers to label by hand
    uv run python -m eval.score agreement results/labels/*.csv           # judge vs hand labels, Cohen's kappa
    uv run python -m eval.score qa --set p1 --modes all current aware --retrieval-only --fake   # no models, no API key

Modes: all, current, aware and lexical as in the pipeline, plus `labelled`: aware retrieval on the
question's labelled as-of date instead of the parsed one, which separates date-extraction errors
from filtering errors.
"""
import argparse
import csv
import json
import math
import os
import random
import re
from datetime import date
from itertools import combinations
from pathlib import Path

import numpy as np

from rag import store
from rag.answer import answer, judge
from rag.ingest import ROOT
from rag.pipeline import ensure_index, retrieve
from versioning import AS_OF_TODAY, parse_as_of

PRICE_IN = float(os.environ.get("GENERATOR_INPUT_USD_PER_MILLION", "0.10"))
PRICE_OUT = float(os.environ.get("GENERATOR_OUTPUT_USD_PER_MILLION", "0.50"))
RESULTS = ROOT / "results"
MODES = ("all", "current", "aware", "lexical", "labelled")
VERSIONED = ("aware", "lexical", "labelled")  # modes whose excerpts carry version and dates
BINARY = ("correct", "hit", "all_sources", "top1_is_gold", "attributed")


def questions(question_set: str) -> list[dict]:
    """The synthetic set plus any hand-written questions in eval/human/<set>.jsonl."""
    rows = []
    for path in (ROOT / "eval" / f"{question_set}.jsonl", ROOT / "eval" / "human" / f"{question_set}.jsonl"):
        if path.exists():
            rows += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows


def gold_date(q: dict) -> date:
    return date.fromisoformat(q["as_of"]) if q["as_of"] else AS_OF_TODAY


def in_effect(chunk: dict, on: date) -> bool:
    return chunk["effective_from"] <= on and (chunk["effective_to"] is None or chunk["effective_to"] > on)


def retrieval(chunks: list[dict], sources: list[dict], on: date) -> dict:
    """Hit, all-source recall and top-1, counted on (document_id, version), never on document alone.
    stale_frac: share of retrieved chunks from versions not in effect on the labelled date."""
    out = {"stale_frac": sum(not in_effect(c, on) for c in chunks) / len(chunks) if chunks else None}
    if not sources:
        return out | {"hit": None, "all_sources": None, "top1_is_gold": None}
    got = [(c["document_id"], c["version"]) for c in chunks]
    want = {(s["document_id"], s["version"]) for s in sources}
    return out | {"hit": float(bool(set(got) & want)), "all_sources": float(want <= set(got)),
                  "top1_is_gold": float(bool(got) and got[0] in want)}


def attributed(text: str, sources: list[dict]) -> float | None:
    """Did the answer name the version it used ("version 3", "v3")? Only meaningful for version-aware answers."""
    if not sources:
        return None
    named = {int(v) for v in re.findall(r"\b(?:version|v)\.?\s*(\d+)\b", text, re.I)}
    return float(bool(named & {s["version"] for s in sources}))


def qa(question_set: str, modes: list[str], split: str | None, k: int, retrieval_only: bool, fake: bool,
       reindex: bool = False) -> dict:
    import anthropic
    from google.genai import errors as gemini_errors

    from rag.models import LLM, Embedder, FakeEmbedder, LLMError, transient
    embedder = FakeEmbedder() if fake else Embedder()
    conn = store.connect()
    ensure_index(conn, embedder, reindex)
    llm = None if retrieval_only else LLM()
    qs = [q for q in questions(question_set) if split is None or q["split"] == split]
    RESULTS.mkdir(exist_ok=True)
    suffix = f"_k{k}" + (f"_{split}" if split else "") + ("_retrieval" if retrieval_only else "") + ("_fake" if fake else "")
    runs = {}
    for mode in modes:
        rows = []
        for q in qs:
            gold, parsed = gold_date(q), parse_as_of(q["question"])
            used = gold if mode == "labelled" else parsed
            row = {"id": q["id"], "type": q["type"], "origin": q["origin"], "split": q["split"], "group": q["group"],
                   "as_of_question": bool(q["as_of"]), "answer_changed": q["answer_changed"],
                   "as_of_gold": gold.isoformat(), "as_of_parsed": parsed.isoformat(), "as_of_used": used.isoformat(),
                   "question": q["question"], "reference_answer": q["reference_answer"]}
            chunks = retrieve(conn, embedder, q["question"], "aware" if mode == "labelled" else mode, used, k)
            if not retrieval_only:
                try:
                    row |= answered(llm, q, chunks, used if mode in VERSIONED else None)
                except (LLMError, anthropic.APIError, gemini_errors.APIError) as e:
                    if not (isinstance(e, LLMError) or transient(e)):
                        raise  # a bad key or model would fail every call the same way
                    row["error"] = f"{type(e).__name__}: {e}"  # left out of the metrics, retried next run
            row |= retrieval(chunks, q["sources"], gold)
            row["retrieved"] = ";".join(f"{c['document_id']}@v{c['version']}" for c in chunks)
            rows.append(row)
        write_csv(RESULTS / f"{question_set}_{mode}{suffix}.csv", rows)
        runs[mode] = rows
    summary = {
        "questions": len(qs),
        "as_of_extraction": extraction(qs),
        "modes": {mode: slices(rows) for mode, rows in runs.items()},
        "deltas": deltas(runs, "hit" if retrieval_only else "correct"),
    }
    (RESULTS / f"{question_set}_summary{suffix}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def usd(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens * PRICE_IN + output_tokens * PRICE_OUT) / 1e6


def answered(llm, q: dict, chunks: list[dict], as_of: date | None) -> dict:
    """Answer and judge one question. A refused or truncated answer is a wrong answer, not a skipped one;
    only a failed judge call leaves the question out."""
    from rag.models import LLMError
    try:
        result = answer(llm, q["question"], chunks, as_of)
    except LLMError as e:
        return {"correct": 0.0, "answer": "", "answer_error": str(e), "generation_usd": usd(e.input_tokens, e.output_tokens)}
    verdict = judge(llm, q["question"], q["reference_answer"], result["text"])
    return {"correct": float(verdict["correct"]), "judge_reason": verdict["reason"],
            "attributed": attributed(result["text"], q["sources"]) if as_of is not None else None,
            "unsupported_citations": len(result["unsupported_citations"]),
            "generation_usd": usd(result["input_tokens"], result["output_tokens"]), "answer": result["text"]}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({key for r in rows for key in r}))
        writer.writeheader()
        writer.writerows(rows)


def extraction(qs: list[dict]) -> dict:
    """How often the regex reads the labelled as-of date, on questions with and without one."""
    out = {}
    for name, group in (("dated", [q for q in qs if q["as_of"]]), ("undated", [q for q in qs if not q["as_of"]])):
        if group:
            misses = [q["id"] for q in group if parse_as_of(q["question"]) != gold_date(q)]
            out[name] = proportion(len(group) - len(misses), len(group)) | {"misses": misses}
    return out


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    """95% Wilson interval for k successes in n. Unlike a bootstrap, it doesn't collapse to [1, 1] at 3/3."""
    p, d = k / n, 1 + z * z / n
    centre, half = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(centre - half, 3), round(centre + half, 3)]


def proportion(k: int, n: int) -> dict:
    return {"k": k, "n": n, "mean": round(k / n, 3), "ci95": wilson(k, n)}


def bootstrap(values: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean and 95% percentile-bootstrap interval, for non-binary metrics."""
    data = np.asarray(values, float)
    means = np.random.default_rng(seed).choice(data, (n, len(data))).mean(axis=1)
    return float(data.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def slices(rows: list[dict]) -> dict:
    """Each metric overall and on the slices P1 reports. Binary metrics get k/n and a Wilson interval,
    which treats questions as independent; deltas between modes resample whole groups instead."""
    groups = {
        "all": rows,
        "changed_answer_now": [r for r in rows if r["answer_changed"] and not r["as_of_question"]],
        "historical": [r for r in rows if r["as_of_question"] and r["type"] != "unanswerable"],
        "unchanged": [r for r in rows if r["answer_changed"] is False],
        "unanswerable": [r for r in rows if r["type"] == "unanswerable"],
        "multi": [r for r in rows if r["type"] == "multi"],
        "origin_synthetic": [r for r in rows if r["origin"] == "synthetic"],
        "origin_human": [r for r in rows if r["origin"] == "human"],
        "split_train": [r for r in rows if r["split"] == "train"],
        "split_test": [r for r in rows if r["split"] == "test"],
    }
    out = {}
    for name, group in groups.items():
        if not group:
            continue
        out[name] = {"n": len(group)}
        for metric in BINARY:
            values = [r[metric] for r in group if r.get(metric) is not None]
            if values:
                out[name][metric] = proportion(int(sum(values)), len(values))
        if stale := [r["stale_frac"] for r in group if r.get("stale_frac") is not None]:
            mean, lo, hi = bootstrap(stale)
            out[name]["stale_frac"] = {"mean": round(mean, 3), "ci95": [round(lo, 3), round(hi, 3)], "n": len(stale)}
        if errors := sum("error" in r for r in group):
            out[name]["errors"] = errors
        if any("generation_usd" in r for r in group):
            out[name]["generation_usd"] = round(sum(r.get("generation_usd", 0.0) for r in group), 5)
    return out


def paired_delta(a: list[dict], b: list[dict], metric: str, n: int = 2000, seed: int = 0) -> dict | None:
    """Mean of b - a over the questions both scored, with a 95% interval from resampling whole groups
    (paraphrases and follow-ups of one fact move together)."""
    by_group: dict[str, list[float]] = {}
    for x, y in zip(a, b, strict=True):
        assert x["id"] == y["id"]
        if x.get(metric) is not None and y.get(metric) is not None:
            by_group.setdefault(x["group"], []).append(y[metric] - x[metric])
    if not by_group:
        return None
    sums = np.array([sum(d) for d in by_group.values()])
    counts = np.array([len(d) for d in by_group.values()])
    picks = np.random.default_rng(seed).integers(0, len(sums), (n, len(sums)))
    means = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    return {"delta": round(float(sums.sum() / counts.sum()), 3),
            "ci95": [round(float(np.percentile(means, 2.5)), 3), round(float(np.percentile(means, 97.5)), 3)],
            "n": int(counts.sum()), "groups": len(sums)}


def deltas(runs: dict[str, list[dict]], metric: str) -> dict:
    """Every later mode minus every earlier one, overall and on the historical and changed-now slices."""
    def keep(rows: list[dict], name: str) -> list[dict]:
        if name == "historical":
            return [r for r in rows if r["as_of_question"] and r["type"] != "unanswerable"]
        if name == "changed_answer_now":
            return [r for r in rows if r["answer_changed"] and not r["as_of_question"]]
        return rows
    out = {}
    for first, second in combinations(runs, 2):
        out[f"{second} - {first}"] = {name: paired_delta(keep(runs[first], name), keep(runs[second], name), metric)
                                      for name in ("all", "historical", "changed_answer_now")}
    return {"metric": metric, "pairs": out}


def labelsheet(run: Path, n: int, seed: int) -> Path:
    """A random sample of answers from one run in results/, without the judge's verdict, to label by hand."""
    rows = [r for r in csv.DictReader(run.open(encoding="utf-8")) if r.get("answer") and not r.get("error")]
    if not rows:
        raise SystemExit(f"{run} has no judged answers (was it a --retrieval-only run?)")
    sample = random.Random(seed).sample(rows, min(n, len(rows)))
    out = RESULTS / "labels" / f"{run.stem}_sheet.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and any(r.get("human_correct", "").strip() for r in csv.DictReader(out.open(encoding="utf-8"))):
        raise SystemExit(f"{out} already has hand marks; move it aside before sampling again")
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "id", "question", "reference_answer", "answer", "human_correct"])
        for r in sample:
            writer.writerow([run.name, r["id"], r["question"], r["reference_answer"], r["answer"], ""])
    print(f"wrote {out}: fill human_correct with 1 or 0, then run `python -m eval.score agreement {out}`")
    return out


def kappa(pairs: list[tuple[int, int]]) -> dict:
    """Agreement and Cohen's kappa between judge (first) and human (second) labels."""
    n = len(pairs)
    observed = sum(j == h for j, h in pairs) / n
    p_judge, p_human = sum(j for j, _ in pairs) / n, sum(h for _, h in pairs) / n
    expected = p_judge * p_human + (1 - p_judge) * (1 - p_human)
    return {"n": n, "agreement": round(observed, 3),
            "kappa": round((observed - expected) / (1 - expected), 3) if expected < 1 else None,
            "judge_yes_human_no": sum(j and not h for j, h in pairs), "judge_no_human_yes": sum(h and not j for j, h in pairs)}


def agreement(sheets: list[Path]) -> dict:
    """Judge vs hand labels, per run (one run is one mode on one question set)."""
    by_run: dict[str, list[tuple[int, int]]] = {}
    verdicts: dict[str, dict] = {}
    stale: dict[str, int] = {}
    for sheet in sheets:
        for r in csv.DictReader(sheet.open(encoding="utf-8")):
            if r["human_correct"].strip() not in ("0", "1"):
                continue
            if r["run"] not in verdicts:
                verdicts[r["run"]] = {x["id"]: x for x in csv.DictReader((RESULTS / r["run"]).open(encoding="utf-8"))}
            judged = verdicts[r["run"]].get(r["id"], {})
            if not judged.get("correct") or judged.get("answer") != r["answer"]:
                stale[r["run"]] = stale.get(r["run"], 0) + 1  # the run was redone since the sheet was made
                continue
            by_run.setdefault(r["run"], []).append((int(float(judged["correct"])), int(r["human_correct"])))
    report = {run: kappa(pairs) | {"stale_labels_skipped": stale.get(run, 0)} for run, pairs in sorted(by_run.items())}
    print(json.dumps(report, indent=2))
    return report


def contradictions(threshold: float, nli_threshold: float, fake: bool, reindex: bool = False) -> dict:
    from rag.models import NLI, Embedder, FakeEmbedder, FakeNLI
    from versioning import candidates, contradiction_score, evaluate_contradictions
    embedder, nli = (FakeEmbedder(), FakeNLI()) if fake else (Embedder(), NLI())
    conn = store.connect()
    ensure_index(conn, embedder, reindex)
    chunks = store.chunks_of_current_versions(conn, AS_OF_TODAY)
    labels = [json.loads(line) for line in (ROOT / "eval" / "contradictions.jsonl").read_text(encoding="utf-8").splitlines()]
    sweep = {}
    for t in (0.5, 0.6, 0.7, 0.8, 0.9):
        surfaced = candidates(chunks, t)
        sweep[t] = {"candidate_pairs": len(surfaced),
                    "candidate_recall": evaluate_contradictions(labels, chunks, {}, surfaced)["candidate_recall"]}
    pairs = candidates(chunks, threshold)
    flagged, evidence = {}, []
    for i, j, sim in pairs:
        p, (a, b) = contradiction_score(chunks[i], chunks[j], embedder, nli)
        flagged[(i, j)] = p
        evidence.append({"a": f"{chunks[i]['document_id']}@v{chunks[i]['version']}:{chunks[i]['section']}",
                         "b": f"{chunks[j]['document_id']}@v{chunks[j]['version']}:{chunks[j]['section']}",
                         "cosine": round(sim, 3), "contradiction": round(p, 3), "sentence_a": a, "sentence_b": b})
    result = {"candidate_threshold": threshold, "nli_threshold": nli_threshold, "candidate_pairs": len(pairs),
              "sweep": sweep, **evaluate_contradictions(labels, chunks, flagged, pairs, nli_threshold)}
    RESULTS.mkdir(exist_ok=True)
    suffix = f"_t{threshold}" + ("_fake" if fake else "")
    (RESULTS / f"contradictions{suffix}.json").write_text(json.dumps(result, indent=2) + "\n")
    (RESULTS / f"contradiction_pairs{suffix}.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in sorted(evidence, key=lambda e: -e["contradiction"])))
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    q = sub.add_parser("qa")
    q.add_argument("--set", choices=["step0", "p1"], required=True)
    q.add_argument("--modes", nargs="+", choices=MODES, default=["all", "current", "aware"])
    q.add_argument("--split", choices=["train", "test"])
    q.add_argument("--k", type=int, default=5)
    q.add_argument("--retrieval-only", action="store_true", help="skip generation and judging (no API key needed)")
    q.add_argument("--fake", action="store_true", help="hashing embedder instead of bge-small (no model download)")
    c = sub.add_parser("contradictions")
    c.add_argument("--threshold", type=float, default=0.7, help="cosine similarity for candidate chunk pairs")
    c.add_argument("--nli-threshold", type=float, default=0.5)
    c.add_argument("--fake", action="store_true", help="hashing embedder and number-mismatch NLI stand-in")
    for sub_parser in (q, c):
        sub_parser.add_argument("--reindex", action="store_true",
                                help="replace an index built with a different embedder (e.g. the API's)")
    s = sub.add_parser("labelsheet")
    s.add_argument("run", type=Path, help="a results/<set>_<mode>_k<k>.csv file from a judged run")
    s.add_argument("--n", type=int, default=50)
    s.add_argument("--seed", type=int, default=0)
    a = sub.add_parser("agreement")
    a.add_argument("sheets", type=Path, nargs="+")
    args = parser.parse_args()
    if args.command == "qa":
        if args.fake and not args.retrieval_only:
            raise SystemExit("--fake only supports --retrieval-only: a fake model can't produce answers worth judging")
        qa(args.set, args.modes, args.split, args.k, args.retrieval_only, args.fake, args.reindex)
    elif args.command == "contradictions":
        contradictions(args.threshold, args.nli_threshold, args.fake, args.reindex)
    elif args.command == "labelsheet":
        labelsheet(args.run, args.n, args.seed)
    else:
        agreement(args.sheets)


if __name__ == "__main__":
    main()
