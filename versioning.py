"""The version-aware parts: as-of dates from questions, change logs, and cross-document contradictions.

    uv run python versioning.py changelog employee-handbook
"""
import calendar
import difflib
import re
import sys
from datetime import date
from itertools import combinations

import numpy as np

from rag.ingest import ROOT, manifest, parse

AS_OF_TODAY = date(2026, 10, 1)  # frozen eval clock
MONTHS = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}
MONTH = "(" + "|".join(MONTHS) + ")"


YEAR = r"((?:19|20)\d\d)"  # other 4-digit numbers ("1000 robots") are not years


def day(year: str, month: int | str, d: str) -> date:
    """A day past the end of its month ("31 February") means the month's last day."""
    year, month = int(year), int(month)
    return date(year, month, min(int(d), calendar.monthrange(year, month)[1]))


def month_end(year: str, month: int) -> date:
    return date(int(year), month, calendar.monthrange(int(year), month)[1])


DATE_PATTERNS = [
    (re.compile(rf"\b{YEAR}-(\d{{2}})-(\d{{2}})\b"), lambda m: day(m[1], m[2], m[3])),
    (re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)? {MONTH},? {YEAR}\b"), lambda m: day(m[3], MONTHS[m[2]], m[1])),
    (re.compile(rf"\b{MONTH} (\d{{1,2}})(?:st|nd|rd|th)?,? {YEAR}\b"), lambda m: day(m[3], MONTHS[m[1]], m[2])),
    (re.compile(rf"\b{MONTH},? {YEAR}\b"), lambda m: month_end(m[2], MONTHS[m[1]])),
    (re.compile(rf"\b(?:in|during|as of|by|for) {YEAR}\b"), lambda m: date(int(m[1]), 12, 31)),
]


def parse_as_of(question: str, today: date = AS_OF_TODAY) -> date:
    """The date a question asks about: '20 February 2026', 'June 2025', 'in 2024', or today.

    A month or a year means its last day ("in 2024" asks about what was true by the end of 2024),
    capped at today. A day past the end of its month is read as the month's last day; a match that still
    isn't a date (month 13, day 0) is skipped.
    """
    q = question.lower()
    for pattern, build in DATE_PATTERNS:
        for m in pattern.finditer(q):
            try:
                return min(build(m), today)
            except ValueError:
                continue
    return today


def changelog(document_id: str) -> str:
    """Markdown change log across a document's versions: sections aligned by heading, lines diffed."""
    rows = sorted((r for r in manifest() if r["document_id"] == document_id), key=lambda r: r["version"])
    if not rows:
        raise SystemExit(f"unknown document: {document_id}")
    out = [f"# Changes to {rows[0]['title']}"]
    for old, new in zip(rows, rows[1:]):
        before = {s.heading: s.text for s in parse(ROOT / old["path"], old["title"])}
        after = {s.heading: s.text for s in parse(ROOT / new["path"], new["title"])}
        out.append(f"\n## Version {old['version']} → {new['version']} (effective {new['effective_from']})")
        for heading in [*after, *(h for h in before if h not in after)]:
            if heading not in before:
                out.append(f"- **Added** {heading}: {after[heading]}")
            elif heading not in after:
                out.append(f"- **Removed** {heading}")
            elif before[heading] != after[heading]:
                diff = difflib.ndiff(before[heading].splitlines(), after[heading].splitlines())
                changed = [line for line in diff if line[:2] in ("- ", "+ ")]
                out.append(f"- **Changed** {heading}:\n" + "\n".join(f"  {line}" for line in changed))
    return "\n".join(out) + "\n"


# Document stamps: version lines, owners, edit dates, breadcrumbs. Matched at the start of a sentence only,
# so "the calendar is published each December" stays in. "Applies to:" stays too: it is a scope qualifier.
STAMP = re.compile(r"[\s_*]*(Version \d+ ·|Effective:? \d{1,2} [a-z]+ \d{4}\W*$|Replaces:|(Page )?owner:|Maintained by"
                   r"|Last (edited|reviewed|updated)|Updated |Published |\S.* > .*\| Updated)", re.I)


def sentences(text: str) -> list[str]:
    """Sentences of a chunk, minus its header line and document stamps (version, owner, dates),
    which differ between any two documents and would read as contradictions."""
    return [s for line in text.splitlines()[1:] for s in re.split(r"(?<=[.!?])\s+", line)
            if len(s.split()) >= 3 and not STAMP.match(s)]


def candidates(chunks: list[dict], threshold: float) -> list[tuple[int, int, float]]:
    """Chunk pairs from different documents whose embeddings have cosine similarity above threshold."""
    vectors = np.array([c["embedding"] for c in chunks], np.float32)
    sims = vectors @ vectors.T
    return [(i, j, float(sims[i, j])) for i, j in combinations(range(len(chunks)), 2)
            if chunks[i]["document_id"] != chunks[j]["document_id"] and sims[i, j] > threshold]


def contradiction_score(a: dict, b: dict, embedder, nli, top: int = 3) -> tuple[float, tuple[str, str]]:
    """Max NLI contradiction probability over the top sentence pairs, checked in both directions.

    Each sentence carries its document title and section so qualifiers ("contractors",
    "full-time employees") survive the split into sentences.
    """
    left = [f"{a['title']}, {a['section']}: {s}" for s in sentences(a["content"])]
    right = [f"{b['title']}, {b['section']}: {s}" for s in sentences(b["content"])]
    if not left or not right:
        return 0.0, ("", "")
    sims = embedder.embed(left) @ embedder.embed(right).T
    order = np.dstack(np.unravel_index(np.argsort(-sims, axis=None), sims.shape))[0][:top]
    pairs = [(left[i], right[j]) for i, j in order]
    forward = nli.contradiction_prob(pairs)
    backward = nli.contradiction_prob([(y, x) for x, y in pairs])
    scores = np.maximum(forward, backward)
    best = int(np.argmax(scores))
    return float(scores[best]), pairs[best]


def key(end: dict) -> tuple:
    return end["document_id"], end["version"], end["section"]


def evaluate_contradictions(labels: list[dict], chunks: list[dict], flagged: dict[tuple[int, int], float],
                            candidate_pairs: list[tuple[int, int, float]], threshold: float = 0.5) -> dict:
    """Each stage scored on its own denominator, then end to end. A labelled pair counts as surfaced
    or flagged if any of its chunk pairs is.

    - candidate_recall: labelled contradictions the similarity filter surfaced
    - nli_recall: of those surfaced, the share NLI flagged (NLI never sees the rest)
    - end_to_end_recall: labelled contradictions flagged
    - precision_on_labelled: flagged labelled pairs that are contradictions; unlabelled flags are counted
      separately because nobody checked them
    """
    def pair_keys(i: int, j: int) -> frozenset:
        return frozenset({key(chunks[i]), key(chunks[j])})
    surfaced = {pair_keys(i, j) for i, j, _ in candidate_pairs}
    predicted = {pair_keys(i, j) for (i, j), p in flagged.items() if p >= threshold}
    pairs = [(frozenset({key(p["a"]), key(p["b"])}), p) for p in labels]
    positives = [k for k, p in pairs if p["label"] == "contradiction"]
    negatives = [k for k, p in pairs if p["label"] == "consistent"]
    seen = [k for k in positives if k in surfaced]
    tp = sum(k in predicted for k in positives)
    fp = sum(k in predicted for k in negatives)
    by_kind: dict[str, dict] = {}
    for k, p in pairs:
        row = by_kind.setdefault(f"{p['label']}/{p['kind']}", {"labelled": 0, "surfaced": 0, "flagged": 0})
        row["labelled"] += 1
        row["surfaced"] += k in surfaced
        row["flagged"] += k in predicted
    return {
        "labelled_contradictions": len(positives),
        "labelled_consistent": len(negatives),
        "candidate_recall": ratio(len(seen), len(positives)),
        "nli_recall": ratio(sum(k in predicted for k in seen), len(seen)),
        "end_to_end_recall": ratio(tp, len(positives)),
        "precision_on_labelled": ratio(tp, tp + fp),
        "false_positives_on_hard_negatives": fp,
        "flagged_unlabelled_pairs": len(predicted - set(positives) - set(negatives)),
        "by_kind": dict(sorted(by_kind.items())),
    }


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "changelog":
        raise SystemExit(__doc__)
    print(changelog(sys.argv[2]))
