"""Check the corpus manifest and eval files against the data contracts. Exit 1 on any error.

    uv run python -m eval.validate
"""
import json
import re
from datetime import date

from rag.ingest import ROOT, manifest, parse

TYPES = {"single", "multi", "unanswerable", "general", "followup"}
FIELDS = {"id": str, "question": str, "reference_answer": str, "sources": list, "type": str, "history": list,
          "as_of": (str, type(None)), "answer_changed": (bool, type(None)), "group": str, "split": str, "origin": str}


def numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in re.findall(r"\d+(?:[.,]\d+)*", text)}


def check() -> list[str]:
    errors: list[str] = []
    rows = manifest()
    by_doc: dict[str, list[dict]] = {}
    for r in rows:
        if not (ROOT / r["path"]).exists():
            errors.append(f"manifest: missing file {r['path']}")
        by_doc.setdefault(r["document_id"], []).append(r)
    for doc, versions in by_doc.items():
        versions.sort(key=lambda r: r["version"])
        if [r["version"] for r in versions] != list(range(1, len(versions) + 1)):
            errors.append(f"manifest: {doc} versions are not 1..n")
        for old, new in zip(versions, versions[1:]):
            if old["uploaded_at"] >= new["uploaded_at"]:
                errors.append(f"manifest: {doc} v{new['version']} was uploaded before v{old['version']}; ingest needs version order")
            if old["effective_to"] != new["effective_from"]:
                errors.append(f"manifest: {doc} v{old['version']} ends {old['effective_to']}, v{new['version']} starts {new['effective_from']}")
        for r in versions:
            if (r["status"] == "current") != (r["effective_to"] is None):
                errors.append(f"manifest: {doc} v{r['version']} status {r['status']} vs effective_to {r['effective_to']}")
    texts = {(r["document_id"], r["version"]): "\n".join(s.heading + "\n" + s.text for s in parse(ROOT / r["path"], r["title"]))
             for r in rows}
    sections = {(r["document_id"], r["version"]): {s.heading or r["title"] for s in parse(ROOT / r["path"], r["title"])}
                for r in rows}

    files = sorted((ROOT / "eval").glob("*.jsonl")) + sorted((ROOT / "eval" / "human").glob("*.jsonl"))
    seen: set[str] = set()
    for path in files:
        if path.name == "contradictions.jsonl":
            continue
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            where = f"{path.relative_to(ROOT)}:{n}"
            q = json.loads(line)
            for field, kind in FIELDS.items():
                if not isinstance(q.get(field), kind):
                    errors.append(f"{where}: {field} missing or not {kind}")
            if q.get("id") in seen:
                errors.append(f"{where}: duplicate id {q['id']}")
            seen.add(q.get("id"))
            if q.get("type") not in TYPES or q.get("split") not in ("train", "test") or q.get("origin") not in ("synthetic", "human"):
                errors.append(f"{where}: bad type, split or origin")
            if q.get("as_of"):
                date.fromisoformat(q["as_of"])
            sources = [(s["document_id"], s["version"]) for s in q.get("sources", [])]
            for s in sources:
                if s not in texts:
                    errors.append(f"{where}: unknown source {s}")
            if q.get("type") == "unanswerable":
                if sources:
                    errors.append(f"{where}: unanswerable question has sources")
                continue
            if not sources:
                errors.append(f"{where}: answerable question has no sources")
            source_numbers = set().union(*(numbers(texts[s]) for s in sources if s in texts))
            if missing := numbers(q["reference_answer"]) - source_numbers:
                errors.append(f"{where}: reference numbers {sorted(missing)} not in its sources")

    for n, line in enumerate((ROOT / "eval" / "contradictions.jsonl").read_text(encoding="utf-8").splitlines(), 1):
        pair = json.loads(line)
        if pair["label"] not in ("contradiction", "consistent") or pair["kind"] not in ("numeric", "policy", "scope"):
            errors.append(f"contradictions.jsonl:{n}: bad label or kind")
        for end in (pair["a"], pair["b"]):
            doc = (end["document_id"], end["version"])
            if end["section"] not in sections.get(doc, set()):
                errors.append(f"contradictions.jsonl:{n}: no section {end['section']!r} in {doc}")
            elif next(r for r in rows if (r["document_id"], r["version"]) == doc)["status"] != "current":
                errors.append(f"contradictions.jsonl:{n}: {doc} is not a current version")
    return errors


if __name__ == "__main__":
    problems = check()
    print("\n".join(problems) or "ok")
    raise SystemExit(1 if problems else 0)
