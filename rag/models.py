"""Embedder, NLI model and model client, each with a deterministic fake for tests and offline runs."""
import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path

import numpy as np

from .ingest import ROOT, words
from .store import DIM

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
GENERATOR = os.environ.get("GENERATOR_MODEL", "test-generator")
JUDGE = os.environ.get("JUDGE_MODEL", "test-judge")
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge v1.5 query instruction


class Embedder:
    name = EMBED_MODEL

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(EMBED_MODEL)

    def embed(self, texts: list[str], query: bool = False) -> np.ndarray:
        texts = [QUERY_PREFIX + t for t in texts] if query else texts
        return self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

    def count(self, text: str) -> int:
        return len(self.model.tokenizer(text, add_special_tokens=True)["input_ids"])


class FakeEmbedder:
    """Hashed bag of words, unit length. Lexical, not semantic: enough to exercise the pipeline."""

    name = "fake-hashing"

    def embed(self, texts: list[str], query: bool = False) -> np.ndarray:
        out = np.zeros((len(texts), DIM), np.float32)
        for row, text in zip(out, texts):
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                row[int(hashlib.blake2b(token.encode(), digest_size=4).hexdigest(), 16) % DIM] += 1
        out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
        return out

    count = staticmethod(words)


class NLI:
    """Contradiction probability from an NLI cross-encoder, label order read from the model config."""

    def __init__(self) -> None:
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(NLI_MODEL)
        labels = {v.lower(): int(k) for k, v in self.model.model.config.id2label.items()}
        self.contradiction = labels["contradiction"]

    def contradiction_prob(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        logits = np.asarray(self.model.predict(pairs, convert_to_numpy=True))
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        return (probs / probs.sum(axis=1, keepdims=True))[:, self.contradiction]


class FakeNLI:
    """Flags a contradiction when the two sentences state different numbers about a shared topic word."""

    def contradiction_prob(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        def numbers(s: str) -> set[str]:
            return set(re.findall(r"\d+(?:\.\d+)?", s.split(":", 1)[-1]))
        return np.array([float(bool(numbers(a)) and bool(numbers(b)) and numbers(a) != numbers(b)) for a, b in pairs])


class LLMError(RuntimeError):
    """A response that can't be used: refused, cut off at max_tokens, stopped for another reason, or not the
    JSON its schema asked for. Carries the tokens it still cost."""

    def __init__(self, message: str, input_tokens: int = 0, output_tokens: int = 0) -> None:
        super().__init__(message)
        self.input_tokens, self.output_tokens = input_tokens, output_tokens


def transient(error: Exception) -> bool:
    """Worth an error row and a retry on the next run: timeouts, rate limits, overload. A 400, 401 or 404
    means a bad key, model or request, and every call would fail the same way."""
    import anthropic
    if isinstance(error, anthropic.APIConnectionError):
        return True
    return isinstance(error, anthropic.APIStatusError) and (error.status_code in (408, 409, 429) or error.status_code >= 500)


class LLM:
    """Model calls cached on disk by a hash of the full request, so reruns are free and repeatable.

    Sampling parameters use the configured models' defaults. Only complete
    responses (stop_reason end_turn) are cached; anything else raises LLMError and is retried next run.
    """

    def __init__(self, client=None, cache_dir: Path = ROOT / ".cache" / "llm") -> None:
        if client is None:
            if not all(os.environ.get(name, "").strip() for name in ("GENERATOR_MODEL", "JUDGE_MODEL")):
                raise ValueError("Set GENERATOR_MODEL and JUDGE_MODEL before making real model calls")
            import anthropic
            client = anthropic.Anthropic(timeout=600.0, max_retries=2)  # judge calls can run for minutes
        self.client, self.cache_dir = client, cache_dir

    def complete(self, model: str, system: str, prompt: str, effort: str, max_tokens: int,
                 schema: dict | None = None) -> dict:
        output_config: dict = {"effort": effort}
        if schema:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        request = {"model": model, "max_tokens": max_tokens, "system": system,
                   "messages": [{"role": "user", "content": prompt}], "output_config": output_config}
        path = self.cache_dir / f"{hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()}.json"
        if path.exists():
            return {**json.loads(path.read_text()), "cached": True}
        start = time.perf_counter()
        response = self.client.messages.create(**request)
        usage = response.usage.input_tokens, response.usage.output_tokens
        if response.stop_reason != "end_turn":
            raise LLMError(f"{model} stopped with {response.stop_reason}: {getattr(response, 'stop_details', None)}", *usage)
        text = next((b.text for b in response.content if b.type == "text"), "")
        if schema:
            try:
                json.loads(text)
            except json.JSONDecodeError as e:
                raise LLMError(f"{model} returned invalid JSON: {e}", *usage) from e
        result = {
            "text": text,
            "model": response.model,
            "input_tokens": usage[0],
            "output_tokens": usage[1],
            "latency_s": round(time.perf_counter() - start, 3),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir=path.parent, suffix=".tmp", delete=False) as f:
            f.write(json.dumps(result))
        os.replace(f.name, path)  # atomic: an interrupted run never leaves a half-written entry
        return {**result, "cached": False}
