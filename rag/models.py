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
    from google.genai import errors as gemini_errors
    if isinstance(error, anthropic.APIConnectionError):
        return True
    if isinstance(error, gemini_errors.APIError):
        return error.code in (408, 409, 429) or error.code >= 500
    return isinstance(error, anthropic.APIStatusError) and (error.status_code in (408, 409, 429) or error.status_code >= 500)


class LLM:
    """Model calls cached on disk by a hash of the full request, so reruns are free and repeatable.

    Model IDs starting with "gemini" go to the Gemini API (GEMINI_API_KEY), all others to the Anthropic API.
    Sampling parameters use the configured models' defaults; `effort` is Anthropic's effort setting or
    Gemini's thinking level. Only complete responses (stop_reason end_turn, finish_reason STOP) are cached;
    anything else raises LLMError and is retried next run.
    """

    def __init__(self, client=None, cache_dir: Path = ROOT / ".cache" / "llm", gemini=None) -> None:
        if client is None and gemini is None:
            if not all(os.environ.get(name, "").strip() for name in ("GENERATOR_MODEL", "JUDGE_MODEL")):
                raise ValueError("Set GENERATOR_MODEL and JUDGE_MODEL before making real model calls")
        self.client, self.gemini, self.cache_dir = client, gemini, cache_dir  # created on first use

    def complete(self, model: str, system: str, prompt: str, effort: str, max_tokens: int,
                 schema: dict | None = None) -> dict:
        if model.startswith("gemini"):
            request = {"provider": "gemini", "model": model, "max_tokens": max_tokens, "system": system,
                       "prompt": prompt, "effort": effort, "schema": schema}
        else:
            output_config: dict = {"effort": effort}
            if schema:
                output_config["format"] = {"type": "json_schema", "schema": schema}
            request = {"model": model, "max_tokens": max_tokens, "system": system,
                       "messages": [{"role": "user", "content": prompt}], "output_config": output_config}
        path = self.cache_dir / f"{hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()}.json"
        if path.exists():
            return {**json.loads(path.read_text()), "cached": True}
        start = time.perf_counter()
        result = self._gemini(request) if "provider" in request else self._anthropic(request)
        if schema:
            try:
                json.loads(result["text"])
            except json.JSONDecodeError as e:
                raise LLMError(f"{model} returned invalid JSON: {e}", result["input_tokens"], result["output_tokens"]) from e
        result["latency_s"] = round(time.perf_counter() - start, 3)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir=path.parent, suffix=".tmp", delete=False) as f:
            f.write(json.dumps(result))
        os.replace(f.name, path)  # atomic: an interrupted run never leaves a half-written entry
        return {**result, "cached": False}

    def _anthropic(self, request: dict) -> dict:
        if self.client is None:
            import anthropic
            self.client = anthropic.Anthropic(timeout=600.0, max_retries=2)  # judge calls can run for minutes
        response = self.client.messages.create(**request)
        usage = response.usage.input_tokens, response.usage.output_tokens
        if response.stop_reason != "end_turn":
            raise LLMError(f"{request['model']} stopped with {response.stop_reason}: "
                           f"{getattr(response, 'stop_details', None)}", *usage)
        return {"text": next((b.text for b in response.content if b.type == "text"), ""), "model": response.model,
                "input_tokens": usage[0], "output_tokens": usage[1]}

    def _gemini(self, request: dict) -> dict:
        from google.genai import types
        if self.gemini is None:
            from google import genai
            retry = types.HttpRetryOptions(attempts=6, initial_delay=2.0, max_delay=60.0,
                                           http_status_codes=[408, 429, 500, 502, 503, 504])
            self.gemini = genai.Client(http_options=types.HttpOptions(timeout=600_000, retry_options=retry))
        config = types.GenerateContentConfig(
            system_instruction=request["system"], max_output_tokens=request["max_tokens"],
            thinking_config=types.ThinkingConfig(thinking_level=request["effort"].upper()),
            **({"response_mime_type": "application/json", "response_json_schema": request["schema"]}
               if request["schema"] else {}))
        response = self.gemini.models.generate_content(model=request["model"], contents=request["prompt"], config=config)
        meta = response.usage_metadata
        # Thinking tokens are billed as output.
        usage = (meta.prompt_token_count or 0, (meta.candidates_token_count or 0) + (meta.thoughts_token_count or 0))
        candidate = response.candidates[0] if response.candidates else None
        if candidate is None or candidate.finish_reason != types.FinishReason.STOP:
            reason = candidate.finish_reason if candidate else getattr(response.prompt_feedback, "block_reason", None)
            raise LLMError(f"{request['model']} stopped with {reason}", *usage)
        return {"text": response.text or "", "model": response.model_version or request["model"],
                "input_tokens": usage[0], "output_tokens": usage[1]}
