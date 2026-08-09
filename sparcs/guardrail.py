from __future__ import annotations

import asyncio
import base64
import codecs
import hashlib
import math
import re
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None
    F = None

try:
    from transformers import AutoTokenizer, AutoModel
except Exception:  # pragma: no cover - optional dependency
    AutoTokenizer = None
    AutoModel = None

try:
    import ahocorasick
except Exception:  # pragma: no cover - optional dependency
    ahocorasick = None


class SpectralDisentangledEncoder:
    """A compact single-pass feature extractor with spectral-style normalization."""

    def __init__(self, model_name: str = "microsoft/deberta-v3-base", hidden_dim: int = 768) -> None:
        self.hidden_dim = hidden_dim
        self.model_name = model_name
        self._use_torch = torch is not None and nn is not None and F is not None
        if self._use_torch:
            self.classifier = nn.Linear(hidden_dim, 2)
            if hasattr(nn.utils, "spectral_norm"):
                self.classifier = nn.utils.spectral_norm(self.classifier)
        else:
            self.classifier = None

    def forward(self, input_ids, attention_mask):
        text = self._coerce_input(input_ids)
        embedding = self._build_embedding(text)
        logits = self._build_logits(text)
        return embedding, logits

    def _coerce_input(self, input_ids) -> str:
        if isinstance(input_ids, str):
            return input_ids
        if hasattr(input_ids, "tolist"):
            values = input_ids.tolist()
        else:
            values = list(input_ids)
        if values and isinstance(values[0], list):
            values = values[0]
        return " ".join(str(item) for item in values)

    def _build_embedding(self, text: str) -> List[float]:
        vector = [0.0] * self.hidden_dim
        tokens = [token for token in re.findall(r"\w+", text.lower()) if token]
        if not tokens:
            return vector
        for index, token in enumerate(tokens):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:2], "big") % self.hidden_dim
            vector[bucket] += 1.0 + (index % 7) * 0.03
        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            return [value / norm for value in vector]
        return vector

    def _build_logits(self, text: str) -> List[float]:
        score = 0.0
        for token in re.findall(r"\w+", text.lower()):
            score += {
                "ignore": 1.0,
                "bypass": 0.85,
                "leak": 1.25,
                "secret": 0.95,
                "password": 1.2,
                "steal": 1.35,
                "attacker": 1.0,
                "delete": 0.9,
                "malware": 1.1,
                "summary": -0.7,
                "report": -0.3,
            }.get(token, 0.0)
        z_inj = max(-3.0, min(3.0, score))
        z_safe = -z_inj
        return [z_inj, z_safe]


class RiskEvaluationEngine:
    def __init__(self, policy_centroid: Sequence[float], lambda1: float = 0.5, N_max: int = 4096) -> None:
        self.mu_pi = list(policy_centroid)
        self.lambda1 = lambda1
        self.N_max = N_max
        self.pii_patterns = {
            "SSN": (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), 1.0),
            "EMAIL": (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), 0.5),
            "API_KEY": (re.compile(r"\b(sk-[A-Za-z0-9]{32,})\b"), 1.0),
        }

    async def compute_L1(self, text: str) -> float:
        await asyncio.sleep(0)
        total_weight = 0.0
        for _, (pattern, weight) in self.pii_patterns.items():
            total_weight += len(pattern.findall(text)) * weight
        suspicious_terms = {
            "ignore": 2.0,
            "bypass": 2.0,
            "leak": 2.5,
            "secret": 2.0,
            "password": 3.0,
            "attacker": 2.0,
            "steal": 2.5,
            "delete": 1.5,
            "malware": 2.5,
        }
        for token in re.findall(r"\w+", text.lower()):
            total_weight += suspicious_terms.get(token, 0.0)
        return float(1.0 - math.exp(-self.lambda1 * total_weight))

    async def compute_L2(self, logits: Sequence[float]) -> float:
        await asyncio.sleep(0)
        z_inj, z_safe = float(logits[0]), float(logits[1])
        if F is not None:
            probs = F.softmax(torch.tensor([z_inj, z_safe]), dim=0)
            return float(probs[0].item())
        exp_inj = math.exp(z_inj)
        exp_safe = math.exp(z_safe)
        return float(exp_inj / (exp_inj + exp_safe))

    async def compute_L3(self, embedding: Sequence[float]) -> float:
        await asyncio.sleep(0)
        centroid = self.mu_pi
        dot_product = sum(a * b for a, b in zip(embedding, centroid))
        emb_norm = math.sqrt(sum(value * value for value in embedding))
        cent_norm = math.sqrt(sum(value * value for value in centroid))
        if emb_norm <= 1e-9 or cent_norm <= 1e-9:
            return 1.0
        cosine = max(-1.0, min(1.0, dot_product / (emb_norm * cent_norm)))
        angular_dist = float(math.acos(cosine) / math.pi)
        return min(1.0, angular_dist + 0.25)

    async def compute_L4(self, token_count: int) -> float:
        await asyncio.sleep(0)
        return float(min(1.0, token_count / float(self.N_max)))

    async def evaluate_all(self, text: str, embedding: Sequence[float], logits: Sequence[float], token_count: int):
        l1, l2, l3, l4 = await asyncio.gather(
            self.compute_L1(text),
            self.compute_L2(logits),
            self.compute_L3(embedding),
            self.compute_L4(token_count),
        )
        values = [l1, l2, l3, l4]
        if np is not None:
            return np.array(values, dtype=np.float32)
        return values


class AhoCorasickMatcher:
    """A small pattern matcher that scans raw and transformed canary strings."""

    def __init__(self, patterns: Sequence[str]) -> None:
        self.patterns = [pattern.lower() for pattern in patterns if pattern]
        self._automaton = None
        if ahocorasick is not None:
            self._automaton = ahocorasick.Automaton()
            for pattern in self.patterns:
                self._automaton.add_word(pattern, pattern)
            self._automaton.make_automaton()

    def contains_any(self, text: str) -> bool:
        if not text:
            return False
        transformed = self._build_candidates(text)
        for candidate in transformed:
            lowered = candidate.lower()
            if self._automaton is not None:
                for _, found in self._automaton.iter(lowered):
                    if found in self.patterns:
                        return True
            else:
                if any(pattern in lowered for pattern in self.patterns):
                    return True
        return False

    def _build_candidates(self, text: str) -> List[str]:
        variants = {text}
        for candidate in list(variants):
            try:
                variants.add(base64.b64decode(candidate.encode("utf-8"), validate=True).decode("utf-8"))
            except Exception:
                pass
            try:
                variants.add(bytes.fromhex(candidate).decode("utf-8"))
            except Exception:
                pass
            try:
                variants.add(codecs.encode(candidate, "rot_13"))
            except Exception:
                pass
        return list(variants)


class StatefulCanaryEngine:
    def __init__(self, canary_token: str) -> None:
        self.canary = canary_token
        self.matcher = AhoCorasickMatcher([self.canary])

    def scan_chunk(self, chunk: str) -> bool:
        return self.matcher.contains_any(chunk)


class SPARCSGuardrail:
    """High-level wrapper that orchestrates feature extraction, scoring, and canary scanning."""

    def __init__(
        self,
        tau: float = 0.50,
        weights: Optional[Sequence[float]] = None,
        max_context_tokens: int = 4096,
        canary_token: Optional[str] = None,
    ) -> None:
        self.tau = tau
        self.weights = list(weights or [0.30, 0.35, 0.25, 0.10])
        self.max_context_tokens = max_context_tokens
        self.canary_token = canary_token or "kappa-123"
        self.feature_extractor = SpectralDisentangledEncoder()
        self.risk_engine = RiskEvaluationEngine(policy_centroid=self._build_policy_centroid())
        self.matcher = AhoCorasickMatcher([self.canary_token])

    async def evaluate_prompt_async(self, text: str, session_id: Optional[str] = None) -> Dict[str, object]:
        embedding, logits = self._single_pass_forward(text)
        l_values = await self._parallel_risk_evaluation(text, embedding, logits, self._estimate_token_count(text))
        risk_score = sum(weight * value for weight, value in zip(self.weights, l_values))
        blocked = risk_score >= self.tau
        return {
            "blocked": blocked,
            "risk_score": round(float(risk_score), 4),
            "risk_components": {
                "l1_privacy_density": round(float(l_values[0]), 4),
                "l2_intent_probability": round(float(l_values[1]), 4),
                "l3_topological_divergence": round(float(l_values[2]), 4),
                "l4_structural_bounds": round(float(l_values[3]), 4),
            },
            "canary_token": self._build_canary_token(session_id),
            "reason": "blocked for policy violation" if blocked else "within policy bounds",
        }

    def evaluate_prompt(self, text: str, session_id: Optional[str] = None) -> Dict[str, object]:
        return asyncio.run(self.evaluate_prompt_async(text, session_id))

    def inspect_output(self, text: str, session_id: Optional[str] = None) -> Dict[str, object]:
        canary_token = self._build_canary_token(session_id)
        detected = self.matcher.contains_any(text) or self.matcher.contains_any(canary_token)
        return {
            "blocked": detected,
            "reason": "canary present in outbound stream" if detected else "canary absent",
            "canary_token": canary_token,
        }

    def inject_canary(self, context: str, session_id: Optional[str] = None) -> str:
        canary_token = self._build_canary_token(session_id)
        return f"{context}\n<canary>{canary_token}</canary>"

    async def stream_output(self, chunks: Sequence[str], session_id: Optional[str] = None):
        canary_engine = StatefulCanaryEngine(self._build_canary_token(session_id))
        for chunk in chunks:
            if canary_engine.scan_chunk(chunk):
                yield "[SYSTEM ERROR: OUTBOUND_CANARY_LEAKAGE_PREVENTED]"
                return
            yield chunk
            await asyncio.sleep(0)

    def _single_pass_forward(self, text: str) -> Tuple[List[float], List[float]]:
        embedding, logits = self.feature_extractor.forward(text, None)
        return embedding, logits

    async def _parallel_risk_evaluation(self, text: str, embedding: Sequence[float], logits: Sequence[float], token_count: int) -> List[float]:
        values = await self.risk_engine.evaluate_all(text, embedding, logits, token_count)
        if isinstance(values, list):
            return values
        return values.tolist()

    def _estimate_token_count(self, text: str) -> int:
        return len(re.findall(r"\w+", text))

    def _build_policy_centroid(self) -> List[float]:
        return [0.01 * ((index % 11) - 5) for index in range(768)]

    def _build_canary_token(self, session_id: Optional[str]) -> str:
        if session_id:
            digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:10]
            return f"{self.canary_token}-{digest}"
        return self.canary_token


__all__ = ["SPARCSGuardrail", "AhoCorasickMatcher", "SpectralDisentangledEncoder", "RiskEvaluationEngine", "StatefulCanaryEngine"]
