"""SPARCS L1-L4 risk signals and adaptive gate."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Sequence

import torch

try:
    from presidio_analyzer import AnalyzerEngine
except Exception:  # pragma: no cover - optional runtime dependency
    AnalyzerEngine = None


@dataclass(frozen=True)
class RiskVector:
    l1: float
    l2: float
    l3: float
    l4: float
    score: float
    decision: bool
    entities: int
    token_count: int


class PrivacyDetector:
    """L1 detector using Presidio, with a narrow regex fallback for portability.

    The fallback is deliberately not presented as equivalent to Presidio.  It only
    keeps the middleware operational when Presidio is unavailable.
    """

    _fallback = re.compile(
        r"(?ix)(?:"
        r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}|"
        r"\b\d{3}-\d{2}-\d{4}\b|"
        r"\b(?:\d[ -]*?){13,19}\b|"
        r"\b(?:\d{1,3}\.){3}\d{1,3}\b|"
        r"\b(?:https?://|www\.)[^\s]+"
        r")"
    )

    def __init__(self, enabled: bool = True, entities: Sequence[str] | None = None):
        self.enabled = bool(enabled)
        self.entities = list(entities or [])
        self.engine = None
        if self.enabled and AnalyzerEngine is not None:
            try:
                self.engine = AnalyzerEngine()
            except Exception:
                self.engine = None

    def count(self, text: str) -> int:
        if not self.enabled:
            return 0
        if self.engine is not None:
            # Keep custom/non-Presidio labels out of AnalyzerEngine; those are
            # covered by the explicit fallback patterns below.
            supported = []
            for entity in self.entities:
                try:
                    self.engine.get_recognizer_registry().get_recognizers(
                        language="en", entities=[entity]
                    )
                    supported.append(entity)
                except Exception:
                    continue
            try:
                results = self.engine.analyze(
                    text=text,
                    entities=supported or None,
                    language="en",
                )
                return len(results) + self._fallback_count(text, exclude_presidio=results)
            except Exception:
                pass
        return len(self._fallback.findall(text))

    @classmethod
    def _fallback_count(cls, text: str, exclude_presidio=None) -> int:
        # API keys are intentionally handled only as an additional narrow signal;
        # avoid trying to emulate Presidio's entire recognizer set.
        patterns = [
            r"\b(?:sk|pk|ghp|github_pat)_[A-Za-z0-9_\-]{12,}\b",
            r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
        ]
        return sum(len(re.findall(p, text)) for p in patterns)


def angular_divergence(x: torch.Tensor, c: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalized angular distance in [0,1]."""
    x = x / x.norm(dim=-1, keepdim=True).clamp(min=eps)
    c = c / c.norm(dim=-1, keepdim=True).clamp(min=eps)
    cos = (x * c).sum(-1).clamp(-1 + eps, 1 - eps)
    return torch.acos(cos) / math.pi


def compute_risk(
    text: str,
    embedding: torch.Tensor,
    logits: torch.Tensor,
    centroid: torch.Tensor,
    cfg: dict,
    privacy_detector: PrivacyDetector,
    token_count: int,
) -> RiskVector:
    """Compute L1-L4 and the configured weighted gate for one request."""
    n_ent = privacy_detector.count(text)
    # L1 is an entity-density signal rather than a raw entity count.  This
    # keeps the signal comparable across short and long requests.
    density = n_ent / max(int(token_count), 1)
    saturation = max(float(cfg["risk"]["l1_saturation"]), 1e-8)
    l1 = min(1.0, density / saturation)

    probs = torch.softmax(logits.float(), dim=-1)
    l2 = float(probs[..., 1].item())
    l3 = float(angular_divergence(embedding.float(), centroid.float()).item())

    max_tok = max(int(cfg["risk"]["l4_max_tokens"]), 1)
    tok = int(token_count)
    l4 = max(0.0, min(1.0, (tok - max_tok) / max_tok))

    weights = [float(x) for x in cfg["risk"]["weights"]]
    if len(weights) != 4:
        raise ValueError("risk.weights must contain exactly four values")
    score = sum(w * x for w, x in zip(weights, (l1, l2, l3, l4)))
    threshold = float(cfg["risk"]["threshold"])
    return RiskVector(l1, l2, l3, l4, score, score >= threshold, n_ent, tok)
