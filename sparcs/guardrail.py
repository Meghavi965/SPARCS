# Copyright (C) 2026 Meghavi Vipulkumar Vyas
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# See LICENSE file in the repository root for full license text.
# For commercial licensing inquiries: meghavi.vyas@svit.ac.in

import math
import asyncio
from typing import Tuple, List, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import ahocorasick

# =====================================================================
# 1. PHASE I: AUTHENTIC SPECTRAL-NORMALIZED DeBERTa-v3 BACKBONE
# =====================================================================

class SpectralDisentangledEncoder(nn.Module):
    """
    Authentic DeBERTa-v3 encoder that outputs continuous pooled embeddings (E_P)
    and uses PyTorch Spectral Normalization on linear heads to enforce local 
    K-Lipschitz continuity as proven in Theorem 1.
    """
    def __init__(self, model_name: str = "microsoft/deberta-v3-base", hidden_dim: int = 768):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Theorem 1 requirement: Spectral Normalization enforces Lipschitz smoothness:
        # ||W||_2 <= 1, preventing sharp gradient changes from adversarial perturbations.
        self.classifier = nn.utils.spectral_norm(
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Disentangled attention extraction across content & relative positions
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Extract sequence representation E_P in R^768 ([CLS] representation)
        E_P = outputs.last_hidden_state[:, 0, :]
        
        # Compute classification logits z(P) = [z_inj, z_safe]
        logits = self.classifier(E_P)
        return E_P, logits


# =====================================================================
# 2. PHASE II: PARALLEL RISK EVALUATION ENGINE (L1 - L4)
# =====================================================================

class SPARCSParallelRiskEngine:
    def __init__(self, policy_centroid: np.ndarray, lambda1: float = 0.5, N_max: int = 4096):
        # Normalize authorized policy centroid ||mu_pi||_2 = 1
        self.mu_pi = policy_centroid / (np.linalg.norm(policy_centroid) + 1e-9)
        self.lambda1 = lambda1
        self.N_max = N_max

    async def compute_L1(self, text: str) -> float:
        """
        L1: Privacy Density Tracking via Presidio NER or deterministic entity spans.
        Mapped to exponential saturation: 1 - exp(-lambda1 * sum(w_e))
        """
        await asyncio.sleep(0)  # Async non-blocking yield
        # Example Presidio integration or fast regex entity weighting
        import re
        pii_patterns = {
            "EMAIL": (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), 0.5),
            "API_KEY": (re.compile(r"\b(sk-[A-Za-z0-9]{20,})\b"), 1.0),
            "SSN": (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), 1.0),
            "JWT": (re.compile(r"\beyJ[A-Za-z0-9-_=]+\.[A-Za-z0-9-_=]+\.?[A-Za-z0-9-_.+/=]*\b"), 1.0)
        }
        total_weight = sum(len(pat.findall(text)) * weight for pat, weight in pii_patterns.values())
        return float(1.0 - math.exp(-self.lambda1 * total_weight))

    async def compute_L2(self, logits: torch.Tensor) -> float:
        """
        L2: Intent Probability derived via Softmax over Phase I logits.
        """
        await asyncio.sleep(0)
        probs = F.softmax(logits, dim=-1)
        # Class 0: Injection/Adversarial probability
        return float(probs[0, 0].item())

    async def compute_L3(self, E_P: np.ndarray) -> float:
        """
        L3: Topological Manifold Divergence (normalized angular arc-cosine distance).
        Metric space satisfying triangle inequality: (1 / pi) * arccos(<E_P, mu_pi>)
        """
        await asyncio.sleep(0)
        norm_E = E_P / (np.linalg.norm(E_P) + 1e-9)
        cosine_sim = float(np.clip(np.dot(norm_E, self.mu_pi), -1.0, 1.0))
        return float(math.acos(cosine_sim) / math.pi)

    async def compute_L4(self, token_count: int) -> float:
        """
        L4: Structural Context Bound min(1.0, |P| / N_max)
        """
        await asyncio.sleep(0)
        return float(min(1.0, token_count / self.N_max))

    async def evaluate_risk_vector(self, text: str, E_P: np.ndarray, logits: torch.Tensor, token_count: int) -> np.ndarray:
        l1, l2, l3, l4 = await asyncio.gather(
            self.compute_L1(text),
            self.compute_L2(logits),
            self.compute_L3(E_P),
            self.compute_L4(token_count)
        )
        return np.array([l1, l2, l3, l4], dtype=np.float32)


# =====================================================================
# 3. PHASE IV / L5: OUTBOUND CANARY ENGINE (AHO-CORASICK AUTOMATON)
# =====================================================================

class StatefulCanaryEngine:
    """
    Scans streaming chunks outbound for canary leaks across 4 encoding planes.
    """
    def __init__(self, canary_token: str):
        self.canary_token = canary_token
        self.automaton = ahocorasick.Automaton()
        import base64
        import codecs

        encodings = {
            "raw": canary_token,
            "b64": base64.b64encode(canary_token.encode()).decode(),
            "hex": canary_token.encode().hex(),
            "rot13": codecs.encode(canary_token, "rot_13")
        }

        for label, val in encodings.items():
            self.automaton.add_word(val, (label, val))
        self.automaton.make_automaton()

    def scan_chunk(self, chunk: str) -> bool:
        for _ in self.automaton.iter(chunk):
            return True  # Leak detected
        return False