from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
import yaml
from transformers import AutoTokenizer

from .models import SPDMD
from .security.canary import CanaryRegistry
from .security.risk import PrivacyDetector, compute_risk


class SPARCS:
    """Reference SPARCS middleware.

    A production/evaluation instance requires a trained checkpoint and a policy
    centroid.  No synthetic/default centroid is silently substituted.
    """

    def __init__(
        self,
        config: str | Path | Mapping = "configs/default.yaml",
        checkpoint: str | Path | None = None,
        centroid: str | Path | None = None,
        device: str | torch.device | None = None,
    ):
        self.cfg = yaml.safe_load(Path(config).read_text()) if isinstance(config, (str, Path)) else dict(config)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        name = self.cfg["model"]["name"]
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = SPDMD(name, self.cfg["model"]["num_labels"]).to(self.device)

        if checkpoint is None:
            raise ValueError("A trained checkpoint is required; refusing an untrained SPARCS model")
        state = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state.get("model", state), strict=True)
        self.model.eval()

        centroid_path = centroid or self.cfg["risk"].get("centroid_path")
        if centroid_path is None:
            raise ValueError("A training-derived policy centroid is required")
        centroid_path = Path(centroid_path)
        if not centroid_path.exists():
            raise FileNotFoundError(f"Policy centroid not found: {centroid_path}")
        self.centroid = torch.load(centroid_path, map_location=self.device, weights_only=False).float()
        if self.centroid.ndim == 1:
            self.centroid = self.centroid.unsqueeze(0)
        if tuple(self.centroid.shape) != (1, self.model.embedding_dim):
            raise ValueError(f"Centroid shape must be (1,{self.model.embedding_dim}), got {tuple(self.centroid.shape)}")
        self.centroid = self.centroid / self.centroid.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        self.privacy = PrivacyDetector(
            self.cfg["privacy"]["enabled"],
            self.cfg["privacy"]["entities"],
        )
        self.canary = CanaryRegistry(
            self.cfg["canary"]["window_chars"],
            self.cfg["canary"].get("enabled", True),
        )

    @torch.inference_mode()
    def inspect(self, text: str) -> dict:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        # L4 must see the true pre-truncation token length; the encoder itself
        # remains bounded by max_length. This is tokenization, not a second
        # encoder/model pass.
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=False,
            add_special_tokens=True,
        )
        token_count = int(enc["attention_mask"].sum().item())
        max_length = int(self.cfg["model"]["max_length"])
        if token_count > max_length:
            # Slice every tensor consistently so tokenizer metadata such as
            # token_type_ids remains aligned. This preserves a single
            # tokenizer pass and a single encoder/model pass.
            enc = {k: v[:, :max_length] for k, v in enc.items() if getattr(v, "ndim", 0) >= 2}
        enc = {k: v.to(self.device) for k, v in enc.items()}
        self.model.reset_forward_counter()
        out = self.model(**enc)
        risk = compute_risk(
            text,
            out.embedding,
            out.logits,
            self.centroid,
            self.cfg,
            self.privacy,
            token_count,
        )
        return {
            "risk": risk,
            "embedding": out.embedding,
            "logits": out.logits,
            "forward_calls": self.model.forward_calls,
        }

    @torch.inference_mode()
    def inspect_batch(self, texts: list[str], batch_size: int = 32) -> list[dict]:
        """Inspect independent requests in batches; each request still uses one encoder pass."""
        if not all(isinstance(t, str) for t in texts):
            raise TypeError("all texts must be strings")
        results=[]; max_length=int(self.cfg["model"]["max_length"])
        for start in range(0, len(texts), max(1, int(batch_size))):
            chunk=texts[start:start+max(1,int(batch_size))]
            raw=self.tokenizer(chunk, padding=True, truncation=False, add_special_tokens=True, return_tensors="pt")
            token_counts=[int(x) for x in raw["attention_mask"].sum(dim=1).tolist()]
            enc={k:v[:, :max_length] for k,v in raw.items() if getattr(v,"ndim",0)>=2}
            enc={k:v.to(self.device) for k,v in enc.items()}
            out=self.model(**enc)
            for j,text in enumerate(chunk):
                risk=compute_risk(text,out.embedding[j:j+1],out.logits[j:j+1],self.centroid,self.cfg,self.privacy,token_counts[j])
                results.append({"risk":risk,"embedding":out.embedding[j:j+1],"logits":out.logits[j:j+1],"forward_calls":1})
        return results

    def create_session(self, session_id=None):
        return self.canary.create(session_id)

    def scan_output(self, session_id: str, chunk: str):
        return self.canary.scan(session_id, chunk)
