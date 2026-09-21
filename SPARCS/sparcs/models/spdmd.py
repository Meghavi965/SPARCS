from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoModel


@dataclass
class SPDMDOutput:
    embedding: torch.Tensor
    logits: torch.Tensor
    attention_mask: torch.Tensor
    token_count: torch.Tensor


class SPDMD(nn.Module):
    """Single-pass encoder + pooled representation + spectral classifier.

    One encoder invocation produces the shared representation used by the
    classifier and downstream L3.  No second encoder pass is performed for L1-L4.
    """

    def __init__(self, model_name="microsoft/deberta-v3-base", num_labels=2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        h = int(self.encoder.config.hidden_size)
        if h != 768:
            raise ValueError(f"SPARCS expects a 768-d encoder; got {h}")
        self.embedding_dim = h
        self.num_labels = int(num_labels)
        self.model_name = model_name
        self.classifier = nn.utils.parametrizations.spectral_norm(nn.Linear(h, num_labels))
        self._forward_calls = 0

    @property
    def forward_calls(self):
        return self._forward_calls

    def reset_forward_counter(self):
        self._forward_calls = 0

    @staticmethod
    def masked_mean_pool(hidden_states, attention_mask):
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        return (hidden_states * mask).sum(1) / mask.sum(1).clamp(min=1)

    def forward(self, input_ids, attention_mask, token_type_ids=None, **kwargs):
        self._forward_calls += 1
        encoder_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "return_dict": True,
        }
        if token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = token_type_ids
        # Ignore unrelated tokenizer metadata (e.g. special-token masks).
        out = self.encoder(**encoder_kwargs)
        emb = self.masked_mean_pool(out.last_hidden_state, attention_mask)
        x = emb.to(dtype=self.classifier.weight.dtype, device=self.classifier.weight.device)
        logits = self.classifier(x)
        token_count = attention_mask.sum(dim=-1)
        return SPDMDOutput(emb, logits, attention_mask, token_count)
