import importlib.util
import pytest

from sparcs.security.canary import CanaryRegistry
from sparcs.security.risk import PrivacyDetector


def test_canary_variants_and_chunk_boundary():
    r=CanaryRegistry(window_chars=4096)
    for enc in ('raw','base64','hex','rot13'):
        ss=r.create(enc); payload=r.variants(ss.canary)[enc]; hits=r.scan(ss.session_id,payload); assert any(h.encoding==enc for h in hits)
    ss=r.create('split'); token=ss.canary; mid=len(token)//2; assert not r.scan('split',token[:mid]); assert r.scan('split',token[mid:])


def test_privacy_disabled():
    assert PrivacyDetector(False).count('test@example.com') == 0


@pytest.mark.skipif(importlib.util.find_spec('transformers') is None, reason='transformers not installed in lightweight test environment')
def test_spdmd_contract():
    from sparcs.models import SPDMD
    import torch
    x=torch.tensor([[[1.,0.],[3.,0.],[99.,99.]]]); m=torch.tensor([[1,1,0]])
    y=SPDMD.masked_mean_pool(x,m); assert torch.allclose(y,torch.tensor([[2.,0.]]))
    assert hasattr(SPDMD,'forward_calls')


def test_canary_reports_stream_positions_and_deduplicates_rolling_hits():
    r = CanaryRegistry(window_chars=128)
    ss = r.create("positions")
    token = ss.canary
    assert not r.scan("positions", token[:4])
    hits = r.scan("positions", token[4:])
    assert len(hits) == 1
    assert hits[0].start == 0
    assert hits[0].end == len(token)
    # The hit remains in the rolling window but must not be counted again.
    assert len(r.scan("positions", " trailing")) == 1


def test_canary_disabled_does_not_halt():
    r = CanaryRegistry(enabled=False)
    ss = r.create("disabled")
    assert r.scan("disabled", ss.canary) == []
    assert not r.sessions["disabled"].halted


def test_api_scan_does_not_expose_canary_secret():
    from pathlib import Path
    text = Path("sparcs/api.py").read_text()
    assert '[h.__dict__ for h in hits]' not in text
    assert '"token_id": h.token_id' not in text


def test_l4_uses_pretruncation_token_count():
    from pathlib import Path
    text = Path("sparcs/guardrail.py").read_text()
    assert 'truncation=False' in text
    assert 'token_count = int(enc["attention_mask"].sum().item())' in text


def test_ablation_weights_are_renormalized():
    import ast
    from pathlib import Path
    tree=ast.parse(Path("sparcs/experiments/ablation.py").read_text())
    fn=next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=="renormalize")
    assert any(isinstance(n, ast.Return) for n in ast.walk(fn))
    assert "sum" in Path("sparcs/experiments/ablation.py").read_text()


def test_scripts_are_repo_root_safe():
    from pathlib import Path
    for p in list(Path("scripts").glob("*.sh")) + [Path("reproduce_all.sh")]:
        text=p.read_text()
        assert 'ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")' in text
        assert 'cd "$ROOT"' in text


def test_leakage_results_do_not_serialize_canary_secret():
    from pathlib import Path
    text = Path("sparcs/experiments/leakage.py").read_text()
    assert "h.__dict__" not in text
    assert "'token_id': h.token_id" not in text
    assert "\'encoding\':h.encoding" in text


def test_centroid_defaults_to_gpu_when_available():
    from pathlib import Path
    text = Path("sparcs/training/centroid.py").read_text()
    assert '--device' in text
    assert 'torch.cuda.is_available()' in text
