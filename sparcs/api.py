"""Small FastAPI adapter around the SPARCS middleware.

The security logic lives in ``SPARCS``; this module is only a transport layer.
"""
from __future__ import annotations

import os

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except Exception:  # pragma: no cover
    FastAPI = None


def create_app(guardrail=None):
    if FastAPI is None:
        raise RuntimeError("fastapi is required for the HTTP adapter")
    from .guardrail import SPARCS
    g = guardrail or SPARCS(
        os.getenv("SPARCS_CONFIG", "configs/calibrated.yaml"),
        os.environ.get("SPARCS_CHECKPOINT"),
        os.environ.get("SPARCS_CENTROID"),
    )
    app = FastAPI(title="SPARCS", version="0.1.0")

    class InspectRequest(BaseModel):
        text: str

    class SessionRequest(BaseModel):
        session_id: str | None = None

    class ScanRequest(BaseModel):
        session_id: str
        chunk: str

    @app.get("/health")
    def health():
        return {"status": "ok", "device": str(g.device)}

    @app.post("/inspect")
    def inspect(req: InspectRequest):
        result = g.inspect(req.text)
        rv = result["risk"]
        return {"allow": not rv.decision, "risk": rv.__dict__, "forward_calls": result["forward_calls"]}

    @app.post("/sessions")
    def sessions(req: SessionRequest):
        s = g.create_session(req.session_id)
        return {"session_id": s.session_id}

    @app.post("/scan")
    def scan(req: ScanRequest):
        try:
            hits = g.scan_output(req.session_id, req.chunk)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        return {"halt": bool(g.canary.sessions[req.session_id].halted), "hits": [{"encoding": h.encoding, "start": h.start, "end": h.end} for h in hits]}

    return app


app = None
if FastAPI is not None:
    # Do not instantiate SPARCS at import time: importing the package must not
    # download/load a 768-d encoder or require a trained artifact.
    app = FastAPI(title="SPARCS", version="0.1.0")
