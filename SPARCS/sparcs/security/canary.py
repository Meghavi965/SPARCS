"""Stateful outbound SPARCS canary and encoded-leakage scanner."""
from __future__ import annotations

import base64
import codecs
import secrets
from collections import deque
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CanaryHit:
    encoding: str
    start: int
    end: int
    token_id: str


@dataclass
class CanarySession:
    session_id: str
    canary: str
    rolling: str = ""
    stream_length: int = 0
    hits: list[CanaryHit] = field(default_factory=list)
    halted: bool = False


class _AhoCorasick:
    """Small dependency-free Aho-Corasick matcher for the four canary variants."""

    def __init__(self, patterns: dict[str, str]):
        self.next = [{}]
        self.fail = [0]
        self.out = [[]]
        for label, pattern in patterns.items():
            node = 0
            for ch in pattern:
                nxt = self.next[node].get(ch)
                if nxt is None:
                    nxt = len(self.next)
                    self.next[node][ch] = nxt
                    self.next.append({})
                    self.fail.append(0)
                    self.out.append([])
                node = nxt
            self.out[node].append((label, len(pattern)))

        q = deque()
        for child in self.next[0].values():
            q.append(child)
            self.fail[child] = 0
        while q:
            node = q.popleft()
            for ch, child in self.next[node].items():
                q.append(child)
                f = self.fail[node]
                while f and ch not in self.next[f]:
                    f = self.fail[f]
                self.fail[child] = self.next[f].get(ch, 0)
                self.out[child].extend(self.out[self.fail[child]])

    def find(self, text: str) -> list[tuple[str, int, int]]:
        node = 0
        hits: list[tuple[str, int, int]] = []
        for i, ch in enumerate(text):
            while node and ch not in self.next[node]:
                node = self.fail[node]
            node = self.next[node].get(ch, 0)
            for label, length in self.out[node]:
                hits.append((label, i + 1 - length, i + 1))
        return hits


class CanaryRegistry:
    """Session registry with chunk-boundary-safe rolling scanning."""

    def __init__(self, window_chars: int = 4096, enabled: bool = True):
        if int(window_chars) <= 0:
            raise ValueError("window_chars must be positive")
        self.window_chars = int(window_chars)
        self.enabled = bool(enabled)
        self.sessions: dict[str, CanarySession] = {}

    def create(self, session_id: str | None = None) -> CanarySession:
        sid = session_id or secrets.token_hex(12)
        if sid in self.sessions:
            raise ValueError(f"session already exists: {sid}")
        raw = f"SPARCS-CANARY-{secrets.token_urlsafe(18)}"
        session = CanarySession(sid, raw)
        self.sessions[sid] = session
        return session

    def close(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)

    @staticmethod
    def variants(token: str) -> dict[str, str]:
        b = token.encode("utf-8")
        return {
            "raw": token,
            "base64": base64.b64encode(b).decode("ascii"),
            "hex": b.hex(),
            "rot13": codecs.encode(token, "rot_13"),
        }

    def scan(self, session_id: str, chunk: str) -> list[CanaryHit]:
        if session_id not in self.sessions:
            raise KeyError(f"unknown session: {session_id}")
        if not isinstance(chunk, str):
            raise TypeError("chunk must be a string")
        session = self.sessions[session_id]
        if not self.enabled or session.halted:
            return list(session.hits[-4:])

        session.stream_length += len(chunk)
        session.rolling = (session.rolling + chunk)[-self.window_chars :]
        base_offset = session.stream_length - len(session.rolling)
        matcher = _AhoCorasick(self.variants(session.canary))
        hits: list[CanaryHit] = []
        existing = {(h.encoding, h.start, h.end) for h in session.hits}
        for encoding, start, end in matcher.find(session.rolling):
            key = (encoding, base_offset + start, base_offset + end)
            if key in existing:
                continue
            hit = CanaryHit(encoding, key[1], key[2], session.canary)
            session.hits.append(hit)
            hits.append(hit)
            existing.add(key)
        if hits:
            session.halted = True
        return list(session.hits[-4:])
