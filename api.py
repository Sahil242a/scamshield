"""REST API + web demo.

    python -m uvicorn api:app         ->  http://127.0.0.1:8000  (web demo)
                                          http://127.0.0.1:8000/docs (Swagger)

Environment variables (set by the Dockerfile for public hosting):
    SCAMSHIELD_PUBLIC=1        public demo mode: the /feedback endpoint is disabled (strangers can't write to
                               the server or poison training data) and the correction buttons are hidden
    SCAMSHIELD_RATE_LIMIT=30   max /analyze requests per IP per minute (0 = off)

Privacy: analysed messages are never logged or stored — only verdict counts. The one exception is when the
user presses a correction button: that message is appended to data/feedback.csv on THIS machine only.
"""
from __future__ import annotations

import csv
import os
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from scamshield.detector import ScamDetector

app = FastAPI(title="ScamShield", version="1.0",
              description="Flags UPI/SMS scam messages in English, Hindi and Hinglish, and explains why.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
detector = ScamDetector()
stats: Counter = Counter()

PUBLIC = os.getenv("SCAMSHIELD_PUBLIC", "0") == "1"
RATE_LIMIT = int(os.getenv("SCAMSHIELD_RATE_LIMIT", "30" if PUBLIC else "0"))
_hits: dict[str, deque] = defaultdict(deque)
_rate_lock = Lock()


def check_rate_limit(request: Request) -> None:
    """Sliding-window limit per client IP. In-memory is fine for one small instance; with several
    replicas you'd move this to Redis or the hosting platform's gateway."""
    if RATE_LIMIT <= 0:
        return
    fwd = request.headers.get("x-forwarded-for", "")
    ip = fwd.split(",")[0].strip() or (request.client.host if request.client else "unknown")
    now = time.monotonic()
    with _rate_lock:
        q = _hits[ip]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= RATE_LIMIT:
            raise HTTPException(429, f"Too many requests — limit is {RATE_LIMIT} per minute. Try again shortly.")
        q.append(now)
        if len(_hits) > 10_000:  # keep memory bounded
            for k in [k for k, v in _hits.items() if not v or now - v[-1] > 60]:
                _hits.pop(k, None)


class Message(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, examples=["Your SBI KYC expired. Update now bit.ly/x"])
    sender: str | None = Field(None, max_length=20, description="Sender ID, e.g. 'VM-HDFCBK' or '9876543210'")


class Feedback(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    sender: str | None = Field(None, max_length=20)
    correct_label: str = Field(..., pattern="^(scam|legit)$")
    predicted: str | None = None


FEEDBACK_FILE = Path(__file__).parent / "data" / "feedback.csv"
_feedback_lock = Lock()


class Batch(BaseModel):
    messages: list[Message] = Field(..., max_length=100)


@app.get("/")
def home():
    return FileResponse(Path(__file__).parent / "web" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "backend": detector.backend, "feedback_enabled": not PUBLIC,
            "thresholds": {"scam": detector.t_scam, "suspicious": detector.t_susp},
            "verdicts_served": dict(stats)}


@app.post("/analyze")
def analyze(msg: Message, request: Request):
    check_rate_limit(request)
    t0 = time.perf_counter()
    v = detector.analyze(msg.text, msg.sender)
    stats[v.label] += 1
    return {**v.to_dict(), "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}


@app.post("/feedback")
def feedback(fb: Feedback):
    """Save a correction LOCALLY (data/feedback.csv). Nothing is sent anywhere.
    Next `python data/build_dataset.py && python train.py` learns from it."""
    if PUBLIC:
        raise HTTPException(403, "Corrections are disabled on the public demo. Run ScamShield locally to use them.")
    with _feedback_lock:
        new = not FEEDBACK_FILE.exists()
        with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["text", "label", "sender", "predicted"])
            w.writerow([fb.text, 1 if fb.correct_label == "scam" else 0, fb.sender or "", fb.predicted or ""])
    with open(FEEDBACK_FILE, encoding="utf-8") as f:
        n = sum(1 for _ in csv.DictReader(f))
    return {"saved": True, "total_corrections": n,
            "next_step": "Retrain to learn from it: python data/build_dataset.py && python train.py"}


@app.post("/analyze/batch")
def analyze_batch(batch: Batch, request: Request):
    check_rate_limit(request)
    if not batch.messages:
        raise HTTPException(400, "messages is empty")
    out = [detector.analyze(m.text, m.sender).to_dict() for m in batch.messages]
    for v in out:
        stats[v["label"]] += 1
    return {"results": out}
