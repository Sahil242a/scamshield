"""The hybrid detector: text model + rules -> stacker -> evidence rules -> verdict + explanation.

    text ─► normalise ─► TF-IDF + LogisticRegression ─► text logit ─┐
      └───► rule engine (links, PIN, OTP, urgency, sender…) ─► 28 flags ─┴─► stacker ─► P(scam)

    P ≥ t_scam → SCAM · t_susp ≤ P < t_scam → SUSPICIOUS · else SAFE, then decide() applies:
      * SCAM needs at least one concrete scam signal (otherwise SUSPICIOUS)
      * a critical signal (PIN-to-receive, OTP request, …) is never SAFE
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import joblib
import numpy as np

from .normalize import clean, for_model
from .rules import RULE_IDS, rule_features, run_rules

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

HINGLISH_WORDS = re.compile(
    r"\b(?:hai|hain|aap|aapka|aapki|aapke|karein|kare|karo|kya|nahi|nahin|ka|ki|ke|ko|se|mein|me|bhai|yaar|"
    r"jayega|jayegi|gaya|raha|rahi|hoga|liye|turant|abhi|aaj|kal|bhej|bhejo|batao|kijiye|wala|wali|paise)\b",
    re.IGNORECASE,
)

# Signals so specific to fraud that a message showing one is never called SAFE.
CRITICAL_SIGNALS = {"pin_to_receive", "credential_ask", "remote_access_app", "lookalike_domain", "apk_link",
                    "new_number_money", "fake_alert_callback", "fee_to_personal_upi"}
NO_EVIDENCE_NOTE = {"signal": "wording_only",
                    "why": "No specific red flag (bad link, PIN/OTP request, fake number…) was found, but the wording "
                           "closely resembles known scam messages. Verify through the official app or website before "
                           "acting.", "evidence": None}


def decide(p: float, scam_signals: set[str], t_scam: float, t_susp: float) -> str:
    """Probability -> verdict, with two evidence rules on top of the thresholds.

    1. SCAM needs evidence: if no concrete scam signal fired, the model's wording-based score alone can
       only reach SUSPICIOUS. Unfamiliar-but-legit messages (new bill formats, promos) fail this way,
       so this caps the damage of the text model's blind spots.
    2. Critical signals are never SAFE: e.g. "enter PIN to receive" is at least SUSPICIOUS whatever
       the trust signals say.
    """
    label = "SCAM" if p >= t_scam else "SUSPICIOUS" if p >= t_susp else "SAFE"
    if label == "SCAM" and not scam_signals:
        label = "SUSPICIOUS"
    if label == "SAFE" and scam_signals & CRITICAL_SIGNALS:
        label = "SUSPICIOUS"
    return label


STOPWORDS = set("""a an the is are to of in on at for and or your you ur my me i it this that be will with from by
se ka ki ke ko me mein hai hain ho aap aapka aapki aapke kar karo karein ye yeh wo woh na hi bhi par pe""".split())

ADVICE = {
    "SCAM": ("Do not click links, call numbers, share OTP/PIN or pay anything. Block the sender. "
             "Report at cybercrime.gov.in or call 1930 (National Cyber Helpline). If you already paid, call 1930 now."),
    "SUSPICIOUS": ("Be careful. Don't act from this message — open your bank/utility app directly or call the number "
                   "printed on your card or bill to check."),
    "SAFE": "No strong scam signs found. Still, never share OTP/PIN with anyone, even if a message looks genuine.",
}


def detect_language(text: str) -> str:
    dev = sum("ऀ" <= c <= "ॿ" for c in text)
    letters = sum(c.isalpha() for c in text) or 1
    if dev / letters > 0.3:
        return "hi"
    return "hinglish" if len(HINGLISH_WORDS.findall(text)) >= 2 else "en"


@dataclass
class Verdict:
    label: str                       # SCAM | SUSPICIOUS | SAFE
    scam_probability: float
    language: str
    reasons: list[dict] = field(default_factory=list)       # rule signals pointing to scam
    trust_signals: list[dict] = field(default_factory=list)  # rule signals pointing to legit
    model_keywords: list[str] = field(default_factory=list)  # words the text model weighted most
    text_model_probability: float = 0.0
    advice: str = ""

    def to_dict(self):
        return asdict(self)


class _TransformerText:
    """Wraps a fine-tuned HF classifier so it exposes the same 'scam log-odds' as the TF-IDF model."""

    def __init__(self, path: Path, max_len: int):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch, self.max_len = torch, max_len
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).eval()

    def logits(self, texts: list[str]) -> np.ndarray:
        with self.torch.no_grad():
            e = self.tok([clean(t) for t in texts], truncation=True, max_length=self.max_len, padding=True,
                         return_tensors="pt")
            z = self.model(**e).logits
        return (z[:, 1] - z[:, 0]).numpy()


class ScamDetector:
    def __init__(self, model_dir: Path | str = MODEL_DIR, backend: str | None = None):
        model_dir = Path(model_dir)
        self.backend = backend or os.getenv("SCAMSHIELD_TEXT_MODEL", "tfidf")
        # The TF-IDF model is always loaded: it's the default scorer and it supplies the
        # human-readable keyword explanation even when the transformer does the scoring.
        self.text_model = joblib.load(model_dir / "text_model.joblib")
        cfg_dir = model_dir / "transformer" if self.backend == "transformer" else model_dir
        self.stacker = joblib.load(cfg_dir / "stacker.joblib")
        cfg = json.loads((cfg_dir / "thresholds.json").read_text())
        self.t_scam, self.t_susp = cfg["t_scam"], cfg["t_suspicious"]
        self._transformer = _TransformerText(cfg_dir, cfg.get("max_len", 96)) if self.backend == "transformer" else None
        vec = self.text_model.named_steps["features"]
        self._feature_names = np.array(vec.get_feature_names_out())
        self._coef = self.text_model.named_steps["clf"].coef_[0]

    # -- scoring -------------------------------------------------------------------
    def text_logit(self, texts: list[str]) -> np.ndarray:
        if self._transformer is not None:
            return self._transformer.logits(texts)
        return self.text_model.decision_function([for_model(t) for t in texts])

    def predict_proba(self, texts: list[str], senders: list[str | None] | None = None) -> np.ndarray:
        senders = senders or [None] * len(texts)
        logits = self.text_logit(texts)
        rules = np.array([rule_features(run_rules(t, s)) for t, s in zip(texts, senders, strict=True)])
        return self.stacker.predict_proba(np.column_stack([logits, rules]))[:, 1]

    # -- explanation ---------------------------------------------------------------
    def _keywords(self, text: str, k: int = 5) -> list[str]:
        """Top word features pushing this message towards 'scam' (char n-grams skipped: unreadable)."""
        X = self.text_model.named_steps["features"].transform([for_model(text)])
        idx = X.nonzero()[1]
        contrib = X.toarray()[0, idx] * self._coef[idx]
        out = []
        for i in idx[np.argsort(-contrib)]:
            name = self._feature_names[i]
            if not name.startswith("word__") or self._coef[i] <= 0:
                continue
            if all(w in STOPWORDS for w in name[6:].split()):
                continue
            word = name[6:].replace("urltoken", "<link>").replace("phonetoken", "<phone no.>") \
                           .replace("amounttoken", "<amount>").replace("numtoken", "<number>")
            out.append(word)
            if len(out) == k:
                break
        return out

    def analyze(self, text: str, sender: str | None = None) -> Verdict:
        text = (text or "").strip()
        if not text:
            return Verdict("SAFE", 0.0, "en", advice="Empty message.")
        rr = run_rules(text, sender)
        logit = float(self.text_logit([text])[0])
        feats = np.array([[logit, *rule_features(rr)]])
        p = float(self.stacker.predict_proba(feats)[0, 1])
        scam_ids = {s.id for s in rr.signals if not s.trust}
        label = decide(p, scam_ids, self.t_scam, self.t_susp)
        seen, reasons, trust = set(), [], []
        for s in rr.signals:
            if s.id in seen:
                continue
            seen.add(s.id)
            (trust if s.trust else reasons).append({"signal": s.id, "why": s.explanation, "evidence": s.evidence})
        if label != "SAFE" and not reasons:
            reasons.append(dict(NO_EVIDENCE_NOTE))
        return Verdict(
            label=label, scam_probability=round(p, 4), language=detect_language(text),
            reasons=reasons, trust_signals=trust, model_keywords=self._keywords(text) if p >= self.t_susp else [],
            text_model_probability=round(float(1 / (1 + np.exp(-logit))), 4), advice=ADVICE[label],
        )

    def stacker_weights(self) -> dict[str, float]:
        names = ["text_model_logit", *RULE_IDS]
        return dict(zip(names, map(float, self.stacker.coef_[0]), strict=True))


def format_verdict(v: Verdict) -> str:
    """Plain-text rendering used by the CLI and the Telegram bot."""
    icon = {"SCAM": "🚨", "SUSPICIOUS": "⚠️", "SAFE": "✅"}[v.label]
    lines = [f"{icon} {v.label}  (scam probability {v.scam_probability:.0%})"]
    if v.reasons:
        lines.append("")
    if v.reasons:
        lines.append("Why:")
        lines += [f"• {r['why']}" + (f"  [{r['evidence']}]" if r.get("evidence") else "") for r in v.reasons]
    if v.model_keywords and v.label != "SAFE":
        lines.append(f"• Language pattern the model flagged: {', '.join(v.model_keywords)}")
    if v.trust_signals:
        lines.append("")
        lines.append("Signs it may be genuine:")
        lines += [f"• {r['why']}" for r in v.trust_signals]
    lines += ["", f"What to do: {v.advice}"]
    return "\n".join(lines)
