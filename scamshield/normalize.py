"""Text normalisation for English, Hindi (Devanagari) and Hinglish SMS.

Why this exists: scammers deliberately vary spelling ("acc0unt", "bl0cked",
"K.Y.C") to dodge keyword filters, and Hinglish has no fixed spelling
("turant" / "turnt", "kijiye" / "kijie"). We normalise the *surface* noise
(digits, homoglyphs, URLs, numbers) and leave spelling variation to the
character n-gram features, which are robust to it.
"""
from __future__ import annotations

import re
import unicodedata

# Devanagari digits ० - ९ -> 0 - 9
_DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

# Common Cyrillic/Greek homoglyphs used to fake Latin letters in links/brands
_HOMOGLYPHS = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x", "у": "y",
    "і": "i", "ј": "j", "ѕ": "s", "Α": "A", "Β": "B", "Ε": "E", "Κ": "K",
    "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Χ": "X",
})

# Leetspeak inside words only (so we don't break real numbers like amounts)
_LEET = {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "@": "a", "$": "s"}
_LEET_WORD = re.compile(r"\b(?=[a-z0-9@$]*[a-z])(?=[a-z0-9@$]*[0-9@$])[a-z0-9@$]{3,}\b")

URL_RE = re.compile(
    r"(?:https?://|www\.)\S+|\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:com|in|co\.in|org|net|xyz|top|icu|"
    r"click|live|online|site|info|buzz|link|ly|gd|me|io|app|sbi|gov\.in|cc|shop|vip|tk)\b(?:/\S*)?",
    re.IGNORECASE,
)
PHONE_RE = re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?!\d)")
AMOUNT_RE = re.compile(r"(?:rs\.?|inr|₹|rupees?|रुपये|रु\.?)\s*[\d,]+(?:\.\d+)?"
                       r"|[\d,]+(?:\.\d+)?\s*(?:rs|rupees?|रुपये|/-)", re.IGNORECASE)
NUMBER_RE = re.compile(r"\d+")


def _unleet(match: re.Match) -> str:
    word = match.group(0)
    # Keep OTPs/refs (many digits) and UPI IDs / emails (name@handle) untouched
    if sum(c.isdigit() for c in word) > 2 or re.search(r"@[a-z]{2,}$", word):
        return word
    return "".join(_LEET.get(c, c) for c in word)


def clean(text: str) -> str:
    """Light clean: unicode-normalise, fix digits/homoglyphs, lowercase, unleet."""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_DEV_DIGITS).translate(_HOMOGLYPHS)
    text = text.lower()
    # "k.y.c" / "o t p" -> "kyc" / "otp"
    text = re.sub(r"\b(?:[a-z][.\s]){2,}[a-z]\b", lambda m: re.sub(r"[.\s]", "", m.group(0)), text)
    text = _LEET_WORD.sub(_unleet, text)
    return re.sub(r"\s+", " ", text).strip()


def for_model(text: str) -> str:
    """Model input: cleaned text with URLs/phones/amounts replaced by tokens.

    Replacing them stops the model memorising specific numbers or domains
    (which never repeat in the wild) and makes it learn the *pattern*
    "message contains a link + an amount".
    """
    t = clean(text)
    t = URL_RE.sub(" urltoken ", t)
    t = PHONE_RE.sub(" phonetoken ", t)
    t = AMOUNT_RE.sub(" amounttoken ", t)
    t = NUMBER_RE.sub(lambda m: " numtoken " if len(m.group(0)) >= 4 else m.group(0), t)
    return re.sub(r"\s+", " ", t).strip()
