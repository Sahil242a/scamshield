"""Telegram bot: forward any suspicious SMS to it and get a verdict + explanation.

Setup (2 minutes):
  1. In Telegram, message @BotFather -> /newbot -> copy the token.
  2. export TELEGRAM_BOT_TOKEN=123456:ABC...
  3. python bot.py

Uses Telegram's long-polling HTTP API directly (no framework), so there's nothing
to deploy publicly — it runs from your laptop. For production you'd switch to a
webhook behind the FastAPI app.

Tip for users: start the message with the sender ID on its own line to use the
sender signal, e.g.
    VM-HDFCBK
    482913 is your OTP ...
"""
from __future__ import annotations

import os
import re
import time

import httpx

from scamshield.detector import ScamDetector, format_verdict

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
API = f"https://api.telegram.org/bot{TOKEN}"
WELCOME = ("👋 Namaste! Forward me any SMS/WhatsApp message you're unsure about and I'll tell you if it looks "
           "like a scam — and why.\n\nWorks with English, Hindi and Hinglish.\n\nOptional: put the sender ID on "
           "the first line (e.g. VM-HDFCBK or 9876543210).\n\nIf you've lost money, call 1930 immediately.")
SENDER_LINE = re.compile(r"^\s*([A-Z]{2}-[A-Z0-9]{6}(?:-[A-Z])?|\+?\d{10,13})\s*\n", re.IGNORECASE)


def split_sender(text: str) -> tuple[str | None, str]:
    m = SENDER_LINE.match(text)
    return (m.group(1).upper(), text[m.end():]) if m else (None, text)


def main():
    if not TOKEN:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN first (get one from @BotFather).")
    detector = ScamDetector()
    offset, client = 0, httpx.Client(timeout=40)
    print("ScamShield bot running. Ctrl+C to stop.")
    while True:
        try:
            r = client.get(f"{API}/getUpdates", params={"timeout": 30, "offset": offset}).json()
        except httpx.HTTPError as e:
            print("network error, retrying:", e)
            time.sleep(3)
            continue
        for upd in r.get("result", []):
            offset = upd["update_id"] + 1
            msg = upd.get("message") or upd.get("edited_message")
            if not msg:
                continue
            text = msg.get("text") or msg.get("caption") or ""
            if not text or text.startswith("/start") or text.startswith("/help"):
                reply = WELCOME
            else:
                sender, body = split_sender(text)
                reply = format_verdict(detector.analyze(body, sender))
            client.post(f"{API}/sendMessage", json={"chat_id": msg["chat"]["id"], "text": reply,
                                                    "reply_to_message_id": msg["message_id"]})


if __name__ == "__main__":
    main()
