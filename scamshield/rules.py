"""Rule-based signals: deterministic, explainable checks that run alongside the model.

Each rule encodes domain knowledge a model trained on a few thousand SMS can't reliably learn
(e.g. "you never need your UPI PIN to *receive* money", or "sbi-kyc-update.xyz is not an SBI
domain"). Every rule produces:
  * a binary feature (fed into the stacker that combines rules with the text model), and
  * a human-readable explanation, with the exact evidence (shown to the user).

Scam signals push towards SCAM; trust signals (prefix ``trust_``) push towards SAFE. Trust
signals matter for precision: a genuine OTP SMS mentions "OTP" and an amount, so without
them it looks scary.

Rules see text after ``clean()`` (lower-cased, leetspeak undone, Devanagari digits mapped),
except link and phone extraction, which use the raw text.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

from .normalize import PHONE_RE, URL_RE, clean

# ------------------------------------------------------------------------------ reference data
OFFICIAL_DOMAINS = {
    # banks, payments, regulators
    "sbi.co.in", "onlinesbi.sbi", "sbi.bank.in", "hdfcbank.com", "hdfc.bank.in", "icicibank.com",
    "icici.bank.in", "axisbank.com", "axis.bank.in", "pnbindia.in", "kotak.com", "bankofbaroda.in",
    "canarabank.com", "unionbankofindia.co.in", "yesbank.in", "idfcfirstbank.com", "npci.org.in", "rbi.org.in",
    "bhimupi.org.in", "paytm.com", "phonepe.com",
    # government
    "incometax.gov.in", "uidai.gov.in", "indiapost.gov.in", "india.gov.in", "digilocker.gov.in",
    "cybercrime.gov.in", "indianrail.gov.in", "irctc.co.in",
    # telecom and utilities
    "jio.com", "airtel.in", "myvi.in", "bsnl.co.in", "bsnl.in", "energy.rajasthan.gov.in", "bsesdelhi.com",
    "mahadiscom.in", "uppcl.org", "tnebltd.gov.in",
    # shopping, delivery, travel, apps
    "amazon.in", "amazon.com", "flipkart.com", "myntra.com", "swiggy.com", "zomato.com", "bigbasket.com",
    "meesho.com", "ajio.com", "nykaa.com", "jiomart.com", "croma.com", "reliancedigital.in", "bluedart.com",
    "delhivery.com", "makemytrip.com", "olacabs.com", "uber.com", "goindigo.in", "airindia.com",
    "google.com", "apple.com", "microsoft.com", "netflix.com", "hotstar.com", "linkedin.com",
    "instagram.com", "whatsapp.com", "naukri.com",
    # insurance, investing, health
    "licindia.in", "hdfclife.com", "policybazaar.com", "zerodha.com", "groww.in", "lalpathlabs.com",
}
# Registries scammers can't buy from
RESTRICTED_SUFFIXES = (".gov.in", ".nic.in", ".bank.in")
# Brand names that, inside a NON-official domain, mark a look-alike (sbi-kyc-update.xyz)
BRAND_TOKENS = [
    "sbi", "hdfc", "icici", "axis", "kotak", "pnb", "baroda", "paytm", "phonepe", "gpay", "googlepay",
    "amazon", "flipkart", "jio", "airtel", "kyc", "incometax", "refund", "bijli", "electricity", "fedex", "dhl",
    "indiapost", "bluedart", "delhivery", "yono", "npci", "rbi", "uidai", "aadhaar", "aadhar", "parivahan",
    "challan", "epfo", "epf", "uan", "lic", "scholarship", "yojana", "subsidy", "lpg", "netbanking", "irctc",
    "netflix", "whatsapp",
]
SHORTENERS = {
    "bit.ly", "tinyurl.com", "cutt.ly", "rb.gy", "is.gd", "t.ly", "goo.gl", "shorturl.at",
    "tiny.cc", "ow.ly", "rebrand.ly", "v.gd", "s.id", "t.co", "bitly.ws",
}
# Deep-link services real apps use in SMS (Branch, Firebase, AppsFlyer, Adjust). NEUTRAL:
# not proof of legitimacy (anyone can create one), but not a phishing domain either.
DEEPLINK_SERVICES = {"app.link", "page.link", "onelink.me", "adj.st", "go.link", "smart.link"}
RISKY_TLDS = {
    "xyz", "top", "icu", "click", "live", "online", "site", "info", "buzz", "link", "shop",
    "vip", "tk", "cc", "rest", "fun", "monster", "cyou", "sbs", "cfd",
}
# TRAI DLT-registered business header, e.g. AX-SBIINB, VM-HDFCBK, JD-JVVNLT-S
DLT_HEADER_RE = re.compile(r"^[A-Z]{2}-[A-Z0-9]{6}(?:-[A-Z])?$")

# English negation comes before the verb ("do not share"); see _negated_after for Hindi.
NEG = r"(?:do\s*not|don'?t|never|not|mat|na|nahi|nahin|kabhi\s*nahi|न|मत|कभी\s*नहीं|ना)"

# ------------------------------------------------------------------------------ signal registry
# Expert prior log-odds per signal: how much each should shift the odds of "scam" before seeing
# data. The stacker starts here and lets the data adjust them. The ORDER of this dict defines
# the model's feature order, so append new signals at the end of their group.
PRIOR_WEIGHTS = {
    "url_shortener": 1.5, "risky_tld": 1.5, "ip_url": 2.0, "lookalike_domain": 2.5, "apk_link": 2.5,
    "pin_to_receive": 3.0, "credential_ask": 3.0, "remote_access_app": 3.0, "urgency": 0.6, "threat": 1.0,
    "kyc_pan_update": 1.0, "prize_lottery": 1.2, "work_from_home": 1.2, "fee_to_personal_upi": 2.0,
    "call_personal_number": 1.0, "impersonation_from_personal_number": 1.5, "sent_by_mistake": 1.5,
    "authority_threat": 1.5, "investment_returns": 2.0, "new_number_money": 2.0, "international_sender": 0.8,
    "fake_alert_callback": 2.5,
    "trust_official_domain": -2.0, "trust_do_not_share": -2.0, "trust_dlt_sender": -1.0, "trust_txn_alert": -2.5,
    "trust_otp_delivery": -2.0, "trust_fraud_advisory": -1.5,
}
RULE_IDS = list(PRIOR_WEIGHTS)


@dataclass
class Signal:
    id: str
    explanation: str
    evidence: str | None = None
    trust: bool = False  # True = evidence the message is legitimate


@dataclass
class RuleResult:
    signals: list[Signal] = field(default_factory=list)

    def fired(self) -> set[str]:
        return {s.id for s in self.signals}

    def add(self, *args, **kwargs) -> None:
        self.signals.append(Signal(*args, **kwargs))


# ------------------------------------------------------------------------------ helpers
def _find(pattern: str, text: str) -> str | None:
    m = re.search(pattern, text, re.IGNORECASE)
    return m.group(0) if m else None


def _negated(text: str, start: int, window: int = 30) -> bool:
    """Negation just BEFORE position `start` ("do not share")."""
    return re.search(NEG + r"\b", text[max(0, start - window):start], re.IGNORECASE) is not None


def _negated_after(text: str, end: int, window: int = 18) -> bool:
    """Negation just AFTER position `end`. Hindi/Hinglish is verb-final (SOV): "share na karein",
    "batana mat", "साझा न करें"."""
    return re.match(r"\W*(?:\w+\s+)?(?:na|mat|nahi|nahin|न|मत|नहीं)\b", text[end:end + window]) is not None


def is_official(reg: str) -> bool:
    return reg in OFFICIAL_DOMAINS or reg.endswith(RESTRICTED_SUFFIXES)


def registered_domain(host: str) -> str:
    """'www.retail.onlinesbi.sbi' -> 'onlinesbi.sbi'; 'x.y.co.in' -> 'y.co.in'."""
    host = host.lower().strip(".").split(":")[0]
    host = host.removeprefix("www.")
    parts = host.split(".")
    if len(parts) >= 3 and ".".join(parts[-2:]) in {"co.in", "gov.in", "org.in", "bank.in", "net.in", "ac.in"}:
        return ".".join(parts[-3:])
    if len(parts) >= 3 and ".".join(parts[-3:]) in OFFICIAL_DOMAINS:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def extract_urls(text: str) -> list[str]:
    return [u.rstrip(".,)!") for u in URL_RE.findall(text)]


def url_host(url: str) -> str:
    if not re.match(r"https?://", url, re.IGNORECASE):
        url = "http://" + url
    return urlparse(url).netloc.lower()


# ------------------------------------------------------------------------------ rule groups
def _link_signals(r: RuleResult, raw: str, t: str) -> None:
    urls = extract_urls(raw)
    hosts = [url_host(u) for u in urls]
    regs = [registered_domain(h) for h in hosts if h]
    for host, reg, url in zip(hosts, regs, urls, strict=False):
        if reg in SHORTENERS or host in SHORTENERS:
            r.add("url_shortener", "Uses a link shortener that hides the real website. Banks and government "
                  "bodies don't send shortened links. / Link chhupaya gaya hai.", url)
        if reg in DEEPLINK_SERVICES:
            continue
        tld = reg.rsplit(".", 1)[-1]
        if tld in RISKY_TLDS:
            r.add("risky_tld", f"Link ends in '.{tld}', a cheap domain extension widely used for phishing.", url)
        if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", host.split(":")[0]):
            r.add("ip_url", "Link points to a raw IP address instead of a website name.", url)
        if not is_official(reg) and reg not in SHORTENERS and any(b in host for b in BRAND_TOKENS):
            r.add("lookalike_domain", f"'{reg}' contains a bank/brand name but is NOT that company's official "
                  "website. Classic phishing look-alike. / Yeh asli website nahi hai.", url)

    apk = next((u for u in urls if u.lower().split("?")[0].endswith(".apk")), None) or _find(r"\bapk\b", t)
    if apk:
        r.add("apk_link", "Asks you to install an app from a link/file (APK). These often steal OTPs and SMS. "
              "Only install apps from the Play Store.", apk)

    # Trust only if EVERY (non-deep-link) link is official: a phishing link can't hide behind a real one
    official = [d for d in regs if is_official(d)]
    non_deeplink = [d for d in regs if d not in DEEPLINK_SERVICES]
    bad_link = r.fired() & {"lookalike_domain", "url_shortener", "risky_tld"}
    if official and len(official) == len(non_deeplink) and not bad_link:
        r.add("trust_official_domain", f"Link goes to an official domain ({official[0]}).", official[0], trust=True)


PIN_TO_RECEIVE = (
    r"(?:upi\s*)?pin.{0,40}(?:receive|get|claim|credit|lene|paane|prapt)"
    r"|(?:receive|claim|lene|paane).{0,40}(?:upi\s*)?pin\b"
    r"|(?:पिन).{0,30}(?:प्राप्त|पाने)"
    r"|(?:refund|cashback|paise\s*aa|money\s*will|रिफंड).{0,80}\bpin\b"
    r"|\bpin\b.{0,50}(?:paise\s*aa|aa\s*jayenge|refund|cashback)"
    # approving a UPI collect request / scanning a QR also SENDS money
    r"|(?:approve|accept|swikar).{0,40}(?:to\s*(?:receive|get)|paise\s*aa|money\s*will|payment\s*will)"
    r"|(?:request|qr).{0,40}(?:approve|accept|scan).{0,30}(?:receive|get\s*(?:the\s*)?(?:money|payment)|paise\s*aa)"
)
CREDENTIAL_WORDS = r"\b(?:otp|cvv|pin|password|m-?pin|card\s*number|ओटीपी|पिन)\b"
ASK_VERBS = (r"share|send|tell|reply|forward|batao|bataye|bataiye|bhejo|bhejein|bhej|provide|give|confirm|"
             r"बताएं|बताओ|भेजें|दें")
# Genuine "handover" OTPs: you give the code to the delivery agent / driver standing in front of you
HANDOVER_OTP = (r"delivery\s*(?:agent|partner|executive|boy)|driver|captain|rider|start\s*(?:the|your)?\s*"
                r"(?:ride|trip)|at\s*the\s*time\s*of\s*delivery|technician|ko\s*batayein|driver\s*ko|"
                r"after\s*the\s*(?:job|service|delivery)|डिलीवरी\s*एजेंट")
CODE_TAKEOVER = (
    r"(?:code|otp|ओटीपी).{0,50}(?:send\s*it\s*back|send\s*(?:it\s*)?to\s*me|forward|wapas\s*bhej|mujhe\s*bhej|"
    r"bhej\s*d[oe]|bata\s*d[oe])"
    r"|(?:galti|mistake|accidentally|by\s*mistake).{0,60}(?:code|otp)"
    r"|(?:code|otp).{0,25}(?:galti|mistake|by\s*mistake)"
    r"|(?:कोड|ओटीपी).{0,60}(?:मुझे\s*भेज|भेज\s*दें|भेज\s*दो|वापस)|गलती\s*से.{0,40}(?:कोड|ओटीपी)"
)
UPI_FEE = (r"(?:send|pay|transfer|bhej\w*|jama|return|wapas|भेजें).{0,40}(?:\b[\w.-]{2,}@(?:ybl|okaxis|oksbi|"
           r"okhdfcbank|okicici|paytm|upi|ibl|axl|apl)\b|upi\s*id)"
           r"|(?:processing|registration|delivery|clearance|verification)\s*(?:fee|charge|charges)")


def _money_signals(r: RuleResult, t: str) -> None:
    m = re.search(PIN_TO_RECEIVE, t)
    if m and not _negated(t, m.start(), 20):
        r.add("pin_to_receive", "Asks you to enter your UPI PIN, scan a QR, or approve a request to *receive* "
              "money. Receiving money never needs any of these — each one SENDS money out of your account. "
              "/ Paise lene ke liye PIN, QR scan ya request approve kabhi nahi karna padta.", m.group(0))

    if not re.search(HANDOVER_OTP, t):
        for m in re.finditer(CREDENTIAL_WORDS, t):
            offset = max(0, m.start() - 40)
            ctx = t[offset: m.end() + 40]
            asks = re.search(ASK_VERBS, ctx)
            if not asks:
                continue
            if not (_negated(t, m.start(), 40) or _negated(t, offset + asks.start(), 25)
                    or _negated_after(t, offset + asks.end())):
                r.add("credential_ask", "Asks you to share an OTP/PIN/CVV/password. No genuine bank, app or "
                      "officer will ever ask for these. / OTP kisi ko mat batao.", ctx.strip())
                break

    m = re.search(CODE_TAKEOVER, t)
    if m and "credential_ask" not in r.fired():
        r.add("credential_ask", "Asks you to send back a code/OTP that 'came to your number by mistake'. That code "
              "is the key to YOUR account (WhatsApp, bank, UPI) — sending it hands the account over.", m.group(0))

    e = _find(r"anydesk|teamviewer|quick\s*support|rustdesk|airdroid|screen\s*shar", t)
    if e:
        r.add("remote_access_app", "Asks you to install a screen-sharing / remote-control app. Scammers use these "
              "to watch your screen and take over your banking apps.", e)

    e = _find(UPI_FEE, t)
    if e:
        r.add("fee_to_personal_upi", "Asks you to pay an upfront 'fee' to claim a prize, loan, job or parcel. "
              "Real prizes/loans don't need you to pay first.", e)


def _pressure_signals(r: RuleResult, t: str) -> None:
    e = _find(r"\b(?:immediately|urgent(?:ly)?|within\s*\d+\s*(?:hrs?|hours?|mins?|minutes?)|today|tonight|"
              r"last\s*(?:date|chance|day)|expire[sd]?\s*today|turant|abhi|aaj\s*raat|aaj\s*hi|jaldi|"
              r"\d+\s*ghante|तुरंत|आज\s*रात|अभी|घंटे)\b", t)
    if e:
        r.add("urgency", "Creates time pressure ('today', 'within 24 hours', 'turant') so you act before thinking.", e)

    e = _find(r"block(?:ed)?|blacklist\w*|restrict\w*|re-?activat\w*|susp\w{0,3}n?d\w*|on\s*hold|deactivat\w*|"
              r"disconnect\w*|(?:will\s*be|be)\s*(?:cut|closed|freeze|frozen)|band\s*ho|kaat\s*di|kat\s*jayega|"
              r"बंद|काट|ब्लॉक|freez\w*|lapse|closure", t)
    if e and "if not you" not in t and "not you?" not in t:
        r.add("threat", "Threatens to block/cut/close your account, SIM or electricity to scare you.", e)

    e = _find(r"\b(?:cbi|police|customs|narcotics|ncb|cyber\s*cell|ed\s*officer|digital\s*arrest|arrest|"
              r"legal\s*action|fir|court|illegal|drugs|mdma|crime\s*branch|money\s*laundering|trai|गिरफ्तार|पुलिस)\b", t)
    if e and not re.search(r"cybercrime\.gov\.in|1930", t):
        r.add("authority_threat", "Uses police/CBI/customs/'digital arrest' threats. Indian agencies never arrest "
              "or demand money over phone or SMS.", e)


def _storyline_signals(r: RuleResult, raw: str, t: str) -> None:
    e = _find(r"\b(?:kyc|e-?kyc|pan\s*card|pan\s*(?:update|link)|aadhaar\s*(?:update|link)|केवाईसी|पैन)\b", t)
    at_branch = re.search(r"visit(?:ing)?\s*(?:your|the)?\s*(?:nearest)?\s*(?:\w+\s)?branch|branch\s*(?:jaa|visit)", t)
    if e and not at_branch:
        r.add("kyc_pan_update", "Asks for KYC/PAN/Aadhaar update via SMS. Banks ask you to update KYC at the branch "
              "or in their official app — not through SMS links or calls.", e)

    e = _find(r"lottery|lucky\s*draw|jackpot|you\s*(?:have\s*)?won|won\s*(?:a|rs|inr|₹)|winner|"
              r"congratulations.{0,40}(?:won|prize|selected|winner|gift)|prize|jeeta|jeet\s*gaye|inaam|इनाम|लॉटरी|"
              r"जीत|kbc|free\s*(?:iphone|gift|recharge)", t)
    if e and "scratch card" not in t:
        r.add("prize_lottery", "Promises a prize/lottery you never entered. Unexpected winnings are almost always "
              "bait.", e)

    e = _find(r"part[\s-]*time|work\s*from\s*home|ghar\s*baith\w*|घर\s*बैठे|daily\s*(?:income|earn|payment)|"
              r"earn\s*(?:rs|₹|inr|\d)|like\s*(?:youtube\s*)?videos?|rate\s*(?:hotels|products)|simple\s*task|"
              r"roz\s*(?:kamaye|kamao)|प्रतिदिन|रोज", t)
    if e:
        r.add("work_from_home", "Offers easy money for simple tasks (liking videos, rating products). This is the "
              "'task scam' that later asks you to 'invest' to unlock earnings.", e)

    e = _find(r"(?:galti|mistake|mistakenly|wrongly|by\s*error).{0,60}(?:transfer|sent|bhej|credit)|"
              r"(?:transfer|sent|bhej|credit\w*).{0,40}(?:galti|mistake|mistakenly|wrongly|by\s*error)", t)
    if e:
        r.add("sent_by_mistake", "Claims money was sent to you 'by mistake' and asks you to return it. Check your "
              "actual bank balance in the bank app first — the 'credit' SMS is often fake.", e)

    e = _find(r"guaranteed\s*(?:returns?|profit|income)|\d{2,4}\s*%\s*(?:returns?|profit)|"
              r"double\s*(?:your|ur)\s*money|stock\s*tips|trading\s*(?:group|tips)|vip\s*(?:group|signals?|tips)|"
              r"crypto\s*(?:profit|investment)|paisa\s*double|t\.me/|telegram\s*group|ipo\s*allotment\s*guaranteed", t)
    if e:
        r.add("investment_returns", "Promises guaranteed/huge investment returns or a 'VIP tips' group. Guaranteed "
              "high returns don't exist; SEBI-registered advisers never promise them.", e)

    new_number = re.search(r"new\s*number|naya\s*number|changed\s*(?:my\s*)?number|number\s*(?:change|badal)|"
                           r"phone\s*(?:is\s*)?(?:lost|broke|broken|fell|stolen|kho|gir)|नया\s*नंबर", t)
    if new_number and re.search(r"send|bhej|transfer|upi|paise|money|pay|₹|rs\.?\s*\d|\d{3,}", t):
        r.add("new_number_money", "Someone 'on a new number' urgently asking for money — the family/friend "
              "impersonation scam. Call the person on their OLD number before paying.")

    alert = re.search(r"debit\w*|withdraw\w*|nikale|kat\s*gaye|transaction|order\s*(?:of|is|has)|confirmed|"
                      r"new\s*device|login|निकाले|डेबिट", t)
    callback = re.search(r"(?:not\s*(?:done\s*by\s*)?you|did\s*not|didn'?t|wasn'?t\s*you|nahi\s*kiya|आपने\s*नहीं)"
                         r".{0,80}(?<![\d])[6-9]\d{9}\b", t)
    if alert and callback:
        phones = PHONE_RE.findall(raw)
        r.add("fake_alert_callback", "A scary 'debit / order / new device' alert that tells you to call a personal "
              "mobile number if it wasn't you. Real banks and shops use toll-free 1800 numbers or their app — "
              "the 'fraud team' on that mobile number is the scammer.", phones[0] if phones else None)


def _number_and_sender_signals(r: RuleResult, raw: str, t: str, sender: str | None) -> None:
    phones = PHONE_RE.findall(raw)
    if phones and re.search(r"call|contact|whatsapp|sampark|baat|संपर्क|कॉल|व्हाट्सएप|officer|adhikari|"
                            r"अधिकारी|executive|press", t):
        r.add("call_personal_number", "Asks you to call/WhatsApp a personal 10-digit mobile number. Real companies "
              "use registered toll-free numbers or official apps.", phones[0])

    s = (sender or "").strip().upper()
    compact = s.replace(" ", "")
    if re.fullmatch(r"\+(?!91)\d{8,14}", compact):
        r.add("international_sender", f"Sent from an international number ({s}). Unexpected messages from foreign "
              "numbers (e.g. 'is this Ravi?', job offers) are a common start of investment scams.", s)
    impersonates = re.search(r"\b(?:bank|sbi|hdfc|icici|axis|kotak|electricity|bijli|बिजली|बैंक|jio|airtel|"
                             r"vodafone|vi\b|income\s*tax|customs|courier|fedex|india\s*post)", t)
    if s and DLT_HEADER_RE.match(s):
        r.add("trust_dlt_sender", f"Sent from a registered business header ({s}). Note: still check the content — "
              "headers lower but don't remove risk.", s, trust=True)
    elif s and re.fullmatch(r"\+?\d{10,13}", compact) and impersonates:
        r.add("impersonation_from_personal_number", f"Claims to be a bank/utility/telecom but was sent from an "
              f"ordinary mobile number ({s}). Real ones send from registered headers like 'VM-HDFCBK'.", s)


def _trust_signals(r: RuleResult, t: str) -> None:
    do_not_share = (
        re.search(NEG + r"\s*(?:\w+\s){0,3}(?:share|disclose|batay|bataye|bataen|batana|बताएं|साझा)", t)
        or re.search(r"(?:share|batana|bataye|bataen|bhejna|बताएं|बताना|साझा)\s*(?:\w+\s)?(?:na|mat|nahi|न|मत)\b", t)
        or re.search(r"never\s*asks?\s*for|kabhi\s*nahi\s*maang|कभी\s*नहीं\s*मांग", t)
    )
    if do_not_share:
        r.add("trust_do_not_share", "Contains a 'do not share' warning, typical of genuine OTP/bank messages.",
              trust=True)

    txn_alert = re.search(
        r"(?:debited|credited|deposited|sent|received|spent)\b.{0,60}(?:a/?c|acct|account|card)\s*(?:no\.?\s*)?"
        r"(?:x+|\*+)\d{3,4}|(?:avl|available)\s*bal|\bbal\s*(?:rs|inr|₹)|upi\s*ref|ref\s*no|"
        r"(?:received|credited)\s*(?:rs\.?|inr|₹)\s*[\d,.]+\s*from\s+[a-z]", t)
    mobile_callback = re.search(r"if\s*(?:this\s*was\s*)?not\s*(?:done\s*by\s*)?you.{0,40}\b[6-9]\d{9}\b|"
                                r"nahi\s*kiya.{0,40}\b[6-9]\d{9}\b", t)
    if txn_alert and not mobile_callback:
        r.add("trust_txn_alert", "Looks like a standard transaction alert (masked account number, UPI reference). "
              "These are informational and ask nothing of you.", trust=True)

    # A genuine OTP SMS *contains* the code; a scam *asks* for it.
    otp_delivered = re.search(
        r"\b\d{4,8}\b\s*(?:is|hai|है)?\s*(?:your|the|aapka|आपका)?\s*(?:\w+\s){0,3}"
        r"(?:otp|one\s*time\s*password|verification\s*code|code|ओटीपी)"
        r"|(?:otp|code)\s*(?:\w+\s){0,6}(?:is|:)\s*\d{4,8}\b|\b(?:otp|ओटीपी)\s*:?\s*\d{4,8}\b", t)
    if otp_delivered and "credential_ask" not in r.fired():
        r.add("trust_otp_delivery", "This message delivers an OTP to you (the code is in it) rather than asking you "
              "for one — that's how genuine OTP messages look. Just don't forward it to anyone.", trust=True)

    if re.search(r"never\s*(?:ask|send)|will\s*never|(?:nahi|नहीं)\s*(?:maang|मांग)|beware\s*of|fraud\s*alert|"
                 r"stay\s*safe|savdhan|सावधान|report\s*(?:fraud|it)?\s*(?:at|to|on)?\s*(?:1930|cybercrime)", t):
        r.add("trust_fraud_advisory", "Reads like an anti-fraud awareness message (warns you, asks nothing).",
              trust=True)


# ------------------------------------------------------------------------------ public API
def run_rules(text: str, sender: str | None = None) -> RuleResult:
    """Run every rule. Order matters: later groups check what earlier ones fired."""
    t = clean(text)
    r = RuleResult()
    _link_signals(r, text, t)
    _money_signals(r, t)
    _pressure_signals(r, t)
    _storyline_signals(r, text, t)
    _number_and_sender_signals(r, text, t, sender)
    _trust_signals(r, t)
    return r


def rule_features(result: RuleResult) -> list[int]:
    fired = result.fired()
    return [int(rid in fired) for rid in RULE_IDS]
