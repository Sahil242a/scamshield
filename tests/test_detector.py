"""Run: python -m pytest -q

These are regression tests for the bugs found during development — each one is
a story you can tell in an interview.
"""
import pytest
from fastapi.testclient import TestClient

from scamshield.detector import ScamDetector, detect_language
from scamshield.normalize import clean, for_model
from scamshield.rules import registered_domain, run_rules


@pytest.fixture(scope="module")
def det():
    return ScamDetector()


# ---------------------------------------------------------------- normalisation
def test_leetspeak_is_undone_but_upi_ids_survive():
    assert "blocked" in clean("Your acc0unt is bl0cked")
    # Bug found in dev: unleet turned 'anjali4413@ibl' into 'anjaliaaieaibl', killing the UPI-ID rule
    assert "anjali4413@ibl" in clean("bhej do anjali4413@ibl pe")


def test_devanagari_digits_and_tokens():
    out = for_model("Call ९८७६५४३२१० now http://x.xyz Rs 500")
    assert "phonetoken" in out and "urltoken" in out and "amounttoken" in out


# ---------------------------------------------------------------- rules
def fired(text, sender=None):
    return run_rules(text, sender).fired()


def test_lookalike_vs_official_domain():
    assert "lookalike_domain" in fired("Update KYC at sbi-kyc-update.xyz")
    assert "lookalike_domain" not in fired("Login at onlinesbi.sbi")
    assert "trust_official_domain" in fired("Pay at energy.rajasthan.gov.in")
    assert registered_domain("www.retail.onlinesbi.sbi") == "onlinesbi.sbi"


def test_otp_asking_vs_otp_delivering():
    assert "credential_ask" in fired("Please share the OTP sent to your phone")
    assert "credential_ask" in fired("Jo OTP aaya hai wo batao")
    legit = fired("482913 is your OTP. Do not share it with anyone.")
    assert "credential_ask" not in legit and {"trust_otp_delivery", "trust_do_not_share"} <= legit


def test_hinglish_verb_final_negation():
    # Hindi is SOV: the negation comes AFTER the verb ("share na karein")
    f = fired("634777 aapka Blinkit login OTP hai. Ise kisi ke saath share na karein.")
    assert "credential_ask" not in f and "trust_do_not_share" in f
    assert "trust_do_not_share" in fired("483920 आपका ओटीपी है। इसे किसी के साथ साझा न करें।")


def test_pin_to_receive():
    assert "pin_to_receive" in fired("Enter your UPI PIN to receive Rs 5000")
    assert "pin_to_receive" in fired("refund aa gaya hai, request me PIN daal do paise aa jayenge")


def test_sender_signals():
    assert "trust_dlt_sender" in fired("Your bill is due", "VM-HDFCBK")
    assert "impersonation_from_personal_number" in fired("Your bank account is blocked", "9876543210")
    assert "international_sender" in fired("Hey is this Ravi?", "+447700900123")


# ---------------------------------------------------------------- end to end
@pytest.mark.parametrize("text,sender", [
    ("Dear Consumer, your electricity will be disconnected tonight at 9.30 pm. Contact officer 9123456789", "9123456789"),
    ("प्रिय ग्राहक, आपका SBI खाता आज बंद कर दिया जाएगा। तुरंत KYC अपडेट करें: http://sbi-kyc.xyz", None),
    ("Aapko Rs 2000 ka cashback mila hai. Paise lene ke liye UPI PIN daalein", None),
    ("Mom this is my new number, phone broke. Send 15000 to ravi22@ybl urgently", "8899776655"),
])
def test_scams_are_flagged(det, text, sender):
    assert det.analyze(text, sender).label == "SCAM"


@pytest.mark.parametrize("text,sender", [
    ("482913 is your OTP for payment of Rs 1,250.00 at Swiggy. Do not share with anyone. -HDFC Bank", "VM-HDFCBK"),
    ("Rs.399 debited from A/c XX4567 on 22-Oct-26 to VPA swiggy@ybl. UPI Ref 629384756123 -SBI", "AX-SBIUPI"),
    ("Bhai 250 bhej de UPI pe, kal chai ka hisaab kar lenge", "9876012345"),
    ("Aaj raat 10 baje tak aa jaunga, darwaza band mat karna", None),
])
def test_legit_messages_are_safe(det, text, sender):
    assert det.analyze(text, sender).label == "SAFE"


def test_explanations_present(det):
    v = det.analyze("Your HDFC account is blocked. Verify at bit.ly/abc123 immediately")
    ids = {r["signal"] for r in v.reasons}
    assert {"url_shortener", "urgency", "threat"} <= ids


def test_language_detection():
    assert detect_language("आपका खाता बंद") == "hi"
    assert detect_language("aapka account band ho jayega") == "hinglish"
    assert detect_language("your account is blocked") == "en"


def test_api():
    from api import app
    c = TestClient(app)
    r = c.post("/analyze", json={"text": "You won Rs 25 lakh in KBC lottery, call 9123456780", "sender": None})
    assert r.status_code == 200 and r.json()["label"] == "SCAM"
    assert c.post("/analyze", json={"text": ""}).status_code == 422
    assert c.get("/health").json()["status"] == "ok"


# First real-world false positive (reported by a user; name/number anonymised):
# app.link (Branch deep links) was treated as a phishing TLD and suppressed the
# official i.airtel.in trust signal; long itemised bills were absent from training.
AIRTEL_BILL = """Hi Rohan, Bill for your Airtel Mobile 9800000001 dated 21-OCT-2026 has been generated.
Amount to be paid: Rs 628.82
Due Date: 24-OCT-2026
This month's charges: Rs 628.82 (inclusive of tax)
Detail Charges-
Plan Rental: Rs 449.00
Other Charges: Rs 83.90
Tax: Rs 95.92

To view your bill or to make payment, please visit https://i.airtel.in/PPBillbreakup
To understand your bill, click https://gh0hi-web.app.link/e/ACYT-UPB"""


def test_deeplink_services_are_neutral():
    f = fired(AIRTEL_BILL)
    assert "risky_tld" not in f and "trust_official_domain" in f
    assert "risky_tld" in fired("Claim reward at rewardz-now.link/claim")  # other .link domains still risky


def test_itemised_airtel_bill_is_safe(det):
    assert det.analyze(AIRTEL_BILL).label == "SAFE"


# Second user report: a genuine Jio Hindi promo (number anonymised). No scam rule fired; the text
# model had learned "Hindi + link + 'आज ही रिचार्ज करें' => scam" because no legit Hindi message in
# training had a link or a promotional offer.
JIO_PROMO_HI = """प्रिय ग्राहक, 9800000002 चालू रखना चाहते हैं?

आपके Jio नंबर पर ‘1 रिचार्ज पर 1 फ्री' ऑफ़र उपलब्ध है। आज ही ₹349 का रिचार्ज करें और ₹349 का अतिरिक्त रिचार्ज फ्री पाएं।

आज ही MyJio, किसी भी पेमेंट ऐप या नज़दीकी स्टोर से रिचार्ज करें, या यहां क्लिक करें www.jio.com/r/U7axGNfgR। जल्दी करें! ऑफ़र सीमित समय के लिए है। नियम और शर्तें लागू।"""


def test_jio_hindi_promo_is_not_scam(det):
    assert det.analyze(JIO_PROMO_HI).label != "SCAM"


def test_official_trust_requires_all_links_official():
    assert "trust_official_domain" in fired("Recharge at www.jio.com/r/abc")
    assert "trust_official_domain" not in fired("Recharge at www.jio.com/r/abc or claim at freegifts4u.com")


# ---------------------------------------------------------------- coverage round 3
def test_handover_otps_are_not_requests():
    for t in ["Share OTP 4821 with the delivery agent to receive your Amazon order.",
              "Ride OTP 6624. Share it with your driver to start the trip.",
              "Technician Meena will visit today. Share service OTP 9915 after the job is done."]:
        f = fired(t)
        assert "credential_ask" not in f and "trust_otp_delivery" in f, t


def test_code_takeover_in_three_languages():
    for t in ["I accidentally sent my WhatsApp code to your number, please send it back to me",
              "Yaar mera code galti se tere number pe aa gaya, jo 6 digit aaya hai wo bhej de",
              "गलती से मेरा कोड आपके नंबर पर आ गया है, कृपया वह कोड मुझे भेज दें।"]:
        assert "credential_ask" in fired(t), t


def test_collect_request_approval_is_pin_to_receive():
    assert "pin_to_receive" in fired("I'll pay the advance through a Google Pay request, just approve it to receive")
    assert "pin_to_receive" not in fired("Approve in your UPI app only if you know the person")


def test_fake_alert_callback_vs_real_alert():
    assert "fake_alert_callback" in fired("Rs 49,999 debited from your account. If not done by you, call 9019283746")
    assert "fake_alert_callback" not in fired("Rs 399 debited from A/c XX4567. Not you? Call 18001234")


def test_wording_only_scam_is_capped_at_suspicious():
    from scamshield.detector import decide
    assert decide(0.99, set(), 0.6, 0.4) == "SUSPICIOUS"
    assert decide(0.99, {"url_shortener"}, 0.6, 0.4) == "SCAM"
    assert decide(0.05, {"pin_to_receive"}, 0.6, 0.4) == "SUSPICIOUS"   # critical signal is never SAFE


def test_feedback_endpoint(tmp_path, monkeypatch):
    import api
    monkeypatch.setattr(api, "FEEDBACK_FILE", tmp_path / "feedback.csv")
    monkeypatch.setattr(api, "PUBLIC", False)
    c = TestClient(api.app)
    r = c.post("/feedback", json={"text": "line one\nline two", "correct_label": "legit", "predicted": "SCAM"})
    assert r.status_code == 200 and r.json()["total_corrections"] == 1
    assert c.post("/feedback", json={"text": "x", "correct_label": "maybe"}).status_code == 422


# ---------------------------------------------------------------- deployment
def test_public_mode_disables_feedback_and_rate_limits(monkeypatch):
    import api
    monkeypatch.setattr(api, "PUBLIC", True)
    monkeypatch.setattr(api, "RATE_LIMIT", 3)
    api._hits.clear()
    c = TestClient(api.app)
    assert c.get("/health").json()["feedback_enabled"] is False
    assert c.post("/feedback", json={"text": "x", "correct_label": "legit"}).status_code == 403
    codes = [c.post("/analyze", json={"text": "hello"}).status_code for _ in range(4)]
    assert codes == [200, 200, 200, 429]
    api._hits.clear()
