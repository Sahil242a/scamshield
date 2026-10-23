"""Build the training dataset.

Sources
  1. UCI SMS Spam Collection (5,574 English SMS, public). Its "spam" class is
     mostly prize / premium-rate fraud, so we map spam -> scam, ham -> legit.
  2. Indian scam + legit messages in English, Hindi and Hinglish, generated from
     templates modelled on scam patterns publicly reported by RBI, NPCI,
     cybercrime.gov.in and news coverage (KYC expiry, electricity cut-off,
     KBC lottery, task scams, UPI-PIN-to-receive, fake customs / digital arrest,
     loan apps, reward points), plus *hard negatives*: real-style bank OTPs,
     debit/credit alerts, genuine bill reminders, delivery updates, chats.

Every templated row keeps a ``group`` id (= its template). The evaluation split
is by group, so the test set contains *templates the model never saw* — this
stops the model scoring 100% by memorising template wording.

Run:  python data/build_dataset.py   ->  data/processed/messages.csv
"""
from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))         # templates_extra
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scamshield package
from templates_extra import EXTRA_LEGIT, EXTRA_SCAM, EXTRA_VALUES  # noqa: E402

from scamshield.detector import detect_language  # noqa: E402

ROOT = Path(__file__).resolve().parent
rng = random.Random(42)

# ---------------------------------------------------------------------------- slot values
BANKS = ["SBI", "HDFC Bank", "ICICI Bank", "Axis Bank", "PNB", "Kotak", "Bank of Baroda", "Canara Bank"]
BANK_SHORT = {"SBI": "SBI", "HDFC Bank": "HDFC", "ICICI Bank": "ICICI", "Axis Bank": "AXIS", "PNB": "PNB",
              "Kotak": "KOTAK", "Bank of Baroda": "BOB", "Canara Bank": "CANARA"}
BOARDS = [("JVVNL", "energy.rajasthan.gov.in"), ("BSES", "bsesdelhi.com"), ("MSEDCL", "mahadiscom.in"),
          ("UPPCL", "uppcl.org"), ("TNEB", "tnebltd.gov.in")]
NAMES = ["Rahul", "Priya", "Amit", "Neha", "Rohit", "Sneha", "Vikram", "Pooja", "Arjun", "Kavya", "Ravi Kumar",
         "Anjali", "Suresh", "Meena", "Karan"]
MERCHANTS = ["AMAZON", "FLIPKART", "SWIGGY", "ZOMATO", "BIGBASKET", "MYNTRA", "IRCTC", "RELIANCE DIGITAL",
             "DMART", "UBER", "OLA"]
APPS = ["Swiggy", "Zomato", "Paytm", "PhonePe", "Instagram", "WhatsApp", "Amazon", "Flipkart", "Ola", "Uber",
        "CRED", "Zepto", "Blinkit"]
GIFTS = ["iPhone 15", "Samsung TV", "Royal Enfield bike", "gold coin", "Rs 25,000 gift voucher", "Maruti car"]
BRANDS = ["Amazon", "Flipkart", "Jio", "Airtel", "Tata", "Reliance", "Myntra"]
ITEMS = ["boAt earphones", "Redmi Note 13", "Nike shoes", "a book", "a kurta", "a pressure cooker"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
COMPANIES = ["TCS", "Infosys", "Wipro", "Zoho", "Razorpay", "Flipkart", "Swiggy", "Accenture"]
SCAM_WORDS = ["kyc", "update", "verify", "secure", "refund", "reward", "bijli", "claim", "help", "care",
              "bonus", "points", "login", "support", "pay", "bill"]
SHORT = ["bit.ly", "cutt.ly", "rb.gy", "tinyurl.com", "is.gd", "t.ly", "shorturl.at"]
BAD_TLDS = ["xyz", "top", "icu", "click", "live", "online", "site", "info", "buzz", "shop", "vip", "in", "com"]


def amt(lo=100, hi=9999):
    return f"{rng.randint(lo, hi):,}"


def big():
    return rng.choice(["25,00,000", "50,00,000", "10,00,000", "5,00,000", "1 crore", "35,00,000", "2,50,000"])


def phone():
    return f"{rng.choice('6789')}{rng.randint(100000000, 999999999)}"


def otp():
    return str(rng.randint(100000, 999999))


def last4():
    return str(rng.randint(1000, 9999))


def date():
    return f"{rng.randint(1, 28):02d}-{rng.choice(MONTHS)}-26"


def ref():
    return str(rng.randint(10**11, 10**12 - 1))


def upi_personal():
    return f"{rng.choice(NAMES).split()[0].lower()}{rng.randint(10, 9999)}@{rng.choice(['ybl', 'okaxis', 'paytm', 'oksbi', 'ibl'])}"


def scam_url(bank=None):
    r = rng.random()
    if r < 0.35:
        return f"https://{rng.choice(SHORT)}/{''.join(rng.choices('abcdefghjkmnpqrstuvwxyz23456789', k=6))}"
    brand = (BANK_SHORT.get(bank, "") if bank else rng.choice(["sbi", "hdfc", "kyc", "bijli", "paytm", ""])).lower()
    word = rng.choice(SCAM_WORDS)
    host = "-".join(x for x in [brand, word, rng.choice(["", "online", "in", "now", "24"])] if x)
    if r < 0.45:
        return f"http://{rng.randint(11, 223)}.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}/{word}"
    return f"{rng.choice(['http://', 'https://', 'www.', ''])}{host}.{rng.choice(BAD_TLDS)}/{rng.choice(['', 'login', word, 'verify'])}"


def apk_url():
    return rng.choice([f"{scam_url()}/app.apk", f"https://{rng.choice(SHORT)}/{''.join(rng.choices('abcdefgh234', k=5))}"
                       f" (download APK)", f"{rng.choice(['quickloan', 'rewardpoints', 'yono-kyc'])}.{rng.choice(BAD_TLDS)}/install.apk"])


def fill(tpl: str) -> str:
    bank = rng.choice(BANKS)
    board, site = rng.choice(BOARDS)
    values = {
        "bank": bank, "bank_short": BANK_SHORT[bank], "url": scam_url(bank), "phone": phone(), "wa": phone(),
        "amt": amt(), "small": amt(300, 900), "mid": amt(1500, 8000), "big": big(), "fee": amt(99, 4999),
        "upi": upi_personal(), "name": rng.choice(NAMES), "gift": rng.choice(GIFTS), "brand": rng.choice(BRANDS),
        "apk": apk_url(), "otp": otp(), "last4": last4(), "date": date(), "ref": ref(),
        "merchant": rng.choice(MERCHANTS), "app": rng.choice(APPS), "bal": amt(1000, 250000),
        "board": board, "site": site, "kno": str(rng.randint(10**9, 10**10)), "month": rng.choice(MONTHS),
        "order": f"{rng.randint(400, 409)}-{rng.randint(10**6, 10**7)}-{rng.randint(10**6, 10**7)}",
        "item": rng.choice(ITEMS), "company": rng.choice(COMPANIES), "time": f"{rng.randint(1, 11)}:{rng.choice(['00', '15', '30', '45'])}",
        "vpa": f"{rng.choice(MERCHANTS).lower().replace(' ', '')}@{rng.choice(['ybl', 'axisbank', 'icici', 'hdfcbank'])}",
        "pct": str(rng.choice([80, 90, 100])), "mins": str(rng.choice([5, 10, 15, 20, 30])),
    }
    values.update({k: rng.choice(v) for k, v in EXTRA_VALUES.items()})
    values.setdefault("gift", rng.choice(GIFTS))
    return tpl.format(**values)


# ---------------------------------------------------------------------------- evasion noise for scams
MISSPELL = {"account": ["acount", "accnt", "a/c", "acc0unt"], "immediately": ["immediatly", "imediately", "immdtly"],
            "blocked": ["bl0cked", "blokd", "blocked!!"], "update": ["updte", "upd8", "UPDATE"],
            "customer": ["costumer", "custmer", "Customer"], "electricity": ["electricty", "elecricity"],
            "verify": ["verfy", "verify!!"], "turant": ["turnt", "turant!!", "TURANT"], "karein": ["kare", "kariye", "karen"]}
EMOJI = ["⚠️", "🚨", "‼️", "🎉", "💰", "🔴", "📢", "🙏"]


def perturb(text: str, strength: float) -> str:
    words = text.split(" ")
    for i, w in enumerate(words):
        key = w.lower().strip(".,!:")
        if key in MISSPELL and rng.random() < strength:
            words[i] = rng.choice(MISSPELL[key])
    text = " ".join(words)
    if rng.random() < strength * 0.6:
        text = rng.choice(EMOJI) + " " + text
    if rng.random() < strength * 0.4:
        text = text.upper()
    if rng.random() < strength * 0.3:
        text = text.replace(" ", "  ", 2)
    return text


def chat_variant(text: str) -> str:
    """Light variation for personal chats (case, punctuation, emoji)."""
    r = rng.random()
    if r < 0.25:
        text = text.lower()
    elif r < 0.4:
        text = text.rstrip("?.!") + rng.choice(["", " 😂", " 🙏", " 👍", "!!", " na", " yaar"])
    if rng.random() < 0.2:
        text = text.replace("?", "??")
    return text


# ---------------------------------------------------------------------------- templates: (category, lang, text)
SCAM = [
    # KYC / account block
    ("kyc", "en", "Dear {bank} customer, your account will be blocked today due to incomplete KYC. Update your PAN immediately: {url}"),
    ("kyc", "en", "Your {bank} YONO account has been suspended. Click {url} to reactivate within 24 hrs."),
    ("kyc", "en", "{bank_short} ALERT: Your KYC has expired. Kindly update now to avoid account closure {url}"),
    ("kyc", "en", "Dear user, your PAN card is not linked with your {bank} account. Account will freeze tonight. Link now {url}"),
    ("kyc", "en", "Dear Customer, your {bank} netbanking will be deactivated today. Please update your KYC documents here {url}"),
    ("kyc", "en", "Your {bank} account is temporarily blocked due to suspicious activity. Verify your identity at {url} to unblock"),
    ("kyc", "hinglish", "Priy grahak, aapka {bank} account aaj raat band ho jayega. KYC turant update karein: {url}"),
    ("kyc", "hinglish", "Aapka KYC expire ho gaya hai. Account block hone se bachane ke liye is link pe click kare {url}"),
    ("kyc", "hinglish", "{bank} se: aapka net banking 24 ghante me band ho jayega, PAN update karo {url}"),
    ("kyc", "hinglish", "Dear customer aapka {bank} khata suspend kar diya gaya hai. Dobara chalu karne ke liye {url} par KYC karein"),
    ("kyc", "hi", "प्रिय ग्राहक, आपका {bank} खाता आज बंद कर दिया जाएगा। तुरंत KYC अपडेट करें: {url}"),
    ("kyc", "hi", "आपका केवाईसी समाप्त हो गया है। अभी अपडेट करें वरना खाता ब्लॉक हो जाएगा {url}"),
    ("kyc", "hi", "{bank} चेतावनी: आपका पैन कार्ड लिंक नहीं है, आज रात खाता बंद होगा। लिंक करें {url}"),
    # SIM KYC
    ("sim", "en", "Your Jio SIM will be blocked in 24 hours due to incomplete e-KYC. Call {phone} to verify"),
    ("sim", "en", "Dear customer your Airtel SIM card KYC is pending. Your number will be deactivated today. Call executive {phone}"),
    ("sim", "hinglish", "Aapka SIM 2 ghante me band ho jayega kyunki aapka KYC adhura hai. Turant call karein {phone}"),
    ("sim", "hi", "आपका सिम कार्ड आज बंद कर दिया जाएगा। केवाईसी के लिए तुरंत कॉल करें {phone}"),
    # Electricity
    ("electricity", "en", "Dear Consumer, your electricity power will be disconnected tonight at 9.30 pm from electricity office because your previous month bill was not updated. Please immediately contact our electricity officer {phone}. Thank you"),
    ("electricity", "en", "Dear customer your electricity connection will be cut today at {time} PM. Last month bill not update. Call officer {phone}"),
    ("electricity", "en", "{board}: Your power supply will be disconnected tonight. Bill payment not reflected. Pay now at {url} or call {phone}"),
    ("electricity", "hinglish", "Priya upbhokta aaj raat {time} baje aapki bijli kaat di jayegi kyunki pichle mahine ka bill update nahi hua. Turant bijli adhikari se sampark karein {phone}"),
    ("electricity", "hinglish", "Aapka bijli connection aaj raat kaat diya jayega. Bill update karne ke liye call karein {phone}"),
    ("electricity", "hi", "प्रिय उपभोक्ता, आज रात {time} बजे आपकी बिजली काट दी जाएगी क्योंकि आपका पिछले महीने का बिल अपडेट नहीं हुआ है। तुरंत बिजली अधिकारी से संपर्क करें {phone}"),
    ("electricity", "hi", "बिजली विभाग: आपका कनेक्शन आज रात काट दिया जाएगा। बिल अपडेट के लिए कॉल करें {phone}"),
    # Prize / lottery
    ("prize", "en", "Congratulations! Your mobile number has won Rs {big} in KBC lucky draw. To claim call {phone} on WhatsApp"),
    ("prize", "en", "You have won a {gift} in the {brand} anniversary lucky draw. Claim now: {url}"),
    ("prize", "en", "WINNER! Your number is selected for {brand} cash prize of Rs {big}. Pay processing fee Rs {fee} to receive. Contact {phone}"),
    ("prize", "hinglish", "Congrats! Aapne {big} rupaye ka lucky draw jeeta hai. Prize lene ke liye processing fee Rs {fee} bhejein UPI {upi}"),
    ("prize", "hinglish", "Badhai ho! KBC lottery me aapka number select hua hai, inaam {big}. Whatsapp karein {wa}"),
    ("prize", "hi", "बधाई हो! आपने {big} रुपये की लॉटरी जीती है। इनाम पाने के लिए {phone} पर व्हाट्सएप करें"),
    ("prize", "hi", "केबीसी लकी ड्रा में आपका नंबर चुना गया है। इनाम पाने के लिए रजिस्ट्रेशन फीस {fee} रुपये भेजें"),
    # Task / job
    ("job", "en", "Hi, we are hiring part time from home. Earn Rs {small}-{mid} daily by liking YouTube videos. Contact on WhatsApp {wa}"),
    ("job", "en", "Part time job: daily income {mid}/day, simple task, no experience needed. Apply {url}"),
    ("job", "en", "Hello, I am HR from {company}. Earn {mid} per day by rating hotels on Google. Interested? Reply YES or WhatsApp {wa}"),
    ("job", "hinglish", "Ghar baithe kamaye {mid} rupaye roz. Sirf YouTube videos like karne hai. Whatsapp karein {wa}"),
    ("job", "hinglish", "Part time kaam chahiye? Roz {mid} kamao, mobile se simple task. Abhi join karo {url}"),
    ("job", "hi", "घर बैठे रोज {mid} रुपये कमाएं, सिर्फ वीडियो लाइक करने हैं। व्हाट्सएप करें {wa}"),
    # UPI PIN to receive / refunds / mistake
    ("upi", "en", "You have received a payment request of Rs {amt} from {name}. Enter your UPI PIN to receive the money."),
    ("upi", "en", "Your refund of Rs {amt} is pending. To receive it, scan the QR and enter UPI PIN {url}"),
    ("upi", "en", "Cashback of Rs {amt} credited to your wallet! Enter PIN to claim it before it expires today {url}"),
    ("upi", "hinglish", "Bhai galti se aapke account me Rs {amt} transfer ho gaye, please wapas bhej do {upi} pe"),
    ("upi", "hinglish", "Aapko Rs {amt} ka cashback mila hai. Paise lene ke liye UPI PIN daalein {url}"),
    ("upi", "hinglish", "Sir mistake se aapko {amt} sent ho gaya hai, urgent hai please return kar do {phone}"),
    ("upi", "hi", "आपको {amt} रुपये का रिफंड मिला है। पैसे प्राप्त करने के लिए अपना यूपीआई पिन डालें {url}"),
    # Courier / customs / digital arrest
    ("courier", "en", "FedEx: Your parcel containing illegal items has been seized by customs. Call {phone} to talk to officer, else legal action will be taken."),
    ("courier", "en", "Your courier is on hold due to incomplete address. Update within 12 hours: {url}"),
    ("courier", "en", "India Post: your package delivery failed. Re-schedule and pay Rs 25 delivery fee {url}"),
    ("courier", "hinglish", "Aapke naam pe parcel me drugs mile hai, CBI case register hoga. Turant call karein {phone}"),
    ("courier", "hinglish", "Mumbai police cyber cell: aapka Aadhaar money laundering case me use hua hai, digital arrest se bachne ke liye call karein {phone}"),
    ("courier", "hi", "आपके नाम के पार्सल में ड्रग्स मिले हैं। गिरफ्तारी से बचने के लिए तुरंत कॉल करें {phone}"),
    # Loan
    ("loan", "en", "Pre-approved instant loan of Rs {big} without documents. Get money in 5 minutes, download app {apk}"),
    ("loan", "en", "Instant loan approved! Rs {big} at 0% interest. Pay processing fee Rs {fee} to {upi}"),
    ("loan", "hinglish", "Bina CIBIL ke {big} ka loan 10 minute me. App download karein {apk}"),
    ("loan", "hi", "बिना दस्तावेज़ {big} रुपये का लोन तुरंत पाएं। ऐप डाउनलोड करें {apk}"),
    # Reward points
    ("reward", "en", "Dear customer your {bank} credit card reward points worth Rs {amt} expire today. Redeem now {url}"),
    ("reward", "en", "Your {bank} credit card points worth Rs {amt} will lapse tonight. Redeem by installing {apk}"),
    ("reward", "hinglish", "Aapke {bank} reward points Rs {amt} aaj expire ho rahe hai, abhi redeem karein {url}"),
    # Remote access / OTP ask
    ("otp_ask", "en", "For your refund complaint, please install AnyDesk app and share the 9 digit code with our executive {phone}"),
    ("otp_ask", "en", "Dear customer, to stop an unauthorized transaction of Rs {amt}, reply with the OTP received on your mobile."),
    ("otp_ask", "en", "This is {bank} customer care. Your card upgrade is pending, please share the OTP sent to you to complete it."),
    ("otp_ask", "hinglish", "Sir main {bank} head office se bol raha hu, aapke card ka upgradation hai, jo OTP aaya hai wo batao"),
    ("otp_ask", "hinglish", "Bank support: KYC verify karne ke liye QuickSupport app download karein aur code batayein {phone}"),
    ("otp_ask", "hi", "आपका कार्ड ब्लॉक होने वाला है। रोकने के लिए आपके फ़ोन पर आया ओटीपी बताएं"),
    # Tax / govt
    ("govt", "en", "Income tax refund of Rs {amt} approved. Verify your account details to receive it at {url}"),
    ("govt", "en", "PM Kisan: your installment of Rs 6000 is on hold. Update Aadhaar details here {url}"),
    ("govt", "hinglish", "Aapka income tax refund Rs {amt} pending hai, bank details verify karein {url}"),
    # Investment / family impersonation / wrong-number openers
    ("investment", "en", "Join our premium trading group for guaranteed returns of {pct}% every week. Limited seats, join now {url}"),
    ("investment", "en", "Earn double your money in 15 days with crypto investment. Our expert will guide you. WhatsApp {wa}"),
    ("investment", "hinglish", "Stock market se roz {mid} kamao, VIP tips group join karo, profit guaranteed hai {url}"),
    ("investment", "hi", "शेयर बाजार में निवेश करें और एक हफ्ते में पैसा डबल करें। ग्रुप जॉइन करें {url}"),
    ("family", "en", "Hi Mom, this is my new number, my old phone broke. Can you send Rs {amt} to {upi}? I'll explain later"),
    ("family", "hinglish", "Papa ye mera naya number hai, purana phone kho gaya. Jaldi {amt} bhej do is UPI pe {upi}, urgent hai"),
    ("family", "hi", "पापा यह मेरा नया नंबर है, पुराना फोन खो गया। जल्दी {amt} रुपये भेज दो"),
    ("wrong_number", "en", "Hello, is this {name}? My friend gave me this number for the property deal"),
    ("wrong_number", "en", "Hi dear, sorry I think I saved the wrong number. Are you not {name} from the gym?"),
]


LEGIT = [
    # OTPs (hard negatives: contain OTP + amount + urgency-ish words)
    ("otp", "en", "{otp} is your OTP for transaction of INR {amt} at {merchant} on your {bank} card ending {last4}. Valid for 10 mins. Do not share it with anyone."),
    ("otp", "en", "OTP for login to {bank_short} NetBanking is {otp}. Never share OTP with anyone. {bank} never asks for OTP."),
    ("otp", "en", "{otp} is the OTP to verify your mobile number on {app}. Do not share this code with anyone."),
    ("otp", "en", "Use {otp} as your verification code for {app}. It expires in {mins} minutes. Don't share it."),
    ("otp", "en", "Dear Customer, {otp} is your OTP to add beneficiary {name} in {bank_short} NetBanking. If not you, call 1800 1234 immediately. Do not share OTP."),
    ("otp", "hinglish", "{otp} aapka {app} login OTP hai. Ise kisi ke saath share na karein."),
    ("otp", "hi", "{otp} आपका {app} ओटीपी है। इसे किसी के साथ साझा न करें।"),
    # Transaction alerts
    ("txn", "en", "Rs.{amt} debited from A/c XX{last4} on {date} to VPA {vpa}. UPI Ref {ref}. Not you? Call 18001234 to block -{bank_short}"),
    ("txn", "en", "Dear Customer, INR {amt} credited to your A/c XX{last4} on {date} by UPI Ref {ref}. Avl Bal INR {bal} -{bank_short}"),
    ("txn", "en", "Sent Rs.{amt} from {bank_short} A/C *{last4} to {name} on {date}. Not You? Call 18002586161/SMS BLOCK UPI to 7308080808"),
    ("txn", "en", "Your {bank} Credit Card XX{last4} has been used for INR {amt} at {merchant} on {date}. Avl limit INR {bal}."),
    ("txn", "en", "ATM withdrawal of Rs {amt} from A/c XX{last4} on {date}. Avl Bal Rs {bal}. -{bank_short}"),
    ("txn", "en", "Your EMI of Rs {amt} for loan a/c XX{last4} is due on {date}. Please keep sufficient balance. -{bank_short}"),
    ("txn", "hi", "आपके खाता XX{last4} से {date} को {amt} रुपये डेबिट हुए। UPI Ref {ref}. -{bank_short}"),
    # Genuine bills / KYC reminders
    ("bill", "en", "Dear Consumer, your electricity bill for K.No {kno} for {month} is Rs {amt}, due on {date}. Pay at {site} or any UPI app. -{board}"),
    ("bill", "en", "Payment of Rs {amt} received towards electricity bill K.No {kno}. Thank you. -{board}"),
    ("bill", "en", "Your Jio number's plan expires on {date}. Recharge via MyJio app or jio.com to continue services."),
    ("bill", "en", "Dear customer, you have used {pct}% of your daily data quota. Speed will reduce after 100%. -Airtel"),
    ("bill", "en", "Your Airtel postpaid bill of Rs {amt} is generated. Due date {date}. Pay on airtel.in or Airtel Thanks app."),
    ("bill", "en", "Dear Customer, as per RBI guidelines please update your KYC by visiting your nearest {bank} branch. {bank} never asks for details over call or SMS links."),
    ("bill", "hinglish", "Aapka bijli bill Rs {amt} {date} tak jama karein. Online bhugtan {site} par karein. -{board}"),
    ("bill", "hi", "प्रिय उपभोक्ता, {month} माह का आपका बिजली बिल {amt} रुपये है। अंतिम तिथि {date}। -{board}"),
    # Itemised telecom / utility bills (long, many amounts, "click to pay" — look scary, are legit)
    ("bill", "en", "Hi {name}, Bill for your Airtel Mobile {phone} dated {date} has been generated.\nAmount to be paid: Rs {amt}\nDue Date: {date}\nPlan Rental: Rs {small}\nOther Charges: Rs {fee}\nTax: Rs {fee}\nTo view your bill or make payment, visit https://i.airtel.in/{order}\nTo understand your bill, click https://airtel.app.link/{order}"),
    ("bill", "en", "Dear {name}, your Jio Postpaid bill for {month} is Rs {amt}. Plan charges Rs {small}, GST Rs {fee}. Due date {date}. Pay via MyJio app or https://www.jio.com/selfcare/paybill"),
    ("bill", "en", "Dear Customer, your Vi bill of Rs {amt} for number {phone} is due on {date}. View detailed bill & pay on Vi App: https://myvi.app.link/{order} or myvi.in"),
    ("bill", "en", "Your BSNL landline bill for {month}: Rental Rs {small}, Usage Rs {fee}, Total Rs {amt}. Last date {date}. Pay at portal.bsnl.in"),
    ("bill", "en", "Hi {name}, your Airtel Xstream Fiber bill of Rs {amt} dated {date} is generated. Due {date}. Pay now on Airtel Thanks app https://airtel.app.link/{order} to avoid late fee."),
    ("bill", "en", "{board}: Bill for K.No {kno} | Units: {pct} | Energy charges Rs {small} | Fixed charges Rs {fee} | Total Rs {amt} | Due {date}. Pay online at {site}"),
    ("bill", "hinglish", "Namaste {name}, aapka Airtel bill Rs {amt} generate ho gaya hai. Due date {date}. Bill dekhne aur pay karne ke liye Airtel Thanks app kholein."),
    # Legit promotional offers (marketing is not fraud): urgency + "free" + link, but official domains
    ("promo", "hi", "प्रिय ग्राहक, आपके Jio नंबर पर विशेष ऑफ़र! आज ही ₹{small} का रिचार्ज करें और {pct}GB अतिरिक्त डेटा पाएं। MyJio या www.jio.com/r/{order} से रिचार्ज करें। नियम और शर्तें लागू।"),
    ("promo", "hi", "आपका Airtel प्लान {date} को समाप्त हो रहा है। Airtel Thanks ऐप या airtel.in से अभी रिचार्ज करें और मुफ्त Disney+ Hotstar पाएं। T&C लागू।"),
    ("promo", "hi", "{bank} क्रेडिट कार्ड पर त्योहार ऑफ़र: {merchant} पर {pct}% तक कैशबैक। ऑफ़र सीमित समय के लिए। नियम व शर्तें लागू।"),
    ("promo", "hi", "Vi ग्राहक, ₹{small} के रिचार्ज पर पाएं अनलिमिटेड कॉल और डेटा। Vi ऐप या myvi.in पर जल्दी करें!"),
    ("promo", "hinglish", "Jio offer: ₹{small} ka recharge karein aur ₹{small} ka extra recharge free payein. Aaj hi MyJio app ya jio.com se recharge karein. T&C apply."),
    ("promo", "hinglish", "Flipkart Big Billion Days shuru! {item} par {pct}% tak off. Jaldi karein, offer seemit samay ke liye. flipkart.com"),
    ("promo", "en", "Recharge today with Rs {small} and get {pct}GB extra data FREE! Offer valid till {date}. Recharge on MyJio app or www.jio.com/r/{order}. T&C apply."),
    ("promo", "en", "Great Indian Festival is LIVE! Up to {pct}% off on {item}. Shop now at amazon.in/deals. Limited time offer."),
    ("promo", "en", "Dear Customer, get {pct}% instant discount on {merchant} with your {bank} Credit Card this weekend. T&C apply."),
    ("promo", "en", "Your Airtel number is eligible for a special offer: Rs {small} pack with unlimited 5G. Recharge on Airtel Thanks app or airtel.in before {date}."),
    # Shopping / delivery
    ("delivery", "en", "Your Amazon order #{order} has been shipped and will be delivered by {date}. Track at amazon.in/orders"),
    ("delivery", "en", "Your Swiggy order is out for delivery. {name} will reach you in {mins} mins."),
    ("delivery", "en", "Your Flipkart order with {item} has been delivered. Rate your experience in the app."),
    ("delivery", "en", "Refund of Rs {amt} for order #{order} has been initiated to your original payment method. It will reflect in 5-7 days. -Amazon"),
    ("delivery", "en", "Congratulations! You've earned a scratch card on Google Pay for your payment to {name}. Open the app to reveal."),
    ("delivery", "en", "Your IRCTC booking PNR {kno} is confirmed. Train departs {date} at {time}. Happy journey!"),
    ("delivery", "hinglish", "Aapka Zomato order 10 min me pahunch jayega. Delivery partner {name}."),
    # Work / college
    ("work", "en", "Your application for Software Engineer Intern at {company} has been received. We'll update you on the careers portal."),
    ("work", "en", "Reminder: project review meeting tomorrow at {time} am in Lab 3. Please bring your report."),
    ("work", "en", "Meeting moved to {time} pm, please join on the same Google Meet link."),
    ("work", "en", "Your fee payment of Rs {amt} for semester 5 has been received by Manipal University Jaipur."),
    ("work", "hinglish", "Sir kal ki class {time} baje hogi, sab log assignment le kar aana."),
    # Fraud advisories from real banks (hard negatives: mention KYC, OTP, fraud, links)
    ("advisory", "en", "Stay safe! {bank} never asks for your OTP, PIN or CVV over call, SMS or email. Report fraud at 1930."),
    ("advisory", "en", "Beware of messages asking you to update KYC through a link. {bank} will never send such links. Report at cybercrime.gov.in"),
    ("advisory", "en", "Fraud alert: do not install screen-sharing apps on the request of unknown callers. -{bank_short}"),
    ("advisory", "hinglish", "Savdhan rahein! {bank} kabhi bhi OTP ya PIN nahi maangta. Kisi ke saath share na karein."),
    ("advisory", "hi", "सावधान! {bank} कभी भी आपसे ओटीपी या पिन नहीं मांगता। धोखाधड़ी की शिकायत 1930 पर करें।"),
    # More everyday Hinglish / Hindi service messages
    ("service", "hinglish", "Aapka {app} order confirm ho gaya hai. Delivery {date} tak ho jayegi."),
    ("service", "hinglish", "Aapke {bank_short} account me Rs {amt} jama ho gaye hai. Shesh rashi Rs {bal}."),
    ("service", "hinglish", "Namaste, aapki doctor appointment kal {time} baje confirm hai. Kripya 10 min pehle aayein."),
    ("service", "hi", "आपका {app} ऑर्डर कन्फर्म हो गया है। डिलीवरी {date} तक होगी।"),
    ("service", "hi", "आपकी गैस सिलेंडर बुकिंग सफल रही। डिलीवरी 2 दिन में होगी। -इंडेन"),
    ("service", "en", "Your transaction of Rs {amt} failed. Any amount debited will be auto-reversed within 5 working days. -{app}"),
    ("service", "en", "Urgent: server maintenance tonight from 11 pm to 1 am. Please save your work. -IT Team"),
]

CHATS = [
    ("chat", "hinglish", t) for t in [
        "Bhai kal college aa raha hai kya? Assignment submit karna hai",
        "Mummy ne bola sabzi le aana ghar aate time",
        "Kitne baje milna hai? Main 6 baje free hu",
        "Ok bro, notes WhatsApp pe bhej de",
        "Happy birthday yaar! Party kab de raha hai",
        "Train late hai, 30 min me pahunch jaunga",
        "Aaj ka match dekha? Kya khela Kohli ne",
        "Paise bhej diye maine, check kar le",
        "Yaar mera charger tere paas reh gaya, kal leke aana",
        "Kal se exams hai, padhai shuru kar di?",
        "Bhai 500 rupaye UPI kar de, kal wapas kar dunga",
        "Dinner pe kya banaya hai aaj?",
        "Mai thodi der me call karta hu, abhi meeting me hu",
        "Pizza order kar de, main paise de dunga",
        "Didi ki shaadi ki date fix ho gayi, 12 December",
        "Gaadi service pe di hai, shaam tak milegi",
        "Bhai project ka code github pe push kar diya",
        "Aaj gym nahi aa paunga, kal pakka",
        "Aaj raat late aaunga, khana rakh dena",
        "Bhai kal ka plan cancel, mummy ki tabiyat theek nahi",
        "Tu abhi kaha hai? Main station pe khada hu",
        "Mera phone silent pe tha, abhi dekha message",
        "Bijli chali gayi hai yaha, inverter bhi band ho gaya",
        "Recharge khatam ho gaya, wifi pe hu abhi",
        "Bhai 300 rupaye Paytm kar de, canteen me card nahi chal raha",
        "Paise mil gaye, thanks bhai",
        "Kal subah jaldi uthna hai, 7 baje ki train hai",
        "Shaadi ka card aa gaya, tu aa raha hai na?",
        "Dukaan band ho gayi thi, kal le aaunga",
        "Exam ka result aa gaya kya? Mera site nahi khul raha",
        "Bhaiya rent is mahine thoda late dunga, 5 tarikh tak",
        "Ghar pahunch ke call karna",
        "Aaj office me bahut kaam hai, der ho jayegi",
        "Movie ke tickets book kar diye, 9 baje ka show hai",
        "Yaar laptop hang ho raha hai, kya karu",
        "Maine bill bhar diya, receipt bhej di hai",
        "Placement ka form bhar diya? Kal last date hai",
        "Teri photo bahut acchi aayi hai yaar",
        "Kal match hai, jaldi aa jana ground pe",
        "Nani ke ghar ja rahe hai is weekend",
        "Doodh khatam ho gaya, aate waqt le aana",
        "Bhai interview kaisa gaya?",
        "Sab theek hai? Kaafi din se baat nahi hui",
        "Uber le le, auto nahi mil raha yaha",
        "Mera OTP aaya hai tere phone pe galti se, ignore kar",
        "Kal se hostel ki mess band rahegi, bahar khana padega",
        "Abhi meeting me hu, baad me baat karta hu",
        "Party ke liye sab 500 500 de do, maine advance de diya",
    ]
] + [
    ("chat", "hi", t) for t in [
        "कल शाम को घर आ जाना, खाना साथ में खाएंगे",
        "मैं पहुंच गया हूँ, चिंता मत करना",
        "दवाई टाइम पर ले लेना",
        "आज बारिश बहुत हो रही है, छाता ले जाना",
        "बेटा फोन उठाओ, जरूरी बात करनी है",
        "मंदिर में आज भंडारा है, सब आ जाना",
        "आज रात को देर से आऊंगा, खाना रख देना",
        "बिजली चली गई है, फोन चार्ज नहीं है",
        "कल सुबह मंदिर चलेंगे, जल्दी उठ जाना",
        "पैसे मिल गए, धन्यवाद",
        "तुम्हारा परिणाम कब आएगा?",
        "शाम को सब्ज़ी ले आना",
        "दादी की तबीयत अब ठीक है",
        "स्कूल की फीस जमा कर दी है",
    ]
] + [
    ("chat", "en", t) for t in [
        "Are we still on for lunch tomorrow?",
        "Can you send me the slides from today's lecture?",
        "Running 10 minutes late, sorry!",
        "Thanks for the help yesterday, really appreciate it",
        "Did you book the cab for the airport?",
        "Let's catch up this weekend",
    ]
]

N_PER_SCAM_TEMPLATE = 26
N_PER_LEGIT_TEMPLATE = 30
N_PER_CHAT = 6


HEADERS = ["SBIINB", "HDFCBK", "ICICIT", "AXISBK", "KOTAKB", "JVVNLT", "BSESRJ", "AMAZON", "FLPKRT",
           "SWIGGY", "JIOPAY", "AIRTEL", "IRCTCI", "GPAYIN", "PHONPE", "MUJAIP"]


def sender_for(label: int, cat: str) -> str:
    """Synthetic sender IDs. In India, business SMS come from DLT headers (VM-HDFCBK);
    scams mostly come from ordinary 10-digit numbers. Noise is added on purpose
    (forwarded legit messages with no sender, a few scams with spoofed headers)
    so the model can't treat the sender as a perfect label."""
    r = rng.random()
    if label == 1:
        if r < 0.65:
            return phone()
        if r < 0.92:
            return ""
        return f"{rng.choice(['VM', 'AX', 'JD'])}-{rng.choice(HEADERS)}"  # spoofed / compromised header
    if cat == "chat":
        return phone() if r < 0.8 else ""
    if r < 0.75:
        return f"{rng.choice(['VM', 'AX', 'JD', 'BP', 'AD', 'VK', 'JM'])}-{rng.choice(HEADERS)}"
    if r < 0.95:
        return ""
    return phone()  # e.g. a colleague forwarding the alert


def build_indian():
    rows = []
    for i, (cat, lang, tpl) in enumerate(SCAM):
        for _ in range(N_PER_SCAM_TEMPLATE):
            rows.append(dict(text=perturb(fill(tpl), 0.35), label=1, category=cat, lang=lang,
                             source="indian_template", group=f"scam_{i}", sender=sender_for(1, cat)))
    for i, (cat, lang, tpl) in enumerate(LEGIT):
        for _ in range(N_PER_LEGIT_TEMPLATE):
            rows.append(dict(text=fill(tpl), label=0, category=cat, lang=lang,
                             source="indian_template", group=f"legit_{i}", sender=sender_for(0, cat)))
    for i, (cat, lang, tpl) in enumerate(CHATS):
        for _ in range(N_PER_CHAT):
            rows.append(dict(text=chat_variant(tpl), label=0, category=cat, lang=lang,
                             source="indian_template", group=f"chat_{i}", sender=sender_for(0, cat)))
    return rows


def build_uci():
    rows, path = [], ROOT / "raw" / "sms.tsv"
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            lab, _, text = line.rstrip("\n").partition("\t")
            rows.append(dict(text=text, label=int(lab == "spam"), category="uci_" + lab, lang="en",
                             source="uci", group=f"uci_{i}", sender=""))
    return rows


SCAM += EXTRA_SCAM
LEGIT += EXTRA_LEGIT


def build_user_rows():
    """Real messages: corrections saved from the web UI (data/feedback.csv) and any datasets you drop in
    data/external/*.csv (columns: text,label  with label 1=scam 0=legit; optional sender).
    Each row is its own group. Real data is worth far more than templates."""
    rows = []
    sources = [(ROOT / "feedback.csv", "feedback")] + [(p, "external") for p in sorted((ROOT / "external").glob("*.csv"))]
    for path, source in sources:
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for i, r in enumerate(csv.DictReader(f)):
                text, label = (r.get("text") or "").strip(), str(r.get("label", "")).strip().lower()
                if not text or label not in {"0", "1", "scam", "legit", "spam", "ham"}:
                    continue
                y = 1 if label in {"1", "scam", "spam"} else 0
                rows.append(dict(text=text, label=y, category=source, lang=detect_language(text), source=source,
                                 group=f"{source}_{path.stem}_{i}", sender=(r.get("sender") or "").strip()))
    if rows:
        print(f"  + {len(rows)} real messages from feedback / external files")
    return rows


def main():
    rows = build_uci() + build_indian() + build_user_rows()
    seen, out = set(), []
    for r in rows:
        key = r["text"].strip().lower()
        if key not in seen:
            seen.add(key)
            out.append(r)
    dest = ROOT / "processed" / "messages.csv"
    dest.parent.mkdir(exist_ok=True)
    with open(dest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["text", "label", "category", "lang", "source", "group", "sender"])
        w.writeheader()
        w.writerows(out)
    n_scam = sum(r["label"] for r in out)
    print(f"wrote {len(out)} rows ({n_scam} scam, {len(out) - n_scam} legit) -> {dest}")


if __name__ == "__main__":
    main()
