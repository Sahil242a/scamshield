# ScamShield — UPI / SMS scam detector for English, Hindi & Hinglish

[![CI](https://github.com/Prabhutvax/scamshield/actions/workflows/ci.yml/badge.svg)](https://github.com/Prabhutvax/scamshield/actions/workflows/ci.yml)
**Live demo:** _add your Hugging Face Space or Render URL here_

Flags fake "KYC expired", "bijli kat jayegi", "you won KBC lottery", "enter PIN to receive" and similar messages,
and **explains why** in plain language. Built to keep false alarms on real bank messages (OTPs, debit alerts,
bills, offers) very low, and to **learn from your corrections**.

```
🚨 SCAM  (scam probability 100%)

Why:
• Creates time pressure ('today', 'within 24 hours', 'turant') so you act before thinking.  [aaj raat]
• Threatens to block/cut/close your account, SIM or electricity to scare you.  [kaat di]
• Asks you to call/WhatsApp a personal 10-digit mobile number...  [9123456789]
• Claims to be a bank/utility/telecom but was sent from an ordinary mobile number...
What to do: Do not click links, call numbers, share OTP/PIN or pay anything... call 1930.
```

## How it works

```
message ─► normalise ─► TF-IDF (word 1-2 + char 2-5 grams) ─► LogisticRegression ─► text log-odds ─┐
   │                                                                                               ├─► stacker ─► P(scam)
   └────► 28 rule signals (lookalike domains, "PIN to receive", OTP ask vs OTP delivery, sender ID…) ┘
                 ─► thresholds + evidence rules ─► SCAM / SUSPICIOUS / SAFE + explanation
```

* **Text model** learns wording (handles Hinglish spelling variants and Devanagari via character n-grams).
* **Rules** encode facts a small dataset can't teach ("you never need a PIN to *receive* money",
  "`sbi-kyc.xyz` is not SBI", "a real OTP SMS *contains* the code; a scam *asks* for it").
* **Stacker**: logistic regression with sign constraints, regularised towards expert prior weights.
* **Evidence rules**: SCAM is only given when a concrete red flag is found; wording alone → SUSPICIOUS.
  A critical red flag (PIN-to-receive, OTP request, remote-access app, fake website…) is never SAFE.
* **Feedback loop**: "wrong verdict?" buttons in the web demo save corrections to `data/feedback.csv` (on your
  machine only); `retrain.bat` learns from them.
* Optional **transformer** text model (MuRIL / IndicBERT / DistilBERT) via `finetune_transformer.py`.

## Results

**On 82 hand-written messages of types the system had never seen** (`data/generalization_set.csv`):

| | Before coverage round | Now |
|---|---|---|
| Genuine messages called SCAM | 16% | **0%** |
| Genuine messages called SAFE | 80% | **95%** |
| Scams flagged (SCAM or SUSPICIOUS) | 87% | **97%** |

**Template-level test, 5 different group splits (test templates never seen in training), mean ± std:**
SCAM verdicts correct 98.4% ± 1.0 · scams flagged 97.3% ± 2.4 · genuine messages called SCAM 0.5% ± 0.3.
Ablation, known misses and the full method: `INTERVIEW_GUIDE.md` §8, `reports/report.md`.

> No detector is right on every message — some scams (e.g. "Hi, wrong number?") are word-for-word identical to
> real texts. Most Indian training data here is template-generated, so real-world numbers will be lower; your
> corrections are what close that gap.

## Quick start (Windows: use `python -m …` as shown)

```bash
python -m pip install -r requirements.txt
python -m uvicorn api:app           # web demo at http://127.0.0.1:8000, API docs at /docs
python -m scamshield "Aapka bijli connection aaj raat kaat diya jayega. Call 9123456789"
python -m pytest -q                 # 29 regression tests
```

**When it gets a message wrong:** press "It's actually a scam / genuine" under the verdict, then run
`retrain.bat` (Windows) or `sh retrain.sh` (~2 min) and restart the server. Each correction counts ×20 in training.
You can also drop any labelled dataset in `data/external/*.csv` (columns `text,label`, 1 = scam) before retraining.

Telegram bot: `set TELEGRAM_BOT_TOKEN=...` then `python bot.py`.
API: `POST /analyze {"text": "...", "sender": "VM-HDFCBK"}` → label, probability, reasons, trust signals, advice.
`POST /feedback {"text": "...", "correct_label": "scam"|"legit"}` → saves a correction locally.

## Deploy

The `Dockerfile` builds the dataset, **trains the model inside the image** (so the model always matches the
library versions that load it), runs the tests, and serves the app in **public mode**: corrections are disabled
(strangers can't write to the server or poison training data) and each IP is limited to 30 checks per minute.

**Hugging Face Spaces (recommended, free CPU):**
1. On huggingface.co → New Space → SDK **Docker** → Blank → create it, then clone it:
   `git clone https://huggingface.co/spaces/<your-username>/scamshield`
2. `python deploy/huggingface/prepare_space.py ../scamshield` (copies the project without models/reports, which
   Hugging Face would reject as binaries, and adds the Space's README header)
3. `cd ../scamshield && git add . && git commit -m "Deploy ScamShield" && git push`
4. The first build takes ~5 minutes (it trains the model). The app is then at
   `https://<your-username>-scamshield.hf.space`.

**Render (alternative):** push this repo to GitHub → render.com → New → Blueprint → pick the repo. `render.yaml`
sets everything up. Free instances sleep after 15 minutes idle and take ~1 minute to wake.

**Any Docker host:** `docker build -t scamshield . && docker run -p 7860:7860 scamshield`

CI: `.github/workflows/ci.yml` rebuilds the data, retrains, runs the tests and uploads the evaluation report on
every push.

### Transformer (optional, on Colab GPU)
```bash
pip install torch transformers accelerate
python finetune_transformer.py --model google/muril-base-cased --epochs 3
SCAMSHIELD_TEXT_MODEL=transformer python -m uvicorn api:app
```
`--smoke-test` checks the code path with a tiny random BERT. (Not executed in the build environment, which had
no PyTorch — run the smoke test first on Colab.)

## Layout
```
data/build_dataset.py      UCI SMS Spam + Indian EN/HI/Hinglish templates + your corrections/external CSVs
data/templates_extra.py    coverage expansion: ~120 templates for message types the first version lacked
data/*_set*.csv            hand-written evaluation sets (challenge set v2, generalization set)
scamshield/normalize.py    unicode/homoglyph/leetspeak/Devanagari-digit cleanup, URL/phone/amount tokens
scamshield/rules.py        28 explainable signals + prior weights
scamshield/stacker.py      sign-constrained, prior-regularised logistic regression
scamshield/detector.py     ScamDetector: score → evidence rules → verdict → explanation
train.py                   group splits ×5, OOF stacking, threshold tuning, ablation, report
retrain.bat / retrain.sh   rebuild data (incl. corrections) and retrain in one step
finetune_transformer.py    MuRIL / IndicBERT / DistilBERT fine-tuning with the same protocol
api.py · web/ · bot.py     FastAPI service (+ /feedback, rate limit, public mode), web demo, Telegram bot
tests/                     regression tests for every bug found in development
Dockerfile · render.yaml   container image (trains at build time) and Render blueprint
deploy/huggingface/        Space README header + prepare_space.py
.github/workflows/ci.yml   build, train, test, upload report on every push
```

## Data credits
UCI SMS Spam Collection (Almeida & Hidalgo, 2011; CC BY 4.0), included at `data/raw/sms.tsv`.
Indian messages are generated templates written for this project; any resemblance to real phone numbers or UPI IDs is random.
