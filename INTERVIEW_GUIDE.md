# ScamShield — Interview Guide

Everything you need to explain this project confidently: the pitch, every design decision and *why*,
the numbers, the bugs you found (your best stories), honest limitations, and likely questions with answers.

---

## 1. The 30-second pitch

> "Indians get a flood of fake 'KYC expired', 'your electricity will be cut tonight' and 'enter your UPI PIN
> to receive cashback' messages. I built ScamShield, which flags these in English, Hindi and Hinglish and
> explains *why* a message is suspicious. It's a hybrid: a text classifier learns the wording, and a rule
> engine encodes facts a small dataset can't teach — like 'you never need a PIN to *receive* money' or
> 'sbi-kyc.xyz isn't an SBI domain'. A constrained stacker combines them, and the verdict has an evidence
> rule: it only says SCAM if it can point to a concrete red flag; wording alone gets an amber SUSPICIOUS.
> On 82 hand-written messages of types the system had never seen, it flags 97% of scams and calls zero
> genuine messages a scam. And when it is wrong, users press 'this verdict is wrong' and the next retrain
> learns from it. It's served as a FastAPI service, a web demo and a Telegram bot."

## 2. The 2-minute architecture walkthrough

```
message ─► normalise ─► TF-IDF (word 1-2 + char 2-5 grams) ─► LogisticRegression ─► text log-odds ─┐
   │                                                                                               ├─► stacker ─► P(scam)
   └────► 28 rule signals (22 scam signals + 6 "trust" signals) ───────────────────────────────────┘      │
                                                                                                          ▼
                     thresholds:  P ≥ 0.70 → SCAM · 0.37–0.70 → SUSPICIOUS · < 0.37 → SAFE
                     evidence rules:  SCAM needs ≥1 concrete scam signal (else → SUSPICIOUS)
                                      a critical signal (PIN-to-receive, OTP ask, …) is never SAFE
                                                                                                          │
   user presses "wrong verdict" ─► data/feedback.csv ─► retrain (corrections weighted ×20) ◄──────────────┘
```

1. **Normalise** — fix tricks scammers use: Devanagari digits (९८७६ → 9876), Cyrillic look-alike letters,
   leetspeak (`bl0cked` → `blocked`), spaced-out letters (`K.Y.C` → `kyc`). Replace URLs, phone numbers and
   amounts with tokens (`urltoken`, `phonetoken`, `amounttoken`) so the model learns *"has a link + an amount"*,
   not specific numbers that never repeat.
2. **Text model** — TF-IDF features from word n-grams *and* character n-grams, fed to logistic regression.
3. **Rules** — 28 deterministic checks, each producing a binary feature *and* a human-readable reason.
4. **Stacker** — a small logistic regression over [text log-odds, 28 rule flags], with sign constraints and
   expert prior weights.
5. **Decision + explanation** — thresholds, then the two evidence rules; list fired rules with the exact
   evidence (the link, the phrase), the words the model weighted most, and "signs it may be genuine".
6. **Feedback loop** — corrections are saved locally and learned at the next retrain.

### Why the evidence rule matters (great point to make)
The text model judges by *surface similarity*. For a message type it has never seen — a long itemised bill, a
festival offer — that similarity is unreliable, and that is exactly how both user-reported false alarms
happened. The evidence rule turns the model's weakest case into a safe outcome: if nothing concrete is wrong
with the message, the worst it can get is "SUSPICIOUS — wording resembles scams, verify in the official app".
A red "SCAM" always comes with a specific, checkable reason. On new-type messages this took legit→SCAM
from 16% to 0%.

---

## 3. Problem framing — the precision vs recall trade-off (they WILL ask this)

* **Precision** = of messages we flagged as scam, how many really were. Low precision = false alarms.
* **Recall** = of all real scams, how many we caught. Low recall = missed scams.
* **False-positive rate (FPR)** = share of legitimate messages wrongly flagged.

**Both errors are costly, differently:**

| Error | Example | Cost |
|---|---|---|
| False negative (miss) | KYC scam marked SAFE | User may lose money — directly harmful |
| False positive (false alarm) | Real HDFC OTP or "Rs 399 debited" marked SCAM | User ignores a real fraud alert from their bank; learns to distrust the tool ("cry wolf") |

The second one is subtle and is the point the brief made: **if we flag the bank's genuine "Rs 50,000 debited,
not you? call…" alert as a scam, the user might ignore a real fraud on their account.**

**My answer: don't force one threshold. Use two.**
* **SCAM** (red, strong words): threshold chosen so precision ≥ 98% → when we shout, we're almost always right.
* **SUSPICIOUS** (amber, "verify through the official app"): threshold chosen so recall ≥ 97%, capped so it
  warns on ≤ 5% of legit messages. The cost of an amber warning is low — it just says "check via the app".
* Thresholds are chosen on dev data only; the test set is used once.

Why **PR-AUC rather than ROC-AUC** as the headline? Scams are the minority class (~29% here, far less in a real
inbox). ROC-AUC includes true negatives, which are plentiful, so it looks great even for a mediocre model.
PR-AUC focuses on how well we rank the positive class, which is what matters for a detector.

---

## 4. Data — and why the split matters most

**Sources (11,101 messages, 3,286 scam):**
1. **UCI SMS Spam Collection** (5,159 after de-dup; public, English). Its spam is mostly prize / premium-rate
   fraud, so spam → scam.
2. **Indian messages from templates** (5,942): **109 scam templates** — KYC, SIM-KYC, electricity cut-off,
   lottery/KBC, task/job, UPI PIN-to-receive and collect-request approval, "sent by mistake", fake
   customs/digital arrest, loan apps, reward points, remote access, investment/crypto, family impersonation,
   fake debit alerts with a callback number, WhatsApp code takeover, fake e-challan, fake govt schemes/subsidies,
   OLX "army buyer", extortion, wallet restrictions — and **138 legitimate templates** across banking (alerts,
   statements, mandates, cheques), insurance, EPF, mutual funds, railways, flights, FASTag, OTT, delivery/ride
   handover OTPs, app security alerts, government and civic notices, schools, hospitals, jobs, promotions with
   codes, itemised bills, plus 68 everyday chats. Each template is filled with random banks, amounts, numbers,
   UPI IDs, vehicles and URLs; scam copies get "evasion noise" (misspellings, emojis, CAPS, leetspeak).
   Hindi and Hinglish each have roughly as many legit as scam messages — this balance is deliberate (§9, story 2).
3. **Hard negatives**: real-style OTPs (including ones you *should* share with a delivery agent or driver),
   debit/credit alerts, bank *anti-fraud* advisories, Google Pay "you won ₹11" scratch cards, "you received
   Rs 500 from Ankit" credits, friends asking for money. They look scary but are legit.
4. **Synthetic sender IDs**: legit business SMS in India come from TRAI-DLT-registered headers like
   `VM-HDFCBK`; scams usually come from 10-digit numbers. Noise is added (spoofed headers, forwarded messages)
   so the model can't treat sender as a perfect label.
5. **Your own data**: corrections from the web UI (`data/feedback.csv`) and any CSV in `data/external/`
   (columns `text,label`). These are real messages, so they're the most valuable rows in the dataset.

**Why templates?** There's no sizeable public labelled dataset of Indian scam SMS. The closest I found, a
CC-BY Hinglish set on Hugging Face, is mostly the same UCI data plus about 60 hand-written Hinglish lines.
Templates let me cover the real scam storylines; the risk is the model memorising template wording. Which
leads to…

**Group split (the most important evaluation decision).** If I split randomly, copies of the same template
land in train and test, and the model scores ~100% by recognising the template. So I split **by template**
(`GroupShuffleSplit`, group = template id): the test set contains ~50 templates the model never saw. This
simulates "a new scam wording appears next month". I repeat this on 5 different splits (§8).

**Plus three hand-written evaluation sets** (never generated from templates):
* challenge set v1 (47 msgs) — used for error analysis while developing, so it stopped being a fair test and
  was retired (removed from the repo).
* `challenge_set_v2.csv` (40 msgs) — written after the first development freeze.
* `generalization_set.csv` (82 msgs) — written **before** the coverage expansion, specifically from message
  *types* the data didn't have yet (EPF, FASTag, flights, handover OTPs, WhatsApp takeover, fake e-challan…).
  I recorded the before score, then deliberately never looked at its individual results while improving the
  system — only the aggregate numbers. It's the best single answer to "does it work on unknown messages?"

---

## 5. Why TF-IDF + logistic regression first (and where transformers fit)

**TF-IDF** turns text into numbers: each feature is a word/character sequence, weighted by how often it appears
in this message (TF) and down-weighted if it appears in every message (IDF), so "the" counts little and
"kyc" counts a lot.

**Why character n-grams (2–5 chars)?** Hinglish has no standard spelling — *turant / turnt*, *karein / kare /
kariye*, *acount / accnt*. Words differ but share character chunks (`tur`, `kar`, `acc`). Char n-grams also work
on Devanagari without a Hindi tokeniser, and survive scammers' misspellings.

**Logistic regression** outputs log-odds = Σ(weight × feature) + bias, then a sigmoid → probability. Fast
(~ms on CPU), well-calibrated-ish, and *interpretable*: each feature's weight × value tells you how much it
pushed the score — that's how I show "words the model flagged".

**Why not start with a transformer?** (a) A strong, cheap baseline tells you whether the expensive model is
worth it. (b) Interpretability. (c) Runs on a phone/CPU. (d) With ~8k mostly templated messages, a 200M+-parameter
model can overfit to template style. The script `finetune_transformer.py` is there to compare **with the same
split, rules, stacker and thresholds**, so the comparison is fair.

**Transformer choices for code-mixed text:**
* **MuRIL** (Google) — pre-trained on 17 Indian languages *including transliterated/Romanised* text. Best fit
  for Hinglish like "aapka account band ho jayega".
* **IndicBERT v2** (AI4Bharat) — strong on native scripts (Devanagari); less Romanised exposure.
* **DistilBERT multilingual** — smaller (~135M params, most of it the large multilingual vocabulary), faster; general-purpose.

**How fine-tuning works (be ready to explain):** a pre-trained transformer already "understands" language
from masked-language-model pre-training (predict hidden words). We add a classification head on the `[CLS]`
token and train the whole thing on our labels for a few epochs with a small learning rate (3e-5) and warm-up,
so we adapt rather than destroy the pre-trained knowledge. Self-attention lets every token look at every
other token, so it can learn that "PIN" near "receive" is dangerous while "PIN" near "never share" isn't —
context that bag-of-n-grams misses. Sub-word tokenisation (WordPiece) splits unseen words like "kariye" into
known pieces, which helps with spelling variants.

*Honest note:* the transformer script was compile-checked but not executed in my build environment (no PyTorch
available there); run `--smoke-test` first on Colab, then the real fine-tune, and report its numbers next to the
TF-IDF ones.

---

## 6. The rule engine — what and why

Rules encode **domain knowledge that a few thousand examples can't teach reliably**, and they make the
explanation trustworthy. Highlights:

| Signal | Logic | Why it's strong |
|---|---|---|
| `pin_to_receive` | "PIN" near receive / refund / cashback / "paise aa jayenge" | You **never** need a UPI PIN to receive money; entering it sends money |
| `credential_ask` vs `trust_otp_delivery` | Asking for an OTP vs a message *containing* the OTP | **Genuine OTP SMS contain the code; scams ask for it.** Same keyword, opposite meaning |
| `lookalike_domain` | brand name in a domain that's not on the official list | `sbi-kyc-update.xyz`, `hdfc-unlock-netbanking.com` |
| `trust_official_domain` | official allowlist + restricted registries (`.gov.in`, `.bank.in`) | scammers can't register these |
| `url_shortener`, `risky_tld`, `ip_url`, `apk_link` | link hygiene | banks don't send bit.ly links or APKs |
| `impersonation_from_personal_number` | claims bank/utility/telecom but sent from a 10-digit number | real ones use DLT headers |
| `new_number_money` | "new number / phone broke" + money ask | family-impersonation scam |
| `remote_access_app` | AnyDesk / TeamViewer / QuickSupport | lets scammer control the phone |
| urgency / threat / authority | "aaj raat", "band ho jayega", "CBI", "digital arrest" | pressure tactics |
| `trust_*` (6) | "do not share", masked a/c XX1234 + UPI ref, fraud advisories, DLT sender | push **towards SAFE** — critical for precision |

**Negation handling in code-mixed text (great detail to mention):** "Do not share your OTP" must not fire
`credential_ask`. English puts negation *before* the verb ("do **not** share"); **Hindi is verb-final (SOV), so
the negation comes *after*: "share **na** karein", "batana **mat**", "साझा **न** करें".** The rules check both
directions.

---

## 7. Combining model + rules — stacking (and the sign-flip story)

**Stacking:** train a second-level model whose inputs are the first model's output plus the rule flags.

**Out-of-fold (OOF) predictions:** if the stacker learns from the text model's predictions *on its own training
data*, those predictions are over-confident (it has memorised those rows), so the stacker would over-trust the
text model. So I use **grouped 5-fold cross-validation**: every dev message gets a text score from a model that
never saw its template. The stacker learns from those honest scores.

**The bug (tell this one):** my first stacker was a plain logistic regression. It learned a **negative weight for
"asks you to share OTP"** — nonsense. Cause: the rules are **collinear** with each other and with the text model
on templated data, and with little data per rule the weights become unstable. Also, a rule that never fires in
training gets weight 0 — silently useless.

**The fix:** a custom logistic regression (`scamshield/stacker.py`) that
* is **sign-constrained** — scam signals can only add risk (w ≥ 0), trust signals can only reduce it (w ≤ 0);
* is **regularised towards expert prior weights** instead of towards zero:
  `loss = mean log-loss + λ/2 · ‖w − w_prior‖²`, solved with L-BFGS-B (bounded optimisation).

This is MAP estimation with a Gaussian prior centred on my expert weights: with little evidence a weight stays
near its prior; with lots of evidence data moves it. *Be honest if asked:* with λ = 0.01 the data moved weights
only modestly (e.g. DLT-sender trust −1.0 → −1.4, transaction-alert trust −2.5 → −2.62), so the final weights
are largely my priors, validated rather than discovered by the data. More real data would let the data speak more
— you'd lower λ.

---

## 8. Results

### The question that matters most: messages of a *type* it has never seen
`generalization_set.csv` — 82 hand-written messages (38 scam, 44 legit; English, Hinglish, Hindi), written
before I broadened the data, scored before and after:

| | Before coverage round | After |
|---|---|---|
| Legit messages called **SCAM** | 16% | **0%** |
| Legit messages called **SAFE** | 80% | **95%** |
| Scams flagged (SCAM or SUSPICIOUS) | 87% | **97%** |
| Scams called SCAM specifically | 87% | 84% |

The last row dropped slightly *by design*: the evidence rule moves wording-only scam calls to SUSPICIOUS. They're
still flagged; the user still gets a warning. That trade — a few red verdicts become amber, and false red verdicts
on genuine messages go to zero — is exactly the precision-vs-recall reasoning from §3.

### Template-level test, repeated on 5 group splits (mean ± std)
What the user actually sees (thresholds + evidence rules):

| Deployed verdicts | Mean ± std |
|---|---|
| When it says SCAM, it's right | 98.4% ± 1.0 |
| Scams flagged (SCAM or SUSPICIOUS) | 97.3% ± 2.4 |
| Scams called SCAM | 79.5% ± 2.3 |
| Legit messages called SCAM | 0.5% ± 0.3 |
| Legit messages called SAFE | 96.5% ± 1.3 |

Most scams that get SUSPICIOUS rather than SCAM are old UK premium-rate spam from the UCI data, which has none of the
Indian red flags the rules look for — so the evidence rule caps them at amber. Correct behaviour.

Model ablation at the probability level (before the evidence rules), same 5 splits:

| Model | Precision @SCAM threshold | Recall |
|---|---|---|
| Naive Bayes baseline | 98.1% ± 3.4 | 44.4% ± 13.9 |
| Rules only | 94.0% ± 3.7 | 77.2% ± 4.2 |
| Text model only | 95.9% ± 5.5 | 58.6% ± 11.9 |
| **Hybrid** | **97.0% ± 1.7** | **94.7% ± 2.6** |

**How to read the ablation:** each half alone either misses a lot (text: 59%) or false-alarms more (rules: 94%
precision). Together: 97% / 95%. Real OTPs, advisories, bills and promos look like scams to a word model, so it
must be conservative; rules supply decisive facts and trust signals; the text model supplies coverage of wordings
the rules don't encode.

**Why 5 splits?** With ~50 templates in a test set, one split's numbers depend on *which* templates land there
(§9, story 9). Mean ± std is the honest headline; the worst split is the floor.

### Known misses — say these out loud
* **"Wrong number" openers** ("Hi, sorry, are you not Riya from the gym?") — the first message of a long
  investment scam, word-for-word identical to a real wrong-number text. No single-message detector can catch it;
  the fix is conversation-level detection (flag the *later* "invest in this app" message).
* **Real achievements that use prize language** — "Congrats! You've won the hackathon… prize ceremony". The
  prize rule can't tell a competition from a lottery. The feedback loop fixes individual cases (tested: a "Best
  Paper Award" message went SCAM 79% → SAFE 8% after one correction, while a KBC lottery scam stayed SCAM 100%).

**Challenge set v2 (40 msgs):** 20/20 scams flagged (19 SCAM, 1 SUSPICIOUS), 0 legit called SCAM, 18/20 legit SAFE.
**User-reported messages:** genuine Airtel itemised bill → SAFE (10%); genuine Hindi Jio offer → SAFE (10%).

Latency: ~30 ms per message on CPU. Model files: ~710 KB.

---

## 9. Bugs I found and fixed (your best interview stories)

1. **Stacker sign flip** — §7. Lesson: don't let a flexible model learn weights it has no data to support; use
   priors and constraints.
2. **Spurious language shortcut** — Hinglish rows were roughly 4–5 scams for every legit message in the first dataset, so the model learned
   "Hinglish ⇒ scam" and flagged ordinary chats like "aaj raat late aaunga". Fixes: (a) added ~45 more legit
   Hinglish/Hindi chats and service messages; (b) **sample weights that balance labels within each language**, so
   language alone carries no information about the label. Note my first weighting attempt equalised *every*
   (language, label) cell, which massively down-weighted English legit messages and raised false positives — the
   correct version keeps each language's total weight and balances labels inside it.
3. **Normaliser broke UPI IDs** — leetspeak reversal turned `anjali4413@ibl` into `anjaliaaieaibl`, so the "pay
   to a personal UPI ID" rule never fired. Fix: skip tokens with >2 digits or `name@handle` shape. Lesson:
   preprocessing bugs silently kill features — test the pipeline, not just the model.
4. **Verb-final negation** — Hinglish OTP messages ("share na karein") were flagged as OTP requests. §6.
5. **Unstable thresholds** — tuned on one validation split, the threshold didn't transfer: one unseen OTP template
   (~30 rows) landed in test above it. Because data comes in clusters (templates), a single split is high-variance.
   Fix: tune thresholds on **grouped 5-fold OOF scores across all dev data** (~5× more templates inform the choice).
6. **Consumed challenge set** — after doing error analysis on challenge v1 it was no longer a fair test, so I wrote
   v2 after freezing development. Lesson: every time you look at a test set and change something, it becomes a
   dev set.
7. **First real-world false positive (user report)** — a genuine Airtel postpaid bill was flagged SCAM at 99.8%. Two
   causes: (a) its "understand your bill" link was `*.app.link` — Branch.io's deep-link service, which many real
   Indian apps use — and `.link` was on my phishing-TLD list; that false alarm also *suppressed* the trust signal for
   the genuine `i.airtel.in` link; (b) training data only had one-line bills, so a long itemised bill with "click"
   and "make payment" looked unfamiliar. Fixes: deep-link services (`app.link`, `page.link`, `onelink.me`) are now
   *neutral* — not trusted, since anyone can create one, but not penalised; and I added itemised telecom/utility
   bill templates. Result: 15% → SAFE, with a regression test (anonymised) so it can't come back. Lesson: this is
   exactly the feedback loop the product needs — and a reminder that **allowlists and blocklists both need care**:
   a rule that's "usually right" about a TLD can be confidently wrong about a specific legitimate service.
8. **Second user report: a genuine Hindi Jio promo flagged SCAM (94%)** — this time *no scam rule fired* and two
   trust signals did; the text model alone said 99%. Its top features were `कर <link>` and `<link>`: no legit Hindi
   message in training contained a link or an offer, so it learned **"Hindi + link + 'आज ही रिचार्ज करें' ⇒ scam"** —
   the same spurious-correlation bug as story 2, in a new place. Fixes: (a) a **label-definition** decision —
   promotional ≠ fraud — so I added genuine promo offers (Jio/Airtel/Vi recharges, sales, card offers) in Hindi,
   Hinglish and English; (b) the official-domain trust signal now requires *every* (non-deep-link) link to be
   official, so a phishing link can't hide behind one real link. Worth saying: UCI's "spam" class includes some
   plain marketing that I labelled scam — a known source of label noise in this project.
9. **Single-split numbers lie** — after story 8, test recall "dropped" from 98.3% to 92.4%. Error analysis showed
   the model hadn't regressed: adding templates had reshuffled the split, moving the unbeatable "wrong number" scam
   into test. Fix: run the full protocol on **5 group splits** and report mean ± std and the worst split. Lesson:
   before believing a metric moved, check whether the *data you measure on* moved.
10. **"Make it right on any unknown message"** — the honest answer is that no detector can be right on every
   message (some scams are word-for-word identical to real texts). So I attacked the problem three ways:
   (a) **measure it**: wrote an 82-message set of *new message types* first and recorded the before score
   (16% of genuine messages called SCAM — far worse than the 99% precision on familiar templates, which proved
   the gap was real); (b) **coverage**: ~120 new templates for message types that were missing, plus rule fixes
   found along the way — handover OTPs ("share OTP with the delivery agent" is legit), WhatsApp code takeover,
   UPI collect-request approval, fake debit alerts with a mobile callback number; (c) **fail safely**: the
   evidence rule (SCAM needs a concrete red flag). Result: 16% → 0% genuine-called-SCAM, 87% → 97% scams flagged.
11. **Feedback loop** — a "this verdict is wrong" button saves corrections locally; retraining weights each one
   ×20 and always keeps them in training data (never in the held-out test). Without the ×20, one correction among
   11,000 rows wouldn't move the model — I checked that it actually flips the verdict end-to-end.

---

## 10. Limitations — say these before they ask

* **Synthetic data** → test numbers are optimistic. Real scams evolve; template wording ≠ real wording.
  The group split and v2 challenge set reduce but don't remove this.
* **UCI data is old UK SMS** (2012), different style from Indian messages.
* **Sender IDs in training are synthetic**; spoofing and compromised headers exist in reality.
* **Rules need maintenance** — new scam types (e.g. new apps, new domains) need new rules/allowlist entries.
* **Adversarial**: scammers adapt. A determined scammer can write a message without any rule trigger — that's
  why the text model matters, and why continuous retraining on reported messages is needed.
* **No images/voice**: QR-code scams and voice calls are out of scope.
* Hindi slice is small; confidence there is low.

**What real validation would look like:** collect messages users forward to the bot (with consent, anonymised:
strip numbers/names), have them labelled, measure precision/recall on that stream, and retrain monthly. Track the
FPR on known-legit senders (DLT headers of major banks) as a guardrail metric.

---

## 11. Deployment & product

* **Docker image that trains itself**: the build rebuilds the dataset, trains, and runs the test suite; if any
  test fails, the image doesn't build. The model is always created by the exact library versions that load it
  (pickled scikit-learn models can break across versions), and the image is reproducible from the repo alone.
* **Public mode** (`SCAMSHIELD_PUBLIC=1`): the correction endpoint is disabled on the public demo — otherwise
  anyone could write to the server or poison the training data — and `/analyze` is rate-limited per IP
  (sliding window, 30/min; in-memory is fine for one instance, Redis or the gateway for several).
* **CI** (GitHub Actions): every push rebuilds data, retrains, runs 29 tests and uploads the evaluation report.
* **Hosting**: Hugging Face Spaces (Docker, free CPU) or Render (free, sleeps after 15 min idle).

* **FastAPI** (`/analyze`, `/analyze/batch`, `/health`), Swagger docs at `/docs`, a web demo at `/`.
* **Telegram bot** — users forward a message; works from a laptop via long polling, webhook for production.
* **WhatsApp**: same detector behind the Meta WhatsApp Cloud API webhook (or Twilio); not built here.
* **Privacy**: message text is never logged or stored; only verdict counts. Corrections are saved only on the
  user's own machine (`data/feedback.csv`), and only when they press the button.
* **Improving over time**: "wrong verdict" button → `data/feedback.csv` → `retrain.bat` (~2 min). In production
  you'd collect opt-in corrections centrally, have them reviewed (to stop people poisoning the model by marking
  scams as safe), and retrain on a schedule.
* **Why CPU-friendly matters**: the TF-IDF model is ~710 KB and ~30 ms — could run **on-device** in an Android
  SMS app (port the vectoriser + weights, or use ONNX), so messages never leave the phone. A transformer would need
  quantisation/distillation for that.
* **Advice is actionable and local**: don't click/call/pay, verify via the official app, report at
  cybercrime.gov.in / call **1930** (National Cyber Crime Helpline).

---

## 12. Likely interview questions — with answers

**Q: Why not just use an LLM / GPT?**
Cost and latency per message, privacy (sending users' bank SMS to a third-party API), and no control over false
positives. An LLM could be a great *labelling assistant* to grow the dataset, or a second opinion for the
SUSPICIOUS band only.

**Q: How do you handle class imbalance?**
`class_weight="balanced"` in the text model and stacker, PR-AUC as the metric, and thresholds tuned to explicit
precision/recall targets instead of the default 0.5.

**Q: Why two thresholds instead of one?**
Asymmetric costs (§3). A single threshold forces a choice between missing scams and crying wolf; the amber band
lets us be conservative with strong claims and generous with gentle warnings.

**Q: Is your probability calibrated?**
Logistic regression on OOF scores is reasonably calibrated, but with synthetic data I wouldn't present "93%" as a
literal frequency to users — I show the verdict and reasons, and treat the number as a ranking score. To check,
I'd plot a reliability diagram on real labelled data and apply isotonic/Platt calibration if needed.

**Q: What's data leakage and how did you prevent it?**
When information from the test set influences training. Here: same template in train and test → group split.
Stacker trained on in-sample predictions → OOF. Tuning on test → thresholds from dev only. Iterating on a
challenge set → wrote a fresh one.

**Q: How does it handle a message type it has never seen?**
Three layers. Rules generalise by *meaning*, not wording — a PIN-to-receive request is caught however it's
phrased. Character n-grams handle spelling variation. And when only the wording looks off, the evidence rule caps
the verdict at SUSPICIOUS, so unfamiliar-but-genuine messages can't get a red SCAM. I measured this directly on 82
new-type messages: 97% of scams flagged, 0% of genuine ones called SCAM.

**Q: Couldn't someone poison it with fake corrections?**
Locally it only affects their own copy. Centrally, corrections would need review or agreement from several users
before training, and I'd monitor the guardrail metric (false alarms on known bank DLT senders).

**Q: How would you handle a brand-new scam type tomorrow?**
Short term: add a rule (minutes, explainable). Medium: add reported examples, retrain (the group split tells me
how well it generalises to *new templates*). Monitor the share of SUSPICIOUS verdicts — a spike suggests drift.

**Q: Why char n-grams handle Hinglish — explain.**
Romanised Hindi has many spellings for one word. Words differ but share substrings, and char n-grams match those.
`char_wb` keeps n-grams within word boundaries so features stay local.

**Q: How would a transformer handle code-mixing better?**
Context via self-attention (negation, "PIN" + "receive" relationships), sub-word tokenisation for unseen
spellings, and pre-training on transliterated Indian text (MuRIL). Cost: size, latency, less interpretable — so
I'd keep the rules for explanations either way.

**Q: How do you explain predictions?**
Three layers: fired rules with evidence (exact link/phrase), top TF-IDF word features × LR weights (local
contribution), and trust signals. For a transformer I'd still show rules, and could add SHAP / integrated
gradients for token attributions.

**Q: Why logistic regression for the stacker instead of XGBoost?**
29 features (text score + 28 rules) and limited, partly synthetic data: a linear model with sign constraints is stable, monotonic (a
scam signal can never make a message *safer*) and each weight reads as "log-odds added". Gradient-boosted trees
could learn interactions but would be harder to constrain and explain; with more real data I'd try monotone-
constrained LightGBM.

**Q: What does the sender ID tell you, and can it be faked?**
In India, business SMS go through TRAI's DLT registration and arrive with headers like `VM-HDFCBK`; most scam SMS
come from personal numbers. It's a strong signal but not proof (spoofing, compromised accounts), so it's a
moderate weight (−1.4 log-odds), not an override.

**Q: What would you improve with more time?**
Real labelled data via the bot; transformer comparison on Colab; link expansion (resolve shorteners, check domain
age / Google Safe Browsing); an Android on-device version; active learning — send SUSPICIOUS cases for labelling
since they're the most informative.

---

## 13. Concept cheat sheet

| Term | One-line definition |
|---|---|
| Precision | TP / (TP + FP) — of flagged, how many were scams |
| Recall | TP / (TP + FN) — of scams, how many we caught |
| FPR | FP / (FP + TN) — of legit, how many we wrongly flagged |
| PR-AUC (average precision) | area under precision-recall curve; good for imbalanced data |
| ROC-AUC | probability a random scam scores above a random legit message |
| TF-IDF | term frequency × inverse document frequency weighting |
| Logit / log-odds | log(p / (1−p)); LR is linear in log-odds |
| Stacking | a model that learns how to combine other models' outputs |
| Out-of-fold | predictions on each fold from a model trained on the other folds |
| Group k-fold | CV where all rows of a group (template) stay in one fold |
| Regularisation | penalty that keeps weights small — or, here, close to priors |
| MAP estimation | maximise likelihood × prior; L2 towards w₀ = Gaussian prior at w₀ |
| Fine-tuning | continue training a pre-trained model on a labelled task |
| Self-attention | each token computes a weighted mix of all tokens' representations |
| Code-mixing | switching languages within a sentence (Hindi + English) |
| Transliteration | writing a language in another script (Hindi in Latin letters) |
| SOV | Subject-Object-Verb order (Hindi); negation often follows the verb |
| Spurious correlation | a feature correlated with the label in training but not causally (language ⇒ scam) |

---

## 14. Demo script (2 minutes, live)

1. `python -m uvicorn api:app` → open `http://127.0.0.1:8000`.
2. Click **Bijli cut-off** → SCAM, point at the 4 reasons.
3. Click **Real OTP** → SAFE; point out *"the message contains the code rather than asking for it"*.
4. Type: `Bhai 250 bhej de UPI pe` → SAFE. Then `Mom this is my new number, send 15000 to ravi22@ybl` → SCAM.
   "Same money request; the difference is the 'new number' storyline."
5. Show `reports/pr_curve.png`: "hybrid curve vs each half alone."
6. Paste "Congratulations! You have won the Best Paper Award at the IEEE conference…" → it may say SCAM. Press
   "It's actually genuine", run `retrain.bat`, restart, paste again → SAFE. Then paste a KBC lottery scam → still
   SCAM. "It learned the difference, not just the word 'congratulations'."
