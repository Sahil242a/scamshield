# Evaluation report

Split (by template group): {'dev': 9460, 'test': 1641, 'dev_groups': 4652, 'unseen_test_templates': 52}

Thresholds: SCAM ≥ 0.699, SUSPICIOUS ≥ 0.374

## Robustness: 5 different group splits (mean ± std; worst split in brackets)

| Model | PR-AUC | Precision @SCAM | Recall @SCAM | FPR @SCAM |
|---|---|---|---|---|
| naive_bayes_baseline | 0.929 ± 0.015 (0.905) | 0.981 ± 0.034 (0.913) | 0.444 ± 0.139 (0.218) | 0.0040 ± 0.0072 (0.0000) |
| rules_only | 0.866 ± 0.014 (0.844) | 0.940 ± 0.037 (0.905) | 0.772 ± 0.042 (0.689) | 0.0203 ± 0.0120 (0.0000) |
| text_model_only | 0.947 ± 0.026 (0.897) | 0.959 ± 0.055 (0.860) | 0.586 ± 0.119 (0.439) | 0.0114 ± 0.0156 (0.0000) |
| hybrid (text + rules) | 0.988 ± 0.010 (0.968) | 0.970 ± 0.017 (0.943) | 0.947 ± 0.026 (0.916) | 0.0112 ± 0.0065 (0.0018) |

## Deployed verdicts (thresholds + evidence rules), 5 splits

| Metric | Mean ± std (worst) |
|---|---|
| scam_precision | 0.984 ± 0.010 (0.968) |
| scam_recall | 0.795 ± 0.023 (0.752) |
| flagged_recall | 0.973 ± 0.024 (0.925) |
| legit_called_scam | 0.005 ± 0.003 (0.002) |
| legit_safe | 0.965 ± 0.013 (0.947) |

## Main split (seed 42) — ablation

| Model | PR-AUC | Precision @SCAM | Recall @SCAM | FPR @SCAM | Recall @SUSPICIOUS | FPR @SUSPICIOUS |
|---|---|---|---|---|---|---|
| naive_bayes_baseline | 0.932 | 1.000 | 0.218 | 0.0000 | 0.794 | 0.0119 |
| rules_only | 0.884 | 0.967 | 0.793 | 0.0137 | 0.811 | 0.0486 |
| text_model_only | 0.955 | 1.000 | 0.695 | 0.0000 | 0.846 | 0.0330 |
| hybrid (text + rules) | 0.968 | 0.996 | 0.916 | 0.0018 | 0.925 | 0.0147 |

## Hybrid model by slice (main split, SCAM threshold)

| Slice | n | Precision | Recall | FPR |
|---|---|---|---|---|
| lang=en | 1248 | 0.996 | 0.854 | 0.0011 |
| lang=hi | 205 | 0.993 | 1.000 | 0.0145 |
| lang=hinglish | 188 | 1.000 | 1.000 | 0.0000 |
| source=indian_template | 871 | 0.995 | 0.911 | 0.0049 |
| source=uci | 770 | 1.000 | 0.943 | 0.0000 |

## Generalization set (82 hand-written messages of new types)

| Slice | SCAM precision | SCAM recall | Flagged recall | Legit called SCAM | Legit SAFE |
|---|---|---|---|---|---|
| all | 1.00 | 0.84 | 0.97 | 0.00 | 0.95 |
| lang=en | 1.00 | 0.85 | 0.95 | 0.00 | 0.96 |
| lang=hi | 1.00 | 0.75 | 1.00 | 0.00 | 1.00 |
| lang=hinglish | 1.00 | 0.90 | 1.00 | 0.00 | 0.90 |

## Challenge set v2 (40 hand-written messages)

| Slice | SCAM precision | SCAM recall | Flagged recall | Legit called SCAM | Legit SAFE |
|---|---|---|---|---|---|
| all | 1.00 | 0.95 | 1.00 | 0.00 | 0.90 |
| lang=en | 1.00 | 1.00 | 1.00 | 0.00 | 0.83 |
| lang=hi | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 |
| lang=hinglish | 1.00 | 0.75 | 1.00 | 0.00 | 1.00 |

### Challenge set v2 verdicts

| Verdict | True label | Note |
|---|---|---|
| SCAM | scam | video KYC lookalike |
| SCAM | scam | electricity cut-off |
| SCAM | scam | Paytm KYC |
| SCAM | scam | gas KYC Hindi |
| SCAM | scam | task scam |
| SCAM | scam | lottery WhatsApp link |
| SUSPICIOUS | scam | mistaken recharge |
| SCAM | scam | lookalike domain |
| SCAM | scam | PIN to receive |
| SCAM | scam | courier duty |
| SCAM | scam | digital arrest |
| SCAM | scam | investment group |
| SCAM | scam | family emergency |
| SCAM | scam | APK shortener |
| SCAM | scam | LIC fee |
| SCAM | scam | PAN Hindi |
| SCAM | scam | non-Indian brand phish |
| SCAM | scam | remote access refund |
| SCAM | scam | smart meter fee |
| SCAM | scam | task scam Telegram |
| SAFE | legit | real debit |
| SAFE | legit | real OTP |
| SUSPICIOUS | legit | advisory |
| SAFE | legit | real bill |
| SAFE | legit | receipt |
| SAFE | legit | delivery |
| SAFE | legit | chat |
| SAFE | legit | college |
| SAFE | legit | chat |
| SAFE | legit | friend money |
| SAFE | legit | family |
| SAFE | legit | real gas booking |
| SAFE | legit | interview |
| SUSPICIOUS | legit | legit 'won' |
| SAFE | legit | reminder |
| SAFE | legit | official gov domain |
| SAFE | legit | real reconnect - hard |
| SAFE | legit | subscription |
| SAFE | legit | real outage notice |
| SAFE | legit | parent warning |

## Stacker weights (log-odds added when a signal fires)

| Feature | Weight |
|---|---|
| credential_ask | +3.030 |
| pin_to_receive | +3.005 |
| remote_access_app | +3.002 |
| trust_txn_alert | -2.563 |
| fake_alert_callback | +2.505 |
| lookalike_domain | +2.504 |
| apk_link | +2.502 |
| new_number_money | +2.276 |
| trust_official_domain | -2.151 |
| fee_to_personal_upi | +2.055 |
| trust_otp_delivery | -2.051 |
| ip_url | +2.006 |
| investment_returns | +2.001 |
| authority_threat | +2.000 |
| trust_do_not_share | -1.846 |
| trust_dlt_sender | -1.580 |
| url_shortener | +1.534 |
| sent_by_mistake | +1.527 |
| trust_fraud_advisory | -1.511 |
| risky_tld | +1.509 |
| impersonation_from_personal_number | +1.356 |
| work_from_home | +1.294 |
| prize_lottery | +1.223 |
| text_model_logit | +1.151 |
| threat | +1.061 |
| call_personal_number | +1.059 |
| kyc_pan_update | +0.988 |
| international_sender | +0.800 |
| urgency | +0.707 |
