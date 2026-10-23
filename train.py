"""Train and evaluate the hybrid scam detector.

    python data/build_dataset.py      # build data/processed/messages.csv
    python train.py                   # ~2 min: trains, tunes thresholds, writes models/ and reports/

Protocol (repeated on 5 different group splits; the first one, seed 42, is the deployed model):
  1. Group split: dev 85% / test 15%, where a group is a template, so test templates are unseen.
     User corrections (source=feedback) always stay in dev — the deployed model must learn them.
  2. Text model: TF-IDF (word 1-2 grams + char 2-5 grams) -> LogisticRegression, with sample
     weights that balance labels within each language (and x20 for user corrections).
  3. Grouped 5-fold out-of-fold (OOF) text scores on dev + rule features -> stacker
     (sign-constrained logistic regression, regularised towards expert prior weights).
     OOF matters: in-sample scores are over-confident, so a stacker trained on them would
     over-trust the text model.
  4. Thresholds from the dev OOF scores: SCAM at precision >= 98%, SUSPICIOUS at recall >= 97%
     (capped so at most 5% of legit messages are flagged). The test set is only scored.
  5. Report: ablation (Naive Bayes / rules only / text only / hybrid), deployed verdicts
     (thresholds + evidence rules), language/source slices, and the hand-written sets.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline

from scamshield.detector import decide
from scamshield.normalize import for_model
from scamshield.rules import PRIOR_WEIGHTS, RULE_IDS, rule_features, run_rules
from scamshield.stacker import PriorLogisticRegression

ROOT = Path(__file__).resolve().parent
MODELS, REPORTS = ROOT / "models", ROOT / "reports"

TARGET_PRECISION, TARGET_RECALL = 0.98, 0.97
SEED = 42                          # split used for the deployed model
EVAL_SEEDS = [42, 1, 7, 13, 29]    # 5 group splits for the robustness estimate
FEEDBACK_WEIGHT = 20.0             # one user correction counts like 20 template rows
HYBRID = "hybrid (text + rules)"
DEPLOYED = "deployed verdicts (hybrid + evidence rules)"
HANDWRITTEN = {                    # never generated from templates
    "challenge_set": "data/challenge_set_v2.csv",       # written after the first development freeze
    "generalization_set": "data/generalization_set.csv",  # message TYPES the first data version lacked
}
SCAM_RULE_IDX = [i for i, r in enumerate(RULE_IDS) if not r.startswith("trust_")]


# ------------------------------------------------------------------------------ models
def make_text_model() -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=40000)),
        # char_wb n-grams: robust to Hinglish spelling variants, misspellings and Devanagari
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True,
                                 max_features=80000)),
    ])
    return Pipeline([("features", features),
                     ("clf", LogisticRegression(C=4.0, class_weight="balanced", max_iter=3000))])


def make_stacker() -> PriorLogisticRegression:
    """Scam signals can only add risk (w >= 0), trust signals can only remove it (w <= 0)."""
    prior = np.array([1.0] + [PRIOR_WEIGHTS[r] for r in RULE_IDS])
    lower = np.array([0.0] + [-np.inf if r.startswith("trust_") else 0.0 for r in RULE_IDS])
    upper = np.array([np.inf] + [0.0 if r.startswith("trust_") else np.inf for r in RULE_IDS])
    return PriorLogisticRegression(prior, lower, upper, lam=0.01)


def make_nb() -> Pipeline:
    return Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)), ("nb", MultinomialNB(alpha=0.1))])


def make_rules_only() -> LogisticRegression:
    return LogisticRegression(class_weight="balanced", max_iter=2000)


# ------------------------------------------------------------------------------ features & weights
def rules_matrix(texts, senders) -> np.ndarray:
    return np.array([rule_features(run_rules(t, s or None)) for t, s in zip(texts, senders, strict=True)])


def lang_balanced_weights(d: pd.DataFrame) -> np.ndarray:
    """Balance labels *within* each language, keeping each language's total weight.

    Otherwise a language with more scam than legit rows teaches the shortcut
    "Hinglish/Hindi => scam" and ordinary chats get flagged.
    """
    cell = d.groupby(["lang", "label"]).text.transform("size")
    lang = d.groupby("lang").text.transform("size")
    w = lang / (2.0 * cell)
    return (w / w.mean()).values


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# ------------------------------------------------------------------------------ thresholds & metrics
def pick_thresholds(y, p) -> tuple[float, float]:
    prec, rec, thr = precision_recall_curve(y, p)
    prec, rec = prec[:-1], rec[:-1]
    ok = np.where(prec >= TARGET_PRECISION)[0]
    t_scam = float(thr[ok[0]]) if len(ok) else 0.9
    # SUSPICIOUS band: catch >= 97% of scams, but warn on at most 5% of legit messages
    ok_r = np.where(rec >= TARGET_RECALL)[0]
    t_recall = float(thr[ok_r[-1]]) if len(ok_r) else 0.3
    neg = np.sort(p[y == 0])
    t_fpr = float(neg[int(np.ceil(0.95 * len(neg))) - 1]) if len(neg) else 0.3
    return t_scam, min(max(t_recall, t_fpr, 0.25), t_scam)


def metrics_at(y, p, t) -> dict:
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(precision=round(float(precision_score(y, pred, zero_division=0)), 4),
                recall=round(float(recall_score(y, pred, zero_division=0)), 4),
                false_positive_rate=round(float(fp / max(fp + tn, 1)), 4),
                tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))


def summary(name, y, p, t_scam, t_susp) -> dict:
    return dict(model=name, roc_auc=round(float(roc_auc_score(y, p)), 4),
                pr_auc=round(float(average_precision_score(y, p)), 4),
                at_scam_threshold=metrics_at(y, p, t_scam), at_suspicious_threshold=metrics_at(y, p, t_susp),
                thresholds=dict(t_scam=round(t_scam, 4), t_suspicious=round(t_susp, 4)))


def verdicts(p, R, t_scam, t_susp) -> np.ndarray:
    """The deployed decision logic (thresholds + evidence rules), applied to a batch."""
    return np.array([decide(float(pi), {RULE_IDS[i] for i in SCAM_RULE_IDX if row[i]}, t_scam, t_susp)
                     for pi, row in zip(p, R, strict=True)])


def verdict_metrics(y, v) -> dict:
    y, v = np.asarray(y), np.asarray(v)
    scam_pred = v == "SCAM"
    tp = int((scam_pred & (y == 1)).sum())
    return dict(scam_precision=round(float(tp / max(scam_pred.sum(), 1)), 4),
                scam_recall=round(float(tp / max((y == 1).sum(), 1)), 4),
                flagged_recall=round(float((v[y == 1] != "SAFE").mean()), 4),
                legit_called_scam=round(float((v[y == 0] == "SCAM").mean()), 4),
                legit_safe=round(float((v[y == 0] == "SAFE").mean()), 4))


def mean_std_min(values) -> dict:
    return {"mean": round(float(np.mean(values)), 4), "std": round(float(np.std(values)), 4),
            "min": round(float(np.min(values)), 4)}


# ------------------------------------------------------------------------------ one split
def split(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    feedback, rest = df[df.source == "feedback"], df[df.source != "feedback"]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    dev_idx, test_idx = next(gss.split(rest, rest.label, rest.group))
    return pd.concat([rest.iloc[dev_idx], feedback]), rest.iloc[test_idx]


def run_split(df: pd.DataFrame, seed: int) -> dict:
    """Full protocol on one group split: OOF scores, final models, thresholds, test scores."""
    dev, test = split(df, seed)
    sw = lang_balanced_weights(dev) * np.where(dev.source.values == "feedback", FEEDBACK_WEIGHT, 1.0)
    R_dev, R_test = rules_matrix(dev.text, dev.sender), rules_matrix(test.text, test.sender)

    # Out-of-fold scores: every dev row is scored by models that never saw its template
    oof = {k: np.zeros(len(dev)) for k in ("text_logit", "nb", "rules_only")}
    for a, b in GroupKFold(n_splits=5).split(dev, dev.label, dev.group):
        d_a, d_b = dev.iloc[a], dev.iloc[b]
        text = make_text_model().fit(d_a.norm, d_a.label, clf__sample_weight=sw[a])
        oof["text_logit"][b] = text.decision_function(d_b.norm)
        oof["nb"][b] = make_nb().fit(d_a.norm, d_a.label).predict_proba(d_b.norm)[:, 1]
        oof["rules_only"][b] = make_rules_only().fit(R_dev[a], d_a.label).predict_proba(R_dev[b])[:, 1]

    text_model = make_text_model().fit(dev.norm, dev.label, clf__sample_weight=sw)
    stacker = make_stacker().fit(np.column_stack([oof["text_logit"], R_dev]), dev.label)
    nb = make_nb().fit(dev.norm, dev.label)
    rules_only = make_rules_only().fit(R_dev, dev.label)

    def hybrid(d, R):
        return stacker.predict_proba(np.column_stack([text_model.decision_function(d.norm), R]))[:, 1]

    dev_scores = {
        "naive_bayes_baseline": oof["nb"],
        "rules_only": oof["rules_only"],
        "text_model_only": sigmoid(oof["text_logit"]),
        HYBRID: stacker.predict_proba(np.column_stack([oof["text_logit"], R_dev]))[:, 1],
    }
    test_scores = {
        "naive_bayes_baseline": nb.predict_proba(test.norm)[:, 1],
        "rules_only": rules_only.predict_proba(R_test)[:, 1],
        "text_model_only": sigmoid(text_model.decision_function(test.norm)),
        HYBRID: hybrid(test, R_test),
    }
    thresholds = {name: pick_thresholds(dev.label.values, s) for name, s in dev_scores.items()}
    results = [summary(name, test.label.values, test_scores[name], *thresholds[name]) for name in test_scores]
    deployed = verdict_metrics(test.label.values, verdicts(test_scores[HYBRID], R_test, *thresholds[HYBRID]))
    return dict(dev=dev, test=test, text_model=text_model, stacker=stacker, hybrid=hybrid,
                test_scores=test_scores, thresholds=thresholds[HYBRID], results=results, deployed=deployed)


def aggregate(runs: list[dict]) -> dict:
    out = {}
    for i, name in enumerate(r["model"] for r in runs[0]["results"]):
        rows = [run["results"][i] for run in runs]
        out[name] = {
            "pr_auc": mean_std_min([r["pr_auc"] for r in rows]),
            "precision": mean_std_min([r["at_scam_threshold"]["precision"] for r in rows]),
            "recall": mean_std_min([r["at_scam_threshold"]["recall"] for r in rows]),
            "fpr": mean_std_min([r["at_scam_threshold"]["false_positive_rate"] for r in rows]),
            "recall_suspicious": mean_std_min([r["at_suspicious_threshold"]["recall"] for r in rows]),
        }
    out[DEPLOYED] = {k: mean_std_min([run["deployed"][k] for run in runs]) for k in runs[0]["deployed"]}
    return out


def score_handwritten(path: str, run: dict) -> tuple[pd.DataFrame, dict]:
    h = pd.read_csv(ROOT / path, keep_default_na=False)
    h["norm"] = h.text.map(for_model)
    R = rules_matrix(h.text, h.sender)
    p = run["hybrid"](h, R)
    h["p_scam"], h["verdict"] = p.round(4), verdicts(p, R, *run["thresholds"])
    metrics = {"all": verdict_metrics(h.label, h.verdict),
               **{f"lang={k}": verdict_metrics(g.label, g.verdict) for k, g in h.groupby("lang")}}
    return h, metrics


# ------------------------------------------------------------------------------ main
def main() -> None:
    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    df = pd.read_csv(ROOT / "data/processed/messages.csv", keep_default_na=False)
    df["norm"] = df.text.map(for_model)

    # With ~50 templates in a test set, one split's numbers depend on WHICH templates land there,
    # so the headline is mean +/- std over 5 splits. The seed-42 run is the one we deploy.
    runs = [run_split(df, seed) for seed in EVAL_SEEDS]
    main_run = runs[EVAL_SEEDS.index(SEED)]
    dev, test, stacker = main_run["dev"], main_run["test"], main_run["stacker"]
    t_scam, t_susp = main_run["thresholds"]

    slices = {}
    for col in ("lang", "source"):
        for key, g in test.assign(p=main_run["test_scores"][HYBRID]).groupby(col):
            if g.label.nunique() == 2:
                slices[f"{col}={key}"] = {**metrics_at(g.label.values, g.p.values, t_scam), "n": len(g)}

    handwritten = {}
    for name, path in HANDWRITTEN.items():
        h, metrics = score_handwritten(path, main_run)
        h[["text", "label", "lang", "note", "p_scam", "verdict"]].to_csv(REPORTS / f"{name}_predictions.csv",
                                                                        index=False)
        handwritten[name] = (h, metrics)

    joblib.dump(main_run["text_model"], MODELS / "text_model.joblib", compress=3)
    joblib.dump(stacker, MODELS / "stacker.joblib")
    (MODELS / "thresholds.json").write_text(json.dumps(
        {"t_scam": t_scam, "t_suspicious": t_susp, "target_precision": TARGET_PRECISION,
         "target_recall": TARGET_RECALL}, indent=2))

    report = dict(
        split=dict(dev=len(dev), test=len(test), dev_groups=int(dev.group.nunique()),
                   unseen_test_templates=int(test[test.source != "uci"].group.nunique())),
        thresholds=dict(t_scam=round(t_scam, 4), t_suspicious=round(t_susp, 4)),
        test_results=main_run["results"],
        across_5_group_splits=aggregate(runs),
        test_slices=slices,
        **{name: metrics for name, (_, metrics) in handwritten.items()},
        stacker_weights=dict(zip(["text_model_logit", *RULE_IDS], np.round(stacker.coef_[0], 3).tolist(),
                                 strict=True)),
    )
    (REPORTS / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    write_markdown(report, handwritten["challenge_set"][0])
    plot_pr(test.label.values, main_run["test_scores"])
    print_summary(report)


# ------------------------------------------------------------------------------ reporting
def _fmt(m: dict, digits: int = 3) -> str:
    return f"{m['mean']:.{digits}f} ± {m['std']:.{digits}f} ({m['min']:.{digits}f})"


def print_summary(rep: dict) -> None:
    th = rep["thresholds"]
    print(f"split: {rep['split']}")
    print(f"thresholds: SCAM >= {th['t_scam']:.3f}   SUSPICIOUS >= {th['t_suspicious']:.3f}")
    print("\nacross 5 group splits — mean ± std (worst):")
    for name, m in rep["across_5_group_splits"].items():
        if name == DEPLOYED:
            continue
        print(f"  {name:24s} precision {_fmt(m['precision'])} | recall {_fmt(m['recall'])} | "
              f"fpr {_fmt(m['fpr'], 4)}")
    print("  deployed verdicts: " + ", ".join(f"{k} {v['mean']:.3f}"
                                              for k, v in rep["across_5_group_splits"][DEPLOYED].items()))
    for name in HANDWRITTEN:
        print(f"{name}: {rep[name]['all']}")


def write_markdown(rep: dict, challenge: pd.DataFrame) -> None:
    L = ["# Evaluation report", "", f"Split (by template group): {rep['split']}", "",
         f"Thresholds: SCAM ≥ {rep['thresholds']['t_scam']:.3f}, SUSPICIOUS ≥ {rep['thresholds']['t_suspicious']:.3f}",
         "", "## Robustness: 5 different group splits (mean ± std; worst split in brackets)", "",
         "| Model | PR-AUC | Precision @SCAM | Recall @SCAM | FPR @SCAM |", "|---|---|---|---|---|"]
    robust = rep["across_5_group_splits"]
    for name, m in robust.items():
        if name != DEPLOYED:
            L.append(f"| {name} | {_fmt(m['pr_auc'])} | {_fmt(m['precision'])} | {_fmt(m['recall'])} | "
                     f"{_fmt(m['fpr'], 4)} |")
    L += ["", "## Deployed verdicts (thresholds + evidence rules), 5 splits", "",
          "| Metric | Mean ± std (worst) |", "|---|---|"]
    L += [f"| {k} | {_fmt(v)} |" for k, v in robust[DEPLOYED].items()]

    L += ["", "## Main split (seed 42) — ablation", "",
          "| Model | PR-AUC | Precision @SCAM | Recall @SCAM | FPR @SCAM | Recall @SUSPICIOUS | FPR @SUSPICIOUS |",
          "|---|---|---|---|---|---|---|"]
    for r in rep["test_results"]:
        a, b = r["at_scam_threshold"], r["at_suspicious_threshold"]
        L.append(f"| {r['model']} | {r['pr_auc']:.3f} | {a['precision']:.3f} | {a['recall']:.3f} | "
                 f"{a['false_positive_rate']:.4f} | {b['recall']:.3f} | {b['false_positive_rate']:.4f} |")
    L += ["", "## Hybrid model by slice (main split, SCAM threshold)", "",
          "| Slice | n | Precision | Recall | FPR |", "|---|---|---|---|---|"]
    L += [f"| {k} | {m['n']} | {m['precision']:.3f} | {m['recall']:.3f} | {m['false_positive_rate']:.4f} |"
          for k, m in rep["test_slices"].items()]

    for name, title in [("generalization_set", "Generalization set (82 hand-written messages of new types)"),
                        ("challenge_set", "Challenge set v2 (40 hand-written messages)")]:
        L += ["", f"## {title}", "",
              "| Slice | SCAM precision | SCAM recall | Flagged recall | Legit called SCAM | Legit SAFE |",
              "|---|---|---|---|---|---|"]
        L += [f"| {k} | {m['scam_precision']:.2f} | {m['scam_recall']:.2f} | {m['flagged_recall']:.2f} | "
              f"{m['legit_called_scam']:.2f} | {m['legit_safe']:.2f} |" for k, m in rep[name].items()]
    L += ["", "### Challenge set v2 verdicts", "", "| Verdict | True label | Note |", "|---|---|---|"]
    L += [f"| {r.verdict} | {'scam' if r.label else 'legit'} | {r.note} |" for r in challenge.itertuples()]

    L += ["", "## Stacker weights (log-odds added when a signal fires)", "", "| Feature | Weight |", "|---|---|"]
    L += [f"| {k} | {v:+.3f} |" for k, v in sorted(rep["stacker_weights"].items(), key=lambda kv: -abs(kv[1]))]
    (REPORTS / "report.md").write_text("\n".join(L) + "\n")


def plot_pr(y, scores: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for name, p in scores.items():
        pr, rc, _ = precision_recall_curve(y, p)
        ax.plot(rc, pr, label=f"{name} (AP {average_precision_score(y, p):.3f})", lw=1.8)
    ax.axhline(TARGET_PRECISION, ls="--", c="grey", lw=1)
    ax.text(0.02, TARGET_PRECISION + 0.003, f"precision target {TARGET_PRECISION:.0%}", fontsize=8, color="grey")
    ax.set(xlabel="Recall (share of scams caught)", ylabel="Precision (share of flags that are real scams)",
           title="Precision-recall on unseen test templates", ylim=(0.6, 1.01), xlim=(0, 1.01))
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(REPORTS / "pr_curve.png", dpi=150)


if __name__ == "__main__":
    main()
