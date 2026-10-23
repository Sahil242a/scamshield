"""Fine-tune a multilingual transformer as the text model (run on Colab / a GPU).

    pip install torch transformers accelerate
    python finetune_transformer.py --model google/muril-base-cased --epochs 3
    python finetune_transformer.py --model ai4bharat/IndicBERTv2-MLM-only
    python finetune_transformer.py --model distilbert-base-multilingual-cased

Which model?
  * MuRIL (Google): pre-trained on 17 Indian languages *including transliterated
    (Romanised) text* — best fit for Hinglish like "aapka account band ho jayega".
  * IndicBERT v2 (AI4Bharat): strong on Devanagari Hindi; less exposure to Romanised Hindi.
  * DistilBERT multilingual: smallest/fastest (~135M params, 6 layers), fine for a CPU demo.

What this script keeps identical to train.py (so the comparison is fair):
  * the same group split (unseen templates in test) and language-balanced weights
  * the same rule features + prior-regularised stacker on top
  * the same threshold rules (SCAM at >= 98% precision, SUSPICIOUS at >= 97% recall)

Output: models/transformer/ (weights + tokenizer). Set SCAMSHIELD_TEXT_MODEL=transformer
before starting the API to use it as the text model; the rules and stacker stay the same.
For speed, the stacker here is trained on held-out *validation* logits rather than 5-fold
OOF (5 transformer fine-tunes is expensive) — say this if asked.

--smoke-test builds a tiny random BERT (no download) to check the code path end to end.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset

from scamshield.normalize import clean
from train import ROOT, SEED, lang_balanced_weights, make_stacker, metrics_at, pick_thresholds, rules_matrix, summary


class SMSDataset(Dataset):
    def __init__(self, enc, labels, weights):
        self.enc, self.labels, self.weights = enc, labels, weights

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(int(self.labels[i]))
        item["sample_weight"] = torch.tensor(float(self.weights[i]))
        return item


def main():
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        BertConfig,
        BertForSequenceClassification,
        BertTokenizerFast,
        Trainer,
        TrainingArguments,
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/muril-base-cased")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=96)  # SMS are short; 96 sub-word tokens covers ~99%
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(ROOT / "data/processed/messages.csv", keep_default_na=False)
    dev_idx, test_idx = next(GroupShuffleSplit(1, test_size=0.15, random_state=SEED).split(df, df.label, df.group))
    dev, test = df.iloc[dev_idx], df.iloc[test_idx]
    tr_idx, val_idx = next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED).split(dev, dev.label, dev.group))
    train, val = dev.iloc[tr_idx], dev.iloc[val_idx]
    if args.smoke_test:
        train, val, test = (train.sample(300, random_state=0), val.sample(150, random_state=0),
                            test.sample(150, random_state=0))

    if args.smoke_test:
        vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + sorted({c for t in df.text for c in clean(t)})
        vf = ROOT / "models" / "smoke_vocab.txt"
        vf.parent.mkdir(exist_ok=True)
        vf.write_text("\n".join(vocab))
        tok = BertTokenizerFast(vocab_file=str(vf), do_lower_case=False, tokenize_chinese_chars=False)
        model = BertForSequenceClassification(BertConfig(vocab_size=len(vocab), hidden_size=64, num_hidden_layers=2,
                                                         num_attention_heads=2, intermediate_size=128, num_labels=2))
    else:
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)

    # Transformers see the cleaned raw text (URLs/numbers kept): sub-word tokenisers
    # can use the shape of a link (".xyz", "bit.ly") themselves.
    enc = lambda d: tok([clean(t) for t in d.text], truncation=True, max_length=args.max_len, padding="max_length")  # noqa
    ds_train = SMSDataset(enc(train), train.label.values, lang_balanced_weights(train))
    ds_val = SMSDataset(enc(val), val.label.values, np.ones(len(val)))

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            w = inputs.pop("sample_weight")
            out = model(**{k: v for k, v in inputs.items() if k != "labels"})
            loss = torch.nn.functional.cross_entropy(out.logits, inputs["labels"], reduction="none")
            loss = (loss * w).sum() / w.sum()
            return (loss, out) if return_outputs else loss

    out_dir = ROOT / "models" / ("transformer_smoke" if args.smoke_test else "transformer")
    targs = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"), num_train_epochs=1 if args.smoke_test else args.epochs,
        learning_rate=args.lr, per_device_train_batch_size=args.batch, per_device_eval_batch_size=64,
        warmup_ratio=0.1, weight_decay=0.01, eval_strategy="epoch", save_strategy="no", logging_steps=50,
        report_to=[], seed=SEED, remove_unused_columns=False, fp16=torch.cuda.is_available(),
    )
    trainer = WeightedTrainer(model=model, args=targs, train_dataset=ds_train, eval_dataset=ds_val)
    trainer.train()

    @torch.no_grad()
    def logits(d):
        model.eval()
        e = tok([clean(t) for t in d.text], truncation=True, max_length=args.max_len, padding=True, return_tensors="pt")
        e = {k: v.to(model.device) for k, v in e.items()}
        z = []
        for i in range(0, len(d), 64):
            z.append(model(**{k: v[i:i + 64] for k, v in e.items()}).logits.cpu())
        z = torch.cat(z)
        return (z[:, 1] - z[:, 0]).numpy()  # scam log-odds, same role as the TF-IDF LR logit

    z_val, z_test = logits(val), logits(test)
    R_val, R_test = rules_matrix(val.text, val.sender), rules_matrix(test.text, test.sender)
    stacker = make_stacker().fit(np.column_stack([z_val, R_val]), val.label)
    p_val = stacker.predict_proba(np.column_stack([z_val, R_val]))[:, 1]
    t_scam, t_susp = pick_thresholds(val.label.values, p_val)
    p_test = stacker.predict_proba(np.column_stack([z_test, R_test]))[:, 1]
    sig = 1 / (1 + np.exp(-z_test))
    report = {
        "model": args.model if not args.smoke_test else "tiny-random-bert (smoke test)",
        "transformer_only": summary("transformer_only", test.label.values, sig,
                                    *pick_thresholds(val.label.values, 1 / (1 + np.exp(-z_val)))),
        "hybrid": summary("transformer + rules", test.label.values, p_test, t_scam, t_susp),
    }
    print(json.dumps(report, indent=2))

    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    import joblib
    joblib.dump(stacker, out_dir / "stacker.joblib")
    (out_dir / "thresholds.json").write_text(json.dumps({"t_scam": t_scam, "t_suspicious": t_susp,
                                                          "max_len": args.max_len}, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    print("saved ->", out_dir, "| test @SCAM:", metrics_at(test.label.values, p_test, t_scam))


if __name__ == "__main__":
    main()
