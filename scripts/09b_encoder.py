"""Step 9b: fine-tuned encoder baseline.

A domain-pretrained encoder is the strongest system in the comparison that is
fully independent of the label source, which is why its numbers carry the
context-scaling claim rather than the language model's.
"""
import json, os, sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT

MODELS = os.environ.get("ENC_MODELS",
    "nlpaueb/legal-bert-base-uncased,distilroberta-base").split(",")
CONTEXTS = os.environ.get("ENC_CONTEXTS", "SENT,W256,W1024,W4096").split(",")
# Batch 8 rather than 16: on unified memory the GPU competes with whatever
# else is resident, and 16 at 512 tokens exhausted it during the corpus pass.
EPOCHS, LR, BS, MAXLEN = 4, 2e-5, int(os.environ.get('ENC_BS', 8)), 512
CLIP = 1.0
DEV = os.environ.get("ENC_DEVICE") or (
    "mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(13)


def encode(tok, issues, ctxs):
    enc = tok(list(issues), list(ctxs), truncation="only_second",
              max_length=MAXLEN, padding="max_length", return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


def run(model_name, df, ctx, device=DEV):
    tok = AutoTokenizer.from_pretrained(model_name)
    tr = df[df["split"] == "DEV"]
    col = f"ctx_{ctx}"
    xi, xm = encode(tok, tr["issue"].astype(str), tr[col].astype(str))
    y = torch.tensor(tr["y"].values)
    ds = TensorDataset(xi, xm, y)
    # A trailing batch of one or two items is where the MPS backend was
    # producing non-finite losses, and it contributes almost nothing to the
    # gradient, so it is dropped whenever there is more than one full batch.
    dl = DataLoader(ds, batch_size=BS, shuffle=True, drop_last=len(ds) > BS)

    dev = device
    m = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=LR)
    w = torch.tensor([1.0, float((y == 0).sum()) / max(1, float((y == 1).sum()))]).to(dev)
    lossf = torch.nn.CrossEntropyLoss(weight=w)
    m.train()
    for ep in range(EPOCHS):
        tot, seen, skipped = 0.0, 0, 0
        for bi, bm, by in dl:
            opt.zero_grad()
            out = m(input_ids=bi.to(dev), attention_mask=bm.to(dev))
            loss = lossf(out.logits, by.to(dev))
            if not torch.isfinite(loss):
                skipped += 1
                continue
            loss.backward()
            # The MPS backend can return non-finite gradients for this model
            # even when the loss itself is finite. Stepping on them poisons
            # every weight, and the next batch's loss is then NaN forever, so
            # the guard has to be on the gradient rather than on the loss.
            gnorm = torch.nn.utils.clip_grad_norm_(m.parameters(), CLIP)
            if not torch.isfinite(gnorm):
                opt.zero_grad(set_to_none=True)
                skipped += 1
                continue
            opt.step()
            tot += float(loss.detach())
            seen += 1
        print(f"    {model_name.split('/')[-1]} {ctx} [{dev}] epoch {ep+1} "
              f"loss {tot/max(1,seen):.4f}"
              + (f" ({skipped} non-finite batches skipped)" if skipped else ""),
              flush=True)
        # If the very first epoch produced nothing usable, the accelerator is
        # the problem rather than the data. Retry once on CPU and say so.
        if ep == 0 and seen < max(1, 0.25 * (seen + skipped)) and dev != "cpu":
            print(f"    {model_name.split('/')[-1]} {ctx}: no finite batch on "
                  f"{dev}; retrying on cpu", flush=True)
            return run(model_name, df, ctx, device="cpu")

    m.eval()
    res = []
    for split in ("TEST", "TEMPORAL", "COURT"):
        te = df[df["split"] == split]
        if not len(te):
            continue
        ti, tm = encode(tok, te["issue"].astype(str), te[col].astype(str))
        preds = []
        with torch.no_grad():
            for i in range(0, len(ti), 32):
                out = m(input_ids=ti[i:i+32].to(dev), attention_mask=tm[i:i+32].to(dev))
                preds.append(out.logits.argmax(-1).cpu().numpy())
        p = np.concatenate(preds)
        res.append({"model": model_name.split("/")[-1], "context": ctx,
                    "split": split, "n": int(len(te)),
                    "acc": float(accuracy_score(te["y"], p)),
                    "macro_f1": float(f1_score(te["y"], p, average="macro",
                                               zero_division=0)),
                    "f1_unresolved": float(f1_score(te["y"], p, pos_label=1,
                                                    zero_division=0))})
        te_out = te.assign(pred=p)
        te_out[["item_id", "stratum", "y", "pred"]].to_csv(
            os.path.join(OUT, f"09b_pred_{model_name.split('/')[-1]}_{ctx}_{split}.csv"),
            index=False)
    del m
    torch.mps.empty_cache() if dev == "mps" else None
    return res


def main():
    df = pd.read_parquet(os.path.join(OUT, "09_items_with_ctx.parquet"))
    print(f"device {DEV}; items {len(df):,}")
    out = []
    for mn in MODELS:
        for ctx in CONTEXTS:
            try:
                out += run(mn, df, ctx)
            except Exception as e:
                print(f"!! {mn} {ctx}: {type(e).__name__}: {e}", flush=True)
    r = pd.DataFrame(out)
    r.to_csv(os.path.join(OUT, "09b_encoder_results.csv"), index=False)
    if len(r):
        print(r.pivot_table(index=["model", "context"], columns="split",
                            values="macro_f1").round(3).to_string())


if __name__ == "__main__":
    main()
