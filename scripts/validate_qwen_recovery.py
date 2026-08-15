"""Compare reconstructed-input Qwen predictions with the archived baseline."""
import argparse

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archived", default="out/09c_llm_preds.qwen.bak.csv")
    parser.add_argument(
        "--recovered", default="out/09c_llm_preds.qwen.recovered.csv"
    )
    parser.add_argument("--min-prediction-match", type=float, default=0.98)
    parser.add_argument("--min-margin-correlation", type=float, default=0.99)
    args = parser.parse_args()

    archived = pd.read_csv(args.archived)
    recovered = pd.read_csv(args.recovered)
    keys = ["item_id", "context"]
    merged = archived.merge(
        recovered, on=keys, how="outer", suffixes=("_archived", "_recovered"),
        indicator=True,
    )
    if len(merged) != 744 or not (merged["_merge"] == "both").all():
        raise SystemExit(
            f"Qwen fingerprint row mismatch: {len(merged)} merged rows; "
            f"{merged['_merge'].value_counts().to_dict()}"
        )
    for column in ("stratum", "split", "y"):
        if not (merged[f"{column}_archived"] == merged[f"{column}_recovered"]).all():
            raise SystemExit(f"Qwen fingerprint metadata mismatch in {column}")

    prediction_match = float(
        (merged["pred_archived"] == merged["pred_recovered"]).mean()
    )
    sign_match = float(
        (np.sign(merged["margin_archived"]) == np.sign(merged["margin_recovered"])).mean()
    )
    correlation = float(
        merged[["margin_archived", "margin_recovered"]].corr().iloc[0, 1]
    )
    median_delta = float(
        (merged["margin_archived"] - merged["margin_recovered"]).abs().median()
    )
    print(
        f"Qwen fingerprint: prediction_match={prediction_match:.6f} "
        f"sign_match={sign_match:.6f} margin_correlation={correlation:.6f} "
        f"median_abs_margin_delta={median_delta:.6f}"
    )
    if (prediction_match < args.min_prediction_match
            or sign_match < args.min_prediction_match
            or correlation < args.min_margin_correlation):
        raise SystemExit("Qwen fingerprint FAILED; recovered prompts are not trusted")
    print("Qwen fingerprint PASSED")


if __name__ == "__main__":
    main()
