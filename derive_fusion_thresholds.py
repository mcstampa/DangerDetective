"""Derive the uncertainty-zone bounds of the custom_uncertain fusion rule.

The bounds are the median CNN danger probabilities of the false-negative (lower bound)
and false-positive clips on the CNN's internal held-out test subset
(412 clips), which is disjoint from the UrbanSound8K evaluation set. The upper bound is then set to 0.60,
below the false-positive median (0.73), to limit activation of the semantic branch.

Input: test_predictions_detailed.csv, saved by the CNN training notebook in
       <run_dir>/selected_by_val_auc_roc/ (columns: prob_danger_risk, prediction_type).
"""
import sys
from decimal import Decimal, ROUND_HALF_UP
import pandas as pd

r2 = lambda x: float(Decimal(f"{x:.4f}").quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

path = sys.argv[1] if len(sys.argv) > 1 else "test_predictions_detailed.csv"
df = pd.read_csv(path)

summary = (df.groupby("prediction_type")["prob_danger_risk"]
             .agg(n_samples="count", median_prob_danger="median", mean_prob_danger="mean")
             .reindex(["TP", "TN", "FP", "FN"]))
print(summary.round(4))
summary.to_csv("cnn_median_mean_probability_by_tp_tn_fp_fn.csv")

low = r2(summary.loc["FN", "median_prob_danger"])       # 0.17
fp_median = r2(summary.loc["FP", "median_prob_danger"])  # 0.73
HIGH = 0.60  # set below the FP median to limit activation of the semantic branch (Section 3.4)
p = df["prob_danger_risk"]
print(f"\nLower bound = FN median = {low:.2f}; FP median = {fp_median:.2f}; upper bound set to {HIGH:.2f}")
print(f"Activation on this subset: {((p >= low) & (p <= fp_median)).mean():.1%} with [FN median, FP median], "
      f"{((p >= low) & (p <= HIGH)).mean():.1%} with [{low:.2f}, {HIGH:.2f}]")
