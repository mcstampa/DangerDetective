"""Final-system evaluation on UrbanSound8K (8419 clips): reproduces Tables 9, 10 and 12.

Part of the code for "Danger Detection through Audio Captioning and Classification Fusion"
(Electronics, MDPI). Paper sections: 4.3 (Table 9), 4.4 (Tables 10 and 12).

Final system: LV-18 CNN (weighted cross-entropy, validation AUC-ROC checkpoint) +
DistilBERT (alignment-stratified split, weighted cross-entropy), custom_uncertain rule:
    if LOW <= p_CNN <= HIGH: decision = DistilBERT, else decision = CNN.
The zone (0.17-0.60) is fixed in advance on the CNN's internal test subset
(see derive_fusion_thresholds.py); nothing here is tuned on UrbanSound8K.

Inputs (prediction files saved by the training notebooks, joined on slice_file_name):
    --cnn     CNN predictions        (columns: slice_file_name, y_true, prob_danger_risk)
    --bert    DistilBERT predictions (columns: slice_file_name, distilbert_danger_prob)
    --sbert   SBERT scorer output    (columns: slice_file_name, sbert_caption_pred)   [optional, Table 9]
    --yamnet  YAMNet baseline        (columns: slice_file_name, yamnet_prob)          [optional, Table 10]

Usage:
    python final_fusion_evaluation.py --cnn cnn.csv --bert distilbert.csv --sbert sbert.csv
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, cohen_kappa_score

LOW, HIGH = 0.17, 0.60
N_BOOT, SEED = 2000, 42


def counts(y, yhat):
    tp = int(((y == 1) & (yhat == 1)).sum()); fn = int(((y == 1) & (yhat == 0)).sum())
    tn = int(((y == 0) & (yhat == 0)).sum()); fp = int(((y == 0) & (yhat == 1)).sum())
    return tn, fp, fn, tp


def prf(y, yhat):
    tn, fp, fn, tp = counts(y, yhat)
    r = tp / (tp + fn) if tp + fn else 0.0
    p = tp / (tp + fp) if tp + fp else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return r, p, f


def bootstrap_ci(y, preds, rng):
    """95% percentile CIs of recall/precision/F1 for each system, and of recall differences."""
    n = len(y)
    draws = {k: [] for k in preds}
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        for k, yhat in preds.items():
            draws[k].append(prf(y[idx], yhat[idx]))
    ci = {k: np.percentile(np.array(v), [2.5, 97.5], axis=0) for k, v in draws.items()}
    return ci, draws


def fused(p_cnn, p_bert, low=LOW, high=HIGH):
    zone = (p_cnn >= low) & (p_cnn <= high)
    yhat = np.where(zone, p_bert >= 0.5, p_cnn >= 0.5).astype(int)
    score = np.where(zone, p_bert, p_cnn)
    return yhat, score, zone


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--cnn", required=True); a.add_argument("--bert", required=True)
    a.add_argument("--sbert"); a.add_argument("--yamnet")
    args = a.parse_args()

    df = (pd.read_csv(args.cnn, low_memory=False)[["slice_file_name", "y_true", "prob_danger_risk"]]
            .merge(pd.read_csv(args.bert, low_memory=False)[["slice_file_name", "distilbert_danger_prob"]],
                   on="slice_file_name"))
    print(f"Clips: {len(df)} ({int(df.y_true.sum())} danger / {int((df.y_true == 0).sum())} no-danger)\n")
    y = df.y_true.to_numpy().astype(int)
    p_cnn, p_bert = df.prob_danger_risk.to_numpy(), df.distilbert_danger_prob.to_numpy()

    y_cnn, y_bert = (p_cnn >= 0.5).astype(int), (p_bert >= 0.5).astype(int)
    y_fus, s_fus, zone = fused(p_cnn, p_bert)
    systems = {"CNN alone": (y_cnn, p_cnn), "DistilBERT alone": (y_bert, p_bert), "Fusion (custom_uncertain)": (y_fus, s_fus)}
    if args.yamnet:
        ym = df[["slice_file_name"]].merge(pd.read_csv(args.yamnet)[["slice_file_name", "yamnet_prob"]], on="slice_file_name")
        p_y = ym.yamnet_prob.to_numpy()
        systems["YAMNet + LR (baseline)"] = ((p_y >= 0.5).astype(int), p_y)

    rng = np.random.default_rng(SEED)
    ci, draws = bootstrap_ci(y, {k: v[0] for k, v in systems.items()}, rng)

    # ---- Table 10
    print(f"Table 10 - zone {LOW:.2f}-{HIGH:.2f}; 95% bootstrap CIs ({N_BOOT} resamples, seed {SEED})")
    for k, (yhat, score) in systems.items():
        r, p, f = prf(y, yhat); tn, fp, fn, tp = counts(y, yhat); lo, hi = ci[k]
        trig = f"{int(zone.sum())} ({zone.mean():.1%})" if k.startswith("Fusion") else ("all" if k.startswith("Distil") else "-")
        print(f"  {k:27s} R {r:.2f} [{lo[0]:.2f}, {hi[0]:.2f}]  P {p:.2f} [{lo[1]:.2f}, {hi[1]:.2f}]  "
              f"F1 {f:.2f} [{lo[2]:.2f}, {hi[2]:.2f}]  PR-AUC {average_precision_score(y, score):.2f}  "
              f"TN/FP {tn}/{fp}  FN/TP {fn}/{tp}  triggered {trig}")

    rf = np.array([d[0] for d in draws["Fusion (custom_uncertain)"]])
    for base in ["CNN alone", "DistilBERT alone"]:
        diff = rf - np.array([d[0] for d in draws[base]])
        print(f"  Recall difference fusion - {base}: {prf(y, y_fus)[0] - prf(y, systems[base][0])[0]:.3f} "
              f"[{np.percentile(diff, 2.5):.3f}, {np.percentile(diff, 97.5):.3f}]")

    # corrections inside the zone, and confident misses below the zone
    z = zone
    corrected = int(((y_cnn != y) & (y_fus == y) & z).sum()); introduced = int(((y_cnn == y) & (y_fus != y) & z).sum())
    miss_fixed = int(((y == 1) & (y_cnn == 0) & (y_fus == 1) & z).sum())
    print(f"\nInside the zone: {corrected} decisions corrected ({miss_fixed} missed dangers, "
          f"{corrected - miss_fixed} false alarms), {introduced} errors introduced")
    conf_miss = int(((y == 1) & (p_cnn < LOW)).sum())
    print(f"Confident misses (danger clips with p_CNN < {LOW}): {conf_miss} of {int(y.sum())} "
          f"({conf_miss / y.sum():.1%}); recall ceiling {1 - conf_miss / y.sum():.2f}")

    # ---- Table 12
    print("\nTable 12 - danger recall / activation rate (%) when the zone bounds are shifted")
    highs = [0.60, 0.63, 0.68, 0.73, 0.78, 0.83]
    print("  lower \\ upper " + "  ".join(f"{h:>11.2f}" for h in highs))
    for lo_ in [0.07, 0.12, 0.17, 0.22, 0.27]:
        cells = []
        for hi_ in highs:
            yh, _, zz = fused(p_cnn, p_bert, lo_, hi_)
            cells.append(f"{prf(y, yh)[0]:.2f} / {100 * zz.mean():4.1f}")
        print(f"  {lo_:>13.2f} " + "  ".join(cells))

    # ---- Table 9
    if args.sbert:
        sb = df.merge(pd.read_csv(args.sbert, low_memory=False)[["slice_file_name", "sbert_caption_pred"]]
                        .drop_duplicates("slice_file_name"), on="slice_file_name")
        ys, yb, yt = sb.sbert_caption_pred.to_numpy().astype(int), (sb.distilbert_danger_prob >= 0.5).astype(int).to_numpy(), sb.y_true.to_numpy().astype(int)
        print("\nTable 9 - semantic classification against the original class labels")
        for k, yh in [("SBERT danger scorer", ys), ("DistilBERT", yb)]:
            r, p, f = prf(yt, yh); tn, fp, fn, tp = counts(yt, yh)
            print(f"  {k:20s} R {r:.2f}  P {p:.2f}  F1 {f:.2f}  TP/FN {tp}/{fn}  TN/FP {tn}/{fp}")
        print(f"  Agreement {np.mean(ys == yb):.1%}, Cohen's kappa {cohen_kappa_score(ys, yb):.2f} (n={len(sb)})")


if __name__ == "__main__":
    main()
