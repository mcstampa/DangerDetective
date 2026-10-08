"""STEP 4 — Expected computational cost per clip (run in either environment).

Combines the measured mean latencies (steps 2 and 3) with the semantic-branch
activation rates measured on UrbanSound8K (8419 clips) for each activation zone:
    fusion cost per clip = CNN + activation x (CoNeTTE + DistilBERT)
"""
import os
import pandas as pd

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
con = pd.read_csv(os.path.join(RES, "conette_latency.csv"))
cb = pd.read_csv(os.path.join(RES, "cnn_distilbert_latency.csv"))
cnn, cap, bert = cb.cnn_ms.mean(), con.conette_ms.mean(), cb.distilbert_ms.mean()

# Activation rate (%) on UrbanSound8K, lower bound (rows) x upper bound (columns)
ACT = {0.07: {0.60: 51.2, 0.63: 53.7, 0.68: 57.9, 0.73: 62.3, 0.78: 67.2, 0.83: 71.2},
       0.12: {0.60: 48.2, 0.63: 50.7, 0.68: 54.9, 0.73: 59.3, 0.78: 64.2, 0.83: 68.2},
       0.17: {0.60: 44.7, 0.63: 47.1, 0.68: 51.3, 0.73: 55.7, 0.78: 60.6, 0.83: 64.7},
       0.22: {0.60: 39.5, 0.63: 42.0, 0.68: 46.2, 0.73: 50.5, 0.78: 55.5, 0.83: 59.5},
       0.27: {0.60: 33.8, 0.63: 36.3, 0.68: 40.5, 0.73: 44.8, 0.78: 49.8, 0.83: 53.8}}

semantic = cap + bert
print(f"Mean latency per clip: CNN {cnn:.1f} ms | CoNeTTE {cap:.0f} ms | DistilBERT {bert:.1f} ms\n")
print(f"CNN alone:               {cnn:.0f} ms per clip")
print(f"Semantic branch on all:  {semantic:.0f} ms per clip (CoNeTTE + DistilBERT)")
f = cnn + 0.447 * semantic
print(f"Fusion, zone 0.17-0.60:  {f:.0f} ms per clip  ({f / semantic:.0%} of always-semantic)\n")

grid = pd.DataFrame({lo: {hi: round(cnn + a / 100 * semantic) for hi, a in row.items()} for lo, row in ACT.items()}).T
grid.index.name, grid.columns.name = "lower", "upper"
print("Expected fusion cost per clip (ms) across activation zones:\n", grid.to_string())
grid.to_csv(os.path.join(RES, "cost_per_clip_grid.csv"))
