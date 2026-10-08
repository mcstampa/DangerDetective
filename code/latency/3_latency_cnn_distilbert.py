"""STEP 3 — CNN and DistilBERT latency (run in the new .venv_latency).

CNN: per clip, audio loading (mono, 22.05 kHz, zero-padded to 5 s) + inference.
DistilBERT: per caption from step 2, tokenization + inference.
Also checks that the local models reproduce the Colab probabilities, so we know
the right models were loaded. Writes results/cnn_distilbert_latency.csv.
"""
import os, time, statistics
import numpy as np, pandas as pd, psutil, librosa
import tensorflow as tf
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PKG = os.path.dirname(os.path.abspath(__file__))
AUDIO, MODELS, RES = (os.path.join(PKG, d) for d in ("audio", "models", "results"))
SR, N = 22050, 5 * 22050
torch.set_num_threads(os.cpu_count())

proc = psutil.Process(os.getpid())
mb = lambda b: b / 2**20
rss = lambda: mb(proc.memory_info().rss)
peak = lambda: mb(getattr(proc.memory_info(), "peak_wset", proc.memory_info().rss))

sample = pd.read_csv(os.path.join(PKG, "latency_sample_200.csv"))
captions = pd.read_csv(os.path.join(RES, "conette_latency.csv"))
df = sample.merge(captions, on="slice_file_name")

# ---------------- CNN ----------------
ram0 = rss()
# The architecture is rebuilt exactly as saved (LV-18: 3 x [Conv1D k=8 -> MaxPool 8 -> Dropout 0.25],
# GlobalAveragePooling, Dense 64, sigmoid output) and only the trained weights are loaded from the
# .keras file. This avoids Keras version mismatches between Colab and the laptop.
L = tf.keras.layers
cnn = tf.keras.Sequential([
    tf.keras.Input(shape=(N, 1)),
    L.Conv1D(32, 8, activation="relu"), L.MaxPooling1D(8), L.Dropout(0.25),
    L.Conv1D(64, 8, activation="relu"), L.MaxPooling1D(8), L.Dropout(0.25),
    L.Conv1D(96, 8, activation="relu"), L.MaxPooling1D(8), L.Dropout(0.25),
    L.GlobalAveragePooling1D(),
    L.Dense(64, activation="relu"),
    L.Dense(1, activation="sigmoid"),
])
cnn.load_weights(os.path.join(MODELS, "cnn_lv18_wce_valauc.keras"))
ram_cnn = rss() - ram0
in_shape = (1,) + tuple(d for d in cnn.input_shape[1:])

def cnn_prob(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    y = np.pad(y[:N], (0, max(0, N - len(y))))
    out = cnn(y.reshape(in_shape).astype("float32"), training=False).numpy().ravel()
    return float(out[-1])  # danger probability (sigmoid output, or the danger column of a softmax)

for f in df.slice_file_name[:3]:  # warm-up
    cnn_prob(os.path.join(AUDIO, f))
cnn_ms, cnn_p = [], []
for f in df.slice_file_name:
    t = time.perf_counter(); p = cnn_prob(os.path.join(AUDIO, f)); cnn_ms.append((time.perf_counter() - t) * 1000); cnn_p.append(p)
peak_cnn = peak()

# ---------------- DistilBERT ----------------
ram1 = rss()
bert_dir = os.path.join(MODELS, "distilbert_stratified_wce")
tok = AutoTokenizer.from_pretrained(bert_dir)
bert = AutoModelForSequenceClassification.from_pretrained(bert_dir).eval()
ram_bert = rss() - ram1

def bert_prob(text):
    enc = tok(text, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        return float(torch.softmax(bert(**enc).logits, dim=-1)[0, 1])

for c in df.caption[:3]:  # warm-up
    bert_prob(c)
bert_ms, bert_p = [], []
for c in df.caption:
    t = time.perf_counter(); p = bert_prob(c); bert_ms.append((time.perf_counter() - t) * 1000); bert_p.append(p)

# ---------------- Save + summary ----------------
df["cnn_ms"], df["cnn_prob_local"] = cnn_ms, cnn_p
df["distilbert_ms"], df["distilbert_prob_local"] = bert_ms, bert_p
df.to_csv(os.path.join(RES, "cnn_distilbert_latency.csv"), index=False)

s = lambda x: f"mean {statistics.mean(x):.1f} ms, SD {statistics.stdev(x):.1f} ms, median {statistics.median(x):.1f} ms"
cnn_diff = np.abs(df.cnn_prob_local - df.cnn_prob_colab).max()
caption_note = "(captions regenerated locally, so small differences are expected)"
bert_agree = ((df.distilbert_prob_local >= 0.5) == (df.distilbert_prob_colab >= 0.5)).mean()
summary = (f"TensorFlow {tf.__version__}, torch {torch.__version__}, {torch.get_num_threads()} threads\n"
           f"CNN: {s(cnn_ms)} (n={len(cnn_ms)}); RAM +{ram_cnn:.0f} MB for the model\n"
           f"DistilBERT: {s(bert_ms)}; RAM +{ram_bert:.0f} MB for the model\n"
           f"Process peak RAM: {peak():.0f} MB\n"
           f"CHECK CNN: max |local - Colab| probability = {cnn_diff:.4f} (should be < 0.01)\n"
           f"CHECK DistilBERT: same decision as Colab for {bert_agree:.0%} of clips {caption_note}\n")
print(summary)
open(os.path.join(RES, "cnn_distilbert_summary.txt"), "w").write(summary)
