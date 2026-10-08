# =============================================================================
# YAMNet baseline (Reviewer 2, Comment 3) — run in Google Colab (GPU or CPU).
# Same protocol as the CNN:
#   train on the CNN training split (1316 clips), choose C on the validation split
#   (330 clips) by AUC-ROC, test on the 8419 UrbanSound8K clips, threshold 0.5.
# YAMNet (pretrained on AudioSet, frozen) -> mean 1024-d embedding per clip
#   -> logistic regression with balanced class weights.
# Upload to the Colab session first:
#   train_manifest_original_only.csv, val_manifest.csv, urbansound8k_test_8419.csv
# Runtime: roughly 15-30 min (mostly reading ~10,000 audio files from Drive).
# =============================================================================

# --- 1. Setup
from google.colab import drive
drive.mount("/content/drive")
import os, time, numpy as np, pandas as pd, librosa
import tensorflow as tf, tensorflow_hub as hub
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix, average_precision_score

ESC50_ROOT = "/content/drive/MyDrive/Colab Notebooks/ESC-50-master/audio"
US8K_ROOT = "/content/drive/MyDrive/Colab Notebooks/UrbanSound8k/audio"
SCREAM_DIR = "/content/drive/MyDrive/MyAudiosetData/screaming_final_5s_wav"
OUT = "/content/drive/MyDrive/Colab Notebooks/FINALS/NEW_DATASET/YAMNet_baseline"
os.makedirs(OUT, exist_ok=True)

yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

# --- 2. Paths (same resolution as the CNN notebook)
def path_of(r):
    if r.dataset_origin == "ESC50":
        return f"{ESC50_ROOT}/{r.filename}"
    if r.dataset_origin == "UrbanSound8K":
        return f"{US8K_ROOT}/fold{int(r.fold)}/{r.filename}"
    return f"{SCREAM_DIR}/{r.filename}"

train = pd.read_csv("/content/train_manifest_original_only.csv")
val = pd.read_csv("/content/val_manifest.csv")
test = pd.read_csv("/content/urbansound8k_test_8419.csv")
train["path"], val["path"] = train.apply(path_of, axis=1), val.apply(path_of, axis=1)
test["path"] = [f"{US8K_ROOT}/fold{f}/{n}" for f, n in zip(test.fold, test.slice_file_name)]
train["y"], val["y"], test["y"] = train.label_int, val.label_int, test.y_true

# --- 3. Embeddings: mono 16 kHz (YAMNet's input), mean over its 0.96 s frames
def embed(paths, name):
    cache = f"{OUT}/emb_{name}.npy"
    if os.path.exists(cache):
        return np.load(cache)
    E = np.zeros((len(paths), 1024), dtype=np.float32)
    for i, p in enumerate(paths):
        y, _ = librosa.load(p, sr=16000, mono=True)
        if len(y) < 16000:  # YAMNet needs at least ~1 s; pad very short clips
            y = np.pad(y, (0, 16000 - len(y)))
        _, emb, _ = yamnet(y.astype(np.float32))
        E[i] = emb.numpy().mean(axis=0)
        if i % 500 == 0:
            print(name, i, "/", len(paths))
    np.save(cache, E)
    return E

Xtr, Xva, Xte = embed(train.path, "train"), embed(val.path, "val"), embed(test.path, "test")

# --- 4. Classifier: C chosen on validation AUC-ROC (as the CNN checkpoint)
best = None
for C in [0.01, 0.1, 1, 10]:
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000).fit(Xtr, train.y)
    auc = roc_auc_score(val.y, clf.predict_proba(Xva)[:, 1])
    print(f"C={C}: validation AUC {auc:.3f}")
    if best is None or auc > best[0]:
        best = (auc, C, clf)
auc, C, clf = best

# --- 5. Test on UrbanSound8K (8419 clips)
p = clf.predict_proba(Xte)[:, 1]
yhat = (p >= 0.5).astype(int)
pr, rc, f1, _ = precision_recall_fscore_support(test.y, yhat, labels=[1], zero_division=0)
tn, fp, fn, tp = confusion_matrix(test.y, yhat, labels=[0, 1]).ravel()
res = (f"YAMNet + logistic regression (C={C}, val AUC {auc:.3f})\n"
       f"UrbanSound8K (8419): recall {rc[0]:.3f}, precision {pr[0]:.3f}, F1 {f1[0]:.3f}, "
       f"PR-AUC {average_precision_score(test.y, p):.3f}\n"
       f"TN {tn}  FP {fp}  FN {fn}  TP {tp}\n")
print(res)
open(f"{OUT}/yamnet_results.txt", "w").write(res)
test.assign(yamnet_prob=p, yamnet_pred=yhat).drop(columns="path").to_csv(f"{OUT}/yamnet_urbansound8k_predictions.csv", index=False)

# --- 6. Optional: latency per clip on this runtime's CPU (embedding + classifier)
with tf.device("/CPU:0"):
    t = []
    for path in test.path[:100]:
        y, _ = librosa.load(path, sr=16000, mono=True)
        s = time.perf_counter(); e = yamnet(y.astype(np.float32))[1].numpy().mean(0); clf.predict_proba(e[None]); t.append((time.perf_counter() - s) * 1000)
print(f"YAMNet latency on Colab CPU (after loading audio): mean {np.mean(t[3:]):.1f} ms per clip")
