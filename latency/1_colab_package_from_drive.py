# =============================================================================
# STEP 1 — Run in Google Colab (one cell, or split at the "# ---" lines).
# Packages everything the latency test needs into ONE zip on your Drive:
#   - the 200 selected UrbanSound8K clips
#   - the final CNN (WCE LV-18, selected by val AUC-ROC)
#   - the final DistilBERT (alignment-stratified, weighted CE)
# Before running: upload latency_sample_200.csv to the Colab session
# (left panel -> Files -> Upload), or put it in your Drive and change SAMPLE_CSV.
# =============================================================================

# --- 1. Mount Drive
from google.colab import drive
drive.mount("/content/drive")

import glob, os, shutil, zipfile
import pandas as pd

# --- 2. Paths (check these match your Drive)
SAMPLE_CSV  = "/content/latency_sample_200.csv"
AUDIO_ROOT  = "/content/drive/MyDrive/Colab Notebooks/UrbanSound8k/audio"
CNN_RUN_DIR = "/content/drive/MyDrive/Colab Notebooks/FINALS/NEW_DATASET/Weighted_CE_CNN/1DWAVE/DangerDetective_1DWAVE_LV-18_s1_p8"
BERT_DIR    = "/content/drive/MyDrive/Colab Notebooks/FINALS/NEW_DATASET/DIstilBERT/training/low_threshold_experiments/low_threshold__alignment_stratified__weighted_ce/best_model"

PKG = "/content/latency_package"
OUT_ZIP = "/content/drive/MyDrive/latency_package.zip"

# --- 3. Copy the 200 clips
os.makedirs(f"{PKG}/audio", exist_ok=True)
sample = pd.read_csv(SAMPLE_CSV)
missing = []
for _, r in sample.iterrows():
    src = f"{AUDIO_ROOT}/fold{r.fold}/{r.slice_file_name}"
    if os.path.exists(src):
        shutil.copy(src, f"{PKG}/audio/{r.slice_file_name}")
    else:
        missing.append(src)
print(f"Copied {len(sample) - len(missing)} / {len(sample)} clips")
if missing:
    print("Missing (check AUDIO_ROOT / folder names):", missing[:5])
shutil.copy(SAMPLE_CSV, f"{PKG}/latency_sample_200.csv")

# --- 4. Copy the CNN model (the val-AUC-ROC checkpoint)
cands = glob.glob(f"{CNN_RUN_DIR}/**/best_model_auc.keras", recursive=True)
print("CNN candidates:", cands)
assert len(cands) >= 1, "best_model_auc.keras not found: check CNN_RUN_DIR"
os.makedirs(f"{PKG}/models", exist_ok=True)
shutil.copy(cands[0], f"{PKG}/models/cnn_lv18_wce_valauc.keras")

# --- 5. Copy the DistilBERT model folder
assert os.path.isdir(BERT_DIR), "DistilBERT best_model folder not found: check BERT_DIR"
shutil.copytree(BERT_DIR, f"{PKG}/models/distilbert_stratified_wce", dirs_exist_ok=True)
print("DistilBERT files:", os.listdir(f"{PKG}/models/distilbert_stratified_wce"))

# --- 6. Zip and save to Drive
with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(PKG):
        for f in files:
            full = os.path.join(root, f)
            z.write(full, os.path.relpath(full, PKG))
print(f"Saved {OUT_ZIP} ({os.path.getsize(OUT_ZIP) / 2**20:.0f} MB). Download it from Drive.")
