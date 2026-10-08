# Latency measurement: step by step

All timing runs on your laptop CPU, one clip at a time, on the same 200 UrbanSound8K
clips (stratified by class, seed 42, from the 8419-clip test set).

## 1. Colab: package everything from Drive
1. Open a new Colab notebook and upload `latency_sample_200.csv` (left panel, Files, Upload).
2. Paste `1_colab_package_from_drive.py` into a cell and run it.
3. Download `latency_package.zip` from the root of your Drive.

## 2. Laptop: set up the folder
1. Unzip it to `C:\Users\maria\Desktop\latency_package\`. Inside: `audio\`, `models\`, `latency_sample_200.csv`.
2. Copy the four `.py` files and both requirements files into the same folder.
3. Open the folder in VS Code (File, Open Folder) and open a terminal (Terminal, New Terminal).
4. Close other heavy programs (browser tabs, Teams) and plug in the charger, so timings are stable.

## 3. CoNeTTE (new environment, same versions as your original captioning run)
```
py -3.10 -m venv .venv_conette
.venv_conette\Scripts\python.exe -m pip install --upgrade pip
.venv_conette\Scripts\python.exe -m pip install torch==1.13.1 torchaudio==0.13.1 --index-url https://download.pytorch.org/whl/cpu
.venv_conette\Scripts\python.exe -m pip install -r requirements_conette_latency.txt
.venv_conette\Scripts\python.exe -m pip show conette torch transformers
.venv_conette\Scripts\python.exe 2_latency_conette.py
```
The `pip show` line should report conette 0.4.0, torch 1.13.1 and transformers 4.30.2,
the versions stated in the paper. The first run downloads the CoNeTTE checkpoint
(Labbeti/conette) from Hugging Face, so you need internet once. It then takes a few
minutes and creates `results\conette_latency.csv` and `results\conette_summary.txt`.

## 4. CNN + DistilBERT (new environment)
```
py -3.10 -m venv .venv_latency
.venv_latency\Scripts\python.exe -m pip install --upgrade pip
.venv_latency\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv_latency\Scripts\python.exe -m pip install -r requirements_latency.txt
.venv_latency\Scripts\python.exe 3_latency_cnn_distilbert.py
```
Check the two CHECK lines at the end: the CNN difference must be below 0.01 (the right
model and preprocessing), and DistilBERT should agree with Colab on almost all clips.

## 5. Cost per clip
```
.venv_latency\Scripts\python.exe 4_combine_cost.py
```

## Send back
- `results\conette_summary.txt`, `results\cnn_distilbert_summary.txt`, the output of step 5
- Your CPU model and RAM (Settings, System, About)
