"""STEP 2 — CoNeTTE latency (run in your existing .venv_conette, Python 3.10).

Captions the 200 clips one at a time and records, per clip, the wall-clock time
of one CoNeTTE call (audio loading + resampling + captioning), plus memory.
Writes results/conette_latency.csv, which step 3 reads for the DistilBERT input.
"""
import os, time, statistics
import pandas as pd
import psutil
import torch
from conette import CoNeTTEConfig, CoNeTTEModel

PKG = os.path.dirname(os.path.abspath(__file__))
AUDIO = os.path.join(PKG, "audio")
os.makedirs(os.path.join(PKG, "results"), exist_ok=True)
torch.set_num_threads(os.cpu_count())  # same CPU setting for every stage

proc = psutil.Process(os.getpid())
mb = lambda b: b / 2**20
peak = lambda: mb(getattr(proc.memory_info(), "peak_wset", proc.memory_info().rss))

ram_start = mb(proc.memory_info().rss)
t = time.perf_counter()
cfg = CoNeTTEConfig.from_pretrained("Labbeti/conette")
model = CoNeTTEModel.from_pretrained("Labbeti/conette", config=cfg)
model.eval()
load_s = time.perf_counter() - t
ram_loaded = mb(proc.memory_info().rss)

sample = pd.read_csv(os.path.join(PKG, "latency_sample_200.csv"))
files = [os.path.join(AUDIO, f) for f in sample.slice_file_name]

for f in files[:3]:  # warm-up, not timed
    model(f)

rows = []
with torch.no_grad():
    for name, f in zip(sample.slice_file_name, files):
        t = time.perf_counter()
        out = model(f)
        ms = (time.perf_counter() - t) * 1000
        rows.append({"slice_file_name": name, "caption": out["cands"][0], "conette_ms": ms})

res = pd.DataFrame(rows)
res.to_csv(os.path.join(PKG, "results", "conette_latency.csv"), index=False)

times = res.conette_ms.tolist()
summary = (f"CoNeTTE (conette 0.4.0, torch {torch.__version__}, {torch.get_num_threads()} threads)\n"
           f"  model load: {load_s:.1f} s\n"
           f"  latency per clip: mean {statistics.mean(times):.0f} ms, SD {statistics.stdev(times):.0f} ms, "
           f"median {statistics.median(times):.0f} ms (n={len(times)})\n"
           f"  RAM: {ram_start:.0f} MB at start, {ram_loaded:.0f} MB after loading, peak {peak():.0f} MB\n")
print(summary)
open(os.path.join(PKG, "results", "conette_summary.txt"), "w").write(summary)
