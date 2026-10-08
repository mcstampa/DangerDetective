"""Caption generation with CoNeTTE (Labbé et al., IEEE/ACM TASLP 2024).

Part of the code for "Danger Detection through Audio Captioning and Classification Fusion"
(Electronics, MDPI). Paper section: 3.3.1.

The model itself is not part of this repository: it is downloaded from the Hugging Face Hub
(checkpoint "Labbeti/conette") through the `conette` package (https://github.com/Labbeti/conette-audio-captioning).
This script only shows how it was applied: one caption per clip, default "clotho" task embedding,
run locally on CPU (environment: environment/requirements_conette.txt; conette 0.4.0, Python 3.10).

Clips too short to be captioned (<= 0.23 s; 33 UrbanSound8K clips) fail and are listed in the
output with status "error"; they are excluded from evaluation (data/excluded/).

Usage:
    python caption_generation_conette.py --input-csv manifest.csv --audio-column path --output-csv captions.csv
"""
import argparse

import pandas as pd
import torch
from conette import CoNeTTEConfig, CoNeTTEModel


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-csv", required=True, help="CSV with one row per audio clip")
    p.add_argument("--audio-column", default="path", help="column with the audio file path")
    p.add_argument("--output-csv", required=True)
    p.add_argument("--task", default="clotho", help="CoNeTTE task embedding (default: clotho)")
    args = p.parse_args()

    cfg = CoNeTTEConfig.from_pretrained("Labbeti/conette")
    model = CoNeTTEModel.from_pretrained("Labbeti/conette", config=cfg)
    model.eval()

    df = pd.read_csv(args.input_csv)
    captions, status = [], []
    with torch.no_grad():
        for i, path in enumerate(df[args.audio_column]):
            try:
                out = model(path, task=args.task)
                captions.append(out["cands"][0])
                status.append("ok")
            except Exception as e:  # e.g. clips too short for the encoder
                captions.append("")
                status.append(f"error: {type(e).__name__}")
            if i % 500 == 0:
                print(i, "/", len(df))

    df["caption"], df["caption_task"], df["status"] = captions, args.task, status
    df.to_csv(args.output_csv, index=False)
    print("saved", args.output_csv, "| failed:", sum(s != "ok" for s in status))


if __name__ == "__main__":
    main()
