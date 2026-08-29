# Danger Detection through Audio Captioning and Classification Fusion
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22165447.svg)](https://doi.org/10.5281/zenodo.22165447)
This repository provides the dataset manifests and supporting information for the paper:

> Stampa, M.-C.; Dimoulas, C.; Stamatiadou, M.-E. *Danger Detection through Audio Captioning and Classification Fusion.* Electronics, 2026. [DOI to be added upon publication]

The audio clips used in this study come from publicly available benchmark datasets and **cannot be redistributed here** due to their respective licenses. Instead, this repository provides **manifests** listing every clip used, its source dataset, its assigned danger/no-danger label, and (where applicable) its fold, so that the dataset can be reconstructed from the original sources.

## Contents

- `train_manifest.csv` — the 2,058 clips used for training (ESC-50, UrbanSound8K, AudioSet).
- `test_manifest.csv` — the 8,419 held-out UrbanSound8K clips used as an independent test set.
- `README.md` — this file.
- `LICENSE` — license for the contents of this repository (CC BY 4.0).

## Training set (`train_manifest.csv`)

2,058 clips: 378 danger (18.4%) and 1,680 no-danger (81.6%). Danger clips comprise eight ESC-50 classes (siren, chainsaw, glass breaking, coughing, sneezing, crying baby, car horn, dog), plus gun_shot from UrbanSound8K and screaming from AudioSet.

Columns:

- `source_dataset` — origin dataset (ESC-50, UrbanSound8K, AudioSet)
- `clip_id` — file name for ESC-50/UrbanSound8K clips; YouTube video ID for AudioSet clips
- `audioset_segment_sec` — start–end time (in seconds) of the AudioSet segment; empty for other sources
- `original_category` — the clip's class label in its source dataset
- `danger_label` — label assigned in this work (1 = danger, 0 = no-danger)
- `esc50_fold` — ESC-50 cross-validation fold; empty for other sources
- `caption` — natural-language caption generated with CoNeTTE (Clotho configuration)

## Test set (`test_manifest.csv`)

8,419 held-out UrbanSound8K clips: 2,419 danger and 6,000 no-danger. These are the UrbanSound8K clips not used in training (after overlap removal and exclusion of the 40 gun_shot clips used for training). Danger classes are gun_shot, siren, car_horn, and dog_bark; all other classes are no-danger.

Columns:

- `source_dataset` — UrbanSound8K
- `clip_id` — UrbanSound8K slice file name
- `original_category` — the clip's UrbanSound8K class label
- `danger_label` — label assigned in this work (1 = danger, 0 = no-danger)
- `urbansound_fold` — UrbanSound8K fold (1–10)
- `caption` — natural-language caption generated with CoNeTTE (Clotho configuration)

## Source datasets

The audio must be obtained from the original providers:

- **ESC-50** — Dataset for Environmental Sound Classification
  https://github.com/karolpiczak/ESC-50
- **UrbanSound8K**
  https://urbansounddataset.weebly.com/urbansound8k.html
- **AudioSet** — Google
  https://research.google.com/audioset/

## How to cite

If you use these manifests or build on this work, please cite:

```bibtex
@article{stampa2026danger,
  title   = {Danger Detection through Audio Captioning and Classification Fusion},
  author  = {Stampa, Maria-Christina and Dimoulas, Charalampos and Stamatiadou, Maria-Eirini},
  journal = {Electronics},
  year    = {2026},
  note    = {DOI to be added upon publication}
}
```

## License

The contents of this repository are released under the Creative Commons Attribution 4.0 International (CC BY 4.0) license. See `LICENSE`.
