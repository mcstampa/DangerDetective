# DangerDetective

Code and data manifests for the article **"Danger Detection through Audio Captioning and Classification Fusion"** (M.-C. Stampa, M.-E. Stamatiadou, C. Dimoulas), submitted to *Electronics* (MDPI).

The system combines a raw-waveform CNN for binary danger classification with a semantic branch (CoNeTTE audio captioning, a lexicon and SBERT scoring stage, and a DistilBERT classifier), linked through an uncertainty-based fusion rule.

Archived version: https://doi.org/10.5281/zenodo.22165447

## Repository structure

```
data/
  train_manifest.csv          Training set (2058 clips: ESC-50, 40 UrbanSound8K gun_shot, 18 AudioSet screaming)
  test_manifest.csv           UrbanSound8K evaluation set (8419 clips)
  splits/                     CNN training / validation / internal test partitions
  excluded/                   UrbanSound8K clips excluded from evaluation
  external/                   DESED and XD-Violence evaluation manifests
code/                         Notebooks and scripts (see below)
environment/                  Package requirements
```

## Data

The audio itself is not redistributed. All clips can be obtained from the original datasets (ESC-50, UrbanSound8K, AudioSet, DESED, XD-Violence) using the file names and identifiers in the manifests.

**Training set.** 2058 clips (378 danger, 1680 no-danger). Split into training, validation, and internal test subsets of 1316, 330, and 412 clips, stratified by original class, with random seed 42. Screaming clips were augmented only in the training subset, after splitting.

**UrbanSound8K evaluation set.** 8419 of the 8732 UrbanSound8K clips:

| Step | Clips removed | Remaining |
|---|---|---|
| Original UrbanSound8K | – | 8732 |
| Overlap with ESC-50 | 240 (99 car_horn, 91 dog_bark, 50 siren) | 8492 |
| gun_shot clips used for training | 40 | 8452 |
| Clips too short for caption generation (≤ 0.23 s) | 33 (24 car_horn, 6 dog_bark, 3 gun_shot) | 8419 |

**Deduplication.** Overlap between ESC-50 and UrbanSound8K was checked between corresponding classes (dog/dog_bark, siren, car_horn) with three criteria: (i) shared Freesound source ID, (ii) identical SHA-256 file hash, and (iii) near-duplicate waveforms (normalized cross-correlation of RMS envelopes ≥ 0.95). The union of the three criteria gives the 240 removed clips.

## Code

| File | Purpose |
|---|---|
| `dedup_esc50_urbansound8k.ipynb` | Overlap detection between ESC-50 and UrbanSound8K |
| `cnn_training_focal_loss.ipynb` | CNN training with focal loss |
| `cnn_training_weighted_ce.ipynb` | CNN training with weighted cross-entropy |
| `distilbert_training.ipynb` | DistilBERT fine-tuning on caption labels |
| `fusion.ipynb` | Decision-fusion rules and evaluation |

## Environment

- `environment/requirements_conette.txt`: caption generation, run locally on CPU (Python 3.10).
- `environment/requirements_colab.txt`: CNN, SBERT, and DistilBERT experiments in Google Colab on an NVIDIA T4 GPU (Python 3.12).

## Citation

If you use this repository, please cite the archived version: https://doi.org/10.5281/zenodo.22165447
