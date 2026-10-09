# DangerDetective

Code and data manifests for the article **"Danger Detection through Audio Captioning and Classification Fusion"** (M.-C. Stampa, M.-E. Stamatiadou, C. Dimoulas), *Electronics* (MDPI).

The system combines a raw-waveform CNN for binary danger classification with a semantic branch (CoNeTTE audio captioning, a lexicon and SBERT scoring stage, and a DistilBERT classifier), linked through an uncertainty-based fusion rule (custom_uncertain, activation zone 0.17–0.60).

Archived version: https://zenodo.org/records/23244035.

## Repository structure

```
data/
  train_manifest.csv            Training set: 2058 clips (ESC-50, 40 UrbanSound8K gun_shot, 18 AudioSet screaming), with captions
  test_manifest.csv             UrbanSound8K evaluation set: 8419 clips, with captions
  splits/                       CNN training (1316) / validation (330) / internal test (412) partitions
  excluded/                     UrbanSound8K clips excluded from evaluation (ESC-50 overlap, too short for captioning)
  external/                     DESED (610 clips) and XD-Violence (1383 segments) manifests, with captions
  predictions/                  Final-system predictions used to compute the reported results
code/                           Notebooks and scripts (see below)
environment/                    Package requirements
```

## Data

The audio itself is not redistributed. All clips can be obtained from the original datasets (ESC-50, UrbanSound8K, AudioSet, DESED, XD-Violence) using the file names and identifiers in the manifests.

**Training set.** 2058 clips (378 danger, 1680 no-danger). Split into training, validation, and internal test subsets of 1316, 330, and 412 clips, stratified by original class, with random seed 42. Screaming clips were augmented only in the training subset, after splitting (`train_manifest_with_screaming_aug.csv`).

**UrbanSound8K evaluation set.** 8419 of the 8732 UrbanSound8K clips:

| Step | Clips removed | Remaining |
|---|---|---|
| Original UrbanSound8K | – | 8732 |
| Overlap with ESC-50 | 240 (99 car_horn, 91 dog_bark, 50 siren) | 8492 |
| gun_shot clips used for training | 40 | 8452 |
| Clips too short for caption generation (≤ 0.23 s) | 33 (24 car_horn, 6 dog_bark, 3 gun_shot) | 8419 |

**Deduplication.** Overlap between ESC-50 and UrbanSound8K was checked between corresponding classes (dog/dog_bark, siren, car_horn) with three criteria: (i) shared Freesound source ID, (ii) identical SHA-256 file hash, and (iii) near-duplicate waveforms (normalized cross-correlation of RMS envelopes ≥ 0.95). The union of the three criteria gives the 240 removed clips.

**External datasets.** DESED: 692 public-evaluation clips, of which 82 containing a dog event were excluded (dog is a danger class here), leaving 610 no-danger clips. XD-Violence: the 1383 violent segments; non-violent segments were not used. The column `included_in_evaluation` marks the clips used.

## Code

| File | Purpose | Paper |
|---|---|---|
| `dedup_esc50_urbansound8k.ipynb` | Overlap detection between ESC-50 and UrbanSound8K | 3.1 |
| `cnn_training_weighted_ce.ipynb` | CNN training with weighted cross-entropy (final model: LV-18, validation AUC-ROC checkpoint) | 3.2, 4.1 |
| `cnn_training_focal_loss.ipynb` | CNN training with focal loss (comparison models) | 4.1 |
| `caption_generation_conette.py` | Caption generation with CoNeTTE (checkpoint `Labbeti/conette`, local CPU) | 3.3.1 |
| `sbert_danger_scoring.py`, `sbert_danger_scoring_runs.ipynb`, `sbert_lexicon/` | Lexicon and SBERT danger scoring of the captions (low threshold profile used for DistilBERT labels) | 3.3.2, 4.2 |
| `distilbert_training.ipynb` | DistilBERT fine-tuning, six configurations (selected: alignment-stratified split, weighted CE) | 3.3.3, 4.3 |
| `derive_fusion_thresholds.py` | Activation zone from the CNN's internal test subset | 3.4 |
| `final_fusion_evaluation.py` | Final results on UrbanSound8K: Tables 9, 10 and 12, bootstrap CIs | 4.3, 4.4 |
| `latency/` | Latency and memory measurements on a laptop CPU | 5 |

## Reproducing the reported results

The final results can be recomputed from the prediction files without re-training:

```
cd data/predictions
python ../../code/derive_fusion_thresholds.py cnn_lv18_wce_internal_test.csv
python ../../code/final_fusion_evaluation.py --cnn cnn_lv18_wce_urbansound8k.csv --bert distilbert_urbansound8k.csv --sbert sbert_urbansound8k.csv
```

The first command gives the activation zone (lower bound 0.17 = median CNN probability of false-negative clips on the internal test subset; upper bound set to 0.60, below the false-positive median of 0.73). The second prints Tables 9, 10 and 12 of the paper.

## Environment

- `environment/requirements_conette.txt`: caption generation, run locally on CPU (Python 3.10, conette 0.4.0).
- `environment/requirements_colab.txt`: CNN, SBERT, and DistilBERT experiments in Google Colab on an NVIDIA T4 GPU (TensorFlow 2.20.0, Keras 3.13.2).
- `code/latency/requirements_*.txt`: latency measurements.

Notebooks were written for Google Colab with the datasets on Google Drive; edit the path cell of each notebook to your own copy.

## Citation

If you use this repository, please cite the article and the archived version: https://zenodo.org/records/23244035.
