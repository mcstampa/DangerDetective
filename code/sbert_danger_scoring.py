"""Colab-ready Sentence-BERT semantic scoring for audio-caption datasets.

Upload these three files to `/content` in Colab:
  - esc50_conette_captions_all_2000.csv
  - class_keywords.json
  - danger_lexicon.json

Then run:
  !pip -q install sentence-transformers pandas
  !python /content/colab_esc50_sentencebert.py --threshold-profile default

Reuse for another dataset, for example UrbanSound8K:
  !python /content/colab_esc50_sentencebert.py \
    --input-csv /content/urbansound8k_conette_captions.csv \
    --dataset-name UrbanSound8K \
    --output-prefix urbansound8k \
    --filename-column slice_file_name \
    --caption-column caption \
    --category-column class

Threshold profiles: default, low, strict.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer


# Edit these defaults in Colab if you do not want to pass paths as arguments.
DEFAULT_INPUT_CSV_PATH = Path("/content/esc50_conette_captions_all_2000.csv")
DEFAULT_CLASS_KEYWORDS_PATH = Path("/content/class_keywords.json")
DEFAULT_DANGER_LEXICON_PATH = Path("/content/danger_lexicon.json")
# Edit this to choose where output CSV files are saved.
DEFAULT_OUTPUT_DIR = Path("/content/sbert_outputs")
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


THRESHOLD_PROFILES = {
    "default": {
        "class": 0.40,
        "event": 0.42,
        "modifier": 0.38,
        "danger": 0.0,
    },
    "low": {
        "class": 0.35,
        "event": 0.38,
        "modifier": 0.35,
        "danger": 0.0,
    },
    "strict": {
        "class": 0.45,
        "event": 0.48,
        "modifier": 0.42,
        "danger": 0.0,
    },
}


@dataclass(frozen=True)
class Prototype:
    name: str
    text: str
    kind: str
    value: float


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_text(text) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"[_/]", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_prototype_text(label: str, terms: list[str], prefix: str) -> str:
    label_text = normalize_text(label)
    unique_terms = [label_text] if label_text else []
    for term in terms:
        cleaned = normalize_text(term)
        if cleaned and cleaned not in unique_terms:
            unique_terms.append(cleaned)
    return f"{prefix}: " + ", ".join(unique_terms)


def build_class_prototypes(class_keywords: dict) -> list[Prototype]:
    return [
        Prototype(
            name=class_name,
            text=build_prototype_text(class_name, keywords, "sound class"),
            kind="class",
            value=1.0,
        )
        for class_name, keywords in class_keywords.items()
    ]


def build_event_prototypes(danger_lexicon: dict) -> list[Prototype]:
    return [
        Prototype(
            name=class_name,
            text=build_prototype_text(
                class_name,
                info.get("keywords", []),
                "danger sound event",
            ),
            kind="event",
            value=float(info.get("score", 0)),
        )
        for class_name, info in danger_lexicon["event_classes"].items()
    ]


def build_group_prototypes(score_dict: dict, kind: str) -> list[Prototype]:
    return [
        Prototype(
            name=term,
            text=f"{kind} cue: {normalize_text(term)}",
            kind=kind,
            value=float(value),
        )
        for term, value in score_dict.items()
    ]


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for column in candidates:
        if column in df.columns:
            return column
    raise ValueError(f"Missing required column. Tried: {', '.join(candidates)}")


def resolve_column(
    df: pd.DataFrame,
    explicit_column: str | None,
    auto_candidates: list[str],
    required: bool,
) -> str | None:
    if explicit_column:
        if explicit_column not in df.columns:
            raise ValueError(f"Column not found: {explicit_column}")
        return explicit_column
    for column in auto_candidates:
        if column in df.columns:
            return column
    if required:
        raise ValueError(f"Missing required column. Tried: {', '.join(auto_candidates)}")
    return None


def similarity(caption_embedding, prototype_embedding) -> float:
    return float(caption_embedding @ prototype_embedding)


def score_dataframe(
    df: pd.DataFrame,
    class_keywords: dict,
    danger_lexicon: dict,
    model_name: str,
    thresholds: dict,
    caption_column: str | None = None,
    category_column: str | None = None,
) -> pd.DataFrame:
    caption_col = resolve_column(
        df,
        caption_column,
        ["caption_clean", "caption"],
        required=True,
    )
    class_col = resolve_column(
        df,
        category_column,
        ["category", "class", "class_name", "label"],
        required=False,
    )

    df = df.copy()
    df["caption_clean_for_sbert"] = df[caption_col].apply(normalize_text)

    class_prototypes = build_class_prototypes(class_keywords)
    event_prototypes = build_event_prototypes(danger_lexicon)
    intensity_prototypes = build_group_prototypes(
        danger_lexicon["intensity_temporal"],
        "intensity",
    )
    context_prototypes = build_group_prototypes(
        danger_lexicon["context_distance"],
        "context",
    )
    media_prototypes = build_group_prototypes(
        danger_lexicon["media_exclusion"],
        "media",
    )

    all_prototypes = (
        class_prototypes
        + event_prototypes
        + intensity_prototypes
        + context_prototypes
        + media_prototypes
    )

    model = SentenceTransformer(model_name)
    prototype_embeddings = model.encode(
        [proto.text for proto in all_prototypes],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    caption_embeddings = model.encode(
        df["caption_clean_for_sbert"].tolist(),
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    prototype_lookup = {
        (proto.kind, proto.name): embedding
        for proto, embedding in zip(all_prototypes, prototype_embeddings)
    }
    class_prototype_by_normalized_name = {
        normalize_text(proto.name): proto for proto in class_prototypes
    }

    results = []
    for idx, row in enumerate(df.itertuples(index=False)):
        caption_embedding = caption_embeddings[idx]
        true_class = getattr(row, class_col) if class_col else None

        best_class_name = ""
        best_class_similarity = -1.0
        for proto in class_prototypes:
            sim = similarity(caption_embedding, prototype_lookup[(proto.kind, proto.name)])
            if sim > best_class_similarity:
                best_class_name = proto.name
                best_class_similarity = sim

        class_alignment_score = None
        class_alignment_details = None
        if true_class is not None and pd.notna(true_class):
            normalized_true_class = normalize_text(str(true_class).strip())
            proto = class_prototype_by_normalized_name.get(normalized_true_class)
            if proto is not None:
                sim = similarity(caption_embedding, prototype_lookup[("class", proto.name)])
                matched = sim >= thresholds["class"]
                class_alignment_score = int(matched)
                class_alignment_details = {
                    "true_class_similarity": round(sim, 6),
                    "threshold": thresholds["class"],
                    "matched": matched,
                    "normalized_true_class": normalized_true_class,
                    "prototype_found": True,
                    "matched_prototype_name": proto.name,
                }
            else:
                class_alignment_score = 0
                class_alignment_details = {
                    "true_class_similarity": 0.0,
                    "threshold": thresholds["class"],
                    "matched": False,
                    "normalized_true_class": normalized_true_class,
                    "prototype_found": False,
                }

        event_score, matched_events = score_prototype_group(
            caption_embedding,
            event_prototypes,
            prototype_lookup,
            thresholds["event"],
        )
        raw_intensity_score, matched_intensity = score_prototype_group(
            caption_embedding,
            intensity_prototypes,
            prototype_lookup,
            thresholds["modifier"],
        )
        raw_context_score, matched_context = score_prototype_group(
            caption_embedding,
            context_prototypes,
            prototype_lookup,
            thresholds["modifier"],
        )
        media_score, matched_media = score_prototype_group(
            caption_embedding,
            media_prototypes,
            prototype_lookup,
            thresholds["modifier"],
        )

        modifier_raw = raw_intensity_score + raw_context_score
        modifier_score_capped = max(min(modifier_raw, 3.0), -3.0)
        total_score = event_score + modifier_score_capped + media_score
        danger_pred = int(total_score > thresholds["danger"])

        results.append(
            {
                "caption_clean": normalize_text(getattr(row, caption_col)),
                "sbert_class_alignment_score": class_alignment_score,
                "sbert_class_alignment_details": json.dumps(
                    class_alignment_details,
                    ensure_ascii=False,
                ),
                "sbert_best_class": best_class_name,
                "sbert_best_class_similarity": round(best_class_similarity, 6),
                "sbert_event_score": round(event_score, 6),
                "sbert_matched_event_classes": json.dumps(
                    matched_events,
                    ensure_ascii=False,
                ),
                "sbert_raw_intensity_score": round(raw_intensity_score, 6),
                "sbert_matched_intensity_terms": json.dumps(
                    matched_intensity,
                    ensure_ascii=False,
                ),
                "sbert_raw_context_score": round(raw_context_score, 6),
                "sbert_matched_context_terms": json.dumps(
                    matched_context,
                    ensure_ascii=False,
                ),
                "sbert_modifier_raw": round(modifier_raw, 6),
                "sbert_modifier_score_capped": round(modifier_score_capped, 6),
                "sbert_media_score": round(media_score, 6),
                "sbert_matched_media_terms": json.dumps(
                    matched_media,
                    ensure_ascii=False,
                ),
                "sbert_danger_score": round(total_score, 6),
                "sbert_caption_pred": danger_pred,
                "sbert_danger_label": (
                    "danger_risk" if danger_pred == 1 else "no_danger_risk"
                ),
            }
        )

    return pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)


def score_prototype_group(
    caption_embedding,
    prototypes: list[Prototype],
    prototype_lookup: dict,
    threshold: float,
) -> tuple[float, dict]:
    total = 0.0
    matched = {}
    for proto in prototypes:
        sim = similarity(caption_embedding, prototype_lookup[(proto.kind, proto.name)])
        if sim >= threshold:
            weighted = sim * proto.value
            matched[proto.name] = {
                "similarity": round(sim, 6),
                "value": proto.value,
                "weighted_score": round(weighted, 6),
            }
            total += weighted
    return total, matched


def optional_column(sbert_df: pd.DataFrame, column: str | None, fallback: str | None = None):
    if column and column in sbert_df.columns:
        return sbert_df[column]
    if fallback and fallback in sbert_df.columns:
        return sbert_df[fallback]
    return pd.NA


def build_final_dataset(
    sbert_df: pd.DataFrame,
    dataset_name: str,
    filename_column: str | None = None,
    caption_column: str | None = None,
    category_column: str | None = None,
    audio_prediction_column: str | None = None,
    keyword_prediction_column: str | None = None,
) -> pd.DataFrame:
    filename_col = resolve_column(
        sbert_df,
        filename_column,
        ["filename", "audio_filename", "slice_file_name", "file", "filepath"],
        required=True,
    )
    caption_col = resolve_column(
        sbert_df,
        caption_column,
        ["caption_clean", "caption"],
        required=True,
    )
    category_col = resolve_column(
        sbert_df,
        category_column,
        ["category", "class", "class_name", "label"],
        required=False,
    )

    final_df = pd.DataFrame(
        {
            "filename": sbert_df[filename_col].astype(str).str.strip(),
            "caption": sbert_df[caption_col].astype(str).str.strip(),
            "semantic_label_final": sbert_df["sbert_danger_label"].astype(str).str.strip(),
            "source_dataset": (
                sbert_df["source_dataset"].astype(str).str.strip()
                if "source_dataset" in sbert_df.columns
                else dataset_name
            ),
            "text_source": caption_col,
            "original_category": (
                sbert_df[category_col].astype(str).str.strip()
                if category_col
                else pd.NA
            ),
            "audio_model_prediction": optional_column(
                sbert_df,
                audio_prediction_column,
                "audio_model_prediction",
            ),
            "keyword_model_prediction": optional_column(
                sbert_df,
                keyword_prediction_column,
                "keyword_model_prediction",
            ),
            "sentencebert_prediction": sbert_df["sbert_caption_pred"],
            "sentencebert_danger_score": sbert_df["sbert_danger_score"],
            "sentencebert_best_class": sbert_df["sbert_best_class"],
            "sentencebert_best_class_similarity": sbert_df[
                "sbert_best_class_similarity"
            ],
            "sentencebert_class_alignment_score": sbert_df[
                "sbert_class_alignment_score"
            ],
        }
    )

    final_df = final_df.dropna(
        subset=["filename", "caption", "semantic_label_final"],
    ).copy()
    final_df = final_df[
        (final_df["filename"] != "")
        & (final_df["caption"] != "")
        & (final_df["semantic_label_final"] != "")
    ]
    return final_df.drop_duplicates(subset=["filename"], keep="first").reset_index(
        drop=True,
    )


def build_class_alignment_summary(
    sbert_df: pd.DataFrame,
    category_column: str | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    category_col = resolve_column(
        sbert_df,
        category_column,
        ["category", "class", "class_name", "label", "original_category"],
        required=False,
    )
    if category_col is None:
        return pd.DataFrame(), pd.Series(dtype="int64")

    metric_df = sbert_df[[category_col, "sbert_class_alignment_score"]].copy()
    metric_df = metric_df.dropna(subset=["sbert_class_alignment_score"])
    metric_df["sbert_class_alignment_score"] = metric_df[
        "sbert_class_alignment_score"
    ].astype(int)
    metric_df["class_name"] = metric_df[category_col].astype(str).str.strip()
    metric_df = metric_df[metric_df["class_name"] != ""]

    counts = metric_df["sbert_class_alignment_score"].value_counts().sort_index()
    summary = (
        metric_df.groupby("class_name")["sbert_class_alignment_score"]
        .agg(
            total="count",
            aligned="sum",
        )
        .reset_index()
    )
    summary["not_aligned"] = summary["total"] - summary["aligned"]
    summary["alignment_rate"] = (summary["aligned"] / summary["total"]).round(6)
    summary["not_alignment_rate"] = (summary["not_aligned"] / summary["total"]).round(6)
    summary = summary[
        [
            "class_name",
            "total",
            "aligned",
            "not_aligned",
            "alignment_rate",
            "not_alignment_rate",
        ]
    ].sort_values(
        ["not_aligned", "aligned", "total", "class_name"],
        ascending=[False, False, False, True],
    )
    return summary, counts


def save_danger_score_boxplot(sbert_df: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    plot_df = sbert_df[["sbert_danger_label", "sbert_danger_score"]].dropna().copy()
    label_order = ["no_danger_risk", "danger_risk"]
    grouped_scores = [
        plot_df.loc[plot_df["sbert_danger_label"] == label, "sbert_danger_score"]
        for label in label_order
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(
        grouped_scores,
        labels=label_order,
        showmeans=True,
        patch_artist=True,
        boxprops={"facecolor": "#d8e6f3", "edgecolor": "#2f4f63"},
        medianprops={"color": "#1f2933", "linewidth": 2},
        meanprops={
            "marker": "o",
            "markerfacecolor": "#b23a48",
            "markeredgecolor": "#b23a48",
        },
    )
    ax.set_title("SBERT Danger Score by Final Label")
    ax.set_xlabel("Final SBERT Label")
    ax.set_ylabel("SBERT danger score")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV_PATH,
    )
    parser.add_argument("--dataset-name", default="ESC50")
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Output filename prefix. Defaults to a normalized dataset name.",
    )
    parser.add_argument(
        "--filename-column",
        default=None,
        help="Audio filename/id column. Auto-detects common names if omitted.",
    )
    parser.add_argument(
        "--caption-column",
        default=None,
        help="Caption text column. Auto-detects caption_clean or caption if omitted.",
    )
    parser.add_argument(
        "--category-column",
        default=None,
        help="Ground-truth sound category column, such as category or class.",
    )
    parser.add_argument(
        "--audio-prediction-column",
        default=None,
        help="Optional existing audio-model prediction column to preserve.",
    )
    parser.add_argument(
        "--keyword-prediction-column",
        default=None,
        help="Optional existing keyword-model prediction column to preserve.",
    )
    parser.add_argument(
        "--class-keywords",
        type=Path,
        default=DEFAULT_CLASS_KEYWORDS_PATH,
    )
    parser.add_argument(
        "--danger-lexicon",
        type=Path,
        default=DEFAULT_DANGER_LEXICON_PATH,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Folder where output CSV files are saved.",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--threshold-profile",
        choices=sorted(THRESHOLD_PROFILES),
        default="default",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    thresholds = THRESHOLD_PROFILES[args.threshold_profile]
    suffix = f"_{args.threshold_profile}_threshold"
    output_prefix = args.output_prefix or normalize_text(args.dataset_name).replace(" ", "_")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sbert_output_path = (
        args.output_dir / f"{output_prefix}_sentencebert_semantic_scored{suffix}.csv"
    )
    final_output_path = (
        args.output_dir / f"{output_prefix}_danger_sound_text_dataset_sentencebert{suffix}.csv"
    )
    alignment_summary_path = (
        args.output_dir / f"{output_prefix}_class_alignment_summary{suffix}.csv"
    )
    danger_score_boxplot_path = (
        args.output_dir / f"{output_prefix}_danger_score_boxplot{suffix}.png"
    )

    class_keywords = load_json(args.class_keywords)
    danger_lexicon = load_json(args.danger_lexicon)
    df = pd.read_csv(args.input_csv)

    print("input_csv =", args.input_csv)
    print("dataset_name =", args.dataset_name)
    print("output_dir =", args.output_dir)
    print("rows =", len(df))
    print("threshold_profile =", args.threshold_profile)
    print("model_name =", args.model_name)

    sbert_df = score_dataframe(
        df=df,
        class_keywords=class_keywords,
        danger_lexicon=danger_lexicon,
        model_name=args.model_name,
        thresholds=thresholds,
        caption_column=args.caption_column,
        category_column=args.category_column,
    )
    sbert_df.to_csv(sbert_output_path, index=False, encoding="utf-8")

    final_df = build_final_dataset(
        sbert_df,
        dataset_name=args.dataset_name,
        filename_column=args.filename_column,
        caption_column=args.caption_column,
        category_column=args.category_column,
        audio_prediction_column=args.audio_prediction_column,
        keyword_prediction_column=args.keyword_prediction_column,
    )
    final_df.to_csv(final_output_path, index=False, encoding="utf-8")

    alignment_summary_df, alignment_counts = build_class_alignment_summary(
        sbert_df,
        category_column=args.category_column,
    )
    if not alignment_summary_df.empty:
        alignment_summary_df.to_csv(
            alignment_summary_path,
            index=False,
            encoding="utf-8",
        )
    save_danger_score_boxplot(sbert_df, danger_score_boxplot_path)

    print("saved:", sbert_output_path)
    print("saved:", final_output_path)
    if not alignment_summary_df.empty:
        print("saved:", alignment_summary_path)
    print("saved:", danger_score_boxplot_path)
    print("\nSentence-BERT labels:")
    print(sbert_df["sbert_danger_label"].value_counts(dropna=False))

    if not alignment_summary_df.empty:
        aligned_count = int(alignment_counts.get(1, 0))
        not_aligned_count = int(alignment_counts.get(0, 0))
        print("\nClass alignment counts:")
        print("aligned =", aligned_count)
        print("not_aligned =", not_aligned_count)
        print("\nClasses with most not aligned captions:")
        print(alignment_summary_df.head(10).to_string(index=False))
        print("\nClasses with most aligned captions:")
        print(
            alignment_summary_df.sort_values(
                ["aligned", "not_aligned", "total", "class_name"],
                ascending=[False, True, False, True],
            )
            .head(10)
            .to_string(index=False)
        )
    else:
        print("\nClass alignment summary skipped: no category/class column found.")

    if "audio_model_prediction" in sbert_df.columns:
        print("\nAudio model vs Sentence-BERT:")
        print(pd.crosstab(sbert_df["audio_model_prediction"], sbert_df["sbert_caption_pred"]))
    if "keyword_model_prediction" in sbert_df.columns:
        print("\nKeyword model vs Sentence-BERT:")
        print(pd.crosstab(sbert_df["keyword_model_prediction"], sbert_df["sbert_caption_pred"]))


if __name__ == "__main__":
    main()
