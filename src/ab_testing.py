"""
A/B Testing Utilities for SentenceMapper

This module provides functions for comparing different configurations:
- Embedding models (potion-base-2M, 4M, 8M, 32M)
- Alpha parameter values (additive vs multiplicative formulas)
- LLM summary quality assessment

Functions:
    compare_embedding_models: Compare ROUGE scores across embedding models
    compare_llm_summaries: Compare LLM-generated summaries with different embeddings
    judge_with_retry: Retry logic for LLM judge calls
"""

import time
from itertools import combinations
from typing import Dict, List, Any

import numpy as np
import pandas as pd
from rouge_score import rouge_scorer
from scipy.stats import spearmanr

from src.sentence_processor import SentenceProcessor


def _compute_target_tokens(
    total_tokens: int,
    objective_percentage: float | None = None,
    objective_tokens: int | None = None,
) -> int:
    """Compute the target token budget from percentage and/or absolute tokens.

    Args:
        total_tokens: Total tokens in the document
        objective_percentage: Target fraction of tokens (0-1), or None
        objective_tokens: Absolute target token count, or None

    Returns:
        Target token budget. If both are provided, returns the minimum.
        If neither is provided, raises ValueError.
    """
    if objective_percentage is None and objective_tokens is None:
        raise ValueError(
            "At least one of objective_percentage or objective_tokens must be provided"
        )

    candidates = []
    if objective_percentage is not None:
        candidates.append(int(total_tokens * objective_percentage))
    if objective_tokens is not None:
        candidates.append(int(objective_tokens))

    return min(candidates)


def compare_embedding_models(
    dataset,
    processors: Dict[str, SentenceProcessor],
    objective_percentage: float | None = None,
    objective_tokens: int | None = None,
    alpha: float = 0.5,
) -> pd.DataFrame:
    """Compare embedding models on extractive selection quality.

    For each document, every processor computes features independently
    (sentence splitting is deterministic — only embeddings differ).
    Sentences are scored with `similarity - α × ratio` and greedily selected.

    Args:
        dataset: Dataset with 'report' and 'summary' fields
        processors: Dict mapping model name to SentenceProcessor instance
        objective_percentage: Target compression ratio (0-1), or None
        objective_tokens: Absolute target token count, or None.
            If both are provided, uses the minimum.
        alpha: Length bias parameter for additive formula

    Returns:
        DataFrame with ROUGE scores, selected tokens, embedding time,
        and pairwise Jaccard overlaps between models.
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    records = []

    for i, example in enumerate(dataset):
        text = example["report"]
        reference = example["summary"]
        row = {"doc": i}

        masks = {}  # model_name → binary mask
        for model_name, proc in processors.items():
            short = model_name.split("/")[-1]  # e.g. "potion-base-8M"

            t0 = time.perf_counter()
            features = proc.compute_document_features(text)
            embed_time = time.perf_counter() - t0

            total_tokens = int(np.sum(features["tokens"]))
            budget = _compute_target_tokens(
                total_tokens, objective_percentage, objective_tokens
            )

            scores = features["similarities"] - alpha * features["ratios"]

            # Greedy fill
            mask = np.zeros(len(scores), dtype=bool)
            current = 0
            for idx in np.argsort(-scores):
                if current + features["tokens"][idx] > budget:
                    continue
                mask[idx] = True
                current += features["tokens"][idx]

            selected_text = proc.select_sentences_with_separators(
                features["sentences"], mask.astype(int)
            )

            rouge = scorer.score(reference, selected_text)
            row[f"{short}_tokens"] = int(np.sum(features["tokens"][mask]))
            row[f"{short}_rouge1"] = rouge["rouge1"].fmeasure
            row[f"{short}_rouge2"] = rouge["rouge2"].fmeasure
            row[f"{short}_rougeL"] = rouge["rougeL"].fmeasure
            row[f"{short}_time"] = embed_time

            masks[short] = mask

        # Pairwise Jaccard overlaps
        shorts = list(masks.keys())
        for a_idx in range(len(shorts)):
            for b_idx in range(a_idx + 1, len(shorts)):
                a, b = shorts[a_idx], shorts[b_idx]
                intersection = (masks[a] & masks[b]).sum()
                union = (masks[a] | masks[b]).sum()
                jaccard = intersection / union if union > 0 else 1.0
                row[f"jaccard_{a}_vs_{b}"] = jaccard

        records.append(row)
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(dataset)} reports...")

    return pd.DataFrame(records)


def compute_spearman_rankings(
    dataset,
    processors: Dict[str, SentenceProcessor],
    alpha: float = 0.5,
) -> pd.DataFrame:
    """Compute Spearman rank correlation between embedding models.

    For each document and model pair, computes:
    - Spearman ρ on raw similarity vectors
    - Spearman ρ on final scores (similarity - α·ratio)

    Args:
        dataset: Dataset with 'report' field
        processors: Dict mapping model name to SentenceProcessor instance
        alpha: Length bias parameter for additive formula

    Returns:
        DataFrame with per-document Spearman correlations for all model pairs
    """
    model_shorts = [m.split("/")[-1] for m in processors.keys()]
    pairs = list(combinations(model_shorts, 2))

    records = []
    for i, example in enumerate(dataset):
        text = example["report"]

        vecs = {}  # short → (similarities, scores)
        for model_name, proc in processors.items():
            short = model_name.split("/")[-1]
            features = proc.compute_document_features(text)
            sims = features["similarities"]
            scores = sims - alpha * features["ratios"]
            vecs[short] = (sims, scores)

        row = {"doc": i}
        for a, b in pairs:
            # Spearman on raw similarities
            rho_sim, _ = spearmanr(vecs[a][0], vecs[b][0])
            # Spearman on final scores (similarity - α·ratio)
            rho_score, _ = spearmanr(vecs[a][1], vecs[b][1])
            row[f"spearman_sim_{a}_vs_{b}"] = rho_sim
            row[f"spearman_score_{a}_vs_{b}"] = rho_score

        records.append(row)
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(dataset)} reports...")

    return pd.DataFrame(records)


def judge_with_retry(
    summarizer, generated: str, reference: str, compression: float, retries: int = 2
) -> Dict[str, Any]:
    """Call judge_with_llm, retrying on empty responses.

    Args:
        summarizer: MapReduceSummarizer instance
        generated: Generated summary text
        reference: Reference/gold summary
        compression: Compression ratio percentage
        retries: Number of retry attempts

    Returns:
        Dictionary with judge evaluation results
    """
    for attempt in range(retries + 1):
        evaluation = summarizer.judge_with_llm(
            generated_summary=generated,
            reference_summary=reference,
            compression_ratio=compression,
        )
        if evaluation.get("overall_score") is not None:
            return evaluation
        if attempt < retries:
            print(f"    ⚠ Empty judge response, retrying ({attempt + 1}/{retries})...")
            time.sleep(2)
    return evaluation  # return last attempt even if failed


def compare_llm_summaries(
    dataset,
    pipelines: Dict[str, Any],
    summarizer,
    objective_percentage: float | None = None,
    objective_tokens: int | None = None,
    n_docs: int = 10,
) -> pd.DataFrame:
    """Compare LLM summary quality across embedding models.

    For each document:
    1. Extract sentences using each embedding model
    2. Generate LLM summaries from extractions
    3. Evaluate with ROUGE and LLM judge

    Args:
        dataset: Dataset with 'report' and 'summary' fields
        pipelines: Dict mapping model short name to pipeline instance
        summarizer: MapReduceSummarizer instance for LLM generation and judging
        objective_percentage: Target compression ratio (0-1), or None
        objective_tokens: Absolute target token count, or None.
            If both are provided, uses the minimum.
        n_docs: Number of documents to process

    Returns:
        DataFrame with ROUGE scores, judge scores, and summaries per model
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    records = []

    for i in range(n_docs):
        text = dataset[i]["report"]
        reference = dataset[i]["summary"]
        row = {"doc": i}

        # Step 1: Extract sentences with each model via pipeline
        extractions = {}
        for short, pipe in pipelines.items():
            result = pipe.process_document(
                text,
                objective_percentage=objective_percentage,
                objective_tokens=objective_tokens,
            )
            extractions[short] = result["selected_text"]

        # Step 2: Generate LLM summaries & evaluate with ROUGE
        for short, selected_text in extractions.items():
            llm_summary = summarizer.summarize_with_llm(text=selected_text)

            rouge = scorer.score(reference, llm_summary)
            row[f"{short}_rouge1"] = rouge["rouge1"].fmeasure
            row[f"{short}_rouge2"] = rouge["rouge2"].fmeasure
            row[f"{short}_rougeL"] = rouge["rougeL"].fmeasure
            row[f"{short}_summary"] = llm_summary

        # Step 3: LLM judge on both summaries
        for short in extractions:
            evaluation = judge_with_retry(
                summarizer,
                generated=row[f"{short}_summary"],
                reference=reference,
                compression=objective_percentage * 100,
            )
            row[f"{short}_judge_score"] = evaluation.get("overall_score", None)
            row[f"{short}_judge"] = evaluation

        records.append(row)
        print(f"  Doc {i}: ", end="")
        for short in extractions:
            print(f"{short} judge={row.get(f'{short}_judge_score')}", end=" ")
        print()

    return pd.DataFrame(records)


def print_embedding_model_summary(
    df_models: pd.DataFrame, model_shorts: List[str], objective_pct: float, alpha: float
):
    """Print summary statistics for embedding model comparison.

    Args:
        df_models: DataFrame from compare_embedding_models
        model_shorts: List of short model names (e.g., ['potion-base-2M', ...])
        objective_pct: Compression percentage used
        alpha: Alpha parameter used
    """
    summary_models = pd.DataFrame(
        {
            short: {
                "ROUGE-1": df_models[f"{short}_rouge1"].mean(),
                "ROUGE-2": df_models[f"{short}_rouge2"].mean(),
                "ROUGE-L": df_models[f"{short}_rougeL"].mean(),
                "Avg tokens": df_models[f"{short}_tokens"].mean(),
                "Embed time (s)": df_models[f"{short}_time"].mean(),
            }
            for short in model_shorts
        }
    )

    print(
        f"Embedding model A/B over {len(df_models)} reports at {objective_pct:.0%} compression (α={alpha})\n"
    )
    print(summary_models.to_string(float_format=lambda x: f"{x:.4f}"))

    # Pairwise Jaccard overlap
    print("\n--- Average pairwise Jaccard overlap ---")
    jaccard_cols = [c for c in df_models.columns if c.startswith("jaccard_")]
    for col in jaccard_cols:
        pair = col.replace("jaccard_", "")
        print(f"  {pair:45s}: {df_models[col].mean():.4f}")

    # Win rates: best model per document
    print("\n--- Win counts (best ROUGE-L per document) ---")
    rougeL_cols = {short: f"{short}_rougeL" for short in model_shorts}
    rougeL_df = df_models[list(rougeL_cols.values())]
    rougeL_df.columns = list(rougeL_cols.keys())
    winner = rougeL_df.idxmax(axis=1)
    print(winner.value_counts().to_string())


def print_spearman_summary(df_spearman: pd.DataFrame, model_shorts: List[str]):
    """Print summary statistics for Spearman rank correlation.

    Args:
        df_spearman: DataFrame from compute_spearman_rankings
        model_shorts: List of short model names
    """
    pairs = list(combinations(model_shorts, 2))

    print(f"\n{'Pair':>45s}  {'ρ (similarity)':>16s}  {'ρ (score)':>12s}")
    print("-" * 80)
    for a, b in pairs:
        rho_sim = df_spearman[f"spearman_sim_{a}_vs_{b}"].mean()
        rho_score = df_spearman[f"spearman_score_{a}_vs_{b}"].mean()
        print(f"  {a} vs {b:>20s}  {rho_sim:>16.4f}  {rho_score:>12.4f}")


def print_llm_summary_comparison(
    df_llm_ab: pd.DataFrame, model_shorts: List[str], objective_pct: float, alpha: float
):
    """Print summary statistics for LLM summary comparison.

    Args:
        df_llm_ab: DataFrame from compare_llm_summaries
        model_shorts: List of short model names
        objective_pct: Compression percentage used
        alpha: Alpha parameter used
    """
    summary_llm = pd.DataFrame(
        {
            short: {
                "ROUGE-1": df_llm_ab[f"{short}_rouge1"].mean(),
                "ROUGE-2": df_llm_ab[f"{short}_rouge2"].mean(),
                "ROUGE-L": df_llm_ab[f"{short}_rougeL"].mean(),
                "Judge score": df_llm_ab[f"{short}_judge_score"].mean(),
            }
            for short in model_shorts
        }
    )
    if len(model_shorts) == 2:
        summary_llm["Δ (difference)"] = (
            summary_llm[model_shorts[0]] - summary_llm[model_shorts[1]]
        )

    print(
        f"LLM Summary A/B — {len(df_llm_ab)} docs, {objective_pct:.0%} compression, α={alpha}\n"
    )
    print(summary_llm.to_string(float_format=lambda x: f"{x:.4f}"))

    # Per-document comparison
    print(f"\n{'Doc':>4s}  ", end="")
    for short in model_shorts:
        print(f"{short} ROUGE-L  {short} Judge  ", end="")
    print()
    print("-" * (20 + 30 * len(model_shorts)))

    for _, row in df_llm_ab.iterrows():
        print(f"{int(row['doc']):>4d}  ", end="")
        for short in model_shorts:
            print(
                f"{row[f'{short}_rougeL']:>11.4f}  {row[f'{short}_judge_score']:>10}  ",
                end="",
            )
        print()

    # Win rates
    if len(model_shorts) == 2:
        print(f"\n--- Win rates ({model_shorts[0]} vs {model_shorts[1]}) ---")
        for metric, label in [("rougeL", "ROUGE-L"), ("judge_score", "Judge")]:
            col_a = f"{model_shorts[0]}_{metric}"
            col_b = f"{model_shorts[1]}_{metric}"
            wins_a = (df_llm_ab[col_a] > df_llm_ab[col_b]).sum()
            wins_b = (df_llm_ab[col_a] < df_llm_ab[col_b]).sum()
            ties = (df_llm_ab[col_a] == df_llm_ab[col_b]).sum()
            print(
                f"  {label:>10s}: {model_shorts[0]} wins {wins_a}, {model_shorts[1]} wins {wins_b}, ties {ties}"
            )
