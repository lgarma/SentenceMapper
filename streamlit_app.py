"""Streamlit app for SentenceMapper extraction visualization."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from datasets import load_dataset
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity

from src.pipeline import SentenceMapperPipeline
from src.sentence_splitter import SentenceSplitter
from src.visualization import SentenceMapperVisualizer

st.set_page_config(page_title="SentenceMapper Comparator", layout="wide")

CUSTOM_PARAMETERS = {
    "prefixes": [
        "H.R",
        "H.Rept",
        "S",
        "P.L",
        "Rep",
        "Sen",
        "S.Rept",
        "U.S",
        "N.Y",
        "Calif",
        "U.N",
    ],
    "additional_replacements": {
        "U.": "U<prd>",
        "S.": "S<prd>",
        "U.S.": "U<prd>S<prd>",
        "U.S.A.": "U<prd>S<prd>A<prd>",
        "i.e.": "i<prd>e<prd>",
        "e.g.": "e<prd>g<prd>",
        "Ph.D.": "Ph<prd>D<prd>",
        "et al.": "et<prd>al<prd>",
    },
}


@st.cache_data(show_spinner=False)
def get_govreport_subset(split: str, size: int):
    return load_dataset("ccdv/govreport-summarization", split=f"{split}[:{size}]")


@st.cache_resource(show_spinner=False)
def get_pipeline(
    embedding_model_name: str,
    context_budget: int,
    min_sentence_length: int,
    length_bias: float = 0.5,
):
    return SentenceMapperPipeline(
        embedding_model_name=embedding_model_name,
        context_budget=context_budget,
        min_sentence_length=min_sentence_length,
        custom_parameters=CUSTOM_PARAMETERS,
        length_bias=length_bias,
    )


def _build_plot(
    result: dict,
    title: str,
    length_bias: float = 0.5,
    objective_percentage: float = 0.3,
) -> go.Figure:
    """Build an interactive scatter comparing baseline vs length-biased selection."""
    similarities = np.asarray(result["similarities"])
    ratios = np.asarray(result["ratios"])
    tokens = np.asarray(result["tokens"])
    sentences = list(result["sentences"])

    fig = SentenceMapperVisualizer.plot_similarity_vs_ratio_interactive(
        similarities=similarities,
        ratios=ratios,
        tokens=tokens,
        sentences=sentences,
        objective_percentage=objective_percentage,
        length_bias=length_bias,
        title=title,
        figsize=(900, 430),
        show=False,
    )

    fig.update_layout(height=430, margin=dict(l=10, r=10, t=45, b=10))
    return fig


def _selected_sentences(result: dict) -> list[str]:
    sentences = result["sentences"]
    mask = np.asarray(result["mask"]).astype(bool)
    return [s for s, keep in zip(sentences, mask) if keep]


def _rouge_scores(reference_summary: str, candidate_text: str) -> dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference_summary, candidate_text)
    return {
        "rouge1": float(scores["rouge1"].fmeasure),
        "rouge2": float(scores["rouge2"].fmeasure),
        "rougeL": float(scores["rougeL"].fmeasure),
    }


def _coverage_redundancy(
    selected_sentences: list[str],
    reference_sentences: list[str],
    embedding_model,
) -> dict[str, float]:
    if not selected_sentences or not reference_sentences:
        return {"coverage": 0.0, "redundancy": 0.0}

    selected_embeddings = embedding_model.encode(selected_sentences)
    reference_embeddings = embedding_model.encode(reference_sentences)

    ref_to_selected_sim = cosine_similarity(reference_embeddings, selected_embeddings)
    coverage = float(np.mean(np.max(ref_to_selected_sim, axis=1)))

    if len(selected_sentences) < 2:
        redundancy = 0.0
    else:
        selected_sim = cosine_similarity(selected_embeddings, selected_embeddings)
        iu = np.triu_indices(selected_sim.shape[0], k=1)
        redundancy = float(np.mean(selected_sim[iu])) if len(iu[0]) > 0 else 0.0

    return {"coverage": coverage, "redundancy": redundancy}


@st.cache_data(show_spinner=False)
def _average_rouge_first_n_reports(
    split: str,
    n_reports: int,
    embedding_model_name: str,
    min_sentence_length: int,
    context_budget: int,
    objective_percentage: float,
    length_bias: float,
) -> dict:
    """Compute average ROUGE over *n_reports*."""
    dataset = load_dataset(
        "ccdv/govreport-summarization", split=f"{split}[:{n_reports}]"
    )

    pipeline = SentenceMapperPipeline(
        embedding_model_name=embedding_model_name,
        context_budget=context_budget,
        min_sentence_length=min_sentence_length,
        length_bias=length_bias,
    )

    acc = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    all_similarities: list[np.ndarray] = []
    all_ratios: list[np.ndarray] = []
    all_tokens: list[np.ndarray] = []

    for example in dataset:
        report = example["report"]
        reference_summary = example["summary"]

        result = pipeline.process_document(
            report, objective_percentage=objective_percentage
        )

        scores = _rouge_scores(reference_summary, result["selected_text"])
        for key in acc:
            acc[key] += scores[key]

        all_similarities.append(np.asarray(result["similarities"], dtype=float))
        all_ratios.append(np.asarray(result["ratios"], dtype=float))
        all_tokens.append(np.asarray(result["tokens"], dtype=float))

    denom = max(len(dataset), 1)
    for key in acc:
        acc[key] /= denom

    if all_similarities:
        similarities_concat = np.concatenate(all_similarities)
        ratios_concat = np.concatenate(all_ratios)
        tokens_concat = np.concatenate(all_tokens)
    else:
        similarities_concat = np.asarray([], dtype=float)
        ratios_concat = np.asarray([], dtype=float)
        tokens_concat = np.asarray([], dtype=float)

    valid_global = (similarities_concat > 0) & (ratios_concat > 0)

    return {
        "count": int(len(dataset)),
        "scores": acc,
        "global": {
            "similarities": similarities_concat[valid_global],
            "ratios": ratios_concat[valid_global],
            "tokens": tokens_concat[valid_global],
        },
    }


def _build_global_ratio_plot(
    similarities: np.ndarray,
    ratios: np.ndarray,
    tokens: np.ndarray,
    objective_percentage: float = 0.3,
    length_bias: float = 0.5,
) -> go.Figure | None:
    if similarities.size < 3 or ratios.size < 3:
        return None

    fig = SentenceMapperVisualizer.plot_similarity_vs_ratio_interactive(
        similarities=similarities,
        ratios=ratios,
        tokens=tokens,
        sentences=None,
        objective_percentage=objective_percentage,
        length_bias=length_bias,
        title="Global Similarity vs Ratio (all evaluated reports pooled)",
        figsize=(1000, 520),
        show=False,
    )

    fig.update_layout(height=520, margin=dict(l=10, r=10, t=45, b=10))
    return fig


@st.cache_data(show_spinner=False)
def get_report_stats(report: str, min_sentence_length: int) -> tuple[int, int]:
    splitter = SentenceSplitter(
        chunk_size=min_sentence_length,
        chunk_overlap=0,
        **CUSTOM_PARAMETERS,
    )
    sentence_count = len(splitter.split_func(report))
    char_count = len(report)
    return char_count, sentence_count


def main() -> None:
    st.title("SentenceMapper — GovReport Explorer")
    st.caption(
        "Extract information-dense sentences using sentence-centered context windows "
        "and score-based ranking."
    )

    with st.sidebar:
        st.header("Configuration")

        split = st.selectbox(
            "GovReport split", ["train", "validation", "test"], index=0
        )
        subset_size = st.slider(
            "Loaded samples", min_value=10, max_value=300, value=50, step=10
        )
        sample_index = st.number_input(
            "Sample index", min_value=0, max_value=subset_size - 1, value=0, step=1
        )

        objective_percentage = st.slider(
            "Target token percentage",
            min_value=0.05,
            max_value=0.80,
            value=0.30,
            step=0.05,
        )

        length_bias = st.number_input(
            "Length bias (α)",
            min_value=0.0,
            max_value=2.0,
            value=0.5,
            step=0.01,
            format="%.2f",
            help=(
                "Linear penalty applied to the length ratio. "
                "score = similarity − α × ratio. "
                "α = 0 → pure similarity ranking. "
                "Higher α penalises longer sentences more."
            ),
        )

        st.divider()
        embedding_model_name = st.text_input(
            "Embedding model", value="minishlab/potion-base-8M"
        )
        min_sentence_length = st.slider(
            "Min sentence length (chars)", 64, 512, 256, step=16
        )
        context_budget = st.slider("Context budget (chars)", 512, 4096, 2048, step=128)
        max_display_tokens = st.slider(
            "Max display tokens (highlighted text)",
            min_value=200,
            max_value=20000,
            value=3000,
            step=200,
        )

        run = st.button("Run extraction", type="primary", use_container_width=True)

    try:
        dataset = get_govreport_subset(split=split, size=subset_size)
    except Exception as exc:  # pragma: no cover
        st.error(f"Could not load GovReport dataset: {exc}")
        return

    example = dataset[int(sample_index)]
    report = example["report"]
    reference_summary = example["summary"]
    char_count, sentence_count = get_report_stats(report, min_sentence_length)

    with st.expander("Dataset example", expanded=False):
        m1, m2 = st.columns(2)
        m1.metric("Characters", f"{char_count:,}")
        m2.metric("Sentences", f"{sentence_count:,}")

        st.markdown("**Report (truncated):**")
        st.write(report[:2500] + ("..." if len(report) > 2500 else ""))
        st.markdown("**Reference summary (truncated):**")
        st.write(
            reference_summary[:1200] + ("..." if len(reference_summary) > 1200 else "")
        )

    if not run and "result" not in st.session_state:
        st.info("Adjust settings and click **Run extraction**.")
        return

    if run:
        with st.spinner("Building pipeline and extracting sentences..."):
            try:
                pipeline = get_pipeline(
                    embedding_model_name=embedding_model_name,
                    context_budget=context_budget,
                    min_sentence_length=min_sentence_length,
                    length_bias=length_bias,
                )

                result = pipeline.process_document(
                    report, objective_percentage=objective_percentage
                )

                selected_sentences = _selected_sentences(result)

                reference_splitter = SentenceSplitter(
                    chunk_size=512,
                    chunk_overlap=0,
                    **CUSTOM_PARAMETERS,
                )
                reference_sentences = [
                    s
                    for s in reference_splitter.split_func(reference_summary)
                    if s.strip()
                ]

                embedding_model = pipeline.processor.embedding_model
                cov_red = _coverage_redundancy(
                    selected_sentences,
                    reference_sentences,
                    embedding_model,
                )

                metrics = {
                    "rouge": _rouge_scores(
                        reference_summary,
                        result["selected_text"],
                    ),
                    "cov_red": cov_red,
                }
            except Exception as exc:  # pragma: no cover
                st.error(f"Processing failed: {exc}")
                return

        st.session_state["result"] = {
            "data": result,
            "objective": objective_percentage,
            "length_bias": length_bias,
            "metrics": metrics,
        }

    stored = st.session_state["result"]
    result = stored["data"]
    objective = stored["objective"]
    stored_length_bias = stored.get("length_bias", 0.5)
    metrics = stored.get("metrics", {})

    st.subheader("Overview")
    o1, o2, o3 = st.columns(3)
    o1.metric("Target", f"{objective:.0%}")
    o2.metric("Selected tokens", f"{int(result['selected_tokens'])}")
    o3.metric("Length bias (α)", f"{stored_length_bias}")

    if metrics:
        st.markdown("### Metrics")
        st.metric(
            "Coverage / Redundancy",
            f"{metrics['cov_red']['coverage']:.3f} / {metrics['cov_red']['redundancy']:.3f}",
        )

        st.markdown("**ROUGE vs reference summary**")
        rouge_table = {
            "Metric": ["ROUGE-1", "ROUGE-2", "ROUGE-L"],
            "Score": [
                metrics["rouge"]["rouge1"],
                metrics["rouge"]["rouge2"],
                metrics["rouge"]["rougeL"],
            ],
        }
        st.dataframe(rouge_table, use_container_width=True)

    total = int(np.sum(result["tokens"]))
    selected = int(result["selected_tokens"])
    st.markdown("### Extraction Results")
    st.caption(
        f"Selected {selected:,} / {total:,} tokens "
        f"({(selected / total if total else 0):.1%})"
    )

    fig = _build_plot(
        result,
        "Similarity vs Ratio",
        length_bias=stored_length_bias,
        objective_percentage=objective,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("### Highlighted document")
    highlighted_html = SentenceMapperVisualizer.display_highlighted_text(
        sentences=list(result["sentences"]),
        mask=np.asarray(result["mask"]),
        title="Sentence Mapping Results",
        dark_mode=False,
        max_display_tokens=max_display_tokens,
    )
    st.components.v1.html(highlighted_html.data, height=320, scrolling=True)

    st.markdown("## Evaluation on first 50 reports")
    if st.button("Run 50-report evaluation", use_container_width=True):
        with st.spinner("Computing ROUGE across first 50 reports..."):
            avg_rouge = _average_rouge_first_n_reports(
                split=split,
                n_reports=50,
                embedding_model_name=embedding_model_name,
                min_sentence_length=min_sentence_length,
                context_budget=context_budget,
                objective_percentage=objective_percentage,
                length_bias=stored_length_bias,
            )

        st.caption(f"Computed on {avg_rouge['count']} reports from split '{split}'.")
        avg_table = {
            "Metric": ["ROUGE-1", "ROUGE-2", "ROUGE-L"],
            "Score (avg)": [
                avg_rouge["scores"]["rouge1"],
                avg_rouge["scores"]["rouge2"],
                avg_rouge["scores"]["rougeL"],
            ],
        }
        st.dataframe(avg_table, use_container_width=True)

        global_eval = avg_rouge.get("global", {})
        similarities_global = np.asarray(
            global_eval.get("similarities", np.asarray([]))
        )
        ratios_global = np.asarray(global_eval.get("ratios", np.asarray([])))
        tokens_global = np.asarray(global_eval.get("tokens", np.asarray([])))

        if similarities_global.size > 0 and ratios_global.size > 0:
            st.markdown("### Global ratio-similarity (all points pooled)")
            global_fig = _build_global_ratio_plot(
                similarities=similarities_global,
                ratios=ratios_global,
                tokens=tokens_global,
                objective_percentage=objective_percentage,
                length_bias=stored_length_bias,
            )
            if global_fig is not None:
                st.caption(f"Points: {similarities_global.size:,}")
                st.plotly_chart(global_fig, use_container_width=True)
        else:
            st.info("Global pooled plot could not be computed for this run.")


if __name__ == "__main__":
    main()
