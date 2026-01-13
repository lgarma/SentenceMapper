"""Prompt templates for map-reduce summarization."""

SYSTEM_PROMPT = """You are a helpful assistant that creates clear, comprehensive summaries of government documents and reports.

You excel at synthesizing information from extracted key sentences and creating coherent, well-structured summaries that capture the essential points of complex documents."""


def get_map_reduce_summary_prompt(extracted_text: str) -> str:
    """Get the prompt for map-reduce summarization.

    Args:
        extracted_text: The key sentences extracted using the map strategy

    Returns:
        Formatted prompt for the LLM
    """
    return f"""Please provide a comprehensive summary of the following document.

# Context
The document comes from the GovReport dataset, which contains about 19.5k reports published by the U.S. Government Accountability Office (GAO) and Congressional Research Service (CRS).
They cover researches on a broad range of national policy issues, including health care, education, national security, economics, and the environment.

The text below consists of key sentences that were algorithmically selected based on their information density.
Short sentences that capture the main ideas of the surrounding text.

# Task
Synthesize these extracted sentences into a coherent, well-structured summary that accurately reflects the main ideas and important details of the original document.

# Guidelines:
- Write in full paragraphs, avoid bullet points or lists.
- Avoid vague statements like "the document mentions" or "the report discusses".
- Include data, statistics to support key points where relevant.
- Prioritize inclusion of specific legislative details:  including bill numbers, actions taken, and outcomes


EXTRACTED KEY SENTENCES:
{extracted_text}

SUMMARY:"""


JUDGE_SYSTEM_PROMPT = """You are an expert evaluator specializing in assessing summary quality for long documents.

Your role is to provide constructive, detailed analysis comparing generated summaries against reference summaries."""


def get_llm_judge_prompt(
    generated_summary: str, reference_summary: str, compression_ratio: float
) -> str:
    """Get the prompt for LLM judge evaluation.

    Args:
        generated_summary: The AI-generated summary to evaluate
        reference_summary: The human-written reference summary
        compression_ratio: The compression ratio from the extraction phase (e.g., 0.2 = 20%)

    Returns:
        Formatted prompt for the LLM judge
    """
    return f"""Please evaluate the GENERATED SUMMARY against the REFERENCE SUMMARY and provide a comprehensive analysis.

## Context0
The summary bellow was generated using a map-reduce strategy.

In the map phase, key sentences were extracted from the original document based on their information density.
Using this strategy, the document was compressed by {compression_ratio:.1f} percent.

These sentences were then given to an AI language model to synthesize into a coherent summary.
The model is prompted with some high-level guidelines.

The document comes from the GovReport dataset, which contains about 19.5k reports published by the U.S. Government Accountability Office (GAO) and Congressional Research Service (CRS).
They cover researches on a broad range of national policy issues, including health care, education, national security, economics, and the environment.
The reference summary is a human-written summary of the same document, and serves as the gold standard for comparison.

## Task

Please provide your evaluation in JSON format with the following sections:

{{
    "strengths": "What aspects of the generated summary work well?",
    "gaps": "What important information from the reference is missing or under-represented?"
    "accuracy_concerns": "Are there any inaccuracies, misrepresentations, or misleading statements?",
    "strategic_recommendations": "High-level guidelines could improve the AI writting for future summaries?",
    "overall_score": "A score from 1 to 10 indicating the overall quality of the generated summary compared to the reference."
}}

Please be specific and cite examples from both summaries where relevant.

## Summaries

Reference summary (Human-written ground truth):
{reference_summary}

---

Generated summary (AI-generated using map-reduce strategy):
{generated_summary}

"""
