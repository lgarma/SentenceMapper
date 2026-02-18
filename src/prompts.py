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

The text below consists of key sentences that were algorithmically selected based on how similar they are to the surrounding context.
When there is a gap between non-consecutive sentences, it is indicated with the string " (...) ".

# Task

Synthesize these extracted sentences into a coherent, well-structured summary that accurately reflects the main ideas of the report.

# Guidelines:

- Write in full paragraphs, avoid bullet points or lists.
- Avoid referring to "the report" or "the document". Focus on summarizing the content directly.
- Ensure the summary adequately reflects the report’s central research questions, findings, and methodology.
- Include quantitative data and statistics to support key points.
- Prioritize inclusion of specific legislative details:  including bill numbers, actions taken, and outcomes.

# Extracted Sentences

{extracted_text}
"""


JUDGE_SYSTEM_PROMPT = """You are an expert evaluator assessing summary quality for long government documents.

Your role is to judge how well a generated summary captures the essential content of a document.
The reference summary is ONE valid summary, not the only valid one — different emphasis or structure is perfectly acceptable as long as the core information is preserved.
Be fair: a summary that covers the main findings and conclusions well deserves a high score, even if it omits secondary details."""


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
    return f"""Evaluate the GENERATED SUMMARY by comparing it to the REFERENCE SUMMARY.

## Context

The summary was generated using a two-step extractive-abstractive pipeline:
1. **Extraction:** Key sentences were selected from the original document based on information density, compressing it to ~{compression_ratio:.0f}% of the original length. Some details are inevitably lost at this compression level — this is expected and should not be heavily penalized.
2. **Abstraction:** An LLM synthesized the extracted sentences into a coherent summary.

The document comes from the GovReport dataset (U.S. GAO and CRS reports on national policy issues).
The reference summary is a human-written summary of the same document. It represents one valid summary, not the only valid one.

## Scoring rubric

- **9-10:** Covers all major findings, conclusions, and recommendations. Minor omissions only.
- **7-8:** Covers most key points. Misses some secondary details but the reader gets an accurate picture.
- **5-6:** Captures the general topic and some findings, but misses important points or has structural issues.
- **3-4:** Significant gaps in coverage or accuracy issues.
- **1-2:** Largely inaccurate or missing most key content.

## Task

Respond in JSON with these fields:

{{
    "strengths": "What the generated summary does well (coverage, clarity, accuracy).",
    "gaps": "Important content present in the reference but missing from the generated summary. Only flag genuinely important omissions, not minor details.",
    "semantic_bias": "A short list of broad thematic keywords (not report-specific terms) that could help the extraction phase surface more relevant sentences. Example: ['legislative actions', 'funding allocations', 'program outcomes']. These should be useful across many GovReport documents, not just this one.",
    "guidance": "One or two general guidelines for the summarizer that would improve summaries across the entire GovReport dataset. Do NOT reference this specific report. Example: 'Always include the main legislative recommendation and its projected impact.'",
    "overall_score": 7
}}

## Summaries

**Reference summary:**
{reference_summary}

---

**Generated summary:**
{generated_summary}

"""
