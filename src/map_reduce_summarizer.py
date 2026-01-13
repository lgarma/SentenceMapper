"""Map-reduce summarization using SentenceMapper and OpenAI."""

import json
import os
from pathlib import Path
from typing import Dict, Any

from openai import OpenAI
from dotenv import load_dotenv
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

try:
    from .pipeline import SentenceMapperPipeline
    from .prompts import (
        SYSTEM_PROMPT,
        get_map_reduce_summary_prompt,
        JUDGE_SYSTEM_PROMPT,
        get_llm_judge_prompt,
    )
except ImportError:
    # Allow running as a script directly
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.pipeline import SentenceMapperPipeline
    from src.prompts import (
        SYSTEM_PROMPT,
        get_map_reduce_summary_prompt,
        JUDGE_SYSTEM_PROMPT,
        get_llm_judge_prompt,
    )


class MapReduceSummarizer:
    """Map-reduce summarizer using SentenceMapper for extraction and OpenAI for summarization."""

    def __init__(
        self,
        embedding_model_name: str = "minishlab/potion-base-8M",
        chunk_size: int = 2048,
        chunk_overlap: int = 0,
        summarize_model: str = "gpt-4o-mini",
        judge_model: str = "gpt-4.1",
        objective_percentage: float = 0.2,
    ):
        """Initialize the map-reduce summarizer.

        Args:
            embedding_model_name: Name of the embedding model for sentence extraction
            chunk_size: Size of text chunks in tokens
            chunk_overlap: Overlap between chunks in tokens
            openai_model: OpenAI model to use for summarization
            objective_percentage: Target percentage of tokens to extract (e.g., 0.2 = 20%)
        """
        # Load environment variables
        load_dotenv()

        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        self.client = OpenAI(api_key=api_key)
        self.summarize_model = summarize_model
        self.judge_model = judge_model

        # Initialize SentenceMapper pipeline
        self.pipeline = SentenceMapperPipeline(
            embedding_model_name=embedding_model_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        self.objective_percentage = objective_percentage

    def extract_key_sentences(self, text: str) -> str:
        """Extract key sentences from document using SentenceMapper.

        Args:
            text: Input document text

        Returns:
            Extracted key sentences as a single string
        """
        result = self.pipeline.process_document(
            text, objective_percentage=self.objective_percentage
        )
        return result["selected_text"]

    def summarize_with_llm(self, text: str, max_tokens: int = 1000) -> str:
        """Summarize text using OpenAI LLM.

        Args:
            text: Text to summarize (extracted sentences)
            max_tokens: Maximum tokens in the summary

        Returns:
            Generated summary
        """
        prompt = get_map_reduce_summary_prompt(text)

        response = self.client.chat.completions.create(
            model=self.summarize_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=max_tokens,
            temperature=0.3,
        )

        return response.choices[0].message.content.strip()

    def judge_with_llm(
        self,
        generated_summary: str,
        reference_summary: str,
        compression_ratio: float,
        max_tokens: int = 2000,
    ) -> Dict[str, Any]:
        """Evaluate generated summary against reference using LLM judge.

        Args:
            generated_summary: The AI-generated summary
            reference_summary: The human-written reference summary
            compression_ratio: The compression ratio from the extraction phase
            max_tokens: Maximum tokens in the evaluation

        Returns:
            Dictionary containing evaluation sections parsed from JSON response
        """
        prompt = get_llm_judge_prompt(
            generated_summary, reference_summary, compression_ratio
        )

        response = self.client.chat.completions.create(
            model=self.judge_model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=max_tokens,
            temperature=0.3,
            response_format={"type": "json_object"},
        )

        # Parse JSON response
        try:
            evaluation = json.loads(response.choices[0].message.content)
            return evaluation
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return {
                "error": "Failed to parse JSON response",
                "raw_response": response.choices[0].message.content,
            }

    def map_reduce_summarize(self, text: str, max_tokens: int = 1000) -> Dict[str, Any]:
        """Perform map-reduce summarization on a document.

        Args:
            text: Input document text
            max_tokens: Maximum tokens in final summary

        Returns:
            Dictionary containing:
                - extracted_text: Key sentences extracted by SentenceMapper
                - summary: Final summary generated by LLM
                - original_tokens: Token count of original document
                - extracted_tokens: Token count of extracted sentences
                - compression_ratio: Ratio of extracted to original tokens
        """
        # Map step: Extract key sentences
        print("Step 1: Extracting key sentences with SentenceMapper...")
        result = self.pipeline.process_document(
            text, objective_percentage=self.objective_percentage
        )

        extracted_text = result["selected_text"]
        original_tokens = result["total_tokens"]
        extracted_tokens = result["selected_tokens"]
        compression_ratio = (
            extracted_tokens / original_tokens if original_tokens > 0 else 0
        )

        print(f"Extracted {extracted_tokens:,} tokens from {original_tokens:,} tokens")
        print(f"Compression ratio: {compression_ratio:.2%}")

        # Reduce step: Summarize extracted sentences
        print("\nStep 2: Generating summary with OpenAI...")
        summary = self.summarize_with_llm(extracted_text, max_tokens=max_tokens)

        return {
            "extracted_text": extracted_text,
            "summary": summary,
            "original_tokens": int(original_tokens),
            "extracted_tokens": int(extracted_tokens),
            "compression_ratio": float(compression_ratio),
        }


def load_jsonl_example(file_path: str, index: int = 0) -> Dict[str, str]:
    """Load a specific example from a JSONL file.

    Args:
        file_path: Path to the JSONL file
        index: Index of the example to load (default: 0)

    Returns:
        Dictionary with keys: id, pid, input, output
    """
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == index:
                return json.loads(line)
    raise IndexError(f"Index {index} not found in file")


def calculate_rouge_scores(reference: str, candidate: str) -> Dict[str, float]:
    """Calculate ROUGE scores between reference and candidate summaries.

    Args:
        reference: Reference summary (ground truth)
        candidate: Candidate summary (generated)

    Returns:
        Dictionary with ROUGE-1, ROUGE-2, and ROUGE-L F1 scores
    """
    try:
        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        scores = scorer.score(reference, candidate)

        return {
            "rouge1": scores["rouge1"].fmeasure,
            "rouge2": scores["rouge2"].fmeasure,
            "rougeL": scores["rougeL"].fmeasure,
        }
    except ImportError:
        print(
            "\nNote: rouge-score package not installed. Install with: pip install rouge-score"
        )
        return None


def calculate_bleu_score(reference: str, candidate: str) -> float:
    """Calculate BLEU score between reference and candidate summaries.

    Args:
        reference: Reference summary (ground truth)
        candidate: Candidate summary (generated)

    Returns:
        BLEU score (0-1)
    """
    try:
        # Tokenize
        reference_tokens = reference.lower().split()
        candidate_tokens = candidate.lower().split()

        # Calculate BLEU with smoothing
        smoothing = SmoothingFunction().method1
        score = sentence_bleu(
            [reference_tokens], candidate_tokens, smoothing_function=smoothing
        )

        return score
    except ImportError:
        print("\nNote: nltk package not installed. Install with: pip install nltk")
        return None


def main():
    """Main function to demonstrate map-reduce summarization on GovReport dataset."""

    # Configuration
    data_file = Path(__file__).parent.parent / "data" / "gov_report" / "train.jsonl"
    example_index = 1  # np.random.randint(0, 200)  # Random index between 0 and 200

    print("=" * 80)
    print("Map-Reduce Summarization with SentenceMapper + OpenAI")
    print("=" * 80)

    # Load example
    print(f"\nLoading example {example_index} from {data_file.name}...")
    example = load_jsonl_example(str(data_file), example_index)

    print(f"Document ID: {example['id']}")
    print(f"Input length: {len(example['input'])} characters")
    print(f"Reference summary length: {len(example['output'])} characters")

    # Initialize summarizer
    print("\nInitializing MapReduceSummarizer...")
    summarizer = MapReduceSummarizer(
        objective_percentage=0.3,
        summarize_model="gpt-4o-mini",
        judge_model="gpt-4.1",
    )

    # Perform map-reduce summarization
    print("\n" + "=" * 80)
    result = summarizer.map_reduce_summarize(example["input"], max_tokens=1000)

    # Display results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    print(f"\nOriginal document: {result['original_tokens']:,} tokens")
    print(
        f"Extracted sentences: {result['extracted_tokens']:,} tokens ({result['compression_ratio']:.2%})"
    )

    print("\n" + "-" * 80)
    print("GENERATED SUMMARY:")
    print("-" * 80)
    print(result["summary"])

    print("\n" + "-" * 80)
    print("REFERENCE SUMMARY:")
    print("-" * 80)
    print(example["output"])

    # Calculate evaluation metrics
    print("\n" + "=" * 80)
    print("EVALUATION METRICS")
    print("=" * 80)

    # ROUGE scores
    rouge_scores = calculate_rouge_scores(example["output"], result["summary"])
    if rouge_scores:
        print("\nROUGE Scores:")
        print(f"  ROUGE-1 F1: {rouge_scores['rouge1']:.4f}")
        print(f"  ROUGE-2 F1: {rouge_scores['rouge2']:.4f}")
        print(f"  ROUGE-L F1: {rouge_scores['rougeL']:.4f}")

    # BLEU score
    bleu_score = calculate_bleu_score(example["output"], result["summary"])
    if bleu_score is not None:
        print(f"\nBLEU Score: {bleu_score:.4f}")

    # Length comparison
    print("\nLength Comparison:")
    print(f"  Reference summary: {len(example['output'])} characters")
    print(f"  Generated summary: {len(result['summary'])} characters")
    print(f"  Ratio: {len(result['summary']) / len(example['output']):.2f}x")

    # LLM Judge Evaluation
    # print("\n" + "="*80)
    # print("LLM JUDGE EVALUATION")
    # print("="*80)
    # print("\nGenerating detailed evaluation with LLM judge...")

    # judge_evaluation = summarizer.judge_with_llm(
    #    result['summary'],
    #    example['output'],
    #    result['compression_ratio']
    # )

    # Display evaluation sections
    # if "error" in judge_evaluation:
    #    print(f"\nError: {judge_evaluation['error']}")
    #    print(judge_evaluation.get('raw_response', ''))
    # else:
    #    print("\n## STRENGTHS")
    #    print(judge_evaluation.get('strengths', 'N/A'))

    #   print("\n## GAPS & MISSING CONTENT")
    #   print(judge_evaluation.get('gaps', 'N/A'))

    #    print("\n## ACCURACY CONCERNS")
    #    print(judge_evaluation.get('accuracy_concerns', 'N/A'))

    #    print("\n## STRATEGIC RECOMMENDATIONS")
    #    print(judge_evaluation.get('strategic_recommendations', 'N/A'))

    #    print("\n## OVERALL ASSESSMENT")
    #    print(judge_evaluation.get('overall_assessment', 'N/A'))

    print("\n" + "=" * 80)
    print("Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
