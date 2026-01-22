"""Custom regex-based sentence splitter with configurable abbreviations.

This module provides a flexible sentence splitter that allows users to define
domain-specific prefixes, suffixes, starters, and acronyms to handle special
cases like government reports where "U.S.", "H.R.", "H.Rept." should not be
treated as sentence boundaries.

Can be used as a standalone sentence splitter or as a langchain TextSplitter
for creating chunks with proper sentence boundaries.

Based on the approach from: https://stackoverflow.com/questions/4576077/
"""

import re
from typing import Any, List

from langchain_text_splitters import TextSplitter


class SentenceSplitter(TextSplitter):
    """Custom regex-based sentence splitter with configurable patterns.

    This splitter extends langchain's TextSplitter to handle domain-specific
    abbreviations. It splits text into sentences respecting abbreviations.

    This splitter uses regex patterns to handle common edge cases in sentence
    splitting such as abbreviations, acronyms, and website domains. Users can
    extend the default patterns with domain-specific terms.

    Example:
        >>> splitter = SentenceSplitter(
        ...     prefixes=["Mr", "Dr", "H.R", "H.Rept"],
        ...     suffixes=["Inc", "Ltd", "Jr"]
        ... )
        >>> text = "H.R. 1234 passed. The U.S. Congress voted."
        >>> chunks = splitter.split_text(text)
    """

    # Default patterns
    DEFAULT_PREFIXES = [
        "Mr",
        "St",
        "Mrs",
        "Ms",
        "Dr",
        "Prof",
        "Capt",
        "Cpt",
        "Lt",
        "Mt",
    ]
    DEFAULT_SUFFIXES = ["Inc", "Ltd", "Jr", "Sr", "Co"]
    DEFAULT_STARTERS = [
        "Mr",
        "Mrs",
        "Ms",
        "Dr",
        "Prof",
        "Capt",
        "Cpt",
        "Lt",
        r"He\s",
        r"She\s",
        r"It\s",
        r"They\s",
        r"Their\s",
        r"Our\s",
        r"We\s",
        r"But\s",
        r"However\s",
        r"That\s",
        r"This\s",
        r"Wherever",
    ]
    DEFAULT_ACRONYMS = r"([A-Z][.][A-Z][.](?:[A-Z][.])?)"
    DEFAULT_WEBSITES = r"[.](com|net|org|io|gov|edu|me|info|biz|online)"
    DEFAULT_DIGITS = r"([0-9])"
    DEFAULT_ALPHABETS = r"([A-Za-z])"
    DEFAULT_MULTIPLE_DOTS = r"\.{2,}"

    def __init__(
        self,
        separator: str = "\n\n",
        prefixes: list[str] | None = None,
        suffixes: list[str] | None = None,
        starters: list[str] | None = None,
        acronyms: str | None = None,
        websites: str | None = None,
        additional_replacements: dict[str, str] | None = None,
        **kwargs: Any,
    ):
        """Initialize the sentence splitter with custom patterns.

        Args:
            separator: Separator to join sentences (default: "\n\n")
            prefixes: List of title/prefix abbreviations (e.g., ["Mr", "Dr", "H.R"])
            suffixes: List of suffix abbreviations (e.g., ["Inc", "Jr"])
            starters: List of sentence starters (can include regex patterns)
            acronyms: Regex pattern for acronyms (e.g., "U.S.A.")
            websites: Regex pattern for website domains
            additional_replacements: Dict of special cases to handle (e.g., {"Ph.D.": "Ph<prd>D<prd>"})
            **kwargs: Additional arguments passed to TextSplitter
        """
        super().__init__(**kwargs)
        self._separator = separator

        # Combine default patterns with user-provided ones
        self.prefixes = self._combine_patterns(self.DEFAULT_PREFIXES, prefixes)
        self.suffixes = self._combine_patterns(self.DEFAULT_SUFFIXES, suffixes)
        self.starters = self._combine_patterns(self.DEFAULT_STARTERS, starters)

        self.acronyms = acronyms if acronyms else self.DEFAULT_ACRONYMS
        self.websites = websites if websites else self.DEFAULT_WEBSITES
        self.digits = self.DEFAULT_DIGITS
        self.alphabets = self.DEFAULT_ALPHABETS
        self.multiple_dots = self.DEFAULT_MULTIPLE_DOTS

        # Special replacements for specific terms
        self.additional_replacements = additional_replacements or {}

        # Compile regex patterns
        self._compile_patterns()

    def _combine_patterns(
        self, defaults: list[str], custom: list[str] | None
    ) -> list[str]:
        """Combine default patterns with custom ones."""
        if custom is None:
            return defaults
        # Merge and deduplicate
        combined = list(set(defaults + custom))
        return combined

    def _compile_patterns(self):
        """Compile all regex patterns for efficiency."""
        # Create regex patterns
        self.prefix_pattern = f"({'|'.join(self.prefixes)})[.]"
        self.suffix_pattern = f"({'|'.join(self.suffixes)})"
        self.starter_pattern = f"({'|'.join(self.starters)})"

    def split_func(self, text: str) -> list[str]:
        """Split the text into sentences.

        If the text contains substrings "<prd>" or "<stop>", they would lead
        to incorrect splitting because they are used as markers for splitting.

        Args:
            text: Text to be split into sentences

        Returns:
            List of sentences
        """
        text = " " + text + "  "
        text = text.replace("\n", " ")

        # Handle prefixes (Mr. Dr. etc.)
        text = re.sub(self.prefix_pattern, r"\1<prd>", text)

        # Handle websites
        text = re.sub(self.websites, r"<prd>\1", text)

        # Handle digits with periods (e.g., 5.5)
        text = re.sub(self.digits + "[.]" + self.digits, r"\1<prd>\2", text)

        # Handle multiple dots (ellipsis)
        text = re.sub(
            self.multiple_dots,
            lambda match: "<prd>" * len(match.group(0)) + "<stop>",
            text,
        )

        # Handle additional special replacements
        for key, value in self.additional_replacements.items():
            if key in text:
                text = text.replace(key, value)

        # Handle single letter abbreviations (e.g., "U. S.")
        text = re.sub(r"\s" + self.alphabets + "[.] ", r" \1<prd> ", text)

        # Handle acronyms followed by starters
        text = re.sub(self.acronyms + " " + self.starter_pattern, r"\1<stop> \2", text)

        # Handle three-letter acronyms
        text = re.sub(
            self.alphabets + "[.]" + self.alphabets + "[.]" + self.alphabets + "[.]",
            r"\1<prd>\2<prd>\3<prd>",
            text,
        )

        # Handle two-letter acronyms
        text = re.sub(
            self.alphabets + "[.]" + self.alphabets + "[.]", r"\1<prd>\2<prd>", text
        )

        # Handle suffixes followed by starters
        text = re.sub(
            " " + self.suffix_pattern + "[.] " + self.starter_pattern,
            r" \1<stop> \2",
            text,
        )

        # Handle suffixes at end of sentence
        text = re.sub(" " + self.suffix_pattern + "[.]", r" \1<prd>", text)

        # Handle single letter at end of sentence
        text = re.sub(" " + self.alphabets + "[.]", r" \1<prd>", text)

        # Handle quotes with punctuation
        if "\u201c" in text:  # Left double quotation mark
            text = text.replace(".\u201d", "\u201d.")  # Right double quotation mark
        if '"' in text:
            text = text.replace('."', '".')
        if "!" in text:
            text = text.replace('!"', '"!')
        if "?" in text:
            text = text.replace('?"', '"?')

        # Mark sentence boundaries
        text = text.replace(".", ".<stop>")
        text = text.replace("?", "?<stop>")
        text = text.replace("!", "!<stop>")

        # Restore periods
        text = text.replace("<prd>", ".")

        # Split and clean
        sentences = text.split("<stop>")
        sentences = [s.strip() for s in sentences]

        # Remove empty strings
        if sentences and not sentences[-1]:
            sentences = sentences[:-1]

        return sentences

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks using sentence boundaries.

        This is the main method required by langchain's TextSplitter interface.
        It splits text into sentences first, then merges them using _merge_splits.

        Args:
            text: Text to be split into chunks

        Returns:
            List of text chunks
        """
        # First split into sentences
        sentences = self.split_func(text)

        return self._merge_splits(sentences, self._separator)

    # Alias for backward compatibility
    def split(self, text: str) -> list[str]:
        """Alias for split_sentences() for backward compatibility."""
        return self.split_sentences(text)
