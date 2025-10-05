"""Tokenization utilities for text processing and token counting."""

import logging
from typing import List

import tiktoken

logger = logging.getLogger(__name__)


class TokenCounter:
    """Token counting utilities using GPT-4o encoding."""

    def __init__(self):
        """Initialize with GPT-4o tokenizer."""
        self.enc = tiktoken.get_encoding('o200k_base')
        logger.info("Initialized token counter with GPT-4o encoding")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using GPT-4o tokenizer.

        Args:
            text: Input text to tokenize

        Returns:
            Number of tokens
        """
        return len(self.enc.encode(text))

    def count_tokens_batch(self, texts: List[str]) -> List[int]:
        """Count tokens for multiple texts efficiently.

        Args:
            texts: List of texts to tokenize

        Returns:
            List of token counts corresponding to input texts
        """
        return [self.count_tokens(text) for text in texts]

    def estimate_word_to_token_ratio(self, sample_texts: List[str]) -> float:
        """Estimate average tokens per word from sample texts.

        Args:
            sample_texts: List of sample texts for estimation

        Returns:
            Average tokens per word ratio
        """
        total_words = sum(len(text.split()) for text in sample_texts)
        total_tokens = sum(self.count_tokens_batch(sample_texts))
        ratio = total_tokens / max(total_words, 1)  # Avoid division by zero
        logger.info(f"Estimated {ratio:.2f} tokens per word from {len(sample_texts)} samples")
        return ratio
