"""Tokenizer used by every language model (composition)."""


class Tokenizer:
    """A simple tokenizer that treats every word as one token."""

    def count_tokens(self, text: str) -> int:
        """Return the number of words in the given text."""
        return len(text.split())

    def __repr__(self) -> str:
        """Return a technical representation of the tokenizer."""
        return "Tokenizer()"
