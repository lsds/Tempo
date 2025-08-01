# Copyright (c) Meta Platforms, Inc. and affiliates.

from abc import ABC, abstractmethod
from typing import List

from tempo.utils import logger

log = logger.get_logger(__name__)


class RuntimeTokenizer(ABC):
    """Abstract base class for tokenizers that can encode/decode text at runtime."""

    @property
    @abstractmethod
    def n_words(self) -> int:
        """Return the vocabulary size."""
        pass

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size."""
        return self.n_words

    @property
    @abstractmethod
    def bos_id(self) -> int:
        """Return the beginning-of-sequence token ID."""
        pass

    @property
    @abstractmethod
    def eos_id(self) -> int:
        """Return the end-of-sequence token ID."""
        pass

    @property
    @abstractmethod
    def pad_id(self) -> int:
        """Return the padding token ID."""
        pass

    @abstractmethod
    def encode(self, s: str, bos: bool = False, eos: bool = False) -> List[int]:
        """
        Encodes a string into a list of token IDs.

        Args:
            s (str): The input string to be encoded.
            bos (bool): Whether to prepend the beginning-of-sequence token.
            eos (bool): Whether to append the end-of-sequence token.

        Returns:
            List[int]: A list of token IDs.
        """
        pass

    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        """
        Decodes a list of token IDs into a string.

        Args:
            t (List[int]): The list of token IDs to be decoded.

        Returns:
            str: The decoded string.
        """
        pass
