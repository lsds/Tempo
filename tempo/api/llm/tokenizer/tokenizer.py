# Copyright (c) Meta Platforms, Inc. and affiliates.

from abc import ABC, abstractmethod

import numpy as np

from tempo.api.recurrent_tensor import RecurrentTensor
from tempo.core.dtype import dtypes
from tempo.utils import logger

log = logger.get_logger(__name__)


class RuntimeTokenizer(ABC):
    """Abstract base class for tokenizers that can encode/decode text at runtime."""

    @property
    @abstractmethod
    def n_words(self) -> int:
        """Return the vocabulary size."""
        ...

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size."""
        return self.n_words

    @property
    @abstractmethod
    def bos_id(self) -> int:
        """Return the beginning-of-sequence token ID."""
        ...

    @property
    @abstractmethod
    def eos_id(self) -> int:
        """Return the end-of-sequence token ID."""
        ...

    @property
    def pad_id(self) -> int:
        """Return the padding token ID. By default, it is the space token."""
        space_enc = self.encode(" ", bos=False, eos=False)
        assert len(space_enc) == 1, "Space encoding should be a single token"
        return space_enc[0]

    @abstractmethod
    def encode(self, s: str, bos: bool = False, eos: bool = False) -> list[int]:
        """
        Encodes a string into a list of token IDs.

        Args:
            s (str): The input string to be encoded.
            bos (bool): Whether to prepend the beginning-of-sequence token.
            eos (bool): Whether to append the end-of-sequence token.

        Returns:
            List[int]: A list of token IDs.
        """
        ...

    @abstractmethod
    def decode(self, tokens: list[int]) -> str:
        """
        Decodes a list of token IDs into a string.

        Args:
            t (List[int]): The list of token IDs to be decoded.

        Returns:
            str: The decoded string.
        """
        ...

    @property
    @abstractmethod
    def stop_tokens(self) -> list[int]:
        """
        Return the list of stop tokens.
        """
        ...

    def left_pad_token_batch(
        self,
        token_lists: list[list[int]],
        max_prompt_len: int,
    ) -> RecurrentTensor:
        """
        Left-pad every sequence in *token_lists* to *max_prompt_len* and return a recurrent tensor.
        """

        # Find token id for " "
        if self.pad_id != -1:
            pad_token = self.pad_id
        else:
            pad_tokens = self.encode(" ", bos=False, eos=False)
            assert len(pad_tokens) == 1, f"Pad token {pad_tokens} is not a single token"
            pad_token = pad_tokens[0]

        padded = [[pad_token] * (max_prompt_len - len(seq)) + seq for seq in token_lists]
        arr = np.asarray(padded, dtype=dtypes.to_np(dtypes.default_int))
        ret = RecurrentTensor.lift(arr)
        return ret
