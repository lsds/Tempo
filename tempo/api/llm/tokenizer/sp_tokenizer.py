from os import PathLike
from pathlib import Path

from sentencepiece import SentencePieceProcessor

from tempo.api.llm.tokenizer.tokenizer import RuntimeTokenizer
from tempo.utils import logger

log = logger.get_logger(__name__)


class SPTokenizer(RuntimeTokenizer):
    """Tokenizing and encoding/decoding text using SentencePiece."""

    def __init__(self, path: PathLike):
        """
        Initializes the Tokenizer with a SentencePiece model.

        Args:
            model_path (str): The path to the SentencePiece model file.
        """
        # reload tokenizer
        path = Path(path)
        assert path.is_file(), f"Tokenizer file {path} does not exist"
        self.sp_model = SentencePieceProcessor(model_file=str(path))
        log.info("Reloaded SentencePiece model from %s", path)

        # BOS / EOS token IDs
        self._n_words: int = self.sp_model.vocab_size()
        self._bos_id: int = self.sp_model.bos_id()
        self._eos_id: int = self.sp_model.eos_id()
        self._pad_id: int = self.sp_model.pad_id()
        log.info(
            "TOKENIZER Initialized: #words: %d - BOS ID: %d - EOS ID: %d",
            self._n_words,
            self._bos_id,
            self._eos_id,
        )
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    @property
    def stop_tokens(self) -> list[int]:
        return [self._eos_id]

    @property
    def n_words(self) -> int:
        """Return the vocabulary size."""
        return self._n_words

    @property
    def bos_id(self) -> int:
        """Return the beginning-of-sequence token ID."""
        return self._bos_id

    @property
    def eos_id(self) -> int:
        """Return the end-of-sequence token ID."""
        return self._eos_id

    @property
    def pad_id(self) -> int:
        """Return the padding token ID."""
        return self._pad_id

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
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self._bos_id] + t
        if eos:
            t = t + [self._eos_id]
        return t  # type: ignore

    def decode(self, t: list[int]) -> str:
        """
        Decodes a list of token IDs into a string.

        Args:
            t (List[int]): The list of token IDs to be decoded.

        Returns:
            str: The decoded string.
        """
        return self.sp_model.decode(t)  # type: ignore
