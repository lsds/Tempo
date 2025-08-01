import base64
from pathlib import Path
from typing import (
    AbstractSet,
    Collection,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Union,
    cast,
)

import tiktoken

from tempo.api.llm.tokenizer.tokenizer import RuntimeTokenizer
from tempo.utils import logger

log = logger.get_logger(__name__)


def load_bpe_file(model_path: Path) -> dict[bytes, int]:
    """
    Load BPE file directly and return mergeable ranks.

    Args:
        model_path (Path): Path to the BPE model file.

    Returns:
        dict[bytes, int]: Dictionary mapping byte sequences to their ranks.
    """
    mergeable_ranks = {}

    with open(model_path, encoding="utf-8") as f:
        content = f.read()

    for line in content.splitlines():
        if not line.strip():  # Skip empty lines
            continue
        try:
            token, rank = line.split()
            mergeable_ranks[base64.b64decode(token)] = int(rank)
        except Exception as e:
            log.warning("Failed to parse line '%s': %s", line, e)
            continue

    return mergeable_ranks


# The tiktoken tokenizer can handle <=400k chars without
# pyo3_runtime.PanicException.
TIKTOKEN_MAX_ENCODE_CHARS = 400_000

# https://github.com/openai/tiktoken/issues/195
# Here we iterate over subsequences and split if we exceed the limit
# of max consecutive non-whitespace or whitespace characters.
MAX_NO_WHITESPACES_CHARS = 25_000


_INSTANCE = None


class TiktokenTokenizer(RuntimeTokenizer):
    """
    Tokenizing and encoding/decoding text using the Tiktoken tokenizer.
    """

    special_tokens: Dict[str, int]

    num_reserved_special_tokens = 256

    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501

    @classmethod
    def get_instance(cls) -> "TiktokenTokenizer":
        global _INSTANCE

        if _INSTANCE is None:
            _INSTANCE = TiktokenTokenizer(Path(__file__).parent / "tokenizer.model")
        return _INSTANCE

    def __init__(self, model_path: Path) -> None:
        """
        Initializes the Tokenizer with a Tiktoken model.

        Args:
            model_path (str): The path to the Tiktoken model file.
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Tokenizer model file not found: {model_path}")

        mergeable_ranks = load_bpe_file(model_path)
        num_base_tokens = len(mergeable_ranks)
        special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|finetune_right_pad_id|>",
            "<|step_id|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|eom_id|>",  # end of message
            "<|eot_id|>",  # end of turn
            "<|python_tag|>",
            "<|image|>",
        ]
        reserved_tokens = [
            f"<|reserved_special_token_{2 + i}|>"
            for i in range(self.num_reserved_special_tokens - len(special_tokens))
        ]
        special_tokens = special_tokens + reserved_tokens

        self.special_tokens = {token: num_base_tokens + i for i, token in enumerate(special_tokens)}
        self.model = tiktoken.Encoding(
            name=model_path.name,
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )

        self._num_base_tokens: int = num_base_tokens
        self._n_words: int = num_base_tokens + len(special_tokens)
        # BOS / EOS token IDs
        self._bos_id: int = self.special_tokens["<|begin_of_text|>"]
        self._eos_id: int = self.special_tokens["<|end_of_text|>"]
        self._eot_id: int = self.special_tokens["<|eot_id|>"]
        self._eom_id: int = self.special_tokens["<|eom_id|>"]
        self._python_tag_id = self.special_tokens["<|python_tag|>"]
        self._pad_id: int = self.special_tokens["<|finetune_right_pad_id|>"]
        self._stop_tokens = [
            self._eos_id,
            self.special_tokens["<|eom_id|>"],
            self.special_tokens["<|eot_id|>"],
        ]
        log.info(
            "TOKENIZER Initialized: #base_tokens: %d - #words: %d - BOS ID: %d - EOS ID: %d - EOT ID: %d - EOM ID: %d - Python Tag ID: %d - Pad ID: %d - Stop Tokens: %s",  # noqa: E501
            self._num_base_tokens,
            self._n_words,
            self._bos_id,
            self._eos_id,
            self._eot_id,
            self._eom_id,
            self._python_tag_id,
            self._pad_id,
            self._stop_tokens,
        )

    @property
    def n_words(self) -> int:
        return self._n_words
        # return self._num_base_tokens

    @property
    def bos_id(self) -> int:
        return self._bos_id

    @property
    def eos_id(self) -> int:
        return self._eos_id

    @property
    def pad_id(self) -> int:
        return self._pad_id

    @property
    def stop_tokens(self) -> List[int]:
        return self._stop_tokens

    def encode(
        self,
        s: str,
        bos: bool = False,
        eos: bool = False,
        allowed_special: Optional[Union[Literal["all"], AbstractSet[str]]] = None,
        disallowed_special: Union[Literal["all"], Collection[str]] = (),
    ) -> List[int]:
        """
        Encodes a string into a list of token IDs.

        Args:
            s (str): The input string to be encoded.
            bos (bool): Whether to prepend the beginning-of-sequence token.
            eos (bool): Whether to append the end-of-sequence token.
            allowed_special ("all"|set[str]): allowed special tokens in string
            disallowed_special ("all"|set[str]): special tokens that raise an error when in string

        Returns:
            list[int]: A list of token IDs.

        By default, setting disallowed_special=() encodes a string by ignoring
        special tokens. Specifically:
        - Setting `disallowed_special` to () will cause all text corresponding
          to special tokens to be encoded as natural text (insteading of raising
          an error).
        - Setting `allowed_special` to "all" will treat all text corresponding
          to special tokens to be encoded as special tokens.
        """
        if allowed_special is None:
            allowed_special = set()
        assert type(s) is str

        substrs = (
            substr
            for i in range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS)
            for substr in self._split_whitespaces_or_nonwhitespaces(
                s[i : i + TIKTOKEN_MAX_ENCODE_CHARS], MAX_NO_WHITESPACES_CHARS
            )
        )
        t: List[int] = []
        for substr in substrs:
            t.extend(
                self.model.encode(
                    substr,
                    allowed_special=allowed_special,
                    disallowed_special=disallowed_special,
                )
            )
        if bos:
            t.insert(0, self._bos_id)
        if eos:
            t.append(self._eos_id)
        return t

    def decode(self, t: Sequence[int]) -> str:
        """
        Decodes a list of token IDs into a string.

        Args:
            t (List[int]): The list of token IDs to be decoded.

        Returns:
            str: The decoded string.
        """
        # Typecast is safe here. Tiktoken doesn't do anything list-related with the sequence.
        return self.model.decode(cast(List[int], t))  # type: ignore

    @staticmethod
    def _split_whitespaces_or_nonwhitespaces(
        s: str, max_consecutive_slice_len: int
    ) -> Iterator[str]:
        """
        Splits the string `s` so that each substring contains
        no more than `max_consecutive_slice_len`
        consecutive whitespaces or consecutive non-whitespaces.
        """
        current_slice_len = 0
        current_slice_is_space = s[0].isspace() if len(s) > 0 else False
        slice_start = 0

        for i in range(len(s)):
            is_now_space = s[i].isspace()

            if current_slice_is_space ^ is_now_space:
                current_slice_len = 1
                current_slice_is_space = is_now_space
            else:
                current_slice_len += 1
                if current_slice_len > max_consecutive_slice_len:
                    yield s[slice_start:i]
                    slice_start = i
                    current_slice_len = 1
        yield s[slice_start:]
