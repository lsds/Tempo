from __future__ import annotations

import tempo.core.index_expr as ie
from tempo.api.nn.linear import Linear
from tempo.api.nn.module import MaybeInitFnOrList, Module
from tempo.api.nn.rope import RopeScalingParams, RotaryPositionalEmbeddings
from tempo.api.recurrent_tensor import RecurrentTensor
from tempo.core.domain import DomainLike
from tempo.core.dtype import DataTypeLike, dtypes
from tempo.core.index_expr import Symbol


class MultiHeadAttention(Module):
    """
    Args:
        embed_dim (int): Total embedding dimension.
        num_heads (int): Number of attention heads.
        seq_len (Symbol): Symbolic representation of sequence length
        dropout (float): Dropout probability (not implemented yet).
        dtype (DataType): Data type for the module.
        domain (DomainLike): Domain for the module.
        independent_domain (DomainLike): Independent dimensions for the module.
        w_init_funs: weights initialization functions for Q, K, V and output projection,
        b_init_funs: bias initialization functions for Q, K, V and output projection
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        seq_len: Symbol,
        dropout: float = 0.0,
        bias: bool = False,
        apply_rope: bool = False,
        dtype: DataTypeLike = None,
        domain: DomainLike = None,
        independent_domain: DomainLike = None,
        w_init_funs: MaybeInitFnOrList = None,
        b_init_funs: MaybeInitFnOrList = None,
        rope_theta: float = 10_000.0,
        rope_scaling_params: RopeScalingParams | None = None,
        num_kv_heads: int | None = None,
    ) -> None:
        super().__init__(domain, independent_domain)

        dtype = dtypes.from_(dtype, none_dtype=dtypes.default_float)

        if embed_dim % num_heads != 0:
            raise ValueError(f"{embed_dim=} must be divisible by {num_heads=}")
        if num_kv_heads is None:
            num_kv_heads = num_heads
        if embed_dim % num_kv_heads != 0:
            raise ValueError(f"{embed_dim=} must be divisible by {num_kv_heads=}")
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"{num_heads=} must be divisible by {num_kv_heads=} for grouped query attention"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_heads
        self.seq_len = seq_len
        self.dropout = dropout
        self.apply_rope = apply_rope

        # Handle w_init_funs: None, single function, or list
        if w_init_funs is None:
            w_init_funs = [None, None, None, None]
        elif not isinstance(w_init_funs, list):
            # If w_init_funs is a single function, convert to a list of 4 identical functions
            w_init_funs = [w_init_funs, w_init_funs, w_init_funs, w_init_funs]
        elif len(w_init_funs) == 1:
            w_init_funs = [w_init_funs[0], w_init_funs[0], w_init_funs[0], w_init_funs[0]]
        elif len(w_init_funs) != 4:
            raise ValueError("w_init_funs must have length 4")

        # Handle b_init_funs: None, single function, or list
        if b_init_funs is None:
            b_init_funs = [None, None, None, None]
        elif not isinstance(b_init_funs, list):
            # If b_init_funs is a single function, convert to a list of 4 identical functions
            b_init_funs = [b_init_funs, b_init_funs, b_init_funs, b_init_funs]
        elif len(b_init_funs) == 1:
            b_init_funs = [b_init_funs[0], b_init_funs[0], b_init_funs[0], b_init_funs[0]]
        elif len(b_init_funs) != 4:
            raise ValueError("b_init_funs must have length 4")

        # Linear projections
        self.q_proj = Linear(
            embed_dim,
            self.num_heads * self.head_dim,
            bias=bias,
            dtype=dtype,
            domain=domain,
            independent_domain=independent_domain,
            w_init_fun=w_init_funs[0],
            b_init_fun=b_init_funs[0],
        )
        self.k_proj = Linear(
            embed_dim,
            self.num_kv_heads * self.head_dim,
            bias=bias,
            dtype=dtype,
            domain=domain,
            independent_domain=independent_domain,
            w_init_fun=w_init_funs[1],
            b_init_fun=b_init_funs[1],
        )
        self.v_proj = Linear(
            embed_dim,
            self.num_kv_heads * self.head_dim,
            bias=bias,
            dtype=dtype,
            domain=domain,
            independent_domain=independent_domain,
            w_init_fun=w_init_funs[2],
            b_init_fun=b_init_funs[2],
        )
        self.output_proj = Linear(
            embed_dim,
            embed_dim,
            bias=bias,
            dtype=dtype,
            domain=domain,
            independent_domain=independent_domain,
            w_init_fun=w_init_funs[3],
            b_init_fun=b_init_funs[3],
        )

        if self.apply_rope:
            self.rope = RotaryPositionalEmbeddings(
                dim=self.head_dim,
                t=seq_len,
                base=rope_theta,
                domain=domain,
                independent_domain=independent_domain,
                rope_scaling_params=rope_scaling_params,
            )

        # Scaling factor for attention
        self.scale = RecurrentTensor.sqrt(self.head_dim)

        self.n_rep = self.num_heads // self.num_kv_heads

    def forward(self, x: RecurrentTensor, pattern: ie.IndexAtom | None = None) -> RecurrentTensor:
        """
        Computes multi-head attention
        Args:
            x: Input tensor.
            pattern: Optional index expression atom, indicating the attention pattern. By default,
            causal attention is used.
        """

        if pattern is None:
            return self.causal_forward(x)

        return self._forward(x, pattern)

    def causal_forward(self, x: RecurrentTensor) -> RecurrentTensor:
        """
        Computes causal multi-head attention.
        Args:
            x: Input tensor.
        """
        return self._forward(x, ie.Slice(ie.ConstInt(0), self.seq_len + 1))

    def windowed_forward(self, x: RecurrentTensor, window_size: int) -> RecurrentTensor:
        """
        Computes windowed multi-head attention.
        Args:
            x: Input tensor.
            window_size: int, the size of the window
        """
        pat = ie.slice_(ie.max(0, self.seq_len - window_size + 1), self.seq_len + 1)
        return self._forward(x, pat)

    def _forward(self, x: RecurrentTensor, pattern: ie.IndexAtom) -> RecurrentTensor:
        seq_len_idx = x.domain.find_variable_index(self.seq_len)
        attention_pattern = x.domain.basis_expr.replace_idx(seq_len_idx, pattern)

        # Project inputs to queries, keys, and values
        q = self.q_proj(x).reshape((self.num_heads, self.head_dim))
        k = self.k_proj(x).reshape((self.num_kv_heads, self.head_dim))
        v = self.v_proj(x).reshape((self.num_kv_heads, self.head_dim))

        if self.apply_rope:
            q = self.rope(q)
            k = self.rope(k)

        Q = q

        K = k[attention_pattern].repeat_interleave(self.n_rep, dim=1).transpose(0, 1)
        V = v[attention_pattern].repeat_interleave(self.n_rep, dim=1).transpose(0, 1)

        # Compute attention scores: Q * K_transpose / sqrt(dim)
        attention_scores = (Q.unsqueeze(1) @ K.transpose(1, 2)) / RecurrentTensor.lift(self.scale)

        # assert attention_scores.dtype == dtypes.float16, "attention_scores.dtype is not float16"

        # Apply softmax to get attention weights
        attention_weights = attention_scores.softmax(dim=-1)
        attention_weights = attention_weights.unsqueeze(1)

        # Apply attention weights to values
        attention_output = attention_weights @ V
        assert attention_output.shape == (
            self.num_heads,
            self.head_dim,
        ), f"Expected shape {self.num_heads, self.head_dim}, got {attention_output.shape}"

        # Reshape back to original shape
        output = attention_output.reshape(self.embed_dim)

        # Final linear projection
        output = self.output_proj(output)  # type: RecurrentTensor

        return output
