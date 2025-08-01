import dataclasses
from typing import Dict, List, Set, Tuple

from tempo.core import index_expr as ie
from tempo.core import tensor_ops as top
from tempo.core.dependence_graph import PDG
from tempo.core.tensor_ops import TensorOp
from tempo.utils import logger

log = logger.get_logger(__name__)

# TODO support vectorizing t:T type expressions. i.e. not const and not point
# TODO: e.g. x[t:T].sum() -> x[0:T].cumsum(0).index_select(0, t)
# Though this really should be done in algebraic simplification

# TODO: if we remove the redundant index check, and instead, in algebraic, we check
# for the redundant index, we get simpler code and more effective optimizations.


@dataclasses.dataclass
class OpVecCtx:
    dg: PDG
    op: top.TensorOp
    vec_dim_symbol: ie.Symbol
    dim_size: ie.IntIndexValueLike
    op_mapping: Dict[TensorOp, TensorOp]
    op_vectorizations: Dict[TensorOp, Tuple[List[ie.Symbol], List[ie.IntIndexValueLike]]]
    ops_to_vectorize: Set[TensorOp]

    @property
    def past_vectorizations(self) -> Tuple[ie.IntIndexValueLike, ...]:
        return tuple(self.op_vectorizations[self.op][1])
