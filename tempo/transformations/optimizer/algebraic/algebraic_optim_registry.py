from typing import Any, Type

from tempo.core import tensor_ops as top
from tempo.core.configs import ExecutionConfig
from tempo.core.subclass_utils import get_all_subclasses
from tempo.core.tensor_op import TensorOp
from tempo.transformations.optimizer.algebraic.implementations import (
    BinEWMovPushdownOptimization,
    ExpandSelectOptimization,
    ExpLnOptimization,
    MatMulConstOptimization,
    MatMulPermuteOptimization,
    MatMulReassocOptimization,
    MatMulToMulOptimization,
    NegNegOptimization,
    OneDivOptimization,
    OneMulOptimization,
    PadExpandElideOptimization,
    PadPushdownOptimization,
    PermutePermuteOptimization,
    RedundantConstIndexSelectOptimization,
    RedundantGatherOptimization,
    RedundantSymbolConstIndexSelectOptimization,
    RedundantSymbolIndexSelectOptimization,
    ReshapeOptimization,
    SliceSliceOptimization,
    SqueezeUnsqueezeOptimization,
    SumDimSizeOneToSqueezeOptimization,
    SumExpandToMulOptimization,
    SumPermuteOptimization,
    UnaryPushdownOptimization,
    ZeroAddOptimization,
    ZeroDivOptimization,
    ZeroMulOptimization,
    ZeroSubOptimization,
)
from tempo.transformations.optimizer.algebraic.match_replacer import (
    MatchReplacer,
    MatchReplacerRegistry,
)
from tempo.transformations.optimizer.algebraic.vec_pattern_lifts import (
    RecurrentCumSumLift,
    RecurrentSumLift,
    SlidingSumLift,
)
from tempo.utils import logger

log = logger.get_logger(__name__)


def build_algebraic_optim_registry(
    exec_cfg: ExecutionConfig, disable_lifts: bool = False
) -> MatchReplacerRegistry:
    registry = MatchReplacerRegistry()

    # Normalize disabled names: lowercase, remove 'optimization'
    disabled = {
        n.lower().replace("optimization", "")
        for n in getattr(exec_cfg, "algebraic_optims_to_disable", [])
    }

    def is_disabled(opt_class: Type[MatchReplacer]) -> bool:
        # Use transform_name if present, else class name
        name = opt_class.transform_name()

        ## If transform_name is a property, get its value
        # if callable(name):
        #    name = name()

        name = name.lower().replace("optimization", "")
        return name in disabled

    def register_if_not_disabled(
        op: Type[TensorOp], opt_class: Type[MatchReplacer], *args: Any, **kwargs: Any
    ) -> None:
        if not is_disabled(opt_class):
            registry.register(op, opt_class(*args, **kwargs))

    # Basic constant optimizations
    register_if_not_disabled(top.AddOp, ZeroAddOptimization)
    register_if_not_disabled(top.SubOp, ZeroSubOptimization)
    register_if_not_disabled(top.MulOp, ZeroMulOptimization)
    register_if_not_disabled(top.MulOp, OneMulOptimization)
    register_if_not_disabled(top.DivOp, ZeroDivOptimization)
    register_if_not_disabled(top.DivOp, OneDivOptimization)

    # Function composition optimizations
    register_if_not_disabled(top.NegOp, NegNegOptimization)
    register_if_not_disabled(top.ExpOp, ExpLnOptimization)
    register_if_not_disabled(top.LnOp, ExpLnOptimization)

    # Shape and movement optimizations
    register_if_not_disabled(top.IndexSliceOp, SliceSliceOptimization)
    register_if_not_disabled(top.SqueezeOp, SqueezeUnsqueezeOptimization)
    register_if_not_disabled(top.UnsqueezeOp, SqueezeUnsqueezeOptimization)
    register_if_not_disabled(top.ReshapeOp, ReshapeOptimization)
    register_if_not_disabled(top.PermuteOp, PermutePermuteOptimization)

    # Sum optimizations
    register_if_not_disabled(top.SumOp, SumDimSizeOneToSqueezeOptimization)
    register_if_not_disabled(top.SumOp, SumExpandToMulOptimization)
    register_if_not_disabled(top.SumOp, SumPermuteOptimization)

    # TODO: Unfortunately, SumSumOptimization seems to fail for some cases. Needs review.
    # register_if_not_disabled(top.SumOp, SumSumOptimization)

    # MatMul optimizations
    register_if_not_disabled(top.MatMulOp, MatMulToMulOptimization)
    register_if_not_disabled(top.MatMulOp, MatMulReassocOptimization)
    register_if_not_disabled(top.MatMulOp, MatMulPermuteOptimization)
    register_if_not_disabled(top.MatMulOp, MatMulConstOptimization)

    # Elementwise optimizations
    for type_ in get_all_subclasses(top.UnaryElementWiseOp):
        register_if_not_disabled(type_, UnaryPushdownOptimization)
        # registry.register(type_, UnaryElementwiseMovementOptimization())
    for type_ in get_all_subclasses(top.BinaryElementWiseOp):
        register_if_not_disabled(type_, BinEWMovPushdownOptimization)

    # Specialized optimizations
    register_if_not_disabled(top.IndexSelectOp, ExpandSelectOptimization)
    register_if_not_disabled(top.IndexSelectOp, RedundantConstIndexSelectOptimization)
    register_if_not_disabled(top.IndexSelectOp, RedundantSymbolIndexSelectOptimization)
    register_if_not_disabled(top.IndexSelectOp, RedundantSymbolConstIndexSelectOptimization)
    register_if_not_disabled(top.GatherOp, RedundantGatherOptimization)
    register_if_not_disabled(top.PadOp, PadExpandElideOptimization)
    register_if_not_disabled(top.PadOp, PadPushdownOptimization)

    # NOTE: Vectorizing Liftings
    if not disable_lifts:  # May want to disable after incrementalization
        register_if_not_disabled(top.MergeOp, RecurrentSumLift)
        register_if_not_disabled(top.MergeOp, RecurrentCumSumLift)
        register_if_not_disabled(top.SumOp, SlidingSumLift)

    opts = [o.transform_name() for o in registry.get_all_registered_optimizations()]
    log.debug(
        "Built algebraic optimizer registry with %d optimizations: %s",
        len(opts),
        opts,
    )

    return registry
