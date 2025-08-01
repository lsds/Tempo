from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union

from tempo.core import tensor_ops as top
from tempo.core.configs import ExecutionConfig
from tempo.core.dependence_graph import PDG
from tempo.core.dg_renderer import DGRenderer

# from tempo.transformations.build_schedule import BuildSchedule
from tempo.transformations.analyse_device_assignment import AnalyseDeviceAssignment
from tempo.transformations.analyse_donations import AnalyseDonations
from tempo.transformations.analyse_storage_methods import AnalyseStorageMethods
from tempo.transformations.clean_up_broadcasting_ops import CleanUpBroadcastingOps
from tempo.transformations.compilation_pass import (
    AbstractCompilationPass,
    CompilationCtx,
    CompilationPass,
    Transformation,
)
from tempo.transformations.conditional_handling.insert_merge_data_dependencies import (
    InsertMergeDataDependencies,
)
from tempo.transformations.conditional_handling.propagate_domain_conditions import (
    PropagateDomainConditions,
)
from tempo.transformations.fold_pads_n_masks_into_storage import FoldPadsNMasksIntoStorage
from tempo.transformations.group_dataflow_regions import GroupDataflowRegions
from tempo.transformations.incrementalization.incrementalize import Incrementalize
from tempo.transformations.incrementalization.statifying_incrementalize import (
    StatifyingIncrementalize,
)
from tempo.transformations.individualize_low_cost import IndividualizeLowCost
from tempo.transformations.merge_copy_analysis import MergeCopyAnalysis
from tempo.transformations.optimizer.algebraic.optimizer import AlgebraicOptimizer
from tempo.transformations.optimizer.constant_folding import ConstantFolding
from tempo.transformations.optimizer.dead_code_elimination import DeadCodeElimination
from tempo.transformations.optimizer.domain_reduction import DomainReduction
from tempo.transformations.optimizer.duplicate_code_elim import (
    DuplicateCodeElimination,
)
from tempo.transformations.scheduling.schedule_execution_transform import (
    ScheduleExecution,
)
from tempo.transformations.statify import Statify
from tempo.transformations.validate_pdg import ValidatePDG
from tempo.transformations.vectorization.vectorize import Vectorize
from tempo.utils.logger import get_logger

log = get_logger(__name__)

UninstantiatedPass = Callable[[CompilationCtx], AbstractCompilationPass]
PipelineDesc = List[UninstantiatedPass]

# TODO eventually, parallel Scheduling + Thunk emission. Needs to be given a merge function.
# TODO transformations need to return a ctx, not just a DG, or the DG needs to contain the thunks.
# class Parallel(Transformation):
#    def __init__(self, ctx: TransformationCtx, transforms: PipelineDesc) -> None:
#        super().__init__(ctx)
#        self.transforms = transforms
#
#    def _run(self) -> Tuple[DependenceGraph, bool]:
#        dg = self.ctx.dg
#        cfg = self.ctx.exec_cfg
#        path = self.ctx.exec_cfg.path
#        any_modified = False
#
#        self._try_visualize(dg, f"{path}/{self.name()}_start")
#        for order_num,


class DebugStopPipeline(Transformation):
    def __init__(self, ctx: CompilationCtx) -> None:
        super().__init__(ctx)

    def _run(self) -> Tuple[PDG, bool]:
        raise ValueError("StopPipeline")

    def name(self) -> str:
        return "StopPipeline"


class Iterate(CompilationPass):
    """An Iterate applies a sequence of CompilationPasses to a computation context
    over several iterations, or until a fix-point is hit."""

    def __init__(
        self,
        ctx: CompilationCtx,
        passes: PipelineDesc,
        max_iter: int = 100,
        semantic_name: str = "Iterate",
        quiet: bool = False,
        visualize: bool = False,
    ) -> None:
        super().__init__(ctx)
        self.passes = passes
        self.max_iter = max_iter
        self.quiet = quiet
        # Profile stores runtime in a nested structure
        self.elapsed_ms_profile: Dict[str, Union[float, Dict[str, Any]]] = {"total": 0}
        self.semantic_name = semantic_name

        self.visualize = visualize

    def _try_visualize(self, ctx: CompilationCtx, name: str, is_debug_stage: bool = False) -> None:
        if (
            ctx.exec_cfg.visualize_pipeline_stages
            or (ctx.exec_cfg.visualize_debug_stages and is_debug_stage)
        ) and self.visualize:
            DGRenderer(ctx, out_fname=name).render()

    def _try_validate(self, ctx: CompilationCtx) -> CompilationCtx:
        if ctx.exec_cfg.validate_pipeline_stages:
            ValidatePDG(ctx).run()
        return ctx

    def _update_elapsed(
        self, profile: Dict[str, Any], inst: AbstractCompilationPass, elapsed: float
    ) -> None:
        """Update the elapsed time profile."""
        if isinstance(inst, Iterate):
            # If the pass is an Iterate, create a nested profile if not already present
            nested_profile = profile.setdefault(inst.name(), {"total": 0})
            nested_profile["total"] += elapsed

            # Merge the nested profile from the pass
            for key, value in inst.per_pass_runtime_profile().items():
                if key == "total":
                    continue
                nested_profile[key] = nested_profile.get(key, {})
                if isinstance(value, dict):
                    nested_profile[key] = {**nested_profile[key], **value}
                else:
                    nested_profile[key] = value

        else:
            # For non-iterates, track time flatly
            profile[inst.name()] = profile.get(inst.name(), 0) + elapsed

        # Update the total time for the top-level profile
        profile["total"] += elapsed

    def _run(self) -> Tuple[CompilationCtx, bool]:
        ctx = self.ctx
        path = str(Path(ctx.exec_cfg.path))

        self._try_visualize(ctx, f"{path}/{self.name()}_start", is_debug_stage=True)
        for iteration in range(self.max_iter):
            any_modified = False
            for order_num, pass_factory in enumerate(self.passes):
                # import time
                # time.sleep(1)
                pass_ = pass_factory(ctx)  # Instantiate the pass
                if not self.quiet:
                    if isinstance(pass_, GroupDataflowRegions):
                        self._try_visualize(
                            ctx,
                            f"{path}/{pass_.name()}_{order_num}_PreGrouping",
                            is_debug_stage=True,
                        )

                    log.info("=== Running %s ===", pass_.name())
                ctx, modified, elapsed = pass_.run()
                if not self.quiet:
                    log.info(
                        "=== Finished %s. %ss elapsed ===", pass_.name(), round(elapsed / 1000, 2)
                    )

                # Update the elapsed time profile
                self._update_elapsed(self.elapsed_ms_profile, pass_, elapsed)

                any_modified |= modified
                if not self.quiet:
                    self._try_visualize(
                        ctx,
                        f"{path}/{self.name()}_iter{iteration}_{order_num}_{pass_.name()}",
                    )
                # ctx, _, _ = Statify(ctx).run()
                ctx = self._try_validate(ctx)
            if not any_modified:
                break

        self._try_visualize(ctx, f"{path}/{self.name()}_end", is_debug_stage=True)
        return ctx, any_modified

    def per_pass_runtime_profile(self) -> Dict[str, Union[float, Dict[str, Any]]]:
        """Returns the runtime profile as a nested dictionary."""
        return self.elapsed_ms_profile

    def name(self) -> str:
        return self.semantic_name


class Pipeline(Iterate):
    """A Pipeline applies a sequence of CompilationPasses in a single iteration."""

    def __init__(
        self,
        ctx: CompilationCtx,
        passes: PipelineDesc,
        semantic_name: str = "Pipeline",
        quiet: bool = False,
    ) -> None:
        super().__init__(
            ctx, passes, max_iter=1, semantic_name=semantic_name, quiet=quiet, visualize=True
        )

    @staticmethod
    def get_default_pipeline(
        cfg: ExecutionConfig,
    ) -> UninstantiatedPass:
        optimizer = partial(
            Iterate,
            passes=[
                *([DeadCodeElimination] if cfg.enable_dead_code_elim else []),
                *([DuplicateCodeElimination] if cfg.enable_duplicate_code_elim else []),
                *([AlgebraicOptimizer] if cfg.enable_algebraic_optimizer else []),
                *([DomainReduction] if cfg.enable_domain_reduction else []),
                Statify,
            ],
            max_iter=10,
        )

        optimizer_with_const_fold = partial(
            Iterate,
            passes=[
                *([CleanUpBroadcastingOps] if cfg.enable_broadcast_elim else []),
                *([DeadCodeElimination] if cfg.enable_dead_code_elim else []),
                *([ConstantFolding] if cfg.enable_constant_folding else []),
                *([DuplicateCodeElimination] if cfg.enable_duplicate_code_elim else []),
                *(
                    [partial(AlgebraicOptimizer, is_post_inc=True)]
                    if cfg.enable_algebraic_optimizer
                    else []
                ),
                *([DomainReduction] if cfg.enable_domain_reduction else []),
                Statify,
            ],
            max_iter=10,
        )

        optim = partial(
            optimizer,
            semantic_name="Optim",
        )

        post_vec_optim = partial(
            optimizer,
            semantic_name="VecOptim",
        )

        # TODO: Why only const fold at the end? Does it affect broadcast clean-up maybe?
        final_optim = partial(
            optimizer_with_const_fold,
            semantic_name="FinalOptim",
        )

        pipeline: UninstantiatedPass = partial(
            Pipeline,
            passes=[
                # NOTE: first so that the data dependencies are inserted before the other passes
                InsertMergeDataDependencies,
                optim,
                *(
                    [
                        # LiftPatterns,
                        partial(
                            IndividualizeLowCost,
                            additional_ops_to_individualize=(top.EvalSymbolOp,),
                        ),
                        Vectorize,
                        post_vec_optim,
                    ]  # type: ignore
                    if cfg.enable_vectorization
                    else []
                ),
                # NOTE: run statifying first, as it may lower memory usage too.
                *(
                    [IndividualizeLowCost, StatifyingIncrementalize]
                    if cfg.enable_statifying_incrementalization
                    else []
                ),
                *([IndividualizeLowCost, Incrementalize] if cfg.enable_incrementalization else []),
                final_optim,
                IndividualizeLowCost,
                *(
                    [FoldPadsNMasksIntoStorage]
                    if cfg.enable_fold_pads_into_storage or cfg.enable_pad_mask_removal
                    else []
                ),
                # ApplyTemporalSlicesLazily,
                # TODO ideally we would run this after grouping to allow for parallel analysis,
                # but can't because grouping relies on isl domains.
                PropagateDomainConditions,
                # TODO: similarly, grouping relies on device assignment to decide group device.
                AnalyseDeviceAssignment,
                *([GroupDataflowRegions] if cfg.enable_dataflow_grouping else []),
                AnalyseStorageMethods,
                ScheduleExecution,
                AnalyseDonations,  # TODO: change this so it adds control edges
                MergeCopyAnalysis,
                # NOTE: if swap or donate, we add more constraints, which may change the schedule
                *([ScheduleExecution] if cfg.enable_swap or cfg.enable_donation_analysis else []),
            ],
            semantic_name="MainPipeline",
        )
        return pipeline
