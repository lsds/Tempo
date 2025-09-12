from collections import Counter

from tempo.core.compilation_ctx import CompilationCtx
from tempo.core.dependence_graph import PDG
from tempo.transformations.compilation_pass import Transformation
from tempo.transformations.optimizer.algebraic.algebraic_optim_registry import (
    build_algebraic_optim_registry,
)
from tempo.transformations.optimizer.dead_code_elimination import DeadCodeElimination
from tempo.utils import logger

log = logger.get_logger(__name__)


class AlgebraicOptimizer(Transformation):
    def __init__(self, ctx: CompilationCtx, is_post_inc: bool = False):
        super().__init__(ctx)
        self.reg = build_algebraic_optim_registry(ctx.exec_cfg, disable_lifts=is_post_inc)

    def _run(self) -> tuple[PDG, bool]:  # noqa: C901
        dg = self.ctx.dg
        new_ctx = self.ctx

        transform_count: Counter[str] = Counter()
        changes = True

        while changes:
            changes = False
            round_transform_count: Counter[str] = Counter()

            for op in list(dg.nodes):
                # Try each replacer until one succeeds
                for mr in self.reg.get_op_match_replacers(op):
                    try:
                        match_result = mr.match(new_ctx, op)
                    except Exception as e:
                        log.debug(
                            "Error while matching %s on %s: \n %s", mr.transform_name(), op, e
                        )
                        continue

                    if match_result:
                        log.debug(
                            "Applying %s with match result %s", mr.transform_name(), match_result
                        )
                        try:
                            mr.replace(new_ctx, op, match_result)
                            log.debug(
                                "Applied %s with match result %s", mr.transform_name(), match_result
                            )
                        except Exception as e:
                            log.error(
                                "Error during %s with match result %s: \n %s",
                                mr.transform_name(),
                                match_result,
                                e,
                            )
                            raise e
                        round_transform_count[mr.transform_name()] += 1
                        changes = True
                        # NOTE: stop after first successful match: move on to next op,
                        #  but we still want to try all the other replacers.
                        break

            ctx, _, _ = DeadCodeElimination(new_ctx).run()
            new_ctx = ctx
            dg = ctx.dg

            # Update transform counts
            transform_count.update(round_transform_count)

        total_transforms = sum(transform_count.values())
        log.info(
            "Performed %d algebraic optimizations: %s",
            total_transforms,
            dict(transform_count),
        )
        return dg, total_transforms > 0
