import islpy as isl

from tempo.core import isl_types as islt
from tempo.core.configs import ExecutionConfig


def get_isl_context(exec_cfg: ExecutionConfig) -> islt.Context:
    ctx = isl.Context()

    SCHED_ALGO_PLUTO = 0
    SCHED_ALGO_FEAUTRIER = 1  # noqa: F841

    # --- Scheduling Options
    ctx.set_schedule_algorithm(SCHED_ALGO_PLUTO)
    # print(self.isl_ctx.get_schedule_algorithm())

    # If this option is set, then all strongly connected components in the dependence
    # graph are serialized as soon as they are detected. This means in particular that
    # instances of statements will only appear in the same band node if these statements
    #  belong to the same strongly connected component at the point where the
    # band node is constructed
    # NOTE: Having this on seems to cause there to be no delays

    ctx.set_schedule_serialize_sccs(0)

    # If this option is set, then entire (weakly) connected components in the dependence graph
    # are scheduled together as a whole. Otherwise, each strongly connected component within
    # such a weakly connected component is first scheduled
    # separately and then combined with other strongly connected components. This
    # option has no effect if schedule_serialize_sccs is set.
    # NOTE: Having this on, produces good schedules for the t+3 example, but not the 0:T example
    # NOTE: Setting this to 1 makes schedule times much longer and schedules worse:
    # meaning swap and gc ops are isolated in loops.
    ctx.set_schedule_whole_component(0)

    # If this option is set then the function isl_schedule_get_map will treat set
    # nodes in the same way as sequence nodes.
    # If the schedule_separate_components
    # option is set, then the order of the children of a set node is explicitly encoded in the
    # result.
    # NOTE: Apparently the default is 1. We want 0 when we need parallelism in the union map
    # NOTE: If we are not using parallel execution, then we should set this to 1
    # NOTE: If we are using parallel execution, then we should set this to 0, so no donations
    # enforce the order of the children of a set node
    ctx.set_schedule_separate_components(1)

    # This option is only effective if the schedule_whole_component option is turned
    # off. If the schedule_maximize_coincidence option is set, then (clusters of)
    # strongly connected components are only combined with each other if this does
    # not reduce the number of coincident band members.
    # NOTE: Only effective if schedule_whole_component is not set
    ctx.set_schedule_maximize_coincidence(0)

    # Defaults
    # ctx.get_schedule_serialize_sccs()=0 ctx.get_schedule_whole_component()=0
    # ctx.get_schedule_separate_components()=1, ctx.get_schedule_maximize_coincidence()=0
    # ctx.get_ast_build_group_coscheduled()=0 ctx.get_schedule_treat_coalescing()=1
    # ctx.get_schedule_maximize_band_depth()=0

    # If this option is set, then the scheduler will try and avoid producing schedules that
    # perform loop coalescing. In particular, for the Pluto-like scheduler, this option
    # places bounds on the schedule coefficients based on the sizes of the instance
    # sets.
    ctx.set_schedule_treat_coalescing(0)

    # If this option is set, then the scheduler tries to maximize the width of the bands.
    # Wider bands give more possibilities for tiling. In particular, if the schedule_whole_component
    # option is set, then bands are split if this might result in wider bands.
    # Otherwise, the effect of this option is to only allow strongly connected components
    # to be combined if this does not reduce the width of the bands.
    ctx.set_schedule_maximize_band_depth(0)

    ctx.set_schedule_max_coefficient(-1)
    ctx.set_schedule_max_constant_term(-1)

    # ctx.set_tile_scale_tile_loops(5)
    # ctx.set_tile_shift_point_loops(2)
    # ctx.set_schedule_carry_self_first(1)
    # ctx.set_gbr_only_first(1)

    # --- AST Build Options
    # IMPLICIT = 1
    # ctx.set_ast_build_separation_bounds(1)
    ctx.set_ast_build_detect_min_max(1)

    # If two domain elements are assigned the same schedule point, then they may be
    # executed in any order and they may even appear in different loops. If this options
    # is set, then the AST generator will make sure that coscheduled domain elements
    # do not appear in separate parts of the AST. This is useful in case of nested AST
    # generation if the outer AST generation is given only part of a schedule and the
    # inner AST generation should handle the domains that are coscheduled by this
    # initial part of the schedule together. For example if an AST is generated for a
    # schedule
    # { A[i] -> [0]; B[i] -> [0] }
    # then the isl_ast_build_set_create_leaf callback described below may
    # get called twice, once for each domain. Setting this option ensures that the call-
    # back is only called once on both domains together.

    ctx.set_ast_build_group_coscheduled(0)

    # Simplify conditions based on bounds of nested for loops. In particular, remove
    # conditions that are implied by the fact that one or more nested loops have at least
    # one iteration, meaning that the upper bound is at least as large as the lower bound.
    ctx.set_ast_build_exploit_nested_bounds(1)

    ctx.set_ast_build_allow_else(1)
    ctx.set_ast_build_allow_or(1)
    # ctx.set_ast_always_print_block(0)
    # ctx.set_ast_build_scale_strides(1)

    # --- AST Print Options
    # NOTE: Always print the block node of an if, even if it only has one child
    # ctx.set_ast_print_outermost_block(1)
    return ctx
