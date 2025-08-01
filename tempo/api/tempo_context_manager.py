from __future__ import annotations

import functools
import threading
from pathlib import Path
from types import TracebackType
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Type, Union

from tempo.api.compiler import Compiler
from tempo.api.recurrent_tensor import RecurrentTensor
from tempo.core import index_expr as ie
from tempo.core import isl_types as islt
from tempo.core.configs import ExecutionConfig
from tempo.core.dependence_graph import PDG
from tempo.core.dg_renderer import raise_error_with_pdg_render
from tempo.core.domain import Domain, DomainLike
from tempo.core.global_objects import (
    DomainCtxManager,
    GroupTagCtxManager,
    NoDedupCtxManager,
    RegionTagCtxManager,
    get_active_exec_cfg,
)
from tempo.core.isl_context_factory import get_isl_context
from tempo.core.utils import make_symbols
from tempo.runtime.executor.executor import Executor
from tempo.utils import logger

log = logger.get_logger(__name__)

Initializer = RecurrentTensor


class CtxManagerJoiner(object):
    def __init__(self, *managers: Any) -> None:
        self.managers = managers

    def __enter__(self) -> None:
        for manager in self.managers:
            manager.__enter__()

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        for manager in self.managers:
            manager.__exit__(exc_type, exc_value, traceback)


class ConditionCtxManager(object):
    def __init__(self, condition: ie.BooleanIndexValue) -> None:
        self.condition = condition

    def __enter__(self) -> None:
        get_active_ctx_manager().active_conditions.append(self.condition)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        get_active_ctx_manager().active_conditions.pop()


class ThreadLocalManagerStack(threading.local):
    def __init__(self) -> None:
        super().__init__()
        self.manager_stack: List[TempoContext] = []


DEFAULT_EXEC_CFG = ExecutionConfig.default()


class TempoContext(object):
    _active = ThreadLocalManagerStack()

    def __init__(
        self,
        execution_config: ExecutionConfig = DEFAULT_EXEC_CFG,
        num_dims: Optional[int] = None,
    ) -> None:
        """Creates a new Tempo context under which your algorithm is recorded.
        Any operations performed on RecurrentTensors under this context are recorded in a dependence
        graph. This graph is then used to compile your algorithm into a backend of your choice.
        To provide dimensions, you must pass either num_dimensions or dimensions, but not both.

        Args:
            num_dimensions (Optional[int], optional): The number of symbolic dimensions your
            algorithm uses. Defaults to None.

        """
        if num_dims is not None and num_dims > 0:
            dimensions = " ".join([f"d{i}" for i in range(num_dims)])
            symbols: List[Tuple[ie.Symbol, ie.Symbol]] = make_symbols(
                dimensions.strip().split(" "), 0
            )
            universe = Domain.from_vars_and_bounds(
                list(zip(*symbols, strict=False))[0], list(zip(*symbols, strict=False))[1]
            )
        else:
            symbols = []
            universe = Domain.from_vars_and_bounds((), ())

        self.dg = PDG(universe)

        from tempo.core import global_objects

        global_objects.set_active_dg(self.dg)
        global_objects.set_active_config(execution_config)
        self.execution_config = execution_config
        self.active_conditions: List[ie.BooleanIndexValue] = []

        # NOTE: other variables may be created after, but should not be visible
        # to ctx
        self.visible_variables = list(universe.variables)
        self.visible_bounds = list(universe.parameters)

    def when(self, condition: ie.BooleanIndexValue) -> ConditionCtxManager:
        return ConditionCtxManager(condition)

    def tag_region(self, tag: str) -> RegionTagCtxManager:
        region_man = RegionTagCtxManager(tag)
        # group_man = GroupTagCtxManager(tag)

        # return CtxManagerJoiner(region_man, group_man)  # type: ignore
        return region_man

    def forbid_dedup(self) -> NoDedupCtxManager:
        return NoDedupCtxManager()

    def tag_group(self, tag: str) -> GroupTagCtxManager:
        return GroupTagCtxManager(tag)

    def domain_ctx(self, domain: DomainLike) -> DomainCtxManager:
        return DomainCtxManager(domain)

    def new_perm_var(self, bound: ie.IntIndexValueLike) -> Tuple[ie.Symbol, ie.Symbol]:
        return self.dg.new_perm_var(bound)

    # def new_temp_var(
    #    self, bound: Optional[ie.IntIndexValueLike] = None
    # ) -> Tuple[ie.Symbol, ie.Symbol]:
    #    return self.dg.new_temp_var(bound)

    @property
    def current_condition(self) -> ie.BooleanIndexValue:
        if len(self.active_conditions) == 0:
            return ie.ConstBool(True)
        elif len(self.active_conditions) == 1:
            return self.active_conditions[0]
        else:
            return functools.reduce(lambda a, b: a & b, self.active_conditions)

    # TODO for some god-forsaken reason, mypy thinks the following 4 functions return Any
    @property
    def universe(self) -> Domain:
        return self.dg.universe  # type: ignore

    @property
    def variables(self) -> Sequence[ie.Symbol]:
        return tuple(self.visible_variables)

    @property
    def upper_bounds(self) -> Sequence[ie.Symbol]:
        return tuple(self.visible_bounds)

    @property
    def variables_and_bounds(self) -> Sequence[Tuple[ie.Symbol, ie.Symbol]]:
        return tuple(zip(self.visible_variables, self.visible_bounds, strict=True))  # type: ignore

    def get_isl_ctx(self) -> islt.Context:
        return get_isl_context(self.execution_config)

    def __enter__(self) -> TempoContext:
        log.debug("Activated Context")
        TempoContext._active.manager_stack.append(self)
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> Optional[bool]:
        # If there was an error, we will render the error graph
        if exc_type is not None:
            raise_error_with_pdg_render(
                self.dg,
                f"Error during graph definition: {exc_val}\n{exc_tb}",
                str(Path(self.execution_config.path) / "graph_def_error_state"),
            )

        log.debug("Exited Active Context")
        TempoContext._active.manager_stack.pop()
        from tempo.core import global_objects

        global_objects.set_active_dg(None)
        global_objects.set_active_config(None)
        return None

    # def visualize(self, name: str = "./dg") -> None:
    #    renderer = DGRenderer(self.dg, name)
    #    renderer.render()

    Function = Any

    def compile(  # noqa: A001, A003
        self,
        bounds: Optional[Mapping[ie.Symbol, Union[int, RecurrentTensor]]] = None,
        use_active_config: bool = False,
    ) -> Executor:
        """Compiles the computation into an executor

        Args:
            bounds (Dict[Symbol, Union[int, RecurrentTensor]]): Can be either a static int, or a
            boolean RecurrentTensor. If it is a RecurrentTensor, then the bound is discovered
            at runtime when the tensor becomes True. e.g. passing the dones tensor for RL.
        Returns:
            Executor: An executor that can be used to execute the computation tick by tick or all
            at once.

        """
        if bounds is None:
            bounds = {}
        self.dg.bound_defs.update(
            {k: (v if isinstance(v, int) else v._underlying) for k, v in bounds.items()}
        )
        for b in self.dg.universe.parameters:
            assert b in self.dg.bound_defs and self.dg.bound_defs[b] is not None, (
                f"Parameter {b} has no bound definition"
            )
        cfg = self.execution_config if not use_active_config else get_active_exec_cfg()
        c = Compiler(self.dg, cfg)
        return c.compile()  # type: ignore


def get_active_ctx_manager() -> TempoContext:
    return TempoContext._active.manager_stack[-1]
