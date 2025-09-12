from __future__ import annotations

from abc import ABC
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, ClassVar

from tempo.core import index_expr as ie
from tempo.core.datatypes import OpId, TensorId
from tempo.core.debug_utils import get_creation_traceback


class InstructionType(IntEnum):
    EXEC = auto()

    DEALLOC = auto()
    OFFLOAD = auto()
    FETCH = auto()

    SEQUENTIAL_BLOCK = auto()
    PARALLEL_BLOCK = auto()

    IF_GUARD = auto()
    FOR_LOOP = auto()


@dataclass(frozen=True, slots=True)
class ScheduleItem(ABC):
    instr_type: ClassVar[InstructionType]
    _creation_traceback: Sequence[str] = field(
        init=False,
        repr=False,
        hash=False,
        compare=False,
        default_factory=lambda: get_creation_traceback(),
    )

    @property
    def creation_traceback(self) -> str:
        return "\n".join([x.strip() for x in self._creation_traceback])

    @property
    def children(self) -> Sequence[ScheduleItem]:
        return []

    @property
    def flat_recursive_tree(self) -> Sequence[ScheduleItem]:
        return [self]

    def render_str(self) -> str:
        return ""

    def __str__(self) -> str:
        return self.render_str().replace("\n", " ")


@dataclass(frozen=True, slots=True)
class ExecInstruction(ScheduleItem):
    op_id: OpId
    # Symbol (e.g. t) and its value in terms of surrounding loop counters (e.g. c0)
    domain_map: Mapping[ie.Symbol, ie.IntIndexValue]
    instr_type: ClassVar[InstructionType] = InstructionType.EXEC

    thunk: Any | None = None

    def render_str(self) -> str:
        return f"EXECUTE {self.op_id}\nDomain map:{self.domain_map}"


@dataclass(frozen=True, slots=True)
class MemManInstr(ScheduleItem):
    tensor_id: TensorId
    index: ie.IndexSequence
    is_point: bool = False
    thunk: Any | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "is_point", self.index.is_point())


@dataclass(frozen=True, slots=True)
class DeallocInstruction(MemManInstr):
    instr_type: ClassVar[InstructionType] = InstructionType.DEALLOC

    def render_str(self) -> str:
        return f"DEALLOCATE\n {self.tensor_id}\nIndex map:{self.index}"


@dataclass(frozen=True, slots=True)
class OffloadInstruction(MemManInstr):
    instr_type: ClassVar[InstructionType] = InstructionType.OFFLOAD

    def render_str(self) -> str:
        return f"OFFLOAD\n{self.tensor_id}\nIndex map:{self.index}"


@dataclass(frozen=True, slots=True)
class FetchInstruction(MemManInstr):
    instr_type: ClassVar[InstructionType] = InstructionType.FETCH

    def render_str(self) -> str:
        return f"FETCH\n{self.tensor_id}\nIndex map:{self.index}"


@dataclass(frozen=True, slots=True)
class IfGuard(ScheduleItem):
    if_cond: ie.BooleanIndexValue
    then_inner: ScheduleItem
    else_inner: ScheduleItem | None

    instr_type: ClassVar[InstructionType] = InstructionType.IF_GUARD

    def render_str(self) -> str:
        return f"If({self.if_cond})"

    @property
    def children(self) -> Sequence[ScheduleItem]:
        return (
            [self.then_inner, self.else_inner] if self.else_inner is not None else [self.then_inner]
        )

    @property
    def flat_recursive_tree(self) -> Sequence[ScheduleItem]:
        return (
            [
                self,
                *self.then_inner.flat_recursive_tree,
                *self.else_inner.flat_recursive_tree,
            ]
            if self.else_inner is not None
            else [self, *self.then_inner.flat_recursive_tree]
        )


@dataclass(frozen=True, slots=True)
class SequentialBlock(ScheduleItem):
    inner_block: Sequence[ScheduleItem]
    instr_type: ClassVar[InstructionType] = InstructionType.SEQUENTIAL_BLOCK
    contains_inline_offload: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "contains_inline_offload",
            any(isinstance(x, OffloadInstruction) for x in self.inner_block),
        )

    def render_str(self) -> str:
        return "Sequence"

    @property
    def children(self) -> Sequence[ScheduleItem]:
        return self.inner_block

    @property
    def flat_recursive_tree(self) -> Sequence[ScheduleItem]:
        tree: list[ScheduleItem] = [self]
        for x in self.inner_block:
            tree.extend(x.flat_recursive_tree)
        return tree


@dataclass(frozen=True, slots=True)
class ParallelBlock(ScheduleItem):
    inner_block: Sequence[ScheduleItem]
    # is_thread_safe: bool = False
    distributable: Sequence[ScheduleItem]
    main_thread_only: Sequence[ScheduleItem]

    instr_type: ClassVar[InstructionType] = InstructionType.PARALLEL_BLOCK

    def render_str(self) -> str:
        return "Parallel"

    def __postinit__(self) -> None:
        # is_safe = all(
        #    not isinstance(item, ForLoop) for item in self.flat_recursive_tree
        # )
        # object.__setattr__(self, "is_thread_safe", is_safe)
        ...

    @property
    def children(self) -> Sequence[ScheduleItem]:
        return self.inner_block

    @property
    def flat_recursive_tree(self) -> Sequence[ScheduleItem]:
        tree: list[ScheduleItem] = [self]
        for x in self.inner_block:
            tree.extend(x.flat_recursive_tree)
        return tree


@dataclass(frozen=True, slots=True)
class ForLoop(ScheduleItem):
    counter: ie.Symbol
    init: ie.IntIndexValue
    cond: ie.BooleanIndexValue
    increment: ie.IntIndexValue
    inner: ScheduleItem
    instr_type: ClassVar[InstructionType] = InstructionType.FOR_LOOP

    def render_str(self) -> str:
        return f"for({self.counter} = {self.init}; {self.cond}; {self.counter}+={self.increment})"

    @property
    def children(self) -> Sequence[ScheduleItem]:
        return [self.inner]

    @property
    def flat_recursive_tree(self) -> Sequence[ScheduleItem]:
        return [self, *self.inner.flat_recursive_tree]


@dataclass(frozen=True, slots=True)
class ExecutionSchedule:
    schedule: ScheduleItem

    def __str__(self) -> str:
        return "ExecutionSchedule"

    def render_to_dot(self, filename: str) -> None:
        from graphviz import Digraph

        def add_nodes_edges(
            item: ScheduleItem, graph: Digraph, parent_name: str | None = None
        ) -> None:
            if item is None:
                return
            # Define the representation of the node
            node_label = item.render_str()
            node_name = str(id(item))
            color_map = {
                ExecInstruction: "lightblue",
                DeallocInstruction: "lightgreen",
                OffloadInstruction: "yellow",
                FetchInstruction: "orange",
                # IfGuard: "lightpink",
                # SequentialBlock: "lightgray",
                ParallelBlock: "lightpink",
                # ForLoop: "lightblue",
            }
            color = color_map.get(type(item), "lightgray")  # type: ignore

            graph.node(node_name, label=node_label, color=color, style="filled")

            if parent_name is not None:
                graph.edge(parent_name, node_name)

            for child in item.children:
                add_nodes_edges(child, graph, node_name)

        # Create a new directed graph
        dot = Digraph(comment="Tempo Schedule")

        # Add nodes and edges
        add_nodes_edges(self.schedule, dot)

        # Write to a DOT file
        dot.render(filename, view=False)
