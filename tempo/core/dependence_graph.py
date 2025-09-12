from __future__ import annotations

import dataclasses
import functools
import typing
import uuid
from collections.abc import (
    Collection,
    Generator,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    Sequence,
)
from dataclasses import dataclass, field
from typing import (
    Any,
    ContextManager,
    cast,
)

import networkx as nx

from tempo.core import index_expr as ie
from tempo.core.datatypes import OpId, OpInId, OpOutId, PDGId, TensorId
from tempo.core.debug_utils import get_creation_traceback
from tempo.core.domain import Domain, DomainLike
from tempo.core.dtype import DataType
from tempo.core.shape import Shape
from tempo.core.tensor_op import TensorOp
from tempo.core.utils import make_symbols
from tempo.utils.logger import get_logger

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class DependencyData:
    expr: ie.IndexSequence
    src_out_idx: OpOutId
    sink_in_idx: OpInId
    cond: ie.BooleanIndexValue | None = None
    _isl_expr: ie.IndexSequence | None = None
    is_control_edge: bool = False
    _creation_traceback: list[str] = field(
        init=False,
        default_factory=lambda: get_creation_traceback(),
        repr=False,
        hash=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.is_control_edge:
            assert self.cond is None
            assert self.src_out_idx is None and self.sink_in_idx is None
            assert self._isl_expr is None

    @property
    def is_data_edge(self) -> bool:
        return not self.is_control_edge

    @staticmethod
    def make_control(expr: ie.IndexSequence) -> DependencyData:
        return DependencyData(
            expr=expr,
            src_out_idx=None,  # type: ignore
            sink_in_idx=None,  # type: ignore
            is_control_edge=True,
        )

    @property
    def isl_expr(self) -> ie.IndexSequence:
        if self._isl_expr is not None:
            return self._isl_expr
        return self.expr

    @property
    def creation_traceback(self) -> str:
        return "\n".join([x.strip() for x in self._creation_traceback])

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, DependencyData):
            return False

        if self.cond is None and other.cond is not None:
            return False
        if self.cond is not None and other.cond is None:
            return False

        if self.cond is not None and other.cond is not None:
            if not self.cond.struct_eq(other.cond):
                return False

        return (
            self.expr.struct_eq(other.expr)
            and self.isl_expr.struct_eq(other.isl_expr)
            and self.src_out_idx == other.src_out_idx
            and self.sink_in_idx == other.sink_in_idx
        )

    def __str__(self) -> str:
        if self.is_control_edge:
            return f"CTRL, expr={self.expr}"
        return (
            f"out_idx={self.src_out_idx}, expr={self.expr}, in_idx={self.sink_in_idx}"
            + (f", when={self.cond}" if self.cond is not None else "")
            + (f", isl_expr={self.isl_expr}" if self._isl_expr is not None else "")
        )

    def is_unconditional(self) -> bool:
        return self.cond is None or self.cond.struct_eq(ie.ConstBool(True))

    # TODO: remove this method. Avoid trusting it, since it only checks for variables,
    # not variables in the right order.
    def is_unconditional_basis(self) -> bool:
        if self.is_control_edge:
            return self.expr.is_basis()
        return self.is_unconditional() and self.expr.is_basis()

    def copy(self) -> DependencyData:
        return DependencyData(
            expr=self.expr,
            src_out_idx=self.src_out_idx,
            sink_in_idx=self.sink_in_idx,
            cond=self.cond,
            _isl_expr=self._isl_expr,
            is_control_edge=self.is_control_edge,
        )


BranchCond = tuple[ie.BooleanIndexValue, TensorId, ie.IndexSequence]


@dataclass(frozen=True)
class OpData:
    op: TensorOp
    output_shapes: dict[OpOutId, Shape]
    output_dtypes: dict[OpOutId, DataType]
    uncommitted_branch_conds: list[BranchCond] = field(default_factory=list)

    @property
    def num_outputs(self) -> int:
        return len(self.output_shapes)


@dataclass(frozen=True)
class ConditionalDefinition:
    when: ie.BooleanIndexValue
    def_tensor: TensorId
    def_index_expr: ie.IndexSequence


@dataclass(frozen=True)
class OpWithIO:
    op: TensorOp
    inputs: Sequence[tuple[TensorOp, DependencyData]]
    outputs: Sequence[tuple[TensorOp, DependencyData]]


class PDG:
    """A polyhedral dependence graph (PDG) is a directed graph that represents the dependencies
    between variables in a SARE.
    An edge from node A to node B indicates that A depends on B. The edge is labeled with an index
    expression that describes how A needs B to be indexed. For example, if A[t] = B[t-2], then there
    is an edge from A to B labeled with "t-2". A is the sink node and B is the source node.
    The edge is A --"t-2"--> B.

    Each node in the PDG corresponds to a TensorOp. This tensor op may in fact return an arbitrary
    pytree of tensors. To get these out, we use project ops.
    """

    def __init__(self, universe: DomainLike) -> None:
        self._G = nx.MultiDiGraph()

        self.universe = Domain.from_(universe)

        # NOTE the compiler externally modifies this field right before compilation
        self.bound_defs: MutableMapping[ie.Symbol, int | ie.IntIndexValue | Any] = {}

        self.next_op_id = 0
        self.ops_by_id: dict[OpId, OpData] = {}

        self.parent_graph: PDG | None = None
        self.dataflow_id: OpId | None = None

        # Generate a random id
        self.pdg_id: PDGId = PDGId(int(uuid.uuid4()))

        # Mutation counter to track graph modifications
        self._mutation_counter: int = 0

    def __hash__(self) -> int:
        return (
            hash((self.pdg_id, self._mutation_counter))
            if (self.dataflow_id is None or self.parent_graph is None)
            else hash(
                (self.parent_graph.pdg_id, self.pdg_id, self.dataflow_id, self._mutation_counter)
            )
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, PDG):
            return False

        return (
            self.pdg_id == other.pdg_id
            and self.dataflow_id == other.dataflow_id
            and self.parent_graph == other.parent_graph
            and self._mutation_counter == other._mutation_counter
        )

    def copy(self) -> PDG:
        new_dg = PDG(self.universe)
        new_dg._G = self._G.copy()
        new_dg.ops_by_id = {op.op_id: self.ops_by_id[op.op_id] for op in new_dg._G.nodes}
        new_dg.bound_defs = {**self.bound_defs}
        new_dg.next_op_id = self.next_op_id
        new_dg.parent_graph = self.parent_graph
        new_dg.dataflow_id = self.dataflow_id

        new_dg.pdg_id = PDGId(int(uuid.uuid4()))
        new_dg._mutation_counter = self._mutation_counter
        return new_dg

    def get_networkx_graph(self) -> nx.MultiDiGraph:
        return self._G

    def get_paths_between(self, snk: TensorOp, src: TensorOp) -> Generator[list[TensorOp]]:
        return nx.all_simple_paths(self._G, source=snk, target=src)  # type: ignore

    def get_ops_between(
        self, snk: TensorOp, src: TensorOp, include_endpoints: bool = False
    ) -> set[TensorOp]:
        # Nodes reachable from snk
        fwd: set[TensorOp] = set(self.recursive_dependency_nodes(snk))
        # Nodes that can reach src
        back: set[TensorOp] = set(self.recursive_dependent_nodes(src))

        nodes: set[TensorOp] = fwd & back
        if include_endpoints:
            nodes.update({snk, src})

        return nodes

    def recursive_dependency_nodes(self, node: TensorOp) -> Iterable[TensorOp]:
        return nx.descendants(self._G, node)  # type: ignore

    def recursive_dependent_nodes(self, node: TensorOp) -> Iterable[TensorOp]:
        return nx.ancestors(self._G, node)  # type: ignore

    @property
    def simple_cycles(self) -> Iterable[list[TensorOp]]:
        return nx.simple_cycles(self._G)  # type: ignore

    def is_dag(self) -> bool:
        return bool(nx.is_directed_acyclic_graph(self._G))

    def is_in_cycle(self, node: TensorOp) -> bool:
        try:
            nx.find_cycle(self._G, source=node)
            return True
        except nx.exception.NetworkXNoCycle:
            return False

    @property
    def static_bounds(self) -> Mapping[ie.Symbol, int]:
        return {k: v for k, v in self.bound_defs.items() if isinstance(v, int)}

    @property
    def dynamic_bounds(self) -> Mapping[ie.Symbol, ie.IntIndexValue]:
        # TODO: note that get_dependents should include dynamic bounds that point to it.
        # This has to be taken into account in some transforms (isolate cond subgraphs)
        return {k: v for k, v in self.bound_defs.items() if not isinstance(v, int)}

    def __str__(self) -> str:
        return (
            f"DG(ops={len(self._G.nodes)}, deps={len(self._G.edges)},"
            # + f" cond_ops={len(self.ops_with_cond_defs)})"
        )

    def extend_universe(
        self, new_var_name: str, ub: ie.IntIndexValueLike | None = None
    ) -> tuple[ie.Symbol, ie.Symbol]:
        ((new_var, new_UB),) = make_symbols((new_var_name,), start_idx=len(self.universe.variables))

        self.universe = self.universe.append_dim(new_var, new_UB)
        self.bound_defs[new_UB] = ub

        return new_var, new_UB

    def remove_universe_dim(self, var: ie.Symbol) -> None:
        self.universe = self.universe.remove_dim(var)
        del self.bound_defs[var.as_bound()]

    # Context manager for creating a temp var with a given bound.
    def _new_var_ctx_man(
        self, bound: ie.IntIndexValueLike | None = None, tmp: bool = False
    ) -> ContextManager[tuple[ie.Symbol, ie.Symbol]]:
        dg = self

        class TempVarCtxManager:
            def __enter__(self) -> tuple[ie.Symbol, ie.Symbol]:
                new_var, new_UB = dg.extend_universe(f"d{len(dg.universe.variables)}", bound)
                self.new_var = new_var
                self.new_UB = new_UB
                return new_var, new_UB

            def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
                if tmp:
                    dg.remove_universe_dim(self.new_var)

        return TempVarCtxManager()

    def new_temp_var(
        self, bound: ie.IntIndexValueLike | None = None
    ) -> ContextManager[tuple[ie.Symbol, ie.Symbol]]:
        return self._new_var_ctx_man(bound, tmp=True)

    def new_perm_var(self, bound: ie.IntIndexValueLike) -> tuple[ie.Symbol, ie.Symbol]:
        return self.extend_universe(f"d{len(self.universe.variables)}", bound)

    __repr__ = __str__

    def get_op_by_id(self, op_id: OpId) -> TensorOp:
        return self.ops_by_id[op_id].op

    def get_next_op_id(self) -> OpId:
        if self.parent_graph is not None:
            return self.parent_graph.get_next_op_id()
        id_ = OpId(self.next_op_id)
        self.next_op_id += 1
        return id_

    def get_all_tensor_descriptions(
        self,
    ) -> list[tuple[TensorId, Shape, DataType, Domain]]:
        # Filter all deps to keep the unique source_op ids and src_out_idx pairs
        descs = []
        op_datas = self.ops_by_id.values()
        for op_data in op_datas:
            for o in range(op_data.num_outputs):
                o_id = OpOutId(o)
                tid = TensorId(op_data.op.op_id, o_id)
                descs.append(
                    (
                        tid,
                        op_data.output_shapes[o_id],
                        op_data.output_dtypes[o_id],
                        op_data.op.domain,
                    )
                )
        return descs

    def create_sub_dg(self, id_: OpId, graph: nx.MultiDiGraph) -> PDG:
        dg = PDG(self.universe.copy())
        dg._G = graph
        dg.ops_by_id = {op.op_id: self.ops_by_id[op.op_id] for op in graph.nodes}
        # dg.ops_with_cond_defs = self.ops_with_cond_defs.intersection(
        #    dg.ops_by_id.keys()
        # )
        dg.parent_graph = self
        dg.dataflow_id = id_
        dg._mutation_counter = self._mutation_counter
        return dg

    def induced_subgraph(self, id_: OpId, ops: Iterable[TensorOp]) -> PDG:
        subgraph = self._G.subgraph(ops).copy()
        result = self.create_sub_dg(id_, subgraph)
        return result

    # def extract_dependency_subgraph(self, op: TensorOp) -> DependenceGraph:
    #    successors: Set[TensorOp] = set()
    #    for n, succ in nx.bfs_successors(self._G, op):
    #        successors = successors.union([n, *succ])

    #    result = self.induced_subgraph(successors)
    #    for op in successors:
    #        self.remove_op(op)
    #    return result

    @property
    def nodes(self) -> Iterable[TensorOp]:
        # return cast(Iterable[TensorOp], list(self._G.nodes))
        return list(self._G.nodes)  # type: ignore

    @property
    def sccs(self) -> Iterable[set[TensorOp]]:
        return cast(Iterable[set[TensorOp]], nx.strongly_connected_components(self._G))

    @property
    def weakly_connected_components(self) -> Iterable[set[TensorOp]]:
        return cast(Iterable[set[TensorOp]], nx.weakly_connected_components(self._G))

    @property
    def nodes_with_no_dependents(self) -> Iterable[TensorOp]:
        return (node for node in self.nodes if len(list(self._G.in_edges(node))) == 0)

    @property
    def nodes_with_no_dependencies(self) -> Iterable[TensorOp]:
        return (node for node in self.nodes if not self._G.out_edges(node))

    @property
    def udf_nodes(self) -> Iterable[TensorOp]:
        return (node for node in self.nodes if node.is_udf())

    @property
    def stateful_udf_nodes(self) -> Iterable[TensorOp]:
        return (node for node in self.udf_nodes if node.is_stateful)

    @property
    def node_datas(self) -> Iterable[OpData]:
        return self.ops_by_id.values()

    @property
    def node_ids(self) -> Collection[OpId]:
        return self.ops_by_id.keys()

    @property
    def nodes_with_io(self) -> Iterable[OpWithIO]:
        ops_with_io_: list[OpWithIO] = []
        nodes = self._G.nodes
        for node in nodes:
            dependents = self.get_flat_direct_dependents(node, include_control=False)
            dependencies = self.get_flat_direct_dependencies(node, include_control=False)
            op_with_io = OpWithIO(node, dependencies, dependents)
            ops_with_io_.append(op_with_io)
        return ops_with_io_

    @property
    def nodes_topologically_sorted(self) -> Iterable[TensorOp]:
        assert self.parent_graph is not None, "Can only sort nodes in a subgraph"
        # We reverse the graph first because this is a dependency graph, not a dataflow graph
        return cast(Iterable[TensorOp], nx.topological_sort(nx.reverse(self._G, copy=True)))

    def get_num_edges_between(self, from_: TensorOp, to: TensorOp) -> int:
        return int(self._G.number_of_edges(from_, to))

    def get_edges_between(
        self, from_: TensorOp, to: TensorOp, include_control: bool = False
    ) -> Sequence[DependencyData]:
        # return [d["dependency_data"] for d in self._G[from_][to].values()]
        edges = []
        for depy_op, depy_data in self.get_flat_direct_dependencies(from_, include_control):
            if depy_op.equivalent(to):
                edges.append(depy_data)
        return edges

    def insert_op(self, op_data: OpData) -> None:
        if op_data.op in self._G.nodes:
            log.warning("node %s already exists in graph.", op_data.op)
        else:
            self._G.add_node(op_data.op)
            self._mutation_counter += 1
        self.ops_by_id[op_data.op.op_id] = op_data

    def replace_op(self, old_op: TensorOp, new_op: TensorOp) -> None:
        if old_op not in self._G.nodes:
            raise ValueError(f"Node {old_op} does not exist in graph.")
        else:
            new_op_data = dataclasses.replace(self.ops_by_id[old_op.op_id], op=new_op)
            self.insert_op(new_op_data)
            self.move_connections(old_op, new_op)
            self.remove_op(old_op)

    def remove_op(self, op: TensorOp) -> None:
        if op not in self._G.nodes:
            log.debug("Warning: trying to remove node %s which does not exist in graph.", op)
        else:
            log.debug("Removing node %s from graph.", op)
            self._G.remove_node(op)
            self._mutation_counter += 1
            del self.ops_by_id[op.op_id]

    def move_dependencies(self, from_: TensorOp, to: TensorOp) -> None:
        """Moves the connections on which from_ "depends on", i.e. their dependencies

        from_ ---> smth, means that from_ reads smth to compute its output.

        #Maps to the irouter
        """
        edges_to_add = []
        edges_to_remove = []
        for dependency, dep_data in self.get_flat_direct_dependencies(from_, include_control=True):
            edges_to_add.append((to, dependency, dep_data))
            edges_to_remove.append((from_, dependency, dep_data))
        for e in edges_to_add:
            self.add_edge(*e)
        for e in edges_to_remove:
            self.remove_edge(*e)

    def move_dependents(self, from_: TensorOp, to: TensorOp, except_to: bool = True) -> None:
        for o in range(from_.num_outputs):
            oid = OpOutId(o)
            self.move_dependents_of_output(from_, to, oid, oid, except_to)

    def move_dependents_of_output(
        self,
        from_: TensorOp,
        to: TensorOp,
        from_out_id: OpOutId,
        to_out_id: OpOutId,
        except_to: bool,
    ) -> None:
        edges_to_add = []
        edges_to_remove = []

        for dependent, dep_data in self.get_flat_direct_dependents(from_, include_control=False):
            if dep_data.src_out_idx == from_out_id:
                if except_to and dependent == to:
                    continue
                new_expr = dep_data.expr
                diff_dom = Domain.difference(from_.domain, to.domain)
                if len(diff_dom) > 0:
                    # We have to remove some dims from the expr
                    idxs_to_rem = []
                    for dim in diff_dom.variables:
                        idx_in_expr = from_.domain.find_variable_index(dim)
                        idxs_to_rem.append((idx_in_expr, dim))
                    for idx_to_rem, dim in sorted(idxs_to_rem, reverse=True):
                        assert ie.struct_eq(new_expr.members[idx_to_rem], dim), (
                            "Can only use move_dependents on simple expressions"
                        )
                        new_expr = new_expr.skip_idx(idx_to_rem)

                new_dep_data = DependencyData(
                    expr=new_expr,
                    src_out_idx=to_out_id,
                    sink_in_idx=dep_data.sink_in_idx,
                    cond=dep_data.cond,
                )
                edges_to_add.append((dependent, to, new_dep_data))
                edges_to_remove.append((dependent, from_, dep_data))
                log.debug(
                    "Adding edge to remove %s -> %s with data %s", dependent, to, new_dep_data
                )
        for e in edges_to_add:
            self.add_edge(*e)
        for e in edges_to_remove:
            self.remove_edge(*e)

    def move_connections(self, from_: TensorOp, to: TensorOp) -> None:
        self.move_dependencies(from_, to)
        self.move_dependents(from_, to)

    def find_edge_key(self, sink: TensorOp, src: TensorOp, dependency_data: DependencyData) -> int:
        for snk, srce, k, data in self._G.edges((sink), data=True, keys=True):
            if snk == sink and srce == src and data["dependency_data"] == dependency_data:
                assert isinstance(k, int)
                return k

        raise ValueError(f"Edge not found: {sink} -> {src}")

    def add_edge(self, sink: TensorOp, src: TensorOp, dependency_data: DependencyData) -> None:
        expr_src_dom_length_mismatch = not len(dependency_data.expr) == len(src.domain)
        domain_mismatch = not Domain.from_(dependency_data.expr.vars_used()).is_contained_in(
            sink.domain
        )
        output_out_of_bounds = False
        input_out_of_bounds = False

        if dependency_data.is_data_edge:
            output_out_of_bounds = dependency_data.src_out_idx > src.num_outputs
            input_out_of_bounds = dependency_data.sink_in_idx > sink.num_inputs

        error = (
            expr_src_dom_length_mismatch
            or domain_mismatch
            or output_out_of_bounds
            or input_out_of_bounds
        )

        if error:
            msg = (
                f"Error adding edge: {expr_src_dom_length_mismatch=}, {domain_mismatch=},"
                + f"{output_out_of_bounds=}, {input_out_of_bounds=}\n"
            )
            msg += f"Edge {sink} --{dependency_data}--> {src}\n"
            msg += f"Sink creation traceback: {sink.creation_traceback}\n"
            msg += f"Edge creation traceback: {dependency_data.creation_traceback}\n"
            msg += f"Src creation traceback: {src.creation_traceback}\n"

            raise ValueError(msg)

            # TODO raise an exception and catch it in compiler, which then renders the graph.
            # from tempo.core.dg_renderer import raise_error_with_pdg_render
            # raise_error_with_pdg_render(self, msg)

        self._G.add_edge(
            sink,
            src,
            dependency_data=dependency_data,
        )
        self._mutation_counter += 1

    def remove_edge(self, sink: TensorOp, src: TensorOp, dependency_data: DependencyData) -> None:
        log.debug("Removing edge %s -> %s with data %s", sink, src, dependency_data)
        key = self.find_edge_key(sink, src, dependency_data)
        self._G.remove_edge(sink, src, key)
        self._mutation_counter += 1

    def get_direct_dependencies(
        self, op: TensorOp, include_control: bool = False
    ) -> Sequence[tuple[TensorOp, tuple[DependencyData, ...]]]:
        """Returns the dependencies of a given op, meaning the nodes that it depends on in order to
        be computed.

        Args:
            op (TensorOp): The TensorOp whose dependencies we want to find.

        Returns:
            Iterable[Tuple[TensorOp, List[DependencyData]]]: An iterable of tuples of the form
            (dep_op, [depency_data]) where dep_op is the TensorOp that op depends on and
            dependency_data is a list of DependencyData objects
            that describe how op depends on dep_op.

        """
        dependencies = []
        for neighbour in self._G.successors(op):
            # Iterate every dependence with this neighbour
            deps = []
            for edge in self._G[op][neighbour].values():
                dep_data = edge.get("dependency_data")
                if not include_control and dep_data.is_control_edge:
                    continue
                deps.append(dep_data)
            dependencies.append((neighbour, tuple(deps)))
        return list(dependencies)  # type: ignore

    def get_flat_recursive_dependencies(
        self, op: TensorOp, start_from_input: OpInId | None = None, include_control: bool = False
    ) -> Sequence[tuple[TensorOp, DependencyData]]:
        visited: set[TensorOp] = set()
        stack: list[TensorOp] = []
        flat_dependencies: list[tuple[TensorOp, DependencyData]] = []
        if start_from_input is not None:
            depy, depy_data = self.get_flat_direct_dependencies(op, include_control)[
                start_from_input
            ]
            stack.append(depy)
            flat_dependencies.append((depy, depy_data))
        else:
            stack.append(op)
        while stack:
            current_op = stack.pop()
            if current_op in visited:
                continue
            visited.add(current_op)
            for next_op, dep_data in self.get_flat_direct_dependencies(current_op, include_control):
                flat_dependencies.append((next_op, dep_data))
                stack.append(next_op)

        return flat_dependencies

    def get_flat_direct_dependencies(
        self, op: TensorOp, include_control: bool = False
    ) -> Sequence[tuple[TensorOp, DependencyData]]:
        dependencies = self.get_direct_dependencies(op, include_control)
        flat_data_dependencies = []
        flat_control_dependencies = []
        for dep_op, dep_datas in dependencies:
            for dep_data in dep_datas:
                if dep_data.is_control_edge:
                    flat_control_dependencies.append((dep_op, dep_data))
                else:
                    flat_data_dependencies.append((dep_op, dep_data))
        flat_data_dependencies = sorted(flat_data_dependencies, key=lambda x: x[1].sink_in_idx)
        # NOTE ensure Control edges are last, since users expect data ordered
        return list(flat_data_dependencies) + list(flat_control_dependencies)

    def get_direct_dependents(
        self, op: TensorOp, include_control: bool = False
    ) -> Sequence[tuple[TensorOp, tuple[DependencyData]]]:
        """Returns the dependents of a given op, meaning the nodes that depend on it in order to be
        computed.

        Args:
            op (TensorOp): The TensorOp whose dependents we want to find.

        Returns:
            Iterable[Tuple[TensorOp, List[DependencyData]]]: An iterable of tuples of the form
            (dep_op, [dependency_data]) where dep_op is the TensorOp that depends on op and
            dependency_data is a list of DependencyData objects
            that describe how dep_op depends on op.

        """
        dependents = set()
        for neighbor in self._G.predecessors(op):
            deps = []
            for edge in self._G[neighbor][op].values():
                dep_data = edge.get("dependency_data")
                if not include_control and dep_data.is_control_edge:
                    continue
                deps.append(dep_data)
            deps = tuple(deps)
            dependents.add((neighbor, deps))
        return list(dependents)  # type: ignore

    def get_flat_direct_dependents(
        self, op: TensorOp, include_control: bool = False
    ) -> Sequence[tuple[TensorOp, DependencyData]]:
        dependents = self.get_direct_dependents(op, include_control)
        flat_data_dependents = []
        flat_control_dependents = []
        for dep_op, dep_datas in dependents:
            for dep_data in dep_datas:
                if dep_data.is_control_edge:
                    flat_control_dependents.append((dep_op, dep_data))
                else:
                    flat_data_dependents.append((dep_op, dep_data))

        flat_data_dependents = sorted(flat_data_dependents, key=lambda x: x[1].src_out_idx)
        # NOTE ensure Control edges are last, since users expect data ordered
        return list(flat_data_dependents) + list(flat_control_dependents)

    def get_tensor_dtype(self, tensor_id: TensorId) -> DataType:
        return self.ops_by_id[tensor_id.op_id].output_dtypes[tensor_id.output_id]

    def get_tensor_flat_direct_dependents(
        self, tensor_id: TensorId
    ) -> Sequence[tuple[TensorOp, DependencyData]]:
        all_dependents = self.get_flat_direct_dependents(
            self.ops_by_id[tensor_id.op_id].op, include_control=False
        )
        only_tensor_dependents = [
            dep for dep in all_dependents if dep[1].src_out_idx == tensor_id.output_id
        ]
        return only_tensor_dependents

    # TODO eventually fix this method
    # def render_as_str(self) -> str:
    #    """Render the DG as a system of recurrence equations."""
    #    equations = []
    #    for op in self._G.nodes:
    #        deps = self.get_direct_dependencies(op)
    #        deps_flat = []
    #        for dep_op, dep_datas in deps:
    #            for dep_data in dep_datas:
    #                deps_flat.append(f"S{dep_op.op_id}[{dep_data}]")  # TODO fix
    #        deps_flat_str = ", ".join(deps_flat)
    #        class_name = op.__class__.__name__
    #        equations.append(f"S{op.op_id}[t] = {class_name}({deps_flat_str})")
    #    return ";".join(equations)

    def get_all_edges(
        self, include_control: bool = False
    ) -> list[tuple[TensorOp, TensorOp, DependencyData]]:
        """Returns all edges (sink, src, dependency_data) in the graph.

        Returns:
            List[Tuple[TensorOp, TensorOp, DependencyData]]: (sink, src, dependency_data)

        """
        edges: list[tuple[TensorOp, TensorOp, DependencyData]] = []
        for snk, src, _, data_dict in self._G.edges(data=True, keys=True):
            d: DependencyData = typing.cast(DependencyData, data_dict["dependency_data"])
            if not include_control and d.is_control_edge:
                continue
            edges.append((snk, src, d))
        return edges

    @functools.lru_cache(maxsize=-1)
    def get_input_shape(self, op: TensorOp, input_idx: OpInId, simplify: bool = True) -> Shape:  # noqa: C901
        # NOTE: first, try to find the input shape in this graph
        for src_op, dep in self.get_flat_direct_dependencies(op, include_control=False):
            if dep.sink_in_idx == input_idx:
                indexing_shape = dep.expr.evaluate_shape(self.static_bounds)
                raw_shape = self.ops_by_id[src_op.op_id].output_shapes[dep.src_out_idx]

                res_shape = Shape.from_((*indexing_shape, *raw_shape._shape), simplify=simplify)
                return res_shape

        # NOTE: if not found, we must be in a dataflow graph, so we need to look in the parent graph
        if self.parent_graph is None or self.dataflow_id is None:
            raise ValueError(f"Input {input_idx} not found for op {op}")

        from tempo.core.dataflow_graph import DataflowGraphI

        # NOTE: During const folding with grouping, we create induced subgraphs with id -1
        # In this case, we need to look in the parent graph for the input shape.
        if self.dataflow_id not in self.parent_graph.ops_by_id:
            return self.parent_graph.get_input_shape(op, input_idx)

        dataflow_op = self.parent_graph.ops_by_id[self.dataflow_id].op
        dataflow: DataflowGraphI = dataflow_op.dataflow  # type: ignore

        # Find the index of the input in the dataflow op inputs
        new_input_idx = -1
        for i, shared_deps in enumerate(dataflow.irouter):
            for op_id, in_idx in shared_deps:
                if op_id == op.op_id and in_idx == input_idx:
                    new_input_idx = i
                    break
        assert new_input_idx != -1, f"Input {input_idx} not found for op {op}"

        # return the input shape from the parent graph dataflow_op
        return self.parent_graph.get_input_shape(dataflow_op, OpInId(new_input_idx))

    def get_input_shapes_list(self, op: TensorOp) -> list[Shape]:
        shapes_dict = self.get_input_shapes(op)
        input_shapes = []
        for i in range(len(shapes_dict)):
            input_shapes.append(shapes_dict[OpInId(i)])
        return input_shapes

    def get_input_shapes(self, op: TensorOp, simplify: bool = True) -> dict[OpInId, Shape]:
        num_inputs = op.num_inputs
        input_shapes: dict[OpInId, Shape] = {}
        for i in range(num_inputs):
            input_shapes[OpInId(i)] = self.get_input_shape(op, OpInId(i), simplify=simplify)
        return input_shapes

    def get_input_dtype(self, op: TensorOp, input_idx: OpInId) -> DataType:
        # First, try to find the input dtype in this graph
        for src_op, dep in self.get_flat_direct_dependencies(op, include_control=False):
            if dep.sink_in_idx == input_idx:
                return self.ops_by_id[src_op.op_id].output_dtypes[dep.src_out_idx]

        # If not found, we must be in a dataflow graph, so we need to look in the parent graph
        if self.parent_graph is None or self.dataflow_id is None:
            raise ValueError(f"Input {input_idx} not found for op {op}")

        from tempo.core.dataflow_graph import DataflowGraphI

        # NOTE: During const folding with grouping, we create induced subgraphs with id -1
        # In this case, we need to look in the parent graph for the input dtype.
        if self.dataflow_id not in self.parent_graph.ops_by_id:
            return self.parent_graph.get_input_dtype(op, input_idx)

        dataflow_op = self.parent_graph.ops_by_id[self.dataflow_id].op
        dataflow: DataflowGraphI = dataflow_op.dataflow  # type: ignore

        # Find the index of the input in the dataflow op inputs
        new_input_idx = -1
        for i, shared_deps in enumerate(dataflow.irouter):
            for op_id, in_idx in shared_deps:
                if op_id == op.op_id and in_idx == input_idx:
                    new_input_idx = i
                    break
        assert new_input_idx != -1, f"Input {input_idx} not found for op {op}"

        # Return the input dtype from the parent graph dataflow_op
        return self.parent_graph.get_input_dtype(dataflow_op, OpInId(new_input_idx))

    def get_input_dtypes_list(self, op: TensorOp) -> list[DataType]:
        dtypes_dict = self.get_input_dtypes(op)
        input_dtypes = []
        for i in range(len(dtypes_dict)):
            input_dtypes.append(dtypes_dict[OpInId(i)])
        return input_dtypes

    def get_input_dtypes(self, op: TensorOp) -> dict[OpInId, DataType]:
        num_inputs = op.num_inputs
        input_dtypes: dict[OpInId, DataType] = {}
        for i in range(num_inputs):
            input_dtypes[OpInId(i)] = self.get_input_dtype(op, OpInId(i))
        return input_dtypes

    def get_output_shapes(self, op: TensorOp) -> dict[OpOutId, Shape]:
        return self.ops_by_id[op.op_id].output_shapes

    def get_output_shape(self, op: TensorOp, out_idx: OpOutId) -> Shape:
        return self.ops_by_id[op.op_id].output_shapes[out_idx]

    def get_output_shapes_list(self, op: TensorOp) -> list[Shape]:
        shapes_dict = self.get_output_shapes(op)
        output_shapes = []
        for i in range(len(shapes_dict)):
            output_shapes.append(shapes_dict[OpOutId(i)])
        return output_shapes

    def get_output_dtypes(self, op: TensorOp) -> dict[OpOutId, DataType]:
        return self.ops_by_id[op.op_id].output_dtypes

    def isolated_nodes(self) -> list[TensorOp]:
        return list(nx.isolates(self._G))

    def reachable_ops(self, op: TensorOp) -> set[TensorOp]:
        return set(nx.descendants(self._G, op))

    def op_cycles(self, op: TensorOp) -> Iterator[list[TensorOp]]:
        dep_ops = {d[0] for d in self.get_flat_direct_dependents(op)}
        for dep_op in dep_ops:
            yield from self.edge_cycles(dep_op, op)

        depy_ops = {d[0] for d in self.get_flat_direct_dependencies(op)}
        for depy_op in depy_ops:
            yield from self.edge_cycles(depy_op, op)

    def edge_cycles(self, snk: TensorOp, src: TensorOp) -> Iterator[list[TensorOp]]:
        current_cycle: list[TensorOp] = []
        stack: list[tuple[TensorOp, TensorOp]] = [(snk, src)]
        visited: set[TensorOp] = set()

        while stack:
            parent, v = stack.pop()

            if v == snk:
                current_cycle.append(v)
                yield current_cycle.copy()  # Make a copy to store the cycle
                current_cycle = self._backtrack_current_cycle(current_cycle, stack)
            elif v not in visited:
                visited.add(v)
                current_cycle.append(v)
                for adj, _ in self.get_flat_direct_dependencies(v, include_control=True):
                    stack.append((v, adj))
            else:
                current_cycle = self._backtrack_current_cycle(current_cycle, stack)

    def _backtrack_current_cycle(
        self, current_cycle: list[TensorOp], stack: list[tuple[TensorOp, TensorOp]]
    ) -> list[TensorOp]:
        if stack:
            next_parent, _ = stack[-1]
            assert next_parent in current_cycle
            parent_index = current_cycle.index(next_parent)
            return current_cycle[: parent_index + 1]
        return current_cycle

    # TODO: remove
    def estimate_tensor_size_bytes(  # noqa: C901
        self,
        op_id: OpId,
        out_idx: OpOutId | None = None,
        in_idx: OpInId | None = None,
        bound_size_estimate: int = 200,
    ) -> int:
        op_data = self.ops_by_id[op_id]

        if out_idx is None and in_idx is None:
            raise ValueError("Either out_idx or in_idx must be provided")

        if out_idx is not None:
            assert in_idx is None
            shape = op_data.output_shapes[out_idx]
            dtype = op_data.output_dtypes[out_idx]

        if in_idx is not None:
            assert out_idx is None
            shape = self.get_input_shape(op_data.op, in_idx)
            dtype = self.get_input_dtype(op_data.op, in_idx)

        if shape.is_static():
            return shape.as_static().prod() * dtype.repr_bytes  # type: ignore
        else:
            # We err on the side of caution and assume that the tensor is large
            statically_known_portion = 1
            dynamic_portion: ie.IntIndexValue = ie.ConstInt(1)
            for dim in shape._shape:
                if isinstance(dim, int):
                    statically_known_portion *= dim
                else:
                    dynamic_portion *= dim  # type: ignore

            statically_known_portion *= dtype.repr_bytes

            # TODO fix this...
            # NOTE: What we can do, is evaluate the dynamic portion at a few different points,
            # take the max, and multiply by the statically known portion.

            dynamic_portion_eval = max(
                [
                    dynamic_portion.evaluate(
                        {
                            **self.static_bounds,
                            **dict.fromkeys(self.dynamic_bounds, bound_size_estimate),
                            **dict.fromkeys(self.universe.variables, v_val),
                        }
                    )
                    for v_val in [
                        0,
                        bound_size_estimate // 4,
                        bound_size_estimate // 4 * 2,
                        bound_size_estimate // 4 * 3,
                        bound_size_estimate,
                    ]
                ]
            )

            size = statically_known_portion * dynamic_portion_eval
            return size
