import functools
from collections.abc import Mapping
from typing import Any, List, Optional, Sequence, Tuple, cast

import islpy as isl

from tempo.core import index_expr as ie
from tempo.core import isl_types as islt
from tempo.core import tensor_op as top
from tempo.core.datatypes import OpId, TensorId
from tempo.core.dependence_graph import DependencyData
from tempo.core.domain import Domain, DomainLike
from tempo.core.isl_context_factory import get_isl_context
from tempo.core.shape import Shape
from tempo.core.statement_type import StmtType
from tempo.utils import logger

log = logger.get_logger(__name__)


empty_str = ""


def get_parameters_and_var_bounds_strs(
    domain: Domain, var_renaming: Optional[Sequence[str]] = None
) -> Tuple[str, str]:
    from tempo.core import global_objects as glob

    dynamic_bounds = dict(glob.get_dynamic_bounds_or_empty())
    dynamic_bounds = {k: v for k, v in dynamic_bounds.items() if domain.has_dim(k.as_var())}
    static_bounds = dict(glob.get_static_bounds_or_empty())
    static_bounds = {k: v for k, v in static_bounds.items() if domain.has_dim(k.as_var())}

    # print(f"dynamic_bounds: {dynamic_bounds}")
    # print(f"static_bounds: {static_bounds}")

    # print(f"domain.ubounds: {domain.ubounds}")
    # print(f"domain.parameters: {domain.parameters}")

    filtered_dyn_bounds = {}
    # NOTE: filter out dynamic bounds that are not in the domain
    for b, bdef in dynamic_bounds.items():
        if all((domain.has_dim(v)) for v in bdef.vars_used()):
            filtered_dyn_bounds[b] = bdef

    # print(f"filtered_dyn_bounds: {filtered_dyn_bounds}")

    parameters = []
    for b in domain.parameters:
        if b in static_bounds:
            parameters.append(f"{b}={static_bounds[b]}")
        else:
            if b not in dynamic_bounds.keys():
                parameters.append(f"{b}")
    parameters_str = ",".join(parameters)

    bounds_g_0 = (
        " and ".join(f"0 < {b}" for b in domain.ubounds if b not in filtered_dyn_bounds) or "true"
    )

    remapped_ubounds = [
        (b if b not in filtered_dyn_bounds else filtered_dyn_bounds[b]) for b in domain.parameters
    ]

    var_names = [
        (var_renaming[i] if var_renaming is not None else f"{v}")
        for i, v in enumerate(domain.variables)
    ]

    # TODO: we still need to more appropriately bound vars either here or in dependence_to_isl_map
    # The issue is related to during backward, we still do not have bounds for the vars.

    var_bounds_str = "true"

    count = 0
    for v, b in zip(var_names, remapped_ubounds, strict=True):
        bound_def_str = "false"
        for c, d in b.enumerate_all_cond_branches():
            if c is None:
                c = ie.ConstBool(True)

            bound_def_str = f"{bound_def_str} or ({c} and 0 <= {v} < {d})"

        var_bounds_str = f"{var_bounds_str} and ({bound_def_str})"
        count += 1
    assert count == len(domain.parameters), (
        f"count: {count}, len(domain.parameters): {len(domain.parameters)}"
    )
    # print(f"var_bounds_str: {var_bounds_str}")
    bounds_str = f"({bounds_g_0}) and ({var_bounds_str})"
    if "DS0" in bounds_str:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"domain: {domain}")
        print(f"bounds_str: {bounds_str}")
        print(f"parameters_str: {parameters_str}")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        raise Exception("DS0 in bounds_str")
    # assert "DS0" not in bounds_str, f"DS0 in bounds_str: {bounds_str}"
    # assert "DS0" not in parameters_str, f"DS0 in parameters_str: {parameters_str}"
    return parameters_str, bounds_str


def op_id_to_exec_name(op_id: OpId) -> str:
    return f"{StmtType.EXECUTE.value}_{op_id}"


def tensor_id_to_gc_stmt(tensor_id: TensorId) -> str:
    return f"{StmtType.DEALLOCATE.value}_{int(tensor_id.op_id)}_{int(tensor_id.output_id)}"


def tensor_id_to_offload_stmt(tensor_id: TensorId, num: int) -> str:
    return f"{StmtType.OFFLOAD.value}_{int(tensor_id.op_id)}_{int(tensor_id.output_id)}_{num}"


def tensor_id_to_fetch_stmt(tensor_id: TensorId, num: int) -> str:
    return f"{StmtType.FETCH.value}_{int(tensor_id.op_id)}_{int(tensor_id.output_id)}_{num}"


def get_params_set(
    symbols_: Mapping[ie.Symbol, int], ctx: Optional[islt.Context] = None
) -> islt.Set:
    only_bounds = [s for s in symbols_.keys() if s.is_bound]
    from tempo.core import global_objects as glob

    dynamic_bounds = glob.get_dynamic_bounds_or_empty()
    # Filter out dynamic bounds
    only_bounds = [b for b in only_bounds if b not in dynamic_bounds]

    joined_params = ", ".join([f"{s}={symbols_[s]}" for s in only_bounds])
    params = isl.Set(f"[{joined_params}] -> {{ : }}", context=ctx)
    return params


def simplify_boolean_index_expr(
    domain: Domain,
    expr: ie.BooleanIndexValue,
    known_symbols: Optional[Mapping[ie.Symbol, int]] = None,
) -> ie.BooleanIndexValue:
    if known_symbols is None:
        known_symbols = {}

    vars_ = expr.vars_used()
    vars_str = ", ".join([str(v) for v in vars_])

    dom_int = domain.intersect(Domain.from_(vars_))

    parameters_str, bounds_str = get_parameters_and_var_bounds_strs(dom_int)

    # req_known_symbols = {k: v for k, v in known_symbols.items() if k in domain.parameters}
    # known_symbols_str = (
    # " and ".join([f"{str(k)}={v}" for k, v in req_known_symbols.items()]) or "true"
    # )

    bool_str = f"[{parameters_str}] -> {{ [{vars_str}] :  {expr}}}"
    # print(f"Simplifying boolean index expr: {bool_str}")
    bool_set = isl.Set(bool_str)

    # NOTE: Both true and false are converted to ConstInt(1) below. We check for emptyness,
    # in order to disambiguate.
    if bool_set.is_empty():
        return ie.ConstBool(False)

    # TODO: replace with below. We havent done it yet because the to bool tempo expr has some
    # exceptions, not covered by the more genral to tempo expr.
    ctx_str = f"[{parameters_str}] -> {{[{vars_str}]: {bounds_str}}}"
    context = isl.Set(ctx_str)
    build = isl.AstBuild.from_context(context)
    ast_expr = build.expr_from_set(bool_set)
    expr = isl_expr_to_tempo_boolean_expr(ast_expr)
    # print(f"Simplified boolean index expr: {expr}")
    return expr


def isl_set_to_index_value(set_: islt.Set, domain: Domain) -> ie.IndexValue:
    parameters_str, bounds_str = get_parameters_and_var_bounds_strs(domain)

    vars_ = list(domain.variables)
    vars_str = ", ".join([str(v) for v in vars_])
    ctx_str = f"[{parameters_str}] -> {{[{vars_str}]: {bounds_str}}}"
    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # print("------------------------------------")
    # print(f"ctx_str: {ctx_str}")
    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # print("------------------------------------")

    context = isl.Set(ctx_str)
    build = isl.AstBuild.from_context(context)

    # NOTE: in case user gives a union set.
    actual_isl_set = isl.Set(str(set_))

    ast_expr = build.expr_from_set(actual_isl_set)

    expr = isl_expr_to_tempo_expr(ast_expr)
    return expr


def simplify_dependence_expr(
    expr: ie.IndexSequence,
    snk_dom: DomainLike,
    src_dom: DomainLike,
    condition: Optional[ie.BooleanIndexValue] = None,
    known_symbols: Optional[Mapping[ie.Symbol, int]] = None,
    ctx: Optional[islt.Context] = None,
) -> ie.IndexSequence:
    # print(f"Calling dependence_to_isl_map with {expr}, {snk_dom}, {src_dom}, {condition}")
    map_ = dependence_to_isl_map(expr, snk_dom, src_dom, condition=condition, ctx=ctx)

    if known_symbols is not None:
        param_set = get_params_set(known_symbols, ctx)
        map_ = map_.gist_params(param_set)
    # print(f"Expr: {expr}, simplified map: {map_}")
    # print(f"{snk_dom=}")
    # print(f"{src_dom=}")
    # print(f"{condition=}")
    return union_map_to_seq_expr(snk_dom, src_dom, map_, ctx=ctx)


def simplify_expr(
    expr: ie.IndexAtomLike,
    condition: Optional[ie.BooleanIndexValue] = None,
    known_symbols: Optional[Mapping[ie.Symbol, int]] = None,
    ctx: Optional[islt.Context] = None,
) -> ie.IndexAtom:
    expr = ie.lift_to_ie_atom(expr)
    if isinstance(expr, ie.Slice):
        return simplify_slice(expr, condition=condition, known_symbols=known_symbols, ctx=ctx)
    elif isinstance(expr, ie.IntIndexValue):
        return simplify_int_index_value(
            expr, condition=condition, known_symbols=known_symbols, ctx=ctx
        )
    else:
        raise ValueError(f"Unknown index value type: {type(expr)}")


def simplify_slice(
    expr: ie.Slice,
    condition: Optional[ie.BooleanIndexValue] = None,
    known_symbols: Optional[Mapping[ie.Symbol, int]] = None,
    ctx: Optional[islt.Context] = None,
) -> ie.Slice:
    lb = simplify_int_index_value(
        expr.start, condition=condition, known_symbols=known_symbols, ctx=ctx
    )
    ub = simplify_int_index_value(
        expr.stop - 1, condition=condition, known_symbols=known_symbols, ctx=ctx
    )
    return ie.Slice(lb, ub + 1)


def simplify_int_index_value(
    expr: ie.IntIndexValueLike,
    condition: Optional[ie.BooleanIndexValue] = None,
    known_symbols: Optional[Mapping[ie.Symbol, int]] = None,
    ctx: Optional[islt.Context] = None,
) -> ie.IntIndexValue:
    expr = ie.lift_to_int_ie(expr)
    # print(f"simp expr: {expr}")
    if isinstance(expr, ie.Symbol):
        if known_symbols is None:
            from tempo.core import global_objects as glob

            known_symbols = glob.get_static_bounds_or_empty()
        return expr.partial_eval(known_symbols)  # if known_symbols is not None else expr

    # domain = Domain.from_(domain)
    pw_aff = int_index_value_to_pw_aff(
        expr, condition=condition, known_symbols=known_symbols, ctx=ctx
    )
    # print(f"pw_aff: {pw_aff}")
    return pw_aff_to_index_value(pw_aff)


def simplify_shape(shape: Shape, known_symbols: Optional[Mapping[ie.Symbol, int]] = None) -> Shape:
    res_shape: List[ie.IntIndexValueLike] = []

    # print(f"Simplifying shape: {shape}")
    for s in shape:
        # print(f"s: {s}, {type(s)}, {s.__class__}")
        if isinstance(s, int):
            # print(f"s is an int: {s}")
            res_shape.append(s)
        elif isinstance(s, ie.ConstInt):
            # print(f"s is a ConstInt: {s}")
            res_shape.append(s.const)
        else:
            # print(f"s is not an int or a ConstInt: {s}, {type(s)}, {s.__class__}")
            try:
                # if not isinstance(s, ie.IntIndexValue):
                #    print(f"s is not an IntIndexValue: {s}, {type(s)}, {s.__class__}")
                assert isinstance(s, ie.IntIndexValue)
                pw_aff = int_index_value_to_pw_aff(s, known_symbols=known_symbols)
                res_shape.append(pw_aff_to_index_value(pw_aff))
            except Exception:
                # print(f"exception simplifying shape: {s}")
                # print(f"exception: {e}")
                res_shape.append(s)
    s = Shape(tuple(res_shape))
    if s.is_static():
        return s.as_static()
    return s
    # map_ = dependence_to_isl_map(expr, snk_dom, src_dom, condition=condition)
    # return _union_map_to_seq_expr(snk_dom, src_dom, map_)


def reverse_dependence_expr(
    expr: ie.IndexSequence,
    snk_dom: DomainLike,
    src_dom: DomainLike,
) -> ie.IndexSequence:
    map_ = dependence_to_isl_map(expr, snk_dom, src_dom)
    # if not map_.is_bijective():
    #    print(f"Map {map_} for expr {expr} with cond {condition} is not bijective")
    # else:
    #    print(f"Map {map_} for expr {expr} with cond {condition} is bijective")
    # print(f"Map {map_} for expr {expr}")
    map_ = map_.reverse()
    # print(f"Reversed map {map_}")
    res = union_map_to_seq_expr(src_dom, snk_dom, map_)
    # print(f"Reversed expr {res}")
    return res


# def reverse_condition(condition: ie.BooleanIndexValue, domain: DomainLike)
#  -> ie.BooleanIndexValue:
#
#    vars_ = condition.vars_used()
#    vars_str = ", ".join([str(v) for v in vars_])
#
#    parameters_str = ", ".join([str(b) for b in domain.parameters])
#
#    dom_int = domain.intersect(Domain.from_(vars_))
#
#    constraints = " and ".join(
#        [
#            f"{lb} <= {v} < {ub}"
#            for lb, v, ub in zip(dom_int.lbounds, vars_, dom_int.ubounds)
#        ]
#    )
#
#    ctx_str = (
#        f"[{parameters_str}] -> {{[{vars_str}]: {constraints}}}"
#    )
#    # print(f"Full str: {ctx_str}")
#    context = isl.Set(ctx_str)
#    build = isl.AstBuild.from_context(context)
#    bool_str = (
#        f"[{parameters_str}] -> {{ [{vars_str}] :  {condition} }}"
#    )
#    # print(f"Bool str: {bool_str}")
#    bool_set = isl.Set(bool_str)
#
#    ast_expr = build.expr_from_set(bool_set)
#    expr = isl_expr_to_tempo_boolean_index_value_expr(ast_expr)
#    return expr

# def isl_point_to_seq_expr(isl_point: isl.Point, domain: DomainLike) -> ie.IndexSequence:
#    sets = union_set_to_seq_expr(isl_point.to_set(), domain)
#    assert len(sets) == 1
#    return sets[0]


# def union_set_to_seq_expr(
#    uset: isl.UnionSet, domain: DomainLike
# ) -> List[ie.IndexSequence]:
#    dom = Domain.from_(domain)
#    vars_ = dom.variables
#    renaming = {f"i{i}": str(v) for i, v in enumerate(vars_)}
#
#    lex_min_set = uset.lexmin()
#    lex_max_set = uset.lexmax()
#
# def point_to_seq_expr(point: isl.Point, domain: DomainLike) -> ie.IndexSequence:
#    return isl_point_to_seq_expr(point.to_set(), domain)


def combine_many_edges(
    ops: List[top.TensorOp],
    edges: List[DependencyData],
    known_symbols: Optional[Mapping[ie.Symbol, int]] = None,
    ctx: Optional[islt.Context] = None,
) -> DependencyData:
    # edges[i] is the edge from ops[i] to ops[i+1]
    assert len(ops) == len(edges) + 1

    combined_edge = edges[0]
    fixed_snk = ops[0]
    current_combined_edge_src = ops[1]
    for i in range(1, len(edges)):
        combined_edge = combine_edges(
            fixed_snk,
            combined_edge,
            current_combined_edge_src,
            edges[i],
            ops[i + 1],
            known_symbols=known_symbols,
            ctx=ctx,
        )
        current_combined_edge_src = ops[i + 1]

    return combined_edge


def can_combine_edges(
    snk: top.TensorOp,
    edge1: DependencyData,
    mid: top.TensorOp,
    edge2: DependencyData,
    src: top.TensorOp,
    known_symbols: Optional[Mapping[ie.Symbol, int]] = None,
    ctx: Optional[islt.Context] = None,
) -> bool:
    from tempo.core import global_objects as glob

    # NOTE: The following is a hack to handle a special case. Remove later
    dg = glob.get_active_dg()
    dyn_bounds = dg.dynamic_bounds
    for B, B_def in dyn_bounds.items():
        b = B.as_var()
        # if len(list(B_def.vars_used())) == 1:
        def_vars = list(B_def.vars_used())
        def_vars_valid = [d for d in def_vars if not d.struct_eq(b)]
        if len(def_vars_valid) == 0:
            continue
        # print(f"{b=} {B=} {B_def=} {def_vars=} {def_vars_valid=}")
        def_var = def_vars_valid[0]

        # NOTE: if src has b and def_var in domain, mid has def_var and snk has neither,
        # then we cannot combine. This is due to mid serving to fetch the correct value
        # from src for the dynamically terminated dimension.
        if (src.domain.has_dim(b) and src.domain.has_dim(def_var)) and (
            mid.domain.has_dim(def_var)
            and not snk.domain.has_dim(b)
            and not snk.domain.has_dim(def_var)
            # and (B in edge2.expr.bound_symbols_used() or B in edge1.expr.bound_symbols_used())
        ):
            return False

    try:
        combine_edges(snk, edge1, mid, edge2, src, known_symbols, ctx)
        return True
    except Exception:
        return False


def combine_edges(
    snk: top.TensorOp,
    edge1: DependencyData,
    mid: top.TensorOp,
    edge2: DependencyData,
    src: top.TensorOp,
    known_symbols: Optional[Mapping[ie.Symbol, int]] = None,
    ctx: Optional[islt.Context] = None,
) -> DependencyData:
    try:
        union_dom = Domain.union(snk.domain, mid.domain, src.domain)
        for v in union_dom.variables:
            if mid.domain.has_dim(v) and src.domain.has_dim(v):
                mid_idx = mid.domain.find_variable_index(v)
                src_idx = src.domain.find_variable_index(v)
                if isinstance(edge1.expr.members[mid_idx], ie.Slice) and isinstance(
                    edge2.expr.members[src_idx], ie.Slice
                ):
                    raise ValueError(f"Cannot combine edges as they both slice dimension {v}")

        return combine_edges_dom(
            snk.domain, edge1, mid.domain, edge2, src.domain, known_symbols, ctx
        )
    except Exception as e:
        # print(f"{snk=} {edge1=} {mid=} {edge2=} {src=}")
        # print(f"Error combining edges: {e}")
        # print()
        # print(f"Edge 1: {edge1.creation_traceback}")
        # print()
        # print(f"Edge 2: {edge2.creation_traceback}")
        # print()
        raise e


def combine_edges_dom(
    snk_dom: DomainLike,
    edge1: DependencyData,
    mid_dom: DomainLike,
    edge2: DependencyData,
    src_dom: DomainLike,
    known_symbols: Optional[Mapping[ie.Symbol, int]] = None,
    ctx: Optional[islt.Context] = None,
) -> DependencyData:
    snk_dom = Domain.from_(snk_dom)
    mid_dom = Domain.from_(mid_dom)
    src_dom = Domain.from_(src_dom)

    dom_union = Domain.union(snk_dom, mid_dom, src_dom)
    parameters, _ = get_parameters_and_var_bounds_strs(dom_union)

    cond: Optional[ie.BooleanIndexValue] = (edge1.cond or ie.ConstBool(True)) & (
        edge2.cond or ie.ConstBool(True)
    )
    if cond == ie.ConstBool(True):
        cond = None
    union_map1 = dependence_to_isl_map(edge1.expr, snk_dom, mid_dom, parameters=parameters, ctx=ctx)
    union_map2 = dependence_to_isl_map(edge2.expr, mid_dom, src_dom, parameters=parameters, ctx=ctx)

    # print(f"Union map 1: {union_map1}")
    # print(f"Union map 2: {union_map2}")

    combined_map = union_map1.apply_range(union_map2)

    if known_symbols is not None:
        param_set = get_params_set(known_symbols, ctx)
        combined_map = combined_map.gist_params(param_set)

    # print(f"Combined map: {combined_map}")
    combined_map = combined_map.coalesce().coalesce()

    expr = union_map_to_seq_expr(snk_dom, src_dom, combined_map)

    assert len(expr.members) == len(src_dom), (
        f"Expected {len(src_dom)} members, got {len(expr.members)}"
    )

    return DependencyData(
        expr, src_out_idx=edge2.src_out_idx, sink_in_idx=edge1.sink_in_idx, cond=cond
    )


def set_to_seq_expr(set_: islt.Set, domain: DomainLike) -> ie.IndexSequence:
    dom = Domain.from_(domain)
    vars_ = dom.variables

    renaming = {f"i{i}": str(v) for i, v in enumerate(vars_)}

    # TODO: may need to mess with this because of dynamic bounds
    renaming.update({str(b): str(b) for b in dom.parameters})

    # Compute lexicographic min and max as PwMultiAffs directly
    lex_min_pw_multi_aff = set_.lexmin_pw_multi_aff()
    lex_max_pw_multi_aff = set_.lexmax_pw_multi_aff()

    new_members: List[ie.IndexAtom] = []

    # Loop over dimensions and analyze bounds
    for i in range(lex_min_pw_multi_aff.n_piece()):
        lex_min_pw_aff = lex_min_pw_multi_aff.get_pw_aff(i)
        lex_max_pw_aff = lex_max_pw_multi_aff.get_pw_aff(i)

        if lex_min_pw_aff == lex_max_pw_aff:
            # Single-point index expression
            ast_expr = isl.AstBuild.from_context(lex_min_pw_aff.domain()).expr_from_pw_aff(
                lex_min_pw_aff
            )
            new_member = isl_expr_to_tempo_int_index_value_expr(ast_expr, renaming)
            new_members.append(new_member)
        else:
            # Range (slice) expression
            lex_min_ast_expr = isl.AstBuild.from_context(lex_min_pw_aff.domain()).expr_from_pw_aff(
                lex_min_pw_aff
            )
            lex_max_ast_expr = isl.AstBuild.from_context(lex_max_pw_aff.domain()).expr_from_pw_aff(
                lex_max_pw_aff.add_constant_val(isl.Val(1))
            )
            lex_min_new_member = isl_expr_to_tempo_int_index_value_expr(lex_min_ast_expr, renaming)
            lex_max_new_member = isl_expr_to_tempo_int_index_value_expr(lex_max_ast_expr, renaming)
            new_members.append(ie.slice_(lex_min_new_member, lex_max_new_member))

    return ie.IndexSequence(tuple(new_members))


def union_map_to_seq_expr(
    snk_dom: DomainLike,
    src_dom: DomainLike,
    map_: islt.UnionMap,
    ctx: Optional[islt.Context] = None,
) -> ie.IndexSequence:
    snk_dom = Domain.from_(snk_dom)
    src_dom = Domain.from_(src_dom)

    union_dom = Domain.union(snk_dom, src_dom)
    vars_ = snk_dom.variables

    renaming = {f"i{i}": str(v) for i, v in enumerate(vars_)}

    # TODO: may need to mess with this because of dynamic bounds
    renaming.update({str(b): str(b) for b in union_dom.parameters})

    # We need to seperate the processing into the case where the
    # lex_min and lex_max match and when they don't
    # so that we can capture slices (e.g. 0:5).
    lex_min_map = map_.lexmin()
    lex_max_map = map_.lexmax()

    lex_min_multi_pw_aff = isl.UnionPwMultiAff.from_union_map(lex_min_map).as_pw_multi_aff()
    lex_max_multi_pw_aff = isl.UnionPwMultiAff.from_union_map(lex_max_map).as_pw_multi_aff()
    new_members: List[ie.IndexAtom] = []
    for i in range(len(src_dom)):  # TODO also use lex_min_multi_pw_aff.n_piece()??
        lex_min_pw_aff = lex_min_multi_pw_aff.get_pw_aff(i)
        lex_max_pw_aff = lex_max_multi_pw_aff.get_pw_aff(i)
        if lex_min_pw_aff == lex_max_pw_aff:
            ast_expr = isl.AstBuild.from_context(lex_min_pw_aff.domain()).expr_from_pw_aff(
                lex_min_pw_aff
            )
            new_member = isl_expr_to_tempo_int_index_value_expr(ast_expr, renaming)
            new_members.append(new_member)
        else:
            lex_min_ast_expr = isl.AstBuild.from_context(lex_min_pw_aff.domain()).expr_from_pw_aff(
                lex_min_pw_aff
            )
            lex_max_ast_expr = isl.AstBuild.from_context(lex_max_pw_aff.domain()).expr_from_pw_aff(
                lex_max_pw_aff.add_constant_val(isl.Val(1))
            )
            lex_min_new_member = isl_expr_to_tempo_int_index_value_expr(lex_min_ast_expr, renaming)
            lex_max_new_member = isl_expr_to_tempo_int_index_value_expr(lex_max_ast_expr, renaming)
            new_members.append(ie.slice_(lex_min_new_member, lex_max_new_member))
    return ie.IndexSequence(tuple(new_members))


def _make_map(map_str: str, ctx: Optional[islt.Context]) -> islt.UnionMap:
    # try:
    map_ = isl.UnionMap(map_str, context=ctx)
    # except Exception as e:
    #    log.error("Error when creating map: %s", map_str)
    #    import traceback
    #    traceback.print_stack()
    #    raise e
    return map_


def int_index_value_to_union_map(
    expr: ie.IntIndexValue,
    domain: DomainLike,
    condition: Optional[ie.BooleanIndexValue] = None,
    ctx: Optional[islt.Context] = None,
) -> islt.UnionMap:
    if ctx is None:
        ctx = get_isl_context(None)  # type: ignore

    # print("Calling int_index_value_to_union_map with:")

    # print(f"Expr: {expr}")

    # snk_dom = Domain.from_(expr.vars_used())

    from tempo.core import global_objects as glob

    dg = glob.get_active_dg()

    # TODO: probably should not provide an upper bound.
    with dg.new_temp_var(ie.ConstInt(123456789)) as (d, D):
        src_dom = Domain.from_((d,))
        # snk_dom = domain.indexed_real_domain(ie.IndexSequence([expr]))
        map_ = dependence_to_isl_map(
            ie.IndexSequence([expr]), domain, src_dom, condition=condition, ctx=ctx
        )
        # print(f"    Domain: {snk_dom}")
        # print(f"    Domain: {src_dom}")
        # print(f"    Map: {map_}")

    return map_


def int_index_value_to_pw_aff(
    expr: ie.IntIndexValue,
    condition: Optional[ie.BooleanIndexValue] = None,
    known_symbols: Optional[Mapping[ie.Symbol, int]] = None,
    ctx: Optional[islt.Context] = None,
) -> islt.PwAff:
    # vars_ = expr.vars_used()
    # vars_ = domain.variables
    # vars_str = ", ".join([str(v) for v in vars_])

    ## TODO: A parameter is currently only included if
    # the corresponding variable is used in the expr.
    ## This is not what we want, so we need to change this.
    ## For example, if the shape is D0, the bound symbol will not be included.
    # bounds_str, constraints = get_parameters_and_var_bounds_strs(domain)

    # condition_str = f"{condition}" if condition is not None else "true"

    # full_str = (
    #    f"[{bounds_str}] -> {{ [{vars_str}] -> [{expr}]: {condition_str} and {constraints} }}"
    # )

    if ctx is None:
        ctx = get_isl_context(None)  # type: ignore

    # if domain is None:
    #    domain = Domain.from_(expr.vars_used())
    domain = Domain.universe()

    map_ = int_index_value_to_union_map(expr, domain, condition, ctx=ctx)
    try:
        pw_aff = union_map_to_pw_aff(map_)
    except Exception as e:
        print(f"Error when creating pw_aff from map: {map_}")
        import traceback

        traceback_ = traceback.format_exc()
        print(f"Traceback: {traceback_}")
        raise e

    if known_symbols is not None:
        param_set = get_params_set(known_symbols, ctx=ctx)
        pw_aff = pw_aff.gist_params(param_set)

    return pw_aff


def union_map_to_pw_aff(map_: islt.UnionMap) -> islt.PwAff:
    m_un_pw_aff = map_.as_multi_union_pw_aff()

    un_pw_aff = m_un_pw_aff.get_union_pw_aff(0)
    assert int(un_pw_aff.n_pw_aff()) == 1
    pw_affs = []
    un_pw_aff.foreach_pw_aff(lambda x: pw_affs.append(x))
    assert len(pw_affs) == 1
    pw_aff = pw_affs[0]
    return pw_aff


# TODO: switch to a symbolic max_val approach
# Here are two ways to do that:
# >>> isl.pw_aff("[D0] -> { [d0] -> [d0+1]: 0 <= d0 < D0 and D0 > 0}")
#     .as_map().range().lexmax_pw_multi_aff().at(0)
# isl.pw_aff("[D0] -> { [(D0)] : D0 > 0 }")
# >>> isl.pw_aff("[D0] -> { [d0] -> [d0+1]: 0 <= d0 < D0 and D0 > 0}")
#      .as_map().range().max_multi_pw_aff().at(0)
# isl.pw_aff("[D0] -> { [(D0)] : D0 > 0 }")
def int_index_val_min(
    expr: ie.IntIndexValue,
    condition: Optional[ie.BooleanIndexValue] = None,
    known_symbols: Optional[Mapping[ie.Symbol, int]] = None,
) -> Optional[int]:
    pw_aff = int_index_value_to_pw_aff(expr, condition=condition, known_symbols=known_symbols)
    min_val = pw_aff.min_val()
    if min_val.is_int():
        x = min_val.to_python()
        assert isinstance(x, int)
        return x
    else:
        return None


def int_index_val_max(
    expr: ie.IntIndexValueLike,
    condition: Optional[ie.BooleanIndexValue] = None,
    known_symbols: Optional[Mapping[ie.Symbol, int]] = None,
) -> Optional[ie.IntIndexValue]:
    # map_ = int_index_value_to_union_map(expr, domain, condition, known_symbols)
    # print(f"MAX VAL MAP: {map_}")
    # print(f"MAX VAL MAP lexmax: {map_.lexmax()}")
    # res = union_map_to_seq_expr(domain, domain, map_)
    # print(f"MAX VAL MAP TO SEQ EXPR: {res}")
    # try:
    #    max_set = map_.range().lexmax()
    # except Exception:
    #    # NOTE: unbounded
    #    return None

    # print(f"MAX VAL SET: {max_set}")

    # as_tempo_expr = isl_set_to_index_value(max_set, domain)
    # print(f"WHOLE EXPR CONVERSION: {as_tempo_expr}")

    # max_sets = isl.Set(str(max_set)).get_basic_sets()
    # cases = []
    # for max_set in max_sets:
    #    constraints = max_set.get_constraints()
    #    constraint_dict = {}
    #    for constraint in constraints:
    #        constraint_set = isl.Set(str(constraint))
    #        constraint_expr = isl_set_to_index_value(constraint_set, domain)
    #        if len(constraint_expr.vars_used()) == 0:
    #            constraint_dict["cond"] = constraint_expr
    #        else:
    #            assert isinstance(constraint_expr, ie.Equal),
    # f"Expected Equal, got {type(constraint_expr)}: {constraint_expr}"
    #            assert isinstance(constraint_expr.left_operand, ie.Symbol),
    # f"Expected Symbol, got {type(constraint_expr.left_operand)}: {constraint_expr.left_operand}"
    #            constraint_dict["value"] = constraint_expr.right_operand

    #    if "cond" not  in constraint_dict:
    #        constraint_dict["cond"] = ie.ConstBool(True)
    #    cases.append(constraint_dict)
    # if len(cases) == 1:
    #    return cases[0]["value"]
    # else:
    #    return ie.Piecewise([(case["cond"], case["value"]) for case in cases])

    # as_tempo_expr = isl_set_to_index_value(max_set, domain)
    # print(f"MAX VAL TEMPO EXPR: {as_tempo_expr}")

    # assert isinstance(as_tempo_expr, ie.Equal), \
    # f"Expected Equal, got {type(as_tempo_expr)}: {as_tempo_expr}"
    # isl.Set(" [D0] -> { [i0 = 5] : D0 >= 6;
    #  [i0 = 4] : D0 = 5;
    # [i0 = -1 + D0] : 2 <= D0 <= 4 }").get_basic_sets()[0].get_constraints()[0]
    # return as_tempo_expr.right_operand

    # return as_tempo_expr

    expr = ie.lift_to_int_ie(expr)

    pw_aff = int_index_value_to_pw_aff(expr, condition=condition, known_symbols=known_symbols)
    max_val = pw_aff.max_val()

    if max_val.is_int():
        x = max_val.to_python()
        return ie.ConstInt(x)
    else:
        return None


def pw_aff_to_index_value(
    pw_aff: islt.PwAff,
) -> ie.IntIndexValue:
    vars_ = Domain.universe().variables
    renaming = {f"i{i}": str(v) for i, v in enumerate(vars_)}

    # ctx = get_isl_context(None)  # type: ignore
    expr = isl.AstBuild.from_context(pw_aff.domain()).expr_from_pw_aff(pw_aff)

    n = isl_expr_to_tempo_int_index_value_expr(expr, renaming)

    return n


def expr_eq(  # noqa: C901
    expr1: Any,
    expr2: Any,
    ctx: Optional[islt.Context] = None,
) -> bool:
    if not isinstance(expr1, ie.IndexExpr):
        expr1 = ie.lift_to_ie(expr1)
    if not isinstance(expr2, ie.IndexExpr):
        expr2 = ie.lift_to_ie(expr2)
    if isinstance(expr1, ie.IndexSequence) != isinstance(expr2, ie.IndexSequence):
        return False
    if not isinstance(expr1, ie.IndexSequence) and not isinstance(expr2, ie.IndexSequence):
        expr1 = ie.IndexSequence([expr1])
        expr2 = ie.IndexSequence([expr2])
    assert isinstance(expr1, ie.IndexSequence) and isinstance(expr2, ie.IndexSequence)

    if len(expr1) != len(expr2):
        return False

    # TODO: this is a problem, because it will make d0-d0 != 0... FFS
    for e1, e2 in zip(expr1.members, expr2.members, strict=False):
        # TODO: this == could be a problem
        if not set(e1.vars_used()) == set(e2.vars_used()):
            return False

    all_bounds = list(set(list(expr1.bound_symbols_used()) + list(expr2.bound_symbols_used())))

    def make_anon_map(expr: ie.IndexSequence) -> islt.UnionMap:
        snk_vars = expr.vars_used()
        dom = Domain.from_(snk_vars)

        sink_vars_str = ", ".join([str(v) for v in dom.variables])
        src_vars = [f"i{i}" for i in range(len(expr.members))]
        src_vars_str = ",".join(src_vars)

        conds_and_branches = expr.enumerate_all_cond_branches()

        i_defs_and_conds = []
        for branch_cond, branch in conds_and_branches:
            assert isinstance(branch, ie.IndexSequence)
            branch_cond_str = f"{branch_cond}" if branch_cond is not None else "true"

            i_defs_ = []
            for src_var, index_expr in zip(src_vars, branch.members, strict=False):
                if isinstance(index_expr, ie.Slice):
                    i_defs_.append(f"{index_expr.start} <= {str(src_var)} < {index_expr.stop}")
                else:
                    i_defs_.append(f"{str(src_var)}={str(index_expr)}")
            i_defs = " and ".join(i_defs_) or "true"
            i_defs_and_conds.append(f"({i_defs} and {branch_cond_str})")

        ored_i_defs_and_conds = " or ".join(i_defs_and_conds) or "true"

        constraint = f"[{sink_vars_str}] -> [{src_vars_str}]: "
        constraint += f"({ored_i_defs_and_conds})"

        parameters = ", ".join([str(b) for b in all_bounds])
        constraint = f"[{parameters}] -> {{ {constraint} }}"

        return _make_map(constraint, ctx).coalesce().coalesce()

    return bool(make_anon_map(expr1) == make_anon_map(expr2))


def dependence_to_isl_map(  # noqa: C901
    expr: ie.IndexSequence,
    snk_dom: DomainLike,
    src_dom: DomainLike,
    sink_name: str = empty_str,
    src_name: str = empty_str,
    condition: Optional[ie.BooleanIndexValue] = None,
    snk_dom_isl: Optional[islt.Set] = None,
    src_dom_isl: Optional[islt.Set] = None,
    ctx: Optional[islt.Context] = None,
    parameters: Optional[str] = None,
) -> islt.UnionMap:
    snk_dom = Domain.from_(snk_dom)
    src_dom = Domain.from_(src_dom)

    from tempo.core import global_objects as glob

    dynamic_bounds = glob.get_dynamic_bounds_or_empty()
    expr = expr.remap(dynamic_bounds)
    condition = condition.remap(dynamic_bounds) if condition is not None else None

    universe = Domain.union(snk_dom, src_dom)

    sink_vars_str = ", ".join([str(v) for v in snk_dom.variables])
    src_vars = [f"i{i}" for i in range(len(src_dom))]
    src_vars_str = ",".join(src_vars)

    cond_str = f"{condition}" if condition is not None else "true"
    _, sink_bounds_str = get_parameters_and_var_bounds_strs(snk_dom)
    _, src_bounds_str = get_parameters_and_var_bounds_strs(src_dom, var_renaming=src_vars)

    conds_and_branches = expr.enumerate_all_cond_branches()

    i_defs_and_conds = []
    for branch_cond, branch in conds_and_branches:
        assert isinstance(branch, ie.IndexSequence)
        branch_cond_str = f"{branch_cond}" if branch_cond is not None else "true"

        i_defs_ = []
        for src_var, index_expr in zip(src_vars, branch.members, strict=False):
            if isinstance(index_expr, ie.Slice):
                i_defs_.append(f"{index_expr.start} <= {str(src_var)} < {index_expr.stop}")
            else:
                i_defs_.append(f"{str(src_var)}={str(index_expr)}")
        i_defs = " and ".join(i_defs_) or "true"
        i_defs_and_conds.append(f"({i_defs} and {branch_cond_str})")

    ored_i_defs_and_conds = " or ".join(i_defs_and_conds) or "true"

    constraint = f"{sink_name}[{sink_vars_str}] -> {src_name}[{src_vars_str}]: "
    constraint += f"({cond_str}) and ({ored_i_defs_and_conds})"
    # if not expr.equivalent(src_dom.basis_expr):
    #    constraint += f" and ({src_bounds_str})"
    # print(f"src_bounds_str: {src_bounds_str}")
    # print(f"sink_bounds_str: {sink_bounds_str}")
    constraint += f" and ({src_bounds_str})"
    constraint += f" and ({sink_bounds_str})"

    if parameters is None:
        parameters, _ = get_parameters_and_var_bounds_strs(universe)
    constraint = f"[{parameters}] -> {{ {constraint} }}"

    # print(f"Constraint: {constraint}")
    map_ = _make_map(constraint, ctx).coalesce().coalesce()
    if snk_dom_isl is not None:
        map_ = map_.intersect_domain(snk_dom_isl)
    if src_dom_isl is not None:
        map_ = map_.intersect_range(src_dom_isl)
    return map_


def try_join_atoms(atoms: Sequence[ie.IndexAtom]) -> Optional[ie.IndexAtom]:
    atom_sets = [index_atom_to_isl_set(atom) for atom in atoms]

    unioned = functools.reduce(lambda x, y: x.union(y), atom_sets).coalesce().coalesce()

    unioned_domain = functools.reduce(
        lambda x, y: Domain.union(x, y),
        [Domain.from_(atom.vars_used()) for atom in atoms],
    )

    if len(unioned_domain) > 1:
        return None

    if unioned.n_set() == 1:
        index_seq = set_to_seq_expr(unioned.as_set(), unioned_domain)
        return index_seq.members[0]
    else:
        return None


def index_atom_to_isl_set(
    seq: ie.IndexAtom, name: Optional[str] = "", ctx: Optional[islt.Context] = None
) -> islt.Set:
    bounds = seq.bound_symbols_used()
    bounds_str = ", ".join([str(b) for b in bounds])
    str_ = f"[{bounds_str}] -> {{ {name}[{seq}] }}"
    return isl.Set(str_, context=ctx)


def index_sequence_to_isl_union_set(
    seq: ie.IndexSequence, name: Optional[str] = "", ctx: Optional[islt.Context] = None
) -> islt.UnionSet:
    bounds = seq.bound_symbols_used()
    bounds_str = ", ".join([str(b) for b in bounds])
    str_ = f"[{bounds_str}] -> {{ {name}[{seq}] }}"
    return isl.UnionSet(str_, context=ctx)


def isl_domain_from_op(op: top.TensorOp, ctx: islt.Context) -> islt.Set:
    domain = op.domain
    exec_name = op_id_to_exec_name(op.op_id)
    variable_strs = ",".join(str(v) for v in domain.variables)

    parameters, bounds_str = get_parameters_and_var_bounds_strs(domain)
    full_str = f"[{parameters}] -> {{ {exec_name}[{variable_strs}]: {bounds_str}}}"

    # print(f"Domain: {full_str}")

    isl_dom = isl.Set.read_from_str(ctx, full_str)
    return isl_dom.coalesce().coalesce()


def isl_sched_to_c(schedule: islt.Schedule) -> str:
    astbuild = isl.AstBuild.from_context(isl.Set("{:}"))
    ast = astbuild.node_from_schedule(schedule)
    p = isl.Printer.to_str(ast.get_ctx())
    p = p.set_output_format(isl.format.C)
    return cast(str, p.print_ast_node(ast).get_str())


def isl_expr_to_tempo_int_index_value_expr(
    e: islt.AstExpr, renaming: Optional[Mapping[str, str]] = None
) -> ie.IntIndexValue:
    res = isl_expr_to_tempo_expr(e, renaming)
    if not isinstance(res, ie.IntIndexValue):
        raise ValueError(f"Expected IntIndexValue, go {type(res)}: {res}")
    return res


def isl_expr_to_tempo_boolean_expr(
    e: islt.AstExpr, renaming: Optional[Mapping[str, str]] = None
) -> ie.BooleanIndexValue:
    res = isl_expr_to_tempo_expr(e, renaming)
    if isinstance(res, ie.ConstInt) and res.const == 0:
        return ie.ConstBool(False)
    if isinstance(res, ie.ConstInt) and res.const == 1:
        return ie.ConstBool(True)
    if not isinstance(res, ie.BooleanIndexValue):
        raise ValueError(f"Expected BooleanIndexValue, go {type(res)}: {res}")
    return res


def rename_union_set_tuples(uset: islt.UnionSet, new_name: str) -> islt.UnionSet:
    renamed_union_set = isl.UnionSet.empty(uset.get_space())

    # Iterate over each set in the UnionSet
    set_list = uset.get_set_list()
    for j in range(set_list.n_set()):
        single_set = set_list.get_at(j)
        # Rename the tuple in the set to "X"
        renamed_set = single_set.set_tuple_name(new_name)

        # Add the renamed set to the new UnionSet
        renamed_union_set = renamed_union_set.union(renamed_set)
    return renamed_union_set


def rename_union_map_tuples(
    umap: islt.UnionMap,
    new_name_in: Optional[str] = None,
    new_name_out: Optional[str] = None,
) -> islt.UnionMap:
    renamed_union_map = isl.UnionMap.empty(umap.get_space())

    # Iterate over each map in the UnionMap
    map_list = umap.get_map_list()
    for j in range(map_list.n_map()):
        single_map = map_list.get_at(j)
        # Rename the tuple in the map to "X"
        if new_name_in is not None:
            single_map = single_map.set_tuple_name(isl.dim_type.in_, new_name_in)
        if new_name_out is not None:
            single_map = single_map.set_tuple_name(isl.dim_type.out, new_name_out)

        # Add the renamed map to the new UnionMap
        renamed_union_map = renamed_union_map.union(single_map)
    return renamed_union_map


def isl_expr_to_tempo_expr(  # noqa: C901
    e: islt.AstExpr, renaming: Optional[Mapping[str, str]] = None
) -> ie.IndexValue:
    outer_expr_type = e.get_type()

    # Symbols
    if outer_expr_type == isl.ast_expr_type.id:
        name = str(e.get_id().name)
        if renaming is not None and renaming.get(name) is not None:
            name = renaming[name]
        if name == "":
            raise ValueError("Empty name")
        else:
            from tempo.core import global_objects as glob

            if name.startswith("c"):
                idx = int(name[1:]) * 2
            else:
                idx = glob.get_active_dg().universe.find_variable_index(ie.Symbol(name.lower())) * 2
                if name[0].isupper():
                    idx += 1
            return ie.Symbol(name, is_bound=name[0].isupper(), idx=idx)

    # INT Consts
    if outer_expr_type == isl.ast_expr_type.int:
        val = e.get_val().to_python()
        return ie.ConstInt(val)

    if outer_expr_type == isl.ast_expr_type.op:
        expr_type = e.get_op_type()

        if expr_type == isl.ast_expr_op_type.le:
            left = isl_expr_to_tempo_int_index_value_expr(e.get_op_arg(0), renaming)
            right = isl_expr_to_tempo_int_index_value_expr(e.get_op_arg(1), renaming)
            return left <= right  # ie.LessThanOrEqual(left, right)
        if expr_type == isl.ast_expr_op_type.lt:
            left = isl_expr_to_tempo_int_index_value_expr(e.get_op_arg(0), renaming)
            right = isl_expr_to_tempo_int_index_value_expr(e.get_op_arg(1), renaming)
            return left < right  # ie.LessThan(left, right)
        if expr_type == isl.ast_expr_op_type.eq:
            left = isl_expr_to_tempo_int_index_value_expr(e.get_op_arg(0), renaming)
            right = isl_expr_to_tempo_int_index_value_expr(e.get_op_arg(1), renaming)
            return left.symb_eq(right)  # ie.Equal(left, right)
        if expr_type == isl.ast_expr_op_type.ge:
            left = isl_expr_to_tempo_int_index_value_expr(e.get_op_arg(0), renaming)
            right = isl_expr_to_tempo_int_index_value_expr(e.get_op_arg(1), renaming)
            return left >= right  # ie.GreaterThanOrEqual(left, right)
        if expr_type == isl.ast_expr_op_type.gt:
            left = isl_expr_to_tempo_int_index_value_expr(e.get_op_arg(0), renaming)
            right = isl_expr_to_tempo_int_index_value_expr(e.get_op_arg(1), renaming)
            return left > right  # ie.GreaterThan(left, right)
        if expr_type == isl.ast_expr_op_type.add:
            left = isl_expr_to_tempo_int_index_value_expr(e.get_op_arg(0), renaming)
            right = isl_expr_to_tempo_int_index_value_expr(e.get_op_arg(1), renaming)
            return left + right  # ie.Add(left, right)
        if expr_type == isl.ast_expr_op_type.sub:
            left = isl_expr_to_tempo_int_index_value_expr(e.get_op_arg(0), renaming)
            right = isl_expr_to_tempo_int_index_value_expr(e.get_op_arg(1), renaming)
            return left - right  # ie.Sub(left, right)
        if expr_type == isl.ast_expr_op_type.mul:
            left = isl_expr_to_tempo_int_index_value_expr(e.get_op_arg(0), renaming)
            right = isl_expr_to_tempo_int_index_value_expr(e.get_op_arg(1), renaming)
            return left * right  # ie.Mul(left, right)
        if expr_type == isl.ast_expr_op_type.pdiv_q:
            left = isl_expr_to_tempo_int_index_value_expr(e.get_op_arg(0), renaming)
            right = isl_expr_to_tempo_int_index_value_expr(e.get_op_arg(1), renaming)
            return left // right  # ie.FloorDivision(left, right)
        if expr_type == isl.ast_expr_op_type.div:
            left = isl_expr_to_tempo_int_index_value_expr(e.get_op_arg(0), renaming)
            right = isl_expr_to_tempo_int_index_value_expr(e.get_op_arg(1), renaming)
            return left // right  # TODO figure out if this is correct or is /
        if expr_type == isl.ast_expr_op_type.fdiv_q:
            left = isl_expr_to_tempo_int_index_value_expr(e.get_op_arg(0), renaming)
            right = isl_expr_to_tempo_int_index_value_expr(e.get_op_arg(1), renaming)
            return left / right
        if expr_type == isl.ast_expr_op_type.zdiv_r:  # NOTE: Divisor may be negative
            left = isl_expr_to_tempo_int_index_value_expr(e.get_op_arg(0), renaming)
            right = isl_expr_to_tempo_int_index_value_expr(e.get_op_arg(1), renaming)
            return left % right
        if expr_type == isl.ast_expr_op_type.pdiv_r:  # NOTE: Divisor known positive
            left = isl_expr_to_tempo_int_index_value_expr(e.get_op_arg(0), renaming)
            right = isl_expr_to_tempo_int_index_value_expr(e.get_op_arg(1), renaming)
            return left % right  # ie.Modulos(left, right)  # TODO probably wrong

        # Unary Int Ops
        if expr_type == isl.ast_expr_op_type.minus:
            arg = isl_expr_to_tempo_int_index_value_expr(e.get_op_arg(0), renaming)
            return -arg

        # N-ary
        if expr_type == isl.ast_expr_op_type.max:
            nargs = e.get_op_n_arg()
            args = tuple(
                isl_expr_to_tempo_int_index_value_expr(e.get_op_arg(i), renaming)
                for i in range(nargs)
            )
            return ie.max(*args)  # ie.Max(args)
        if expr_type == isl.ast_expr_op_type.min:
            nargs = e.get_op_n_arg()
            args = tuple(
                isl_expr_to_tempo_int_index_value_expr(e.get_op_arg(i), renaming)
                for i in range(nargs)
            )
            return ie.min(*args)
        # Boolean Binary Ops
        if expr_type == isl.ast_expr_op_type.or_:
            left_b = isl_expr_to_tempo_boolean_expr(e.get_op_arg(0), renaming)
            right_b = isl_expr_to_tempo_boolean_expr(e.get_op_arg(1), renaming)
            return left_b | right_b  # ie.Or(left_b, right_b)
        if expr_type == isl.ast_expr_op_type.and_:
            left_b = isl_expr_to_tempo_boolean_expr(e.get_op_arg(0), renaming)
            right_b = isl_expr_to_tempo_boolean_expr(e.get_op_arg(1), renaming)
            return left_b & right_b  # ie.And(left_b, right_b)

        if expr_type == isl.ast_expr_op_type.select:
            cond = isl_expr_to_tempo_boolean_expr(e.get_op_arg(0), renaming)
            true = isl_expr_to_tempo_int_index_value_expr(e.get_op_arg(1), renaming)
            false = isl_expr_to_tempo_int_index_value_expr(e.get_op_arg(2), renaming)
            return ie.piecewise(((cond, true), (~cond, false)))

        if expr_type == isl.ast_expr_op_type.error:
            raise NotImplementedError("Error not supported")

        raise NotImplementedError(f"No support for {e} op_ty={expr_type}")

    if outer_expr_type == isl.ast_expr_type.error:
        raise NotImplementedError
    else:
        raise ValueError("Unknown expression type: %d" % (outer_expr_type,))
