from tempo.core import index_expr as ie
from tempo.core.domain import Domain
from tempo.core.dependence_graph import PDG, DependencyData
from tempo.core import tensor_ops as top
from tempo.core.datatypes import OpId
from tempo.utils.isl import combine_edges, combine_edges_dom

from tempo.api.tempo_context_manager import TempoContext
from tempo.core.configs import ExecutionConfig


def test_combine_piecewise_with_modulo_block(exec_cfg: ExecutionConfig):
    # Set up the context with all required dimensions

    tpo_ctx = TempoContext(exec_cfg, num_dims=2)
    (t, T), (b, B) = tpo_ctx.variables_and_bounds
    pdg = PDG(universe=tpo_ctx.variables)
    from tempo.core import global_objects as glob
    glob.set_active_dg(pdg)


    pdg.bound_defs[B] = ie.Ceil(t / ie.ConstInt(64))
    pdg.bound_defs[T] = 1024
    with tpo_ctx:

        # Create the dependency edges
        edge1 = DependencyData(
            expr=ie.IndexSequence((
                ie.min((b + 1) * 64 - 1, t),
            )),
            src_out_idx=0,
            sink_in_idx=0
        )

        edge2 = DependencyData(
            expr=ie.IndexSequence((
                ie.slice_(t-(t%64),t),
            )),
            src_out_idx=0,
            sink_in_idx=0
        )

        # Known symbols dictionary
        known_symbols = pdg.static_bounds

        # Combine the edges
        combined_edge = combine_edges_dom(
            snk_dom=(t, b),
            edge1=edge1,
            mid_dom=(t,),
            edge2=edge2,
            src_dom=(t,),
            known_symbols=known_symbols,
            ctx=tpo_ctx.get_isl_ctx()
        )

        print(combined_edge)




def test_specific_edge_combination(exec_cfg: ExecutionConfig):
    # Set up the context with all required dimensions

    #Define symbols once
    #d0, d1, d2, di0, di1 = ie.symbols("d0 d1 d2 di0 di1")
    #D0, D1, D2, DI0, DI1 = ie.bound_symbols("D0 D1 D2 DI0 DI1")

    tpo_ctx = TempoContext(exec_cfg, num_dims=5)

    with tpo_ctx:
        (d0, D0), (d1, D1), (d2, D2), (di0, DI0), (di1, DI1) = tpo_ctx.variables_and_bounds

        #src = RecurrentTensor.rand((5,5), domain=(d1,d2))
        #mid = src.negate()

        # Define the domains with their variables and bounds
        permute_domain = Domain(
            [d1, di0, di1],
            [D1, DI0, DI1]
        )

        ident_domain = Domain(
            [d1, di0, di1],
            [D1, DI0, DI1]
        )

        sub_domain = Domain(
            [d1, d2],
            [D1, D2]
        )

        # Create the operations
        permute_op = top.PermuteOp(
            op_id=OpId(2156),
            domain=permute_domain,
            dims=(1, 0, 2),
            tags={}
        )

        ident_op = top.IdentOp(
            op_id=OpId(2202),
            domain=ident_domain,
            tags={},
        )

        sub_op = top.SubOp(
            op_id=OpId(1560),
            domain=sub_domain,
            tags={},
        )

        # Create the dependency edges
        edge1 = DependencyData(
            expr=ie.IndexSequence((
                d1,
                di0,
                di1
            )),
            src_out_idx=0,
            sink_in_idx=0
        )

        edge2 = DependencyData(
            expr=ie.IndexSequence((
                d1,
                ie.slice_(
                    di0 * 40 + di1 * 10,
                    (di0 * 40 + (di1 + 1) * 10) - 1
                )
            )),
            src_out_idx=0,
            sink_in_idx=0
        )

        # Known symbols dictionary
        known_symbols = {
            D0: 16,
            D1: 20,
            D2: 200,
            DI0: 5,
            DI1: 4
        }

        # Combine the edges
        combined_edge = combine_edges(
            snk=permute_op,
            edge1=edge1,
            mid=ident_op,
            edge2=edge2,
            src=sub_op,
            known_symbols=known_symbols,
            ctx=tpo_ctx.get_isl_ctx()
        )

    # Verify the result
    assert len(combined_edge.expr.enumerate_all_cond_branches()) == 1

    assert combined_edge is not None
    assert len(combined_edge.expr.members) == 2  # Should have 2 dimensions matching sub_domain
    assert combined_edge.sink_in_idx == 0
    assert combined_edge.src_out_idx == 0

if __name__ == "__main__":
    #pytest.main([__file__])
    test_combine_piecewise_with_modulo_block(ExecutionConfig(
        path="./",
        visualize_pipeline_stages=False,
        dev="cpu",
        backend="torch",
        deterministic=True,
        seed=0,
        executor_debug_mode=False,
        gc_bytes_min=1024 * 1024,  # 1MiB
        enable_dataflow_grouping=False,
        enable_constant_folding=False,
        enable_dead_code_elim=False,
        enable_duplicate_code_elim=False,
        enable_algebraic_optimizer=False,
        enable_broadcast_elim=False,
        enable_incrementalization=False,
        enable_vectorization=False,
        enable_gc=False,
        enable_swap=False,
        enable_parallel_block_detection=False,
        enable_donation_analysis=False,
        enable_hybrid_tensorstore=False,
        enable_x64=True,
    )
    )
