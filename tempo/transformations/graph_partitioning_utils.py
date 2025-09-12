from collections.abc import Sequence

import networkx as nx
import numpy as np

from tempo.core import index_expr as ie
from tempo.core.datatypes import OpId, OpInId, OpOutId
from tempo.core.dependence_graph import PDG, DependencyData
from tempo.core.tensor_ops import TensorOp
from tempo.utils import logger
from tempo.utils.memory_estimator import MemoryEstimator

log = logger.get_logger(__name__)


def ilp_based_cut(
    dg: PDG,
    node_group: set[TensorOp],
    router_info: tuple[
        tuple[tuple[tuple[OpId, OpInId], ...], ...],
        tuple[tuple[OpId, OpOutId], ...],
        dict[tuple[OpId, OpOutId, ie.IndexSequence], int],
    ],
    bytes_importance: float = 0.75,
    max_allowed_imbalance_percent: float = 0.75,
    lambda_balance: float = 100,
    required_cut_edges: Sequence[tuple[TensorOp, TensorOp, DependencyData]] = (),  # NEW
    mem_est: MemoryEstimator | None = None,
) -> tuple[set[TensorOp], set[TensorOp], int]:
    if bytes_importance > 0:
        assert mem_est is not None, "Memory estimator is required when bytes_importance > 0"

    # First, insert fake ops for the irouter
    _, _, inp_index_tracker = router_info
    fake_ops = {dg.get_op_by_id(op_id) for op_id, _, _ in inp_index_tracker.keys()}

    group_dg = dg.induced_subgraph(OpId(-1), {*node_group, *fake_ops})

    import pulp

    G = group_dg.get_networkx_graph().copy()
    max_bytes = mem_est.get_max_tensor_out_bytes() if mem_est is not None else 0

    for edge in G.edges:
        data: DependencyData = G.edges[edge]["dependency_data"]

        if data.is_control_edge:
            raise ValueError("Control edges should not be present in the group")

        capacity = (1 - bytes_importance) * 1

        if mem_est is not None:
            bytes_num = mem_est.estimate_tensor_point_size_bytes(edge[1].op_id, data.src_out_idx)
            bytes_normalized = bytes_num / max_bytes
            capacity += bytes_importance * bytes_normalized

        G.edges[edge]["capacity"] = capacity

    # assert nx.is_directed_acyclic_graph(G), "The graph must be acyclic."

    nodes = list(G.nodes())
    real_nodes = list(set(nodes) - fake_ops)
    edges = list(G.edges(keys=True))  # Include keys for multidigraph

    prob = pulp.LpProblem("GraphPartitioning", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", nodes, cat="Binary")  # Cluster assignments
    y = pulp.LpVariable.dicts("y", edges, cat="Binary")  # Cut edges

    # Objective Function 1: Minimize total cut weight
    min_cut_edges = pulp.lpSum(
        [G.get_edge_data(u, v, key)["capacity"] * y[(u, v, key)] for (u, v, key) in edges]
    )

    # NOTE: Enabling this objective makes optimization incredibly more expensive.
    ## Objective Function 2: Minimize imbalance
    ## Introduce auxiliary variable for imbalance
    # z = pulp.LpVariable("z", lowBound=0)  # Imbalance penalty variable

    ## Add constraints to model z as the absolute value of (2 * s - total_nodes)
    # prob += z >= (2 * s - total_nodes), "ImbalancePositive"
    # prob += z >= -(2 * s - total_nodes), "ImbalanceNegative"

    ## Add both objectives to the problem (with a trade-off parameter lambda)
    # prob += min_cut_edges + lambda_balance * z  # Minimize cut weight and imbalance

    prob += min_cut_edges  # Minimize cut weight

    # Balance Constraints
    if max_allowed_imbalance_percent < 1.0:
        total_nodes = len(nodes) - len(fake_ops)
        s = pulp.lpSum([x[i] for i in real_nodes])  # Number of nodes in Cluster 1
        max_allowed_imbalance = int(max_allowed_imbalance_percent * total_nodes)
        prob += 2 * s - total_nodes <= max_allowed_imbalance, "BalanceConstraintUpper"
        prob += 2 * s - total_nodes >= -max_allowed_imbalance, "BalanceConstraintLower"

    # Constraints to prevent edges from cluster 1 to cluster 0
    for snk, src, key in edges:
        # Enforce x_u <= x_v to prevent edges from cluster 1 to cluster 0
        prob += x[snk] <= x[src], f"EdgeDirection_{snk}_{src}_{key}"

        # Define y_{uv} = x_v - x_u
        prob += (
            y[(snk, src, key)] == x[src] - x[snk],
            f"CutEdge_{snk}_{src}_{key}",
        )

    # Enforce required cut edges
    for snk, src, data in required_cut_edges:
        # enforce that snk  and src are in different clusters
        # prob += x[src] - x[snk] >= 1, f"RequiredCut_{snk}_{src}_{key}"
        key = dg.find_edge_key(snk, src, data)
        prob += y[(snk, src, key)] == 1, f"RequiredCut_{snk}_{src}_{key}"

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    success = prob.status == 1
    if not success:
        raise ValueError(f"ILP solver failed to find a solution. Got status {prob.status}")

    # Output the results
    cluster_0 = []
    cluster_1 = []
    for node in real_nodes:
        if pulp.value(x[node]) == 0:
            cluster_0.append(node)
        else:
            cluster_1.append(node)

    # Display cut edges
    cut_edges = [(u, v, key) for (u, v, key) in edges if pulp.value(y[(u, v, key)]) == 1]
    # print("Cut Edges:", cut_edges)

    return set(cluster_0), set(cluster_1), len(cut_edges)


def spectral_min_cut_weighted(
    g: PDG, alpha: float = 0.5
) -> tuple[set[TensorOp], set[TensorOp], int]:
    """Perform spectral clustering on a MultiDiGraph with weighted edges.

    Parameters
    ----------
    - G: nx.MultiDiGraph
    - alpha: (0 <= alpha <= 1) controlling the importance of edge weights vs. edge counts.
             alpha = 1 focuses entirely on minimizing edge weights.
             alpha = 0 focuses entirely on minimizing the number of edges.

    Returns
    -------
    - set1, set2: The two partitions of the nodes
    - cut_set: The set of edges in the cut

    """
    # Step 1: Convert the directed graph to an undirected graph
    # undirected_G = multidigraph_to_undirected(G)
    G_bytes = nx.Graph()
    G_count = nx.Graph()

    for snk, src, data in g.get_all_edges():
        num_bytes = g.estimate_tensor_size_bytes(src.op_id, data.src_out_idx)

        if G_bytes.has_edge(snk.op_id, src.op_id):
            G_bytes[snk.op_id][src.op_id]["weight"] += num_bytes
            G_count[snk.op_id][src.op_id]["weight"] += 1
        else:
            G_bytes.add_edge(snk.op_id, src.op_id, weight=num_bytes)
            G_count.add_edge(snk.op_id, src.op_id, weight=1)

    # Step 2: Create the weighted adjacency matrix A
    A_bytes = nx.to_numpy_array(G_bytes, weight="weight")  # Uses edge weights if present
    A_count = nx.to_numpy_array(G_count, weight="weight")  # Uses edge weights if present

    # Step 4: Normalize both matrices
    # Normalize A_bytes (weighted adjacency matrix) by its maximum value (to limit byte dominance)
    if np.max(A_bytes) > 0:
        A_bytes = A_bytes / np.max(A_bytes)

    # Normalize A_count (binary adjacency matrix) by the total number of edges
    num_edges = np.max(A_count)
    if num_edges > 0:
        A_count = A_count / num_edges

    # Step 5: Combine weighted and binary matrices using alpha
    A_combined = alpha * A_bytes + (1 - alpha) * A_count

    # Step 6: Degree matrix for the combined adjacency matrix
    degrees = np.sum(A_combined, axis=1)
    D = np.diag(degrees)

    # Step 7: Compute the unnormalized Laplacian matrix L = D - A_combined
    L = D - A_combined

    # Step 8: Compute the eigenvalues and eigenvectors of the Laplacian
    eigenvalues, eigenvectors = np.linalg.eigh(L)

    # Step 9: Get the second smallest eigenvector (Fiedler vector)
    fiedler_vector = eigenvectors[:, 1]

    # Step 10: Partition the nodes based on the sign of the Fiedler vector
    partition1 = [i for i, value in enumerate(fiedler_vector) if value < 0]
    partition2 = [i for i, value in enumerate(fiedler_vector) if value >= 0]

    # Step 11: Recover set1 and set2 as sets of ops
    order = list(G_bytes.nodes)
    set1: set[OpId] = {order[i] for i in partition1}
    set2: set[OpId] = {order[i] for i in partition2}
    set1_nodes = {g.get_op_by_id(x) for x in set1}
    set2_nodes = {g.get_op_by_id(x) for x in set2}
    assert len(set1) + len(set2) == len(order)

    # Step 11: Compute the cut set (edges between set1 and set2)
    len_cut_set = len(list(nx.edge_boundary(G_count, set1, set2)))

    return set1_nodes, set2_nodes, len_cut_set
