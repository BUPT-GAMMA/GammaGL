import sys
import json
import numpy as np

try:
    import graph_tool.all as gt
    from scipy.stats import chi2
except ImportError:
    print("ERROR: graph-tool and scipy required", file=sys.stderr)
    sys.exit(1)

def main():
    data = json.loads(sys.stdin.read())
    edges = data['edges']
    p_intra = data['p_intra']
    p_inter = data['p_inter']
    strict = data['strict']
    refinement_steps = data['refinement_steps']

    gt_g = gt.Graph()
    if edges:
        gt_g.add_edge_list(edges)
        
    try:
        state = gt.minimize_blockmodel_dl(gt_g)
    except ValueError:
        print("False")
        return

    # Refine using merge-split MCMC
    for _ in range(refinement_steps):
        state.multiflip_mcmc_sweep(beta=np.inf, niter=10)

    b = gt.contiguous_map(state.get_blocks())
    state = state.copy(b=b)
    e = state.get_matrix()
    n_blocks = state.get_nonempty_B()
    node_counts = state.get_nr().get_array()[:n_blocks]
    edge_counts = e.todense()[:n_blocks, :n_blocks]

    if strict:
        if (node_counts > 40).sum() > 0 or (node_counts < 20).sum() > 0 or n_blocks > 5 or n_blocks < 2:
            print("False")
            return

    max_intra_edges = node_counts * (node_counts - 1)
    est_p_intra = np.diagonal(edge_counts) / (max_intra_edges + 1e-6)

    max_inter_edges = node_counts.reshape((-1, 1)) @ node_counts.reshape((1, -1))
    np.fill_diagonal(edge_counts, 0)
    est_p_inter = edge_counts / (max_inter_edges + 1e-6)

    W_p_intra = (est_p_intra - p_intra) ** 2 / (est_p_intra * (1 - est_p_intra) + 1e-6)
    W_p_inter = (est_p_inter - p_inter) ** 2 / (est_p_inter * (1 - est_p_inter) + 1e-6)

    W = W_p_inter.copy()
    np.fill_diagonal(W, W_p_intra)
    p = 1 - chi2.cdf(abs(W), 1)
    p = p.mean()
    
    if p > 0.9:
        print("True")
    else:
        print("False")

if __name__ == "__main__":
    main()
