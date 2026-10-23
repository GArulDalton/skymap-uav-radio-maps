"""Graph construction, risk-aware planning and the length-budgeted planner (Algorithm 3)."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra


def graph(S):
    """20 m planning graph with 26-neighbourhood over the valid airspace."""
    i, j, k = S.ijk(np.arange(len(S.P)))
    node = np.where((i % 2 == 0) & (j % 2 == 0) & S.valid)[0]
    pos = -np.ones(len(S.P), dtype=int); pos[node] = np.arange(len(node))
    src, dst = [], []
    for di in (-2, 0, 2):
        for dj in (-2, 0, 2):
            for dk in (-1, 0, 1):
                if di == dj == dk == 0:
                    continue
                ii, jj, kk = i[node] + di, j[node] + dj, k[node] + dk
                ok = (ii >= 0) & (ii < S.nx) & (jj >= 0) & (jj < S.nx) & (kk >= 0) & (kk < S.nz)
                nb = S.grid_index(ii[ok], jj[ok], kk[ok])
                good = S.valid[nb] & (pos[nb] >= 0)
                src.append(np.arange(len(node))[ok][good]); dst.append(pos[nb[good]])
    E = np.c_[np.concatenate(src), np.concatenate(dst)]
    length = np.linalg.norm(S.P[node[E[:, 0]]] - S.P[node[E[:, 1]]], axis=1)
    return node, E, length


def plan(node, E, length, c, s, t, lam):
    """Dijkstra on edge cost l_e * [1 + lam * (c_u + c_v)/2]  (Eq. 14)."""
    w = length * (1 + lam * 0.5 * (c[E[:, 0]] + c[E[:, 1]]))
    G = coo_matrix((w, (E[:, 0], E[:, 1])), shape=(len(node),) * 2).tocsr()
    _, pred = dijkstra(G, indices=s, return_predecessors=True)
    path = [t]
    while path[-1] != s:
        path.append(pred[path[-1]])
    return path[::-1]


def path_stats(S, node, path, field):
    """length-weighted mean of a per-node field along the path, and path length."""
    P = S.P[node[path]]; seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    f = field[path]; fw = 0.5 * (f[:-1] + f[1:])
    return float((seg * fw).sum() / seg.sum()), float(seg.sum())


def budget_plan(S, node, E, length, c, s, t, delta, lam_max=64.0, iters=12):
    """Algorithm 3 (LB-RAP): largest risk weight whose path respects length <= (1+delta) * shortest.
    Bisection on lam, using the monotone length/risk trade-off of the Lagrangian relaxation."""
    p0 = plan(node, E, length, c, s, t, 0.0); L0 = path_stats(S, node, p0, c)[1]
    budget = (1 + delta) * L0
    pmax = plan(node, E, length, c, s, t, lam_max)
    if path_stats(S, node, pmax, c)[1] <= budget:
        return pmax, lam_max
    lo, hi, best, best_lam = 0.0, lam_max, p0, 0.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        p = plan(node, E, length, c, s, t, mid)
        if path_stats(S, node, p, c)[1] <= budget:
            lo, best, best_lam = mid, p, mid
        else:
            hi = mid
    return best, best_lam
