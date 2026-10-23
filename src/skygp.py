"""SKYMAP core estimator: antenna-aware physics prior + multi-cell, LoS-heteroscedastic GP residual field."""
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from env import NB, PT, bs_gain
from methods import prior_design


def matern32(r):
    a = np.sqrt(3.0) * r
    return (1 + a) * np.exp(-a)


class SkyGP:
    """Joint GP over (position, cell).  K = s_i s_j C[b_i,b_j] k_M32(x_i,x_j) + sigma_n^2 I
       s = sigma_L * exp(-kappa z) (LoS) | sigma_N (NLoS),  C = (1-rho) I + rho 11^T."""

    def __init__(self, S, antenna=True, multicell=True, hetero=True, n_hyp=300, seed=0):
        self.S, self.antenna, self.multicell, self.hetero = S, antenna, multicell, hetero
        self.n_hyp = n_hyp; self.rng = np.random.default_rng(seed)

    # ---- mean function: PT + known BS vertical pattern - LoS-aware two-slope path loss (LS-calibrated)
    def _mean(self, rows, b):
        g = bs_gain(self.S.BS[b], self.S.P[rows]) if self.antenna else 0.0
        return g + prior_design(self.S, rows, b) @ self.theta[b]

    def _calib(self, idx, Y, ridge=1.0):
        self.theta = []
        for b in range(NB):
            g = bs_gain(self.S.BS[b], self.S.P[idx]) if self.antenna else 0.0
            y = Y[:, b] - g
            A = prior_design(self.S, idx, b)
            A0 = np.c_[np.ones(len(idx)), 10 * np.log10(self.S.D[idx, b])]
            t0, *_ = np.linalg.lstsq(A0, y, rcond=None)
            tref = np.array([t0[0], t0[1], t0[0], t0[1]])
            self.theta.append(np.linalg.solve(A.T @ A + ridge * np.eye(4), A.T @ y + ridge * tref))

    def _unpack(self, p):
        lh, lv, r, sL, kap, sN, sn = p
        rho = 0.95 * np.tanh(r) if self.multicell else 0.0
        if not self.hetero:
            sN = sL; kap = 0.0
        return np.exp(lh), np.exp(lv), rho, np.exp(sL), kap, np.exp(sN), np.exp(sn)

    def _scale(self, rows, b, p):
        _, _, _, sL, kap, sN, _ = self._unpack(p)
        z = self.S.P[rows, 2] / 100.0
        los = self.S.LOS[rows, b]
        return np.where(los > 0.5, sL * np.exp(-kap * z), sN)

    def _stack(self, rows):
        """observation list: (row, cell) for all cells"""
        R = np.repeat(rows, NB); Bc = np.tile(np.arange(NB), len(rows))
        return R, Bc

    def _K(self, R1, B1, R2, B2, p):
        lh, lv, rho, *_ = self._unpack(p)
        X1 = self.S.P[R1]; X2 = self.S.P[R2]
        d2 = ((X1[:, None, 0] - X2[None, :, 0]) ** 2 + (X1[:, None, 1] - X2[None, :, 1]) ** 2) / lh ** 2 \
            + (X1[:, None, 2] - X2[None, :, 2]) ** 2 / lv ** 2
        k = matern32(np.sqrt(d2))
        C = np.where(B1[:, None] == B2[None, :], 1.0, rho)
        s1 = np.empty(len(R1)); s2 = np.empty(len(R2))
        for b in range(NB):
            m1 = B1 == b; m2 = B2 == b
            s1[m1] = self._scale(R1[m1], b, p); s2[m2] = self._scale(R2[m2], b, p)
        return s1[:, None] * s2[None, :] * C * k

    def _nll(self, p, R, Bc, r):
        sn = self._unpack(p)[-1]
        K = self._K(R, Bc, R, Bc, p) + (sn ** 2 + 1e-6) * np.eye(len(R))
        try:
            c = cho_factor(K, lower=True)
        except np.linalg.LinAlgError:
            return 1e10
        a = cho_solve(c, r)
        return 0.5 * r @ a + np.log(np.diag(c[0])).sum()

    def fit(self, idx, Y, n_anchor=None):
        self.idx = idx
        na = len(idx) if n_anchor is None else n_anchor   # calibrate prior + hyper-params on a random anchor subset only
        self._calib(idx[:na], Y[:na])
        res = np.stack([Y[:, b] - self._mean(idx, b) for b in range(NB)], 1)
        # hyper-parameters by type-II ML on a random subset
        si = np.arange(na) if na <= self.n_hyp else self.rng.choice(na, self.n_hyp, replace=False)
        R, Bc = self._stack(idx[si]); r = res[si].ravel()
        p0 = np.array([np.log(40.0), np.log(25.0), 0.3, np.log(4.0), 0.3, np.log(6.0), np.log(2.0)])
        bounds = [(np.log(5), np.log(400)), (np.log(5), np.log(300)), (-2, 2), (np.log(0.3), np.log(20)),
                  (0.0, 2.0), (np.log(0.3), np.log(20)), (np.log(0.3), np.log(8))]
        opt = minimize(self._nll, p0, args=(R, Bc, r), method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 60})
        self.p = opt.x
        # full posterior
        self.R, self.Bc = self._stack(idx); self.r = res.ravel()
        sn = self._unpack(self.p)[-1]
        K = self._K(self.R, self.Bc, self.R, self.Bc, self.p) + (sn ** 2 + 1e-6) * np.eye(len(self.R))
        self.Lc = np.linalg.cholesky(K)
        self.alpha = cho_solve((self.Lc, True), self.r)
        return self

    def hyper(self):
        lh, lv, rho, sL, kap, sN, sn = self._unpack(self.p)
        return dict(l_h=lh, l_v=lv, rho=rho, sigma_L=sL, kappa=kap, sigma_N=sN, sigma_n=sn)

    def predict(self, rows=None, return_std=False, chunk=1500):
        if rows is None:
            rows = np.arange(len(self.S.P))
        mu = np.zeros((len(rows), NB)); sd = np.zeros((len(rows), NB))
        for s in range(0, len(rows), chunk):
            rr = rows[s:s + chunk]
            for b in range(NB):
                Bq = np.full(len(rr), b)
                Ks = self._K(rr, Bq, self.R, self.Bc, self.p)
                mu[s:s + chunk, b] = self._mean(rr, b) + Ks @ self.alpha
                if return_std:
                    V = solve_triangular(self.Lc, Ks.T, lower=True)
                    prior_var = self._scale(rr, b, self.p) ** 2
                    sd[s:s + chunk, b] = np.sqrt(np.maximum(prior_var - (V ** 2).sum(0), 1e-6))
        return (mu, sd) if return_std else mu


def fit_skymap(S, idx, Y, **kw):
    return SkyGP(S, **kw).fit(idx, Y).predict()


def ivr_scores(g, ref, cand):
    """Integrated variance reduction: expected drop of total posterior variance over `ref` if `cand` is measured (per cell, summed)."""
    from env import NB
    sn2 = g._unpack(g.p)[-1] ** 2
    tot = np.zeros(len(cand))
    for b in range(NB):
        Br = np.full(len(ref), b); Bcn = np.full(len(cand), b)
        Vr = solve_triangular(g.Lc, g._K(ref, Br, g.R, g.Bc, g.p).T, lower=True)
        Vc = solve_triangular(g.Lc, g._K(cand, Bcn, g.R, g.Bc, g.p).T, lower=True)
        Kx = g._K(ref, Br, cand, Bcn, g.p) - Vr.T @ Vc
        vc = g._scale(cand, b, g.p) ** 2 - (Vc ** 2).sum(0)
        tot += (Kx ** 2).sum(0) / (np.maximum(vc, 1e-6) + sn2)
    return tot
