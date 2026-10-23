"""Radio-map estimators. All operate on a Scenario S with sampled indices idx and noisy measurements Y (n,B)."""
import numpy as np
from scipy.spatial import cKDTree
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from env import L, NB, PT

ZSCALE = 1.0


def coords(S, idx):
    return S.P[idx] / np.array([L, L, 150.0])


# ---------------------------------------------------------------- physics-only baselines
def fit_ldpl(S, idx, Y):
    out = np.zeros((len(S.P), NB))
    for b in range(NB):
        A = np.c_[np.ones(len(idx)), 10 * np.log10(S.D[idx, b])]
        th, *_ = np.linalg.lstsq(A, Y[:, b], rcond=None)
        out[:, b] = th[0] + th[1] * 10 * np.log10(S.D[:, b])
    return out


def prior_design(S, rows, b):
    los = S.LOS[rows, b]; ld = 10 * np.log10(S.D[rows, b])
    return np.c_[los, los * ld, 1 - los, (1 - los) * ld]


def fit_twoslope(S, idx, Y, ridge=1.0):
    """LoS-aware two-slope prior; ridge-shrunk toward the pooled LDPL fit (stable when a class has few samples)."""
    out = np.zeros((len(S.P), NB)); thetas = []
    for b in range(NB):
        A = prior_design(S, idx, b)
        A0 = np.c_[np.ones(len(idx)), 10 * np.log10(S.D[idx, b])]
        t0, *_ = np.linalg.lstsq(A0, Y[:, b], rcond=None)
        tref = np.array([t0[0], t0[1], t0[0], t0[1]])
        th = np.linalg.solve(A.T @ A + ridge * np.eye(4), A.T @ Y[:, b] + ridge * tref)
        thetas.append(th)
        out[:, b] = prior_design(S, np.arange(len(S.P)), b) @ th
    return out, thetas


# ---------------------------------------------------------------- interpolation baselines
def fit_idw(S, idx, Y, k=8, p=2.0):
    Xs = coords(S, idx); Xa = coords(S, np.arange(len(S.P)))
    tr = cKDTree(Xs); d, j = tr.query(Xa, k=min(k, len(idx)))
    w = 1.0 / np.maximum(d, 1e-6) ** p
    w /= w.sum(1, keepdims=True)
    return np.einsum("nk,nkb->nb", w, Y[j])


def _gp(Xs, y, Xa, return_std=False):
    k = C(1.0) * RBF(length_scale=[0.1, 0.1, 0.3], length_scale_bounds=(1e-3, 10)) + WhiteKernel(0.05, (1e-4, 1))
    gp = GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=0, random_state=0)
    gp.fit(Xs, y)
    return gp.predict(Xa, return_std=return_std)


def fit_kriging(S, idx, Y):
    Xs = coords(S, idx); Xa = coords(S, np.arange(len(S.P)))
    return np.stack([_gp(Xs, Y[:, b], Xa) for b in range(NB)], 1)


def fit_pr_kriging(S, idx, Y):
    """Physics-residual kriging: two-slope prior + GP on residuals (strong hybrid baseline)."""
    prior, _ = fit_twoslope(S, idx, Y)
    Xs = coords(S, idx); Xa = coords(S, np.arange(len(S.P)))
    return prior + np.stack([_gp(Xs, Y[:, b] - prior[idx, b], Xa) for b in range(NB)], 1)


# ---------------------------------------------------------------- tiny numpy MLP + Adam
class MLP:
    def __init__(self, sizes, rng, act="tanh"):
        self.W = [rng.standard_normal((a, b)) * np.sqrt(1.0 / a) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]
        self.W[-1] *= 0.1
        self.act = act
        self.m = [np.zeros_like(p) for p in self.W + self.b]
        self.v = [np.zeros_like(p) for p in self.W + self.b]
        self.t = 0

    def forward(self, X):
        self.cache = [X]; h = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            if i < len(self.W) - 1:
                h = np.tanh(z) if self.act == "tanh" else np.maximum(z, 0)
            else:
                h = z
            self.cache.append(h)
        return h

    def backward(self, g):
        gW, gb = [None] * len(self.W), [None] * len(self.b)
        for i in reversed(range(len(self.W))):
            h_in = self.cache[i]
            gW[i] = h_in.T @ g; gb[i] = g.sum(0)
            if i > 0:
                g = g @ self.W[i].T
                h = self.cache[i]
                g = g * (1 - h ** 2) if self.act == "tanh" else g * (h > 0)
        return gW + gb

    def step(self, grads, lr, b1=0.9, b2=0.999, wd=1e-5):
        self.t += 1
        params = self.W + self.b
        for i, (p, g) in enumerate(zip(params, grads)):
            if i < len(self.W):
                g = g + wd * p
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * g * g
            mh = self.m[i] / (1 - b1 ** self.t); vh = self.v[i] / (1 - b2 ** self.t)
            p -= lr * mh / (np.sqrt(vh) + 1e-8)


def fourier(Xn, Bf):
    z = 2 * np.pi * Xn @ Bf
    return np.c_[np.sin(z), np.cos(z)]


# ---------------------------------------------------------------- neural field variants
class NeuralField:
    """Generic coordinate network.  Variants:
       mlp      : raw coords -> P
       ff       : Fourier features -> P (NeRF/SIREN-style, no physics)
       skymap   : physics prior + residual net on [FF, LoS, log d] + physics-consistency losses
    Flags allow ablations."""

    def __init__(self, S, variant="skymap", seed=0, use_prior=True, use_phys=True, use_los=True,
                 iters=1500, hidden=96, nfreq=48, fscale=4.0, lam_s=0.5, lam_m=0.5, lam_r=0.02, lr=3e-3):
        self.S, self.variant = S, variant
        self.rng = np.random.default_rng(seed)
        self.use_prior = use_prior and variant == "skymap"
        self.use_phys = use_phys and variant == "skymap"
        self.use_los = use_los and variant == "skymap"
        self.iters, self.lam_s, self.lam_m, self.lam_r, self.lr = iters, lam_s, lam_m, lam_r, lr
        self.Bf = self.rng.standard_normal((3, nfreq)) * fscale * np.array([[1], [1], [0.5]])
        self.Fall = self._feat(np.arange(len(S.P)))
        fdim = self.Fall.shape[1]
        self.net = MLP([fdim, hidden, hidden, NB], self.rng, act="tanh" if variant != "mlp" else "relu")
        self._build_pairs()

    def _feat(self, rows):
        Xn = coords(self.S, rows)
        if self.variant == "mlp":
            return Xn
        F = fourier(Xn, self.Bf)
        if self.use_los:
            ld = (10 * np.log10(self.S.D[rows]) - 25) / 5
            F = np.c_[F, self.S.LOS[rows], ld]
        return F

    def _build_pairs(self):
        """Neighbour pairs for smoothness, and radially-outward pairs (same LoS state) for monotonic decay."""
        S = self.S; v = S.idx_valid
        i, j, k = S.ijk(v)
        sm = []
        for di, dj, dk in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
            ok = (i + di < S.nx) & (j + dj < S.nx) & (k + dk < S.nz)
            a = v[ok]; bb = S.grid_index(i[ok] + di, j[ok] + dj, k[ok] + dk)
            ok2 = S.valid[bb]
            sm.append(np.c_[a[ok2], bb[ok2]])
        self.smooth_pairs = np.vstack(sm)
        mono = []
        for b in range(NB):
            best = np.full(len(v), -1); gain = np.zeros(len(v))
            for di, dj, dk in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1)]:
                ok = (i + di >= 0) & (i + di < S.nx) & (j + dj >= 0) & (j + dj < S.nx) & (k + dk < S.nz)
                nb_ = np.full(len(v), -1)
                nb_[ok] = S.grid_index(i[ok] + di, j[ok] + dj, k[ok] + dk)
                g = np.where(nb_ >= 0, S.D[np.maximum(nb_, 0), b] - S.D[v, b], -1)
                upd = g > gain
                best[upd] = nb_[upd]; gain[upd] = g[upd]
            ok = (best >= 0)
            ok &= S.valid[np.maximum(best, 0)]
            ok &= S.LOS[v, b] == S.LOS[np.maximum(best, 0), b]
            ok &= S.D[v, b] > 60   # far-field
            mono.append((b, np.c_[v[ok], best[ok]]))
        self.mono_pairs = mono

    def prior(self, rows):
        if not self.use_prior:
            return np.zeros((len(rows), NB)) + self.mu0
        return np.stack([prior_design(self.S, rows, b) @ self.theta[b] for b in range(NB)], 1)

    def fit(self, idx, Y):
        S = self.S
        self.mu0 = Y.mean(0)
        if self.use_prior:
            _, self.theta = fit_twoslope(S, idx, Y)
        self.Pall = self.prior(np.arange(len(S.P)))
        sc = 10.0   # work in units of 10 dB
        F_obs = self.Fall[idx]; base_obs = self.prior(idx)
        n = len(idx)
        for it in range(self.iters):
            lr = self.lr * (0.5 * (1 + np.cos(np.pi * it / self.iters)) * 0.95 + 0.05)
            rows = [F_obs]; parts = [("obs", n)]
            if self.use_phys:
                sp = self.smooth_pairs[self.rng.integers(0, len(self.smooth_pairs), 512)]
                rows += [self.Fall[sp[:, 0]], self.Fall[sp[:, 1]]]; parts += [("sa", 512), ("sb", 512)]
                mp_all = []
                for b, mp in self.mono_pairs:
                    m = mp[self.rng.integers(0, len(mp), 256)]
                    mp_all.append((b, m))
                    rows += [self.Fall[m[:, 0]], self.Fall[m[:, 1]]]; parts += [("ma", 256), ("mb", 256)]
            X = np.vstack(rows)
            R = self.net.forward(X)       # residual (units of 10 dB)
            G = np.zeros_like(R)
            o = 0
            # data term
            pred = base_obs / sc + R[:n]
            err = pred - Y / sc
            G[:n] = 2 * err / n
            o = n
            if self.use_phys:
                sa = sp[:, 0]; sb = sp[:, 1]
                Ra, Rb = R[o:o + 512], R[o + 512:o + 1024]
                # smoothness on residual (shadowing is spatially correlated) + residual energy prior
                dR = Ra - Rb
                G[o:o + 512] += self.lam_s * 2 * dR / 512 + self.lam_r * 2 * Ra / 512
                G[o + 512:o + 1024] += -self.lam_s * 2 * dR / 512
                o += 1024
                for b, m in mp_all:
                    Pa = self.Pall[m[:, 0], b] / sc + R[o:o + 256, b]
                    Pb = self.Pall[m[:, 1], b] / sc + R[o + 256:o + 512, b]
                    viol = Pb - Pa   # outward point should not be stronger (tolerance 0.2 dB)
                    act = viol > 0.02
                    G[o:o + 256, b] += -self.lam_m * 2 * viol * act / 256
                    G[o + 256:o + 512, b] += self.lam_m * 2 * viol * act / 256
                    o += 512
            grads = self.net.backward(G)
            self.net.step(grads, lr)
        return self

    def predict(self, rows=None):
        if rows is None:
            rows = np.arange(len(self.S.P))
        out = np.zeros((len(rows), NB))
        for s in range(0, len(rows), 8192):
            r = rows[s:s + 8192]
            out[s:s + 8192] = self.Pall[r] + 10.0 * self.net.forward(self.Fall[r])
        return out


def fit_nf(S, idx, Y, variant, seed=0, **kw):
    return NeuralField(S, variant, seed=seed, **kw).fit(idx, Y).predict()


def fit_ensemble(S, idx, Y, M=5, seed=0, **kw):
    """Deep ensemble with bootstrap resampling -> mean and epistemic std."""
    rng = np.random.default_rng(seed)
    preds = []
    for m in range(M):
        bi = rng.integers(0, len(idx), len(idx))
        preds.append(NeuralField(S, "skymap", seed=seed * 100 + m, **kw).fit(idx[bi], Y[bi]).predict())
    P = np.stack(preds)
    return P.mean(0), P.std(0)
