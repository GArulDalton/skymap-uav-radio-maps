"""SKYMAP scenario generator: 3D urban city + 3GPP TR 36.777 UMa-AV ground-truth radio maps."""
import numpy as np
from scipy.ndimage import gaussian_filter

L = 600.0          # area side (m)
DXY = 10.0         # map grid (m)
ZS = np.arange(30, 151, 15.0)  # UAV altitudes (m)
FC = 2.0           # GHz
PT = 46.0          # dBm
NB = 3             # base stations
NOISE_DBM = -174 + 10 * np.log10(10e6) + 7  # 10 MHz, NF 7 dB
TILT = 12.0; HPBW_V = 65.0; SLA = 30.0      # BS vertical pattern (3GPP 38.901)
MEAS_STD = 2.0
RHO_SITE = 0.5    # inter-site shadowing correlation     # dB measurement noise


def buildings(rng, alpha=0.3, beta=500, gamma=15.0, res=2.0):
    """ITU-R P.1410 statistical city -> height raster (res m)."""
    n = int(L / res)
    H = np.zeros((n, n))
    nb = int(beta * (L / 1000) ** 2)
    w = np.sqrt(alpha / beta) * 1000.0            # building width (m)
    pitch = L / np.sqrt(nb)
    k = int(np.sqrt(nb))
    for i in range(k):
        for j in range(k):
            cx = (i + 0.5) * pitch + rng.uniform(-3, 3)
            cy = (j + 0.5) * pitch + rng.uniform(-3, 3)
            ww = w * rng.uniform(0.8, 1.2); hh = w * rng.uniform(0.8, 1.2)
            h = np.clip(rng.rayleigh(gamma), 6, 70)
            x0, x1 = int((cx - ww / 2) / res), int((cx + ww / 2) / res)
            y0, y1 = int((cy - hh / 2) / res), int((cy + hh / 2) / res)
            H[max(x0, 0):min(x1, n), max(y0, 0):min(y1, n)] = h
    return H, res


def height_at(H, res, x, y):
    n = H.shape[0]
    ix = np.clip((x / res).astype(int), 0, n - 1)
    iy = np.clip((y / res).astype(int), 0, n - 1)
    return H[ix, iy]


def los_mask(H, res, bs, P, nstep=120):
    """Geometric LoS between BS (3,) and points P (N,3) by ray marching over the raster."""
    t = np.linspace(0.02, 0.98, nstep)[None, :]
    X = bs[0] + (P[:, [0]] - bs[0]) * t
    Y = bs[1] + (P[:, [1]] - bs[1]) * t
    Z = bs[2] + (P[:, [2]] - bs[2]) * t
    blocked = (Z < height_at(H, res, X, Y)).any(axis=1)
    return (~blocked).astype(float)


def pathloss(d3, h, los):
    pl_los = 28.0 + 22 * np.log10(d3) + 20 * np.log10(FC)
    pl_nlos = -17.5 + (46 - 7 * np.log10(h)) * np.log10(d3) + 20 * np.log10(40 * np.pi * FC / 3)
    pl_nlos = np.maximum(pl_nlos, pl_los)
    return np.where(los > 0.5, pl_los, pl_nlos)


def bs_gain(bs, P):
    dz = P[:, 2] - bs[2]
    dh = np.linalg.norm(P[:, :2] - bs[:2], axis=1)
    elev = np.degrees(np.arctan2(dz, dh))           # + upward
    theta = 90 - elev                               # zenith angle
    return 8.0 - np.minimum(12 * ((theta - 90 - TILT) / HPBW_V) ** 2, SLA)


class Scenario:
    def __init__(self, seed, rho_site=RHO_SITE, sigma_const=None):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.H, self.res = buildings(rng)
        xs = np.arange(0, L + 1e-6, DXY)
        self.nx = len(xs); self.nz = len(ZS)
        GX, GY, GZ = np.meshgrid(xs, xs, ZS, indexing="ij")
        self.P = np.stack([GX.ravel(), GY.ravel(), GZ.ravel()], 1)
        self.valid = self.P[:, 2] > height_at(self.H, self.res, self.P[:, 0], self.P[:, 1]) + 3
        # base stations: spread, rooftop-mounted
        self.BS = []
        anchors = [(0.2, 0.25), (0.75, 0.3), (0.45, 0.8)]
        for (ax, ay) in anchors:
            x, y = ax * L + rng.uniform(-40, 40), ay * L + rng.uniform(-40, 40)
            h = max(25.0, height_at(self.H, self.res, np.array([x]), np.array([y]))[0] + 3)
            self.BS.append(np.array([x, y, h]))
        self.BS = np.array(self.BS)
        N = len(self.P)
        wc = rng.standard_normal((self.nx, self.nx, self.nz))
        common = gaussian_filter(wc, sigma=(3.0, 3.0, 1.2), mode="wrap").ravel(); common /= common.std()
        self.LOS = np.zeros((N, NB)); self.D = np.zeros((N, NB)); self.PR = np.zeros((N, NB))
        for b in range(NB):
            d3 = np.maximum(np.linalg.norm(self.P - self.BS[b], axis=1), 10.0)
            los = los_mask(self.H, self.res, self.BS[b], self.P)
            pl = pathloss(d3, self.P[:, 2], los)
            # 3D-correlated log-normal shadowing (decorrelation ~50 m horiz, ~30 m vert)
            w = rng.standard_normal((self.nx, self.nx, self.nz))
            s = gaussian_filter(w, sigma=(3.0, 3.0, 1.2), mode="wrap").ravel()
            s /= s.std()
            s = np.sqrt(rho_site) * common + np.sqrt(1 - rho_site) * s   # inter-site correlation (3GPP TR 36.814)
            sig = np.where(los > 0.5, 4.64 * np.exp(-0.0066 * self.P[:, 2]), 6.0)
            if sigma_const is not None:
                sig = np.full_like(sig, sigma_const)
            self.LOS[:, b] = los; self.D[:, b] = d3
            self.PR[:, b] = PT + bs_gain(self.BS[b], self.P) - pl + sig * s
        self.idx_valid = np.where(self.valid)[0]

    def measure(self, idx):
        return self.PR[idx] + MEAS_STD * self.rng.standard_normal((len(idx), NB))

    # neighbour structure on the grid (for physics collocation)
    def grid_index(self, i, j, k):
        return (i * self.nx + j) * self.nz + k

    def ijk(self, idx):
        k = idx % self.nz; j = (idx // self.nz) % self.nx; i = idx // (self.nz * self.nx)
        return i, j, k


def sinr_db(PRdbm):
    """Best-server SINR (dB) from per-BS received power maps (N,B)."""
    lin = 10 ** (PRdbm / 10)
    s = lin.max(1); tot = lin.sum(1)
    return 10 * np.log10(s / (tot - s + 10 ** (NOISE_DBM / 10)))
