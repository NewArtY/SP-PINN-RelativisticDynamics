"""
Run 4 -- end-to-end pipeline closure: integrate the LEARNED Hamiltonian H_theta
with the actual Stage-2 Tao SYMPLECTIC map (not RK4) and confirm it gives the
same trajectory as RK4.  This closes the title's pipeline and shows directly
that the residual laser-surrogate error is set by the surrogate, not the
integrator (symplectic and RK4 follow the same slightly-imperfect H_theta).

The learned vector-potential surrogate is wrapped in a field adapter exposing the
relsim interface (A, gradH_P, gradH_r) so the *tested* relsim.TaoSymplectic drives
it unchanged.  We first unit-test the adapter's analytic gradients against finite
differences of H_theta, then integrate.

Run from experiments/ :   python run4_pipeline_closure.py    (CPU ~8 min)
                          SMOKE=1 python run4_pipeline_closure.py   (coarse, fast)
"""
import _paths  # noqa: F401
import os, math
import numpy as np
import torch
import torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from relsim.integrators import TaoSymplectic
from relsim.fields import GaussianLaserPulse
from relsim.plotstyle import apply_style, COLORS

SMOKE = bool(os.environ.get("SMOKE"))
CKPT = os.path.join(_paths.ROOT, "..", "result_Aresidual", "sp_pinn_laser_A.pt")
ch = -1.0; a0 = 5.0; lam = 2 * math.pi; k0 = 2 * math.pi / lam
w0 = 5 * lam; tau = 30.0; zR = k0 * w0 ** 2 / 2


def A_pw(z, t):
    return a0 * torch.exp(-(z - t) ** 2 / tau ** 2) * torch.cos(k0 * (z - t))


# ---- load the learned A-residual surrogate ----
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
HARM = ckpt["HARM"]


class ANet(nn.Module):
    def __init__(s, width=256, depth=6):
        super().__init__(); ind = 4 + 2 * len(HARM)
        L = [nn.Linear(ind, width), nn.Tanh()]
        for _ in range(depth - 1):
            L += [nn.Linear(width, width), nn.Tanh()]
        L += [nn.Linear(width, 1)]; s.net = nn.Sequential(*L)
        s.register_buffer("scale", torch.ones(4)); s.register_buffer("center", torch.zeros(4))
        s.register_buffer("harm", torch.tensor(HARM, dtype=torch.float32))

    def forward(s, q):
        sn = (q - s.center) / s.scale; ang = (q[..., 2:3] - q[..., 3:4]) * s.harm
        return A_pw(q[..., 2], q[..., 3]) + s.net(torch.cat([sn, torch.sin(ang), torch.cos(ang)], -1)).squeeze(-1)


model = ANet(); model.load_state_dict(ckpt["state_dict"]); model.eval()
print("loaded surrogate eps_A=%.3e eps_gradA=%.3e" % (ckpt["eps_A"], ckpt["eps_gradA"]))


class LearnedLaserField:
    """Adapter: learned A_theta(x,y,z,t) -> relsim field interface used by the
    Tao map and the canonical RK4 (same H_theta = sqrt(1+|P-ch A_theta|^2))."""
    ch = -1.0

    def __init__(self, model):
        self.model = model

    def _A_and_grad(self, r, t):
        dt_ = next(self.model.parameters()).dtype
        q = np.column_stack([r[:, 0], r[:, 1], r[:, 2], np.full(len(r), float(t))])
        q = torch.tensor(q, dtype=dt_, requires_grad=True)
        A = self.model(q)
        g = torch.autograd.grad(A.sum(), q)[0]
        return A.detach().numpy(), g.detach().numpy()[:, :3]   # Ax, [dAx/dx,dAx/dy,dAx/dz]

    def A(self, r, t):
        Ax, _ = self._A_and_grad(r, t); out = np.zeros_like(r); out[:, 0] = Ax; return out

    def _p_gamma(self, r, P, t):
        Ax, dA = self._A_and_grad(r, t)
        p = P.copy(); p[:, 0] = p[:, 0] - self.ch * Ax
        g = np.sqrt(1.0 + np.sum(p * p, axis=-1, keepdims=True))
        return p, g, dA

    def H(self, r, P, t):
        p, g, _ = self._p_gamma(r, P, t); return g[..., 0]

    def gradH_P(self, r, P, t):
        p, g, _ = self._p_gamma(r, P, t); return p / g

    def gradH_r(self, r, P, t):
        p, g, dA = self._p_gamma(r, P, t)
        px = p[:, 0:1]
        return (1.0 / g) * px * (-self.ch * dA)


learned = LearnedLaserField(model)
analytic = GaussianLaserPulse(a0=a0, w0=w0, tau=tau, ch=ch, lam=lam)

# ---- unit test: adapter analytic gradients vs finite differences of H_theta ----
# Use a float64 copy of the model so the FD reference is not limited by float32
# round-off (catastrophic cancellation in the spatial-gradient FD numerator).
model_d = ANet(); model_d.load_state_dict(ckpt["state_dict"]); model_d.double().eval()
learned_d = LearnedLaserField(model_d)
rng = np.random.default_rng(0)
rt = rng.standard_normal((6, 3)) * np.array([0.4, 0.4, 5.0]); Pt = rng.standard_normal((6, 3)) * 2.0; tt = -5.0
h = 1e-5; ok = True
learned = learned_d  # alias for the test rows below; reset to float adapter afterwards
gP = learned.gradH_P(rt, Pt, tt); gr = learned.gradH_r(rt, Pt, tt)
for i in range(3):
    dP = np.zeros((6, 3)); dP[:, i] = h
    fdP = (learned.H(rt, Pt + dP, tt) - learned.H(rt, Pt - dP, tt)) / (2 * h)
    dR = np.zeros((6, 3)); dR[:, i] = h
    fdr = (learned.H(rt + dR, Pt, tt) - learned.H(rt - dR, Pt, tt)) / (2 * h)
    eP = np.abs(gP[:, i] - fdP).max(); er = np.abs(-gr[:, i] - (-fdr)).max()
    ok = ok and eP < 1e-3 and er < 1e-3
    print("  comp %d: |gradH_P-FD|=%.2e  |gradH_r-FD|=%.2e" % (i, eP, er))
print("ADAPTER GRADIENT UNIT TEST:", "PASS" if ok else "FAIL")
assert ok, "adapter gradients disagree with finite differences"
learned = LearnedLaserField(model)   # back to the float32 model for integration (matches training)

# ---- integrators (canonical r,P; both consume field.gradH_P/gradH_r) ----
def gamma_kin(field, r, P, t):
    p = P - field.ch * field.A(r, t); return float(np.sqrt(1 + np.sum(p * p)))

def rk4_canonical(field, ts, dt):
    r = np.zeros((1, 3)); P = np.zeros((1, 3)); g = np.ones(len(ts))
    for k in range(len(ts) - 1):
        t = ts[k]
        def f(r, P, t): return field.gradH_P(r, P, t), -field.gradH_r(r, P, t)
        k1r, k1P = f(r, P, t); k2r, k2P = f(r + .5 * dt * k1r, P + .5 * dt * k1P, t + .5 * dt)
        k3r, k3P = f(r + .5 * dt * k2r, P + .5 * dt * k2P, t + .5 * dt); k4r, k4P = f(r + dt * k3r, P + dt * k3P, t + dt)
        r = r + dt / 6 * (k1r + 2 * k2r + 2 * k3r + k4r); P = P + dt / 6 * (k1P + 2 * k2P + 2 * k3P + k4P)
        g[k + 1] = gamma_kin(field, r, P, t + dt)
    return g

def tao_canonical(field, ts, dt, omega=40.0):
    tao = TaoSymplectic(field, omega=omega)
    r = np.zeros((1, 3)); P = np.zeros((1, 3)); xx, yy = tao.init_copies(r, P); g = np.ones(len(ts))
    for k in range(len(ts) - 1):
        r, P, xx, yy = tao.step(r, P, xx, yy, ts[k], dt)
        g[k + 1] = gamma_kin(field, r, P, ts[k + 1])
    return g


dt = 0.4 if SMOKE else 0.02
t0 = -3 * tau; t1 = 3 * tau; ts = t0 + dt * np.arange(int((t1 - t0) / dt) + 1)
print("integrating  (dt=%.3f, %d steps) ..." % (dt, len(ts) - 1))
g_ana = rk4_canonical(analytic, ts, dt);   print("  analytic-H RK4 done")
g_rk4 = rk4_canonical(learned, ts, dt);     print("  learned-H  RK4 done")
g_tao = tao_canonical(learned, ts, dt);     print("  learned-H  Tao done")

terr_rk4 = float((np.abs(g_rk4 - g_ana) / g_ana).max())
terr_tao = float((np.abs(g_tao - g_ana) / g_ana).max())
idiff = float(np.abs(g_tao - g_rk4).max())
idiff_rel = float((np.abs(g_tao - g_rk4) / np.maximum(g_ana, 1.0)).max())
print("\n=== RESULT ===")
print("traj err (learned-H, RK4)            : %.4f" % terr_rk4)
print("traj err (learned-H, Tao symplectic) : %.4f" % terr_tao)
print("max |gamma_Tao - gamma_RK4|          : %.4e  (rel %.2e)" % (idiff, idiff_rel))
print("--> symplectic and RK4 of the learned H are indistinguishable vs the surrogate error"
      if idiff_rel < 0.05 * max(terr_rk4, 1e-9) or idiff < 5e-2 else "--> integrators DIFFER -- investigate")

apply_style()
fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.5))
ax[0].plot(ts / tau, g_ana, "k--", lw=1.4, label="analytic")
ax[0].plot(ts / tau, g_rk4, color=COLORS["RK4"], ls="-", lw=1.3, label="learned, RK4")
ax[0].plot(ts / tau, g_tao, color=COLORS["SP-PINN"], ls=":", lw=1.6, label="learned, Tao symplectic")
ax[0].set_xlabel(r"$t/\tau_L$"); ax[0].set_ylabel(r"$\gamma(t)$"); ax[0].legend(loc="upper left"); ax[0].set_title("(a)")
ax[1].semilogy(ts / tau, np.maximum(np.abs(g_tao - g_rk4), 1e-16), color="0.3")
ax[1].set_xlabel(r"$t/\tau_L$"); ax[1].set_ylabel(r"$|\gamma_{\rm Tao}-\gamma_{\rm RK4}|$"); ax[1].set_title("(b)")
fig.savefig(os.path.join(_paths.FIG, "run4_pipeline_closure.png"))
plt.close(fig)
print("wrote run4_pipeline_closure.png to", _paths.FIG)
