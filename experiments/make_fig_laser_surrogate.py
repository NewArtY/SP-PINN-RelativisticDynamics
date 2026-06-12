"""
Figure: the LEARNED Stage-1 surrogate in action (Figure 7).

Loads the trained A-residual vector-potential surrogate A_theta(x,y,z,t) =
A_planewave(eta) + NN(x,y,z,t) and integrates the on-axis electron in the focused
Gaussian pulse with the *learned* Hamiltonian H_theta = sqrt(1+|P-ch A_theta|^2),
comparing gamma(t) against the analytic reference.  This is the one figure that
shows the learned pipeline (the rest of the paper isolates the integrator with the
analytic Hamiltonian).  The checkpoint is the best A-residual run (Run 1, single GPU,
eps_theta ~ 3.4e-4); see EXPERIMENTS.md.

Run from the experiments/ directory:  python make_fig_laser_surrogate.py
"""
import _paths  # noqa: F401  (puts repo root on sys.path, defines FIG/DATA)
import math
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from relsim.plotstyle import apply_style, COLORS

# checkpoint lives in the top-level results directory (best A-residual run)
CKPT = os.path.join(_paths.ROOT, "..", "result_Aresidual", "sp_pinn_laser_A.pt")

# ---- analytic laser field (identical to the training script) ----
ch = -1.0; a0 = 5.0; lam = 2 * math.pi; k0 = 2 * math.pi / lam
w0 = 5 * lam; tau = 30.0; zR = k0 * w0 ** 2 / 2


def A_full(x, y, z, t):
    wz = w0 * torch.sqrt(1 + (z / zR) ** 2)
    Rinv = z / (z ** 2 + zR ** 2)
    gouy = torch.atan2(z, zR * torch.ones_like(z))
    env = torch.exp(-(x ** 2 + y ** 2) / wz ** 2 - (z - t) ** 2 / tau ** 2)
    phase = k0 * (z - t) + k0 * (x ** 2 + y ** 2) * 0.5 * Rinv - gouy
    return a0 * (w0 / wz) * env * torch.cos(phase)


def A_pw(z, t):
    return a0 * torch.exp(-(z - t) ** 2 / tau ** 2) * torch.cos(k0 * (z - t))


def main():
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    HARM = ckpt["HARM"]

    class ANet(nn.Module):
        def __init__(s, width=256, depth=6):
            super().__init__()
            ind = 4 + 2 * len(HARM)
            L = [nn.Linear(ind, width), nn.Tanh()]
            for _ in range(depth - 1):
                L += [nn.Linear(width, width), nn.Tanh()]
            L += [nn.Linear(width, 1)]
            s.net = nn.Sequential(*L)
            s.register_buffer("scale", torch.ones(4))
            s.register_buffer("center", torch.zeros(4))
            s.register_buffer("harm", torch.tensor(HARM, dtype=torch.float32))

        def forward(s, q):
            sn = (q - s.center) / s.scale
            ang = (q[..., 2:3] - q[..., 3:4]) * s.harm
            return A_pw(q[..., 2], q[..., 3]) + s.net(
                torch.cat([sn, torch.sin(ang), torch.cos(ang)], -1)).squeeze(-1)

    model = ANet()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print("loaded surrogate; eps_A=%.3e eps_gradA=%.3e"
          % (ckpt["eps_A"], ckpt["eps_gradA"]))

    def H_full7(s):
        x, y, z, Px, Py, Pz, t = [s[..., i] for i in range(7)]
        px = Px - ch * A_full(x, y, z, t)
        return torch.sqrt(1 + px ** 2 + Py ** 2 + Pz ** 2)

    def H_learned(s):
        x, y, z, Px, Py, Pz, t = [s[..., i] for i in range(7)]
        Ath = model(torch.stack([x, y, z, t], dim=-1))
        px = Px - ch * Ath
        return torch.sqrt(1 + px ** 2 + Py ** 2 + Pz ** 2)

    def vf(use_model, r, P, t):
        s = torch.tensor([[r[0], r[1], r[2], P[0], P[1], P[2], t]],
                         dtype=torch.float32, requires_grad=True)
        H = H_learned(s) if use_model else H_full7(s)
        g = torch.autograd.grad(H.sum(), s)[0][0]
        return g[3:6].detach().numpy(), -g[0:3].detach().numpy()

    def rk4(use_model, dt=0.02, t0=-3 * tau, t1=3 * tau):
        n = int((t1 - t0) / dt); ts = t0 + dt * np.arange(n + 1)
        r = np.zeros(3); P = np.zeros(3); gg = np.ones(n + 1)
        for k in range(n):
            t = ts[k]
            k1r, k1p = vf(use_model, r, P, t)
            k2r, k2p = vf(use_model, r + .5 * dt * k1r, P + .5 * dt * k1p, t + .5 * dt)
            k3r, k3p = vf(use_model, r + .5 * dt * k2r, P + .5 * dt * k2p, t + .5 * dt)
            k4r, k4p = vf(use_model, r + dt * k3r, P + dt * k3p, t + dt)
            r = r + dt / 6 * (k1r + 2 * k2r + 2 * k3r + k4r)
            P = P + dt / 6 * (k1p + 2 * k2p + 2 * k3p + k4p)
            Ax = float(A_full(*[torch.tensor(v) for v in (r[0], r[1], r[2], t + dt)]))
            gg[k + 1] = math.sqrt(1 + (P[0] - ch * Ax) ** 2 + P[1] ** 2 + P[2] ** 2)
        return ts, gg

    print("integrating analytic and learned trajectories ...")
    ts, g_ana = rk4(False)
    _, g_pin = rk4(True)
    err = np.abs(g_pin - g_ana) / g_ana
    print("TRAJ_MAX_REL_ERR %.4f" % float(err.max()))

    apply_style()
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.5))
    ax[0].plot(ts / tau, g_ana, color="k", ls="--", lw=1.6, label="analytic")
    ax[0].plot(ts / tau, g_pin, color=COLORS["SP-PINN"], ls="-", lw=1.5,
               label=r"learned $\mathcal{H}_\theta$ (A-residual)")
    ax[0].set_xlabel(r"$t/\tau_L$"); ax[0].set_ylabel(r"$\gamma(t)$")
    ax[0].set_title("(a)"); ax[0].legend(loc="upper left")
    ax[1].semilogy(ts / tau, np.maximum(err, 1e-12), color=COLORS["SP-PINN"])
    ax[1].set_xlabel(r"$t/\tau_L$")
    ax[1].set_ylabel(r"Relative $\gamma$ error")
    ax[1].set_title("(b)")
    for f in ("figure7.pdf", "figure7.png"):
        fig.savefig(os.path.join(_paths.FIG, f))
    plt.close(fig)
    print("wrote figure7 to", _paths.FIG)


if __name__ == "__main__":
    main()
