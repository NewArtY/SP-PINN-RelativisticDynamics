"""
Stage-1 surrogate training for the time-dependent 3+1D Gaussian laser pulse:
the A-RESIDUAL light-cone formulation (the formulation that makes the learned
pipeline work for the laser; see paper Section 6.3 and EXPERIMENTS.md).

We learn the four-dimensional vector potential as a light-cone residual,

    A_theta(x,y,z,t) = A_planewave(eta = z - ct) + NN(x,y,z,t),

and reconstruct the Hamiltonian analytically,  H_theta = sqrt(1 + |P - ch A_theta|^2).
The mass-shell constraint then holds by construction and the Lorentz force is an
exact function of the learned dA_theta, so the network only has to fit a smooth
4-D scalar.  Training uses an lr warmup + cosine decay, gradient clipping, an
upweighted gradient-matching term, and an L-BFGS polish (Run 1 config: this is
what keeps the fit stable and reaches eps_theta ~ 3.4e-4).

This is the device-agnostic, dependency-light reference version of the GPU
training job; on CPU it is slow but correct. It writes the checkpoint and the
diagnostic figures next to the other outputs.  GPU strongly recommended.

Run from experiments/ :   python train_laser_A_residual.py
"""
import _paths  # noqa: F401
import math
import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev)
torch.set_default_dtype(torch.float32)
torch.manual_seed(0)
OUT = _paths.DATA

# ---- analytic laser field (code units c = 1, k0 = omega0 = 1) ----
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


def Afull4(q):
    x, y, z, t = [q[..., i] for i in range(4)]
    return A_full(x, y, z, t)


def grads4(fn, q):
    q = q.clone().requires_grad_(True)
    A = fn(q)
    g = torch.autograd.grad(A.sum(), q)[0]
    return A.detach(), g.detach()


# ---- reference trajectory (full analytic H) to define the sampling tube ----
def H_full7(s):
    x, y, z, Px, Py, Pz, t = [s[..., i] for i in range(7)]
    px = Px - ch * A_full(x, y, z, t)
    return torch.sqrt(1 + px ** 2 + Py ** 2 + Pz ** 2)


def deriv(s7):
    s7 = s7.view(1, 7).to(dev).clone().requires_grad_(True)
    H = H_full7(s7)
    g = torch.autograd.grad(H.sum(), s7)[0][0]
    return g[3:6], -g[0:3]


def integrate_traj(dt=0.05, t0=-3 * tau, t1=3 * tau):
    n = int((t1 - t0) / dt)
    r = torch.zeros(3, device=dev); P = torch.zeros(3, device=dev); rows = []
    for k in range(n):
        t = t0 + k * dt
        rows.append(torch.cat([r, P, torch.tensor([t], device=dev)]).detach().cpu().numpy())
        def sb(rr, PP, tt):
            return torch.cat([rr, PP, torch.tensor([tt], device=dev)])
        k1r, k1p = deriv(sb(r, P, t))
        k2r, k2p = deriv(sb(r + .5 * dt * k1r, P + .5 * dt * k1p, t + .5 * dt))
        k3r, k3p = deriv(sb(r + .5 * dt * k2r, P + .5 * dt * k2p, t + .5 * dt))
        k4r, k4p = deriv(sb(r + dt * k3r, P + dt * k3p, t + dt))
        r = r + dt / 6 * (k1r + 2 * k2r + 2 * k3r + k4r)
        P = P + dt / 6 * (k1p + 2 * k2p + 2 * k3p + k4p)
    return np.array(rows)


HARM = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16]


def main():
    traj = integrate_traj()
    traj4 = torch.tensor(traj[:, [0, 1, 2, 6]], device=dev, dtype=torch.float32)
    tube = torch.tensor([0.4, 0.4, 0.8, 0.8], device=dev)   # Run-1 tube

    def sample(n):
        i = torch.randint(0, traj4.shape[0], (n,), device=dev)
        return traj4[i] + torch.randn(n, 4, device=dev) * tube

    N_train = 400_000
    Q = sample(N_train)
    A_t, GA_t = grads4(Afull4, Q)
    scale = Q.std(0) + 1e-3
    center = Q.mean(0)

    class ANet(nn.Module):
        def __init__(s, width=256, depth=6):                # Run-1 size (keep small!)
            super().__init__()
            ind = 4 + 2 * len(HARM)
            L = [nn.Linear(ind, width), nn.Tanh()]
            for _ in range(depth - 1):
                L += [nn.Linear(width, width), nn.Tanh()]
            L += [nn.Linear(width, 1)]
            s.net = nn.Sequential(*L)
            for m in s.net:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
            s.register_buffer("scale", scale)
            s.register_buffer("center", center)
            s.register_buffer("harm", torch.tensor(HARM, dtype=torch.float32))

        def forward(s, q):
            sn = (q - s.center) / s.scale
            ang = (q[..., 2:3] - q[..., 3:4]) * s.harm
            return A_pw(q[..., 2], q[..., 3]) + s.net(
                torch.cat([sn, torch.sin(ang), torch.cos(ang)], -1)).squeeze(-1)

    model = ANet().to(dev)
    print("parameters", sum(p.numel() for p in model.parameters()))
    w_g = 24.0                                              # upweight the force term

    def loss_on(idx):
        q = Q[idx].clone().requires_grad_(True)
        A = model(q)
        g = torch.autograd.grad(A.sum(), q, create_graph=True)[0]
        La = ((A - A_t[idx]) ** 2).mean()
        Lg = ((g - GA_t[idx]) ** 2).mean()
        return La + w_g * Lg, La.detach(), Lg.detach()

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    EPOCHS, BATCH, WARMUP = 20000, 24000, 500

    def lr_lambda(ep):                                      # warmup then cosine (stable)
        if ep < WARMUP:
            return (ep + 1) / WARMUP
        prog = (ep - WARMUP) / max(1, EPOCHS - WARMUP)
        return 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    hist = []; t0 = time.time()
    for ep in range(EPOCHS):
        idx = torch.randint(0, N_train, (BATCH,), device=dev)
        opt.zero_grad()
        loss, La, Lg = loss_on(idx)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if ep % 500 == 0:
            hist.append((ep, float(loss), float(La), float(Lg)))
            print("ep %5d loss %.3e A %.3e gradA %.3e" % (ep, float(loss), float(La), float(Lg)), flush=True)
    print("ADAM_TIME %.1f s" % (time.time() - t0))

    try:
        idx = torch.randint(0, N_train, (60000,), device=dev)
        o2 = torch.optim.LBFGS(model.parameters(), max_iter=800, history_size=100,
                               line_search_fn="strong_wolfe")
        def cl():
            o2.zero_grad(); l, _, _ = loss_on(idx); l.backward(); return l
        t0 = time.time(); o2.step(cl); print("LBFGS_TIME %.1f s" % (time.time() - t0))
    except RuntimeError as e:
        print("LBFGS skipped", e)

    Qte = sample(60000)
    Ate, GAte = grads4(Afull4, Qte)
    q = Qte.clone().requires_grad_(True)
    A = model(q)
    g = torch.autograd.grad(A.sum(), q)[0]
    rms = lambda a: float((a ** 2).mean().sqrt())
    eA = rms(A.detach() - Ate); eg = rms(g.detach() - GAte); eps_rms = max(eA, eg)
    print("EPS_A_RMS %.4e  EPS_gradA_RMS %.4e  EPS_THETA_RMS %.4e" % (eA, eg, eps_rms))
    torch.save({"state_dict": model.state_dict(), "scale": scale.cpu(),
                "center": center.cpu(), "HARM": HARM, "eps_A": eA, "eps_gradA": eg},
               os.path.join(OUT, "sp_pinn_laser_A.pt"))
    json.dump({"eps_A": eA, "eps_gradA": eg, "eps_theta_rms": eps_rms},
              open(os.path.join(OUT, "sp_pinn_laser_A_results.json"), "w"), indent=2)
    print("saved checkpoint to", OUT,
          "-- run make_fig_laser_surrogate.py to plot the learned trajectory")


if __name__ == "__main__":
    main()
