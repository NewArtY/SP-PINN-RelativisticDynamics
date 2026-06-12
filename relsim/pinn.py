"""
Stage 1 of the SP-PINN method: an unsupervised physics-informed neural
network that learns the relativistic Hamiltonian surrogate H_theta(r, P)
directly from the governing relations, with no trajectory data.

The loss combines
    L_constraint = (H_theta - gamma)^2            (relativistic mass shell)
    L_eqs        = ||dH_theta/dP - v||^2 + ||dH_theta/dr - f_r||^2
                                                  (Hamilton's equations)
where gamma, v and f_r are the analytic targets for the chosen static field.

After training, ``PINNField`` wraps the network so that it exposes exactly
the interface (H, gradH_r, gradH_P) consumed by the TaoSymplectic
integrator, enabling a genuine end-to-end "learned-Hamiltonian + symplectic
integration" run whose error floor is the training error eps_theta.
"""

from __future__ import annotations
import numpy as np

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:                       # pragma: no cover
    _HAS_TORCH = False


if _HAS_TORCH:

    class HamiltonianNet(nn.Module):
        def __init__(self, in_dim=6, width=128, depth=5, scale=None):
            super().__init__()
            layers = [nn.Linear(in_dim, width), nn.Tanh()]
            for _ in range(depth - 1):
                layers += [nn.Linear(width, width), nn.Tanh()]
            layers += [nn.Linear(width, 1)]
            self.net = nn.Sequential(*layers)
            for m in self.net:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)
            if scale is None:
                scale = torch.ones(in_dim)
            self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float64))

        def forward(self, rP):
            return self.net(rP / self.scale)


def train_pinn(field, domain, n_coll=8000, epochs=8000, lam_eqs=1.0,
               lam_constraint=10.0, seed=0, verbose=False, lbfgs_iters=400):
    """
    Train a Hamiltonian surrogate for a *static* field (FreeField,
    ConstMagneticField).  ``domain`` = dict with 'r' and 'P' half-widths.
    Inputs are normalized by the domain half-widths; training is Adam followed
    by an L-BFGS polish.  Returns (model, history, eps_theta).
    """
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch is required for the PINN stage.")
    torch.manual_seed(seed)
    Lr = domain["r"]; Lp = domain["P"]

    scale = torch.tensor([Lr, Lr, Lr, Lp, Lp, Lp], dtype=torch.float64)
    model = HamiltonianNet(in_dim=6, scale=scale).double()

    # quasi-random collocation points (Sobol via torch)
    sob = torch.quasirandom.SobolEngine(dimension=6, scramble=True, seed=seed)
    u = sob.draw(n_coll).numpy()
    r = (2 * u[:, :3] - 1) * Lr
    P = (2 * u[:, 3:] - 1) * Lp
    rP = np.concatenate([r, P], axis=1)
    rP_t = torch.tensor(rP, dtype=torch.float64, requires_grad=True)

    # analytic targets
    t0 = 0.0
    gamma_t = torch.tensor(field.H(r, P, t0), dtype=torch.float64).reshape(-1, 1)
    v_t = torch.tensor(field.gradH_P(r, P, t0), dtype=torch.float64)
    fr_t = torch.tensor(field.gradH_r(r, P, t0), dtype=torch.float64)

    def compute_loss():
        H = model(rP_t)
        grad = torch.autograd.grad(H.sum(), rP_t, create_graph=True)[0]
        L_con = ((H - gamma_t) ** 2).mean()
        L_eqs = ((grad[:, 3:] - v_t) ** 2).mean() + ((grad[:, :3] - fr_t) ** 2).mean()
        return lam_eqs * L_eqs + lam_constraint * L_con, L_con, L_eqs

    history = []
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(epochs):
        opt.zero_grad()
        loss, L_con, L_eqs = compute_loss()
        loss.backward()
        opt.step(); sched.step()
        if ep % 400 == 0:
            history.append((ep, float(loss), float(L_con), float(L_eqs)))
            if verbose:
                print(f"[adam] ep {ep:5d}  loss {float(loss):.3e}  "
                      f"L_con {float(L_con):.3e}  L_eqs {float(L_eqs):.3e}")

    # L-BFGS polish
    opt2 = torch.optim.LBFGS(model.parameters(), max_iter=lbfgs_iters,
                             history_size=50, line_search_fn="strong_wolfe",
                             tolerance_grad=1e-12, tolerance_change=1e-14)

    def closure():
        opt2.zero_grad()
        loss, _, _ = compute_loss()
        loss.backward()
        return loss
    opt2.step(closure)
    loss, L_con, L_eqs = compute_loss()
    history.append((epochs, float(loss), float(L_con), float(L_eqs)))
    if verbose:
        print(f"[lbfgs] loss {float(loss):.3e}  L_con {float(L_con):.3e}  "
              f"L_eqs {float(L_eqs):.3e}")

    # eps_theta on a fresh test set
    u2 = sob.draw(2000).numpy()
    rt = (2 * u2[:, :3] - 1) * Lr
    Pt = (2 * u2[:, 3:] - 1) * Lp
    eps = _eval_eps(model, field, rt, Pt)
    return model, history, eps


def _eval_eps(model, field, r, P):
    import torch
    rP = torch.tensor(np.concatenate([r, P], axis=1), dtype=torch.float64,
                      requires_grad=True)
    H = model(rP)
    grad = torch.autograd.grad(H.sum(), rP, create_graph=False)[0].detach().numpy()
    Hn = H.detach().numpy().reshape(-1)
    H0 = field.H(r, P, 0.0)
    v0 = field.gradH_P(r, P, 0.0)
    fr0 = field.gradH_r(r, P, 0.0)
    eH = np.max(np.abs(Hn - H0))
    ev = np.max(np.abs(grad[:, 3:] - v0))
    efr = np.max(np.abs(grad[:, :3] - fr0))
    rmsH = float(np.sqrt(np.mean((Hn - H0) ** 2)))
    rmsv = float(np.sqrt(np.mean((grad[:, 3:] - v0) ** 2)))
    rmsfr = float(np.sqrt(np.mean((grad[:, :3] - fr0) ** 2)))
    return {"H": float(eH), "gradP": float(ev), "gradr": float(efr),
            "eps_theta_max": float(max(eH, ev, efr)),
            "rmsH": rmsH, "rmsgradP": rmsv, "rmsgradr": rmsfr,
            "eps_theta": float(max(rmsH, rmsv, rmsfr))}


class PINNField:
    """Adapter exposing a trained HamiltonianNet through the field interface
    consumed by TaoSymplectic (static field only)."""

    name = "pinn"

    def __init__(self, model, base_field):
        self.model = model
        self.ch = base_field.ch
        self.base = base_field

    def A(self, r, t):
        return self.base.A(r, t)

    def _eval(self, r, P):
        import torch
        rP = torch.tensor(np.concatenate([np.atleast_2d(r), np.atleast_2d(P)],
                          axis=1), dtype=torch.float64, requires_grad=True)
        H = self.model(rP)
        grad = torch.autograd.grad(H.sum(), rP, create_graph=False)[0]
        return (H.detach().numpy().reshape(-1),
                grad[:, :3].detach().numpy(),
                grad[:, 3:].detach().numpy())

    def H(self, r, P, t):
        H, _, _ = self._eval(r, P)
        return H.reshape(np.asarray(r).shape[:-1])

    def gradH_r(self, r, P, t):
        _, gr, _ = self._eval(r, P)
        return gr.reshape(np.asarray(r).shape)

    def gradH_P(self, r, P, t):
        _, _, gP = self._eval(r, P)
        return gP.reshape(np.asarray(r).shape)
