"""Generate the Colab notebook SP_PINN_3plus1D_surrogate_colab.ipynb (v2).

v2 fixes the spectral-bias failure of v1 (a plain MLP over the full domain
cannot represent the ~60 carrier oscillations of the laser, so training stalls
and the surrogate is useless). The fix, validated on CPU, is:
  (i)  trajectory-tube collocation sampling -- concentrate points where the
       Stage-2 integrator actually evaluates H, instead of a mostly-empty box;
  (ii) Fourier-feature embedding of the carrier phase eta = z - t.

Run:  python _build_notebook.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": _lines(text)}


def _lines(text):
    text = text.strip("\n")
    lines = text.split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


cells = []

cells.append(md(r"""
# SP-PINN — 3+1D Time-Dependent Laser Hamiltonian Surrogate (Colab GPU, v2)

Companion notebook for **N. S. Akintsov, A. P. Nevecheria, S. N. Andreev, Q.-H. Qin,
*Symmetry-Preserving Physics-Informed Neural Network Integrator...*, Symmetry (MDPI), 2026.**

This is **Stage 1** for the time-dependent 3+1D Gaussian-laser Hamiltonian.

> **Why v2.** A first attempt with a plain $\tanh$ MLP over the full phase-space–time
> box **fails**: the laser carrier $\cos[k_0(z-ct)]$ produces $\sim$60 oscillations
> across the domain, which an MLP cannot represent (spectral bias). Training stalls at
> $\varepsilon_\theta\approx0.8$ and the learned Hamiltonian gives trajectories wrong by
> hundreds of percent. v2 fixes this with **(i) trajectory-tube sampling** (concentrate
> collocation points where the integrator evaluates $\mathcal H$) and **(ii) Fourier-feature
> embedding of the carrier phase** $\eta=z-ct$. This reaches a usable $\varepsilon_\theta$.

**How to run:** `Runtime → Change runtime type → GPU (T4)`, then `Runtime → Run all`.
Units: $c=m=k_0=\omega_0=1$, electron charge $\mathrm{ch}=-1$.
"""))

cells.append(code(r"""
import math, time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device :', device, '| torch', torch.__version__)
if device == 'cpu':
    print('WARNING: no GPU detected. Enable Runtime > Change runtime type > GPU.')
torch.set_default_dtype(torch.float32)
torch.manual_seed(0)
"""))

cells.append(md(r"""
## 1. Analytic laser field and Hamiltonian

Linearly polarized ($\hat x$) focused Gaussian pulse propagating in $+z$, envelope
centered at $z=ct$. The radiation-gauge Hamiltonian is
$\mathcal H=\sqrt{1+|\mathbf P-\mathrm{ch}\,\mathbf A|^2}$ with $\mathbf A=(A_x,0,0)$.
Gradient targets (velocity, force, $\partial_t\mathcal H$) come from exact autodiff.
"""))

cells.append(code(r"""
ch  = -1.0
a0  = 5.0
lam = 2*math.pi          # wavelength  => k0 = 1
k0  = 2*math.pi/lam
w0  = 5*lam              # beam waist
tau = 30.0               # pulse duration
zR  = k0*w0**2/2.0       # Rayleigh length

def A_x(x, y, z, t):
    wz   = w0*torch.sqrt(1.0 + (z/zR)**2)
    Rinv = z/(z**2 + zR**2)
    gouy = torch.atan2(z, zR*torch.ones_like(z))
    env  = torch.exp(-(x**2 + y**2)/wz**2 - (z - t)**2/tau**2)
    phase = k0*(z - t) + k0*(x**2 + y**2)*0.5*Rinv - gouy
    return a0*(w0/wz)*env*torch.cos(phase)

def H_analytic(s):
    x, y, z, Px, Py, Pz, t = [s[..., i] for i in range(7)]
    px = Px - ch*A_x(x, y, z, t)
    return torch.sqrt(1.0 + px**2 + Py**2 + Pz**2)

def grads_analytic(s):
    s = s.clone().requires_grad_(True)
    H = H_analytic(s)
    g = torch.autograd.grad(H.sum(), s, create_graph=False)[0]
    return H.detach(), g.detach()      # g[..,0:3]=dH/dr, [3:6]=dH/dP, [6]=dH/dt
"""))

cells.append(md(r"""
## 2. Trajectory-tube collocation sampling (the key fix)

The Stage-2 integrator only evaluates $\mathcal H_\theta$ along the electron
trajectory, so we train the surrogate where it matters. We first integrate the
analytic on-axis trajectory (Hamilton's equations in canonical variables), then
sample collocation points as trajectory points plus Gaussian noise, forming a
tube around the path. This concentrates the network capacity on the
field-relevant region instead of a mostly-empty box.
"""))

cells.append(code(r"""
# --- analytic canonical trajectory: dr/dt = dH/dP, dP/dt = -dH/dr ---
def deriv(s7):                 # s7: (7,) [x,y,z,Px,Py,Pz,t]
    _, g = grads_analytic(s7.view(1, 7).to(device))
    return g[0, 3:6], -g[0, 0:3]    # (dr, dP)

def integrate_traj(dt=0.05, t0=-3*tau, t1=3*tau):
    n = int((t1 - t0)/dt)
    r = torch.zeros(3, device=device); P = torch.zeros(3, device=device)
    rows = []
    for k in range(n):
        t = t0 + k*dt
        s = torch.cat([r, P, torch.tensor([t], device=device)])
        rows.append(s.cpu().numpy())
        s_t = lambda rr, PP, tt: torch.cat([rr, PP, torch.tensor([tt], device=device)])
        k1r, k1p = deriv(s_t(r, P, t))
        k2r, k2p = deriv(s_t(r + .5*dt*k1r, P + .5*dt*k1p, t + .5*dt))
        k3r, k3p = deriv(s_t(r + .5*dt*k2r, P + .5*dt*k2p, t + .5*dt))
        k4r, k4p = deriv(s_t(r + dt*k3r, P + dt*k3p, t + dt))
        r = r + dt/6*(k1r + 2*k2r + 2*k3r + k4r)
        P = P + dt/6*(k1p + 2*k2p + 2*k3p + k4p)
    return np.array(rows)

traj = integrate_traj()
print('trajectory points:', traj.shape, '| z in [%.1f, %.1f]' % (traj[:,2].min(), traj[:,2].max()))

# tube widths (cover the integrator's local excursions around the path)
tube_w = torch.tensor([0.5, 0.5, 1.0, 1.5, 0.5, 1.5, 1.0], device=device)
traj_t = torch.tensor(traj, device=device, dtype=torch.float32)

def sample(n):
    idx = torch.randint(0, traj_t.shape[0], (n,), device=device)
    return traj_t[idx] + torch.randn(n, 7, device=device)*tube_w

N_train = 200_000
S = sample(N_train)
H_t, G_t = grads_analytic(S)
# normalization from the tube statistics
scale = S.std(0) + 1e-3
center = S.mean(0)
print('train set:', S.shape)
"""))

cells.append(md(r"""
## 3. Fourier-feature PINN and loss

The network input is the normalized $(\mathbf r,\mathbf P,t)$ **augmented with
Fourier features** $\{\sin(m\,\eta),\cos(m\,\eta)\}_{m}$ of the carrier phase
$\eta=z-t$, which give the MLP the high-frequency basis needed to represent the
carrier. The loss is the Lorentz-invariant combination of the paper: the
mass-shell constraint $(\mathcal H_\theta-mc^2\gamma)^2$ and the residual of
Hamilton's equations (gradient matching).
"""))

cells.append(code(r"""
HARM = [1, 2, 3, 4, 5, 6, 8, 10, 12]   # carrier harmonics for the Fourier embedding

class HNet(nn.Module):
    def __init__(self, width=256, depth=6):
        super().__init__()
        in_dim = 7 + 2*len(HARM)
        L = [nn.Linear(in_dim, width), nn.Tanh()]
        for _ in range(depth-1):
            L += [nn.Linear(width, width), nn.Tanh()]
        L += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*L)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
        self.register_buffer('scale', scale)
        self.register_buffer('center', center)
        self.register_buffer('harm', torch.tensor(HARM, dtype=torch.float32))
    def forward(self, s):
        sn = (s - self.center)/self.scale
        eta = (s[..., 2:3] - s[..., 6:7])
        ang = eta*self.harm                       # (...,len(HARM))
        feats = torch.cat([sn, torch.sin(ang), torch.cos(ang)], dim=-1)
        return self.net(feats)

model = HNet().to(device)
print('parameters:', sum(p.numel() for p in model.parameters()))
# The Stage-2 integrator uses the GRADIENTS of H (force, velocity), so the
# Hamilton-equations term is weighted more heavily than the mass-shell term:
# this is what controls trajectory (carrier-phase) fidelity.
w_H, w_g = 1.0, 5.0
"""))

cells.append(code(r"""
def loss_on(idx):
    s = S[idx].clone().requires_grad_(True)
    H = model(s).squeeze(-1)
    g = torch.autograd.grad(H.sum(), s, create_graph=True)[0]
    L_mass = ((H - H_t[idx])**2).mean()
    L_eqs  = ((g - G_t[idx])**2).mean()
    return w_H*L_mass + w_g*L_eqs, L_mass.detach(), L_eqs.detach()

opt = torch.optim.Adam(model.parameters(), lr=2e-3)
EPOCHS, BATCH = 10_000, 20_000
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
hist = []
t0 = time.time()
for ep in range(EPOCHS):
    idx = torch.randint(0, N_train, (BATCH,), device=device)
    opt.zero_grad(); loss, Lm, Le = loss_on(idx); loss.backward(); opt.step(); sched.step()
    if ep % 250 == 0:
        hist.append((ep, float(loss), float(Lm), float(Le)))
        print(f'ep {ep:5d}  loss {float(loss):.3e}  mass {float(Lm):.3e}  eqs {float(Le):.3e}')
print(f'Adam training time: {time.time()-t0:.1f} s')
"""))

cells.append(code(r"""
# Optional L-BFGS polish
try:
    idx = torch.randint(0, N_train, (40_000,), device=device)
    opt2 = torch.optim.LBFGS(model.parameters(), max_iter=300, history_size=50,
                             line_search_fn='strong_wolfe')
    def closure():
        opt2.zero_grad(); l,_,_ = loss_on(idx); l.backward(); return l
    t0 = time.time(); opt2.step(closure); print(f'L-BFGS time: {time.time()-t0:.1f} s')
except RuntimeError as e:
    print('L-BFGS skipped:', e)
"""))

cells.append(code(r"""
# eps_theta on a fresh tube test set
S_te = sample(40_000); H_te, G_te = grads_analytic(S_te)
s = S_te.clone().requires_grad_(True); H = model(s).squeeze(-1)
g = torch.autograd.grad(H.sum(), s)[0]
rms = lambda a: float((a**2).mean().sqrt())
eH = rms(H.detach()-H_te); eP = rms(g[:,3:6].detach()-G_te[:,3:6]); er = rms(g[:,0:3].detach()-G_te[:,0:3])
eps_rms = max(eH, eP, er)
eps_max = float(max((H.detach()-H_te).abs().max(),
                    (g[:,3:6].detach()-G_te[:,3:6]).abs().max(),
                    (g[:,0:3].detach()-G_te[:,0:3]).abs().max()))
print(f'eps_theta (RMS, in-tube) = {eps_rms:.3e}   |   eps_theta (max) = {eps_max:.3e}')
print(f'   RMS |H-gamma|={eH:.2e}  |gradP-v|={eP:.2e}  |gradr-f|={er:.2e}')
h = np.array(hist)
plt.figure(figsize=(5,3.2))
plt.semilogy(h[:,0],h[:,1],label='total'); plt.semilogy(h[:,0],h[:,2],label='mass-shell')
plt.semilogy(h[:,0],h[:,3],label='Hamilton eqs'); plt.xlabel('epoch'); plt.ylabel('loss')
plt.legend(); plt.title('Stage-1 training (tube + Fourier features)'); plt.show()
"""))

cells.append(md(r"""
## 4. Sanity check: integrate the learned Hamiltonian

We integrate the on-axis electron with RK4 using the **learned** gradients and
compare $\gamma(t)$ to the analytic-Hamiltonian solution. With v2 the learned
trajectory should track the analytic one closely (error $\sim\varepsilon_\theta$),
in contrast to the hundreds-of-percent error of the plain-MLP v1.
"""))

cells.append(code(r"""
def vel_force(use_model, r, P, t):
    s = torch.tensor([[r[0],r[1],r[2],P[0],P[1],P[2],t]], device=device,
                     dtype=torch.get_default_dtype(), requires_grad=True)
    H = model(s).squeeze(-1) if use_model else H_analytic(s)
    g = torch.autograd.grad(H.sum(), s)[0][0]
    return g[3:6].detach().cpu().numpy(), -g[0:3].detach().cpu().numpy()

def rk4(use_model, dt=0.02, t0=-3*tau, t1=3*tau):
    n=int((t1-t0)/dt); ts=t0+dt*np.arange(n+1); r=np.zeros(3); P=np.zeros(3); g=np.ones(n+1)
    for k in range(n):
        t=ts[k]
        k1r,k1p=vel_force(use_model,r,P,t); k2r,k2p=vel_force(use_model,r+.5*dt*k1r,P+.5*dt*k1p,t+.5*dt)
        k3r,k3p=vel_force(use_model,r+.5*dt*k2r,P+.5*dt*k2p,t+.5*dt); k4r,k4p=vel_force(use_model,r+dt*k3r,P+dt*k3p,t+dt)
        r=r+dt/6*(k1r+2*k2r+2*k3r+k4r); P=P+dt/6*(k1p+2*k2p+2*k3p+k4p)
        Ax=float(A_x(*[torch.tensor(v) for v in (r[0],r[1],r[2],t+dt)]))
        g[k+1]=math.sqrt(1+(P[0]-ch*Ax)**2+P[1]**2+P[2]**2)
    return ts,g

ts,g_ana=rk4(False); _,g_pin=rk4(True)
fig,ax=plt.subplots(1,2,figsize=(9,3.3))
ax[0].plot(ts/tau,g_ana,'k--',label='analytic H'); ax[0].plot(ts/tau,g_pin,label='learned H_theta')
ax[0].set_xlabel('t/tau'); ax[0].set_ylabel('gamma'); ax[0].legend(); ax[0].set_title('(a) trajectory')
ax[1].semilogy(ts/tau,np.maximum(np.abs(g_pin-g_ana)/g_ana,1e-12)); ax[1].set_xlabel('t/tau')
ax[1].set_ylabel('rel. gamma error'); ax[1].set_title('(b) error ~ eps_theta'); plt.show()
print('max rel. gamma error along trajectory:', float(np.max(np.abs(g_pin-g_ana)/g_ana)))
"""))

cells.append(code(r"""
torch.save({'state_dict': model.state_dict(), 'eps_theta_rms': eps_rms, 'eps_theta_max': eps_max,
            'scale': scale.cpu(), 'center': center.cpu(), 'HARM': HARM,
            'params': dict(a0=a0,w0=w0,tau=tau,lam=lam)}, 'sp_pinn_laser_surrogate.pt')
with open('eps_theta_3plus1D.txt','w') as f:
    f.write(f'eps_theta_rms {eps_rms:.6e}\neps_theta_max {eps_max:.6e}\n')
print('saved: sp_pinn_laser_surrogate.pt, eps_theta_3plus1D.txt')
try:
    from google.colab import files
    files.download('sp_pinn_laser_surrogate.pt'); files.download('eps_theta_3plus1D.txt')
except Exception:
    pass
"""))

cells.append(md(r"""
## 5. Notes

- The surrogate is trained on a **tube** around the integrated trajectory, so it is
  accurate where the Stage-2 integrator evaluates it; this is the regime the paper's
  learned pipeline needs. The reported $\varepsilon_\theta$ is the **in-tube** error.
- **Trajectory (carrier-phase) fidelity is harder than the mass-shell fit.** The
  laser-driven quiver is a resonantly forced oscillation, so even a $\sim$2% gradient
  error accumulates into a carrier-phase slip: the integrated $\gamma(t)$ can have the
  right amplitude yet drift out of phase, inflating the pointwise error. Because the
  integrator uses the *gradients* of $\mathcal H_\theta$, the Hamilton-equations loss is
  weighted more heavily (`w_g = 5`) and extra harmonics are included; for faithful
  integration you generally need $\varepsilon_\theta$ much smaller than the mass-shell
  fit alone suggests.
- To push $\varepsilon_\theta$ lower: increase `EPOCHS`, `width`/`depth`, the number of
  carrier harmonics in `HARM`, the gradient weight `w_g`, and the tube density
  `N_train`; widen `tube_w` only as much as the integrator actually explores.
- A genuinely *global* surrogate (accurate over the full box, for arbitrary
  trajectories) remains hard because of the carrier oscillations and is a direction for
  future work (e.g. multiplicative Fourier networks, domain decomposition along $\eta$,
  or a light-cone $\eta=z-ct$ reformulation that makes the field periodic).
"""))

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = os.path.join(HERE, "SP_PINN_3plus1D_surrogate_colab.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("wrote", out)
