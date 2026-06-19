# B2 - PARAMETRIC surrogate (Kaggle GPU): ONE network A_theta(x,y,z,t; tau)
# for a FAMILY of laser pulses parameterised by the pulse duration tau (genuinely
# nonlinear in the envelope; a0 is excluded because A is exactly linear in a0).
# Trains over tau in [20,40], evaluates eps_theta(tau) including interpolation and
# extrapolation (tau=15,45) -> demonstrates the symmetry-constrained surrogate
# generalises across a pulse family (the "framework" claim, not a single pulse).
#
# Local smoke test (CPU, ~60 s):  SMOKE=1 python param.py
import os, sys, subprocess, math, time, json
ON_KAGGLE = os.path.isdir("/kaggle/working")
if ON_KAGGLE:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--force-reinstall", "--no-deps",
                    "torch", "--index-url", "https://download.pytorch.org/whl/cu121"])
import numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
dev = "cuda" if torch.cuda.is_available() else "cpu"
SMOKE = bool(os.environ.get("SMOKE")); OUT = "/kaggle/working" if ON_KAGGLE else "."
print("GPU", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "SMOKE", SMOKE, flush=True)
torch.set_default_dtype(torch.float32); torch.manual_seed(0)
ch = -1.0; a0 = 5.0; lam = 2*math.pi; k0 = 2*math.pi/lam; w0 = 5*lam; zR = k0*w0**2/2
TAU_LO, TAU_HI = 20.0, 40.0; TAU0 = 30.0

def A_full(x, y, z, t, tau):
    wz = w0*torch.sqrt(1+(z/zR)**2); Rinv = z/(z**2+zR**2); gouy = torch.atan2(z, zR*torch.ones_like(z))
    env = torch.exp(-(x**2+y**2)/wz**2-(z-t)**2/tau**2); phase = k0*(z-t)+k0*(x**2+y**2)*0.5*Rinv-gouy
    return a0*(w0/wz)*env*torch.cos(phase)
def A_pw(z, t, tau): return a0*torch.exp(-(z-t)**2/tau**2)*torch.cos(k0*(z-t))
def A_full5(q): return A_full(q[..., 0], q[..., 1], q[..., 2], q[..., 3], q[..., 4])
def grads4(q):                                   # grad of A_full wrt (x,y,z,t) only (tau is a parameter)
    q = q.clone().requires_grad_(True); A = A_full5(q); g = torch.autograd.grad(A.sum(), q)[0][..., :4]
    return A.detach(), g.detach()

# reference on-axis trajectory at tau0 for the tube geometry (reused across tau)
def H7(s, tau):
    x, y, z, Px, Py, Pz, t = [s[..., i] for i in range(7)]; px = Px-ch*A_full(x, y, z, t, tau); return torch.sqrt(1+px**2+Py**2+Pz**2)
def deriv(s7):
    s7 = s7.view(1, 7).to(dev).clone().requires_grad_(True); H = H7(s7, torch.tensor(TAU0)); g = torch.autograd.grad(H.sum(), s7)[0][0]; return g[3:6], -g[0:3]
def traj(dt):
    n = int(6*TAU0/dt); r = torch.zeros(3, device=dev); P = torch.zeros(3, device=dev); rows = []
    for k in range(n):
        t = -3*TAU0+k*dt; rows.append(torch.cat([r, P, torch.tensor([t], device=dev)]).detach().cpu().numpy())
        sb = lambda rr, PP, tt: torch.cat([rr, PP, torch.tensor([tt], device=dev)])
        k1r, k1p = deriv(sb(r, P, t)); k2r, k2p = deriv(sb(r+.5*dt*k1r, P+.5*dt*k1p, t+.5*dt))
        k3r, k3p = deriv(sb(r+.5*dt*k2r, P+.5*dt*k2p, t+.5*dt)); k4r, k4p = deriv(sb(r+dt*k3r, P+dt*k3p, t+dt))
        r = r+dt/6*(k1r+2*k2r+2*k3r+k4r); P = P+dt/6*(k1p+2*k2p+2*k3p+k4p)
    return np.array(rows)
T = traj(0.5 if SMOKE else 0.05); traj4 = torch.tensor(T[:, [0, 1, 2, 6]], device=dev, dtype=torch.float32)
TUBE = torch.tensor([0.4, 0.4, 0.8, 0.8], device=dev)
def sample(n, taus=None):
    i = torch.randint(0, traj4.shape[0], (n,), device=dev); xyzt = traj4[i]+torch.randn(n, 4, device=dev)*TUBE
    if taus is None: taus = TAU_LO+(TAU_HI-TAU_LO)*torch.rand(n, 1, device=dev)
    return torch.cat([xyzt, taus], -1)
HARM = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16]; rms = lambda a: float((a**2).mean().sqrt())

N = 4000 if SMOKE else 200_000
Q = sample(N); A_t, GA_t = grads4(Q); scale = Q.std(0)+1e-3; center = Q.mean(0)

class PNet(nn.Module):
    def __init__(s, width=256, depth=6):
        super().__init__(); ind = 5+2*len(HARM); L = [nn.Linear(ind, width), nn.Tanh()]
        for _ in range(depth-1): L += [nn.Linear(width, width), nn.Tanh()]
        L += [nn.Linear(width, 1)]; s.net = nn.Sequential(*L)
        for m in s.net:
            if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
        s.register_buffer("scale", scale); s.register_buffer("center", center); s.register_buffer("harm", torch.tensor(HARM, dtype=torch.float32))
    def forward(s, q):
        sn = (q-s.center)/s.scale; ang = (q[..., 2:3]-q[..., 3:4])*s.harm
        return A_pw(q[..., 2], q[..., 3], q[..., 4])+s.net(torch.cat([sn, torch.sin(ang), torch.cos(ang)], -1)).squeeze(-1)
model = PNet().to(dev); print("params", sum(p.numel() for p in model.parameters()), flush=True)
w_g = 32.0
def loss_on(idx):
    q = Q[idx].clone().requires_grad_(True); A = model(q)
    g = torch.autograd.grad(A.sum(), q, create_graph=True)[0][..., :4]
    return ((A-A_t[idx])**2).mean()+w_g*((g-GA_t[idx])**2).mean()
# eval set per tau (tau0 tube geometry, fixed tau column)
def eval_eps(tau):
    torch.manual_seed(777); q = sample(2000 if SMOKE else 20000, taus=torch.full((2000 if SMOKE else 20000, 1), float(tau), device=dev))
    At, GAt = grads4(q); qq = q.clone().requires_grad_(True); A = model(qq); g = torch.autograd.grad(A.sum(), qq)[0][..., :4]
    return max(rms(A.detach()-At), rms(g.detach()-GAt))

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
EPOCHS = 60 if SMOKE else 25000; BATCH = 512 if SMOKE else 20000; WARMUP = 10 if SMOKE else 500
ADAPT_START = int(0.4*EPOCHS); ADAPT_EVERY = max(1, EPOCHS//20)
def adaptive_weights():
    res = torch.empty(N, device=dev); cs = 50000
    for i in range(0, N, cs):
        q = Q[i:i+cs].clone().requires_grad_(True); A = model(q); g = torch.autograd.grad(A.sum(), q)[0][..., :4]; res[i:i+cs] = ((g-GA_t[i:i+cs])**2).mean(1)
    return res+0.25*res.mean()
def lr_lambda(ep):
    if ep < WARMUP: return (ep+1)/WARMUP
    prog = (ep-WARMUP)/max(1, EPOCHS-WARMUP); return 0.5*(1+math.cos(math.pi*prog))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda); pw = None; t0 = time.time(); best = {"e": 1e9, "st": None}
for ep in range(EPOCHS):
    if ep >= ADAPT_START and (ep-ADAPT_START) % ADAPT_EVERY == 0: pw = adaptive_weights()
    idx = torch.multinomial(pw, BATCH, replacement=True) if pw is not None else torch.randint(0, N, (BATCH,), device=dev)
    opt.zero_grad(); loss = loss_on(idx); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    if ep % max(1, EPOCHS//10) == 0:
        e = eval_eps(TAU0)
        if e < best["e"]: best = {"e": e, "st": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        print("ep %5d loss %.3e  eps(tau0) %.4e" % (ep, float(loss), e), flush=True)
if best["st"] is not None: model.load_state_dict(best["st"])
try:
    idx = torch.randint(0, N, (40000 if not SMOKE else 2000,), device=dev)
    o2 = torch.optim.LBFGS(model.parameters(), max_iter=15 if SMOKE else 500, history_size=80, line_search_fn="strong_wolfe")
    pre = eval_eps(TAU0)
    def cl(): o2.zero_grad(); l = loss_on(idx); l.backward(); return l
    o2.step(cl)
    if eval_eps(TAU0) > pre and best["st"] is not None: model.load_state_dict(best["st"])
except RuntimeError as e: print("lbfgs skip", e, flush=True)
print("TRAIN_TIME %.1f s" % (time.time()-t0), flush=True)

taus = [15, 20, 25, 30, 35, 40, 45]
eps_tau = {int(tt): eval_eps(tt) for tt in taus}
for tt in taus: print("eps_theta(tau=%2d) = %.3e  %s" % (tt, eps_tau[tt], "(train)" if TAU_LO <= tt <= TAU_HI else "(extrap)"), flush=True)
summary = {"eps_tau": eps_tau, "tau_range_train": [TAU_LO, TAU_HI], "train_time_s": time.time()-t0,
           "eps_in_range_mean": float(np.mean([eps_tau[t] for t in [20, 25, 30, 35, 40]]))}
torch.save({"state_dict": model.state_dict(), "scale": scale.cpu(), "center": center.cpu(), "HARM": HARM}, OUT+"/param_model.pt")
json.dump(summary, open(OUT+"/results.json", "w"), indent=2)
plt.figure(figsize=(5.2, 3.6))
tt = sorted(taus); yy = [eps_tau[t] for t in tt]
plt.semilogy(tt, yy, "o-")
plt.axvspan(TAU_LO, TAU_HI, color="green", alpha=.12, label="training range")
plt.xlabel(r"pulse duration $\tau$ ($\omega_0^{-1}$)"); plt.ylabel(r"$\varepsilon_\theta(\tau)$")
plt.title("One surrogate for a pulse family"); plt.legend(fontsize=8); plt.grid(True, which="both", alpha=.3)
plt.tight_layout(); plt.savefig(OUT+"/param_eps_tau.png", dpi=140); print("DONE", flush=True)
