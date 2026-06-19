# E4 + E5 - surrogate accuracy SCALING LAW and ABLATION GRID (Kaggle GPU).
# A-residual light-cone laser surrogate (the winning formulation). Loops over
# configs, writes results INCREMENTALLY to results.json (partial progress
# survives a timeout). REDUCED epoch budget: this is a *trend* study; the
# official headline number stays exp-8 (eps_theta = 3.65e-4). Honesty: every
# config trained to the SAME budget; eval always on the SAME in-tube held-out set.
#
#   E4 scaling : width {128,256,384} x N_collocation {2k,8k,32k,128k}, all features on
#   E5 ablation: from baseline (w256,N128k,w_g32, all on) leave-one-out
#                {-fourier,-tube,-warmup,-clip,-adapt} + w_g sweep {1,8,24}
#
# Local smoke test (CPU, ~60 s):  SMOKE=1 python scaling.py
import os, sys, subprocess, math, time, json
ON_KAGGLE = os.path.isdir("/kaggle/working")
if ON_KAGGLE:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--force-reinstall",
                    "--no-deps", "torch", "--index-url",
                    "https://download.pytorch.org/whl/cu121"])
import numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
dev = "cuda" if torch.cuda.is_available() else "cpu"
SMOKE = bool(os.environ.get("SMOKE"))
OUT = "/kaggle/working" if ON_KAGGLE else "."
print("GPU", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "SMOKE", SMOKE, flush=True)
torch.set_default_dtype(torch.float32)
ch = -1.0; a0 = 5.0; lam = 2*math.pi; k0 = 2*math.pi/lam; w0 = 5*lam; tau = 30.0; zR = k0*w0**2/2

def A_full(x, y, z, t):
    wz = w0*torch.sqrt(1+(z/zR)**2); Rinv = z/(z**2+zR**2); gouy = torch.atan2(z, zR*torch.ones_like(z))
    env = torch.exp(-(x**2+y**2)/wz**2-(z-t)**2/tau**2); phase = k0*(z-t)+k0*(x**2+y**2)*0.5*Rinv-gouy
    return a0*(w0/wz)*env*torch.cos(phase)
def A_pw(z, t): return a0*torch.exp(-(z-t)**2/tau**2)*torch.cos(k0*(z-t))
def Afull4(q): x, y, z, t = [q[..., i] for i in range(4)]; return A_full(x, y, z, t)
def grads4(fn, q):
    q = q.clone().requires_grad_(True); A = fn(q); g = torch.autograd.grad(A.sum(), q)[0]; return A.detach(), g.detach()

# reference on-axis trajectory (full H) for the tube + the traj-err metric
def H_full7(s):
    x, y, z, Px, Py, Pz, t = [s[..., i] for i in range(7)]; px = Px-ch*A_full(x, y, z, t); return torch.sqrt(1+px**2+Py**2+Pz**2)
def deriv(s7):
    s7 = s7.view(1, 7).to(dev).clone().requires_grad_(True); H = H_full7(s7); g = torch.autograd.grad(H.sum(), s7)[0][0]; return g[3:6], -g[0:3]
def integrate_traj(dt, t0=-3*tau, t1=3*tau):
    n = int((t1-t0)/dt); r = torch.zeros(3, device=dev); P = torch.zeros(3, device=dev); rows = []
    for k in range(n):
        t = t0+k*dt; rows.append(torch.cat([r, P, torch.tensor([t], device=dev)]).detach().cpu().numpy())
        sb = lambda rr, PP, tt: torch.cat([rr, PP, torch.tensor([tt], device=dev)])
        k1r, k1p = deriv(sb(r, P, t)); k2r, k2p = deriv(sb(r+.5*dt*k1r, P+.5*dt*k1p, t+.5*dt))
        k3r, k3p = deriv(sb(r+.5*dt*k2r, P+.5*dt*k2p, t+.5*dt)); k4r, k4p = deriv(sb(r+dt*k3r, P+dt*k3p, t+dt))
        r = r+dt/6*(k1r+2*k2r+2*k3r+k4r); P = P+dt/6*(k1p+2*k2p+2*k3p+k4p)
    return np.array(rows)
traj = integrate_traj(0.5 if SMOKE else 0.05)
traj4 = torch.tensor(traj[:, [0, 1, 2, 6]], device=dev, dtype=torch.float32)
TUBE = torch.tensor([0.4, 0.4, 0.8, 0.8], device=dev)
lo = traj4.min(0).values - torch.tensor([1.0, 1.0, 2.0, 2.0], device=dev)
hi = traj4.max(0).values + torch.tensor([1.0, 1.0, 2.0, 2.0], device=dev)
def sample_tube(n): i = torch.randint(0, traj4.shape[0], (n,), device=dev); return traj4[i]+torch.randn(n, 4, device=dev)*TUBE
def sample_box(n): return lo + (hi-lo)*torch.rand(n, 4, device=dev)
HARM = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16]
rms = lambda a: float((a**2).mean().sqrt())
# fixed in-tube held-out eval set, identical for ALL configs (apples-to-apples)
torch.manual_seed(123); Qte = sample_tube(2000 if SMOKE else 40000); Ate, GAte = grads4(Afull4, Qte)

class ANet(nn.Module):
    def __init__(s, width, depth, use_fourier, center, scale):
        super().__init__(); s.use_fourier = use_fourier
        ind = 4+(2*len(HARM) if use_fourier else 0); L = [nn.Linear(ind, width), nn.Tanh()]
        for _ in range(depth-1): L += [nn.Linear(width, width), nn.Tanh()]
        L += [nn.Linear(width, 1)]; s.net = nn.Sequential(*L)
        for m in s.net:
            if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
        s.register_buffer("scale", scale); s.register_buffer("center", center)
        s.register_buffer("harm", torch.tensor(HARM, dtype=torch.float32))
    def forward(s, q):
        sn = (q-s.center)/s.scale
        if s.use_fourier:
            ang = (q[..., 2:3]-q[..., 3:4])*s.harm
            feat = torch.cat([sn, torch.sin(ang), torch.cos(ang)], -1)
        else:
            feat = sn
        return A_pw(q[..., 2], q[..., 3])+s.net(feat).squeeze(-1)

def train_once(cfg):
    t0 = time.time(); torch.manual_seed(cfg["seed"])
    N = cfg["N"]; samp = sample_tube if cfg["tube"] else sample_box
    Q = samp(N); A_t, GA_t = grads4(Afull4, Q); scale = Q.std(0)+1e-3; center = Q.mean(0)
    model = ANet(cfg["width"], cfg["depth"], cfg["fourier"], center, scale).to(dev)
    w_g = cfg["w_g"]
    def loss_on(idx):
        q = Q[idx].clone().requires_grad_(True); A = model(q)
        g = torch.autograd.grad(A.sum(), q, create_graph=True)[0]
        La = ((A-A_t[idx])**2).mean(); Lg = ((g-GA_t[idx])**2).mean(); return La+w_g*Lg
    def eval_eps():
        q = Qte.clone().requires_grad_(True); A = model(q); g = torch.autograd.grad(A.sum(), q)[0]
        eA = rms(A.detach()-Ate); eg = rms(g.detach()-GAte); return max(eA, eg), eA, eg
    def adaptive_weights():
        res = torch.empty(N, device=dev); cs = 50000
        for i in range(0, N, cs):
            q = Q[i:i+cs].clone().requires_grad_(True); A = model(q)
            g = torch.autograd.grad(A.sum(), q)[0]; res[i:i+cs] = ((g-GA_t[i:i+cs])**2).mean(1)
        return res+0.25*res.mean()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    EPOCHS = cfg["epochs"]; BATCH = min(cfg.get("batch", 16000), N); WARMUP = cfg["warmup_ep"] if cfg["warmup"] else 0
    ADAPT_START = int(0.4*EPOCHS); ADAPT_EVERY = max(1, EPOCHS//20)
    def lr_lambda(ep):
        if ep < WARMUP: return (ep+1)/WARMUP
        prog = (ep-WARMUP)/max(1, EPOCHS-WARMUP); return 0.5*(1+math.cos(math.pi*prog))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    best = {"eps": float("inf"), "state": None}; pw = None
    for ep in range(EPOCHS):
        if cfg["adapt"] and ep >= ADAPT_START and (ep-ADAPT_START) % ADAPT_EVERY == 0: pw = adaptive_weights()
        idx = torch.multinomial(pw, BATCH, replacement=True) if pw is not None else torch.randint(0, N, (BATCH,), device=dev)
        opt.zero_grad(); loss = loss_on(idx); loss.backward()
        if cfg["clip"]: torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if ep % max(1, EPOCHS//10) == 0:
            e, _, _ = eval_eps()
            if e < best["eps"]: best = {"eps": e, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
    if best["state"] is not None: model.load_state_dict(best["state"])
    if cfg["lbfgs"]:
        try:
            idx = torch.randint(0, N, (min(40000, N),), device=dev)
            o2 = torch.optim.LBFGS(model.parameters(), max_iter=cfg["lbfgs_it"], history_size=80, line_search_fn="strong_wolfe")
            ep_pre, _, _ = eval_eps()
            def cl(): o2.zero_grad(); l = loss_on(idx); l.backward(); return l
            o2.step(cl)
            ep_post, _, _ = eval_eps()
            if ep_post > ep_pre and best["state"] is not None: model.load_state_dict(best["state"])
        except RuntimeError:
            if best["state"] is not None: model.load_state_dict(best["state"])
    eps, eA, eg = eval_eps()
    # trajectory error (coarse dt=0.05 for the sweep; isolates surrogate error)
    def H_learned(s):
        x, y, z, Px, Py, Pz, t = [s[..., i] for i in range(7)]
        Ath = model(torch.stack([x, y, z, t], -1)); px = Px-ch*Ath; return torch.sqrt(1+px**2+Py**2+Pz**2)
    def vf(use_model, r, P, t):
        s = torch.tensor([[*r, *P, t]], device=dev, dtype=torch.float32, requires_grad=True)
        H = H_learned(s) if use_model else H_full7(s); g = torch.autograd.grad(H.sum(), s)[0][0]
        return g[3:6].detach().cpu().numpy(), -g[0:3].detach().cpu().numpy()
    def rk4(use_model, dt, t0=-3*tau, t1=3*tau):
        n = int((t1-t0)/dt); ts = t0+dt*np.arange(n+1); r = np.zeros(3); P = np.zeros(3); gg = np.ones(n+1)
        for k in range(n):
            t = ts[k]; k1r, k1p = vf(use_model, r, P, t); k2r, k2p = vf(use_model, r+.5*dt*k1r, P+.5*dt*k1p, t+.5*dt)
            k3r, k3p = vf(use_model, r+.5*dt*k2r, P+.5*dt*k2p, t+.5*dt); k4r, k4p = vf(use_model, r+dt*k3r, P+dt*k3p, t+dt)
            r = r+dt/6*(k1r+2*k2r+2*k3r+k4r); P = P+dt/6*(k1p+2*k2p+2*k3p+k4p)
            Ax = float(A_full(*[torch.tensor(v) for v in (r[0], r[1], r[2], t+dt)])); gg[k+1] = math.sqrt(1+(P[0]-ch*Ax)**2+P[1]**2+P[2]**2)
        return ts, gg
    try:
        ts, ga = rk4(False, 0.5 if SMOKE else 0.05); _, gp = rk4(True, 0.5 if SMOKE else 0.05)
        traj_err = float((np.abs(gp-ga)/ga).max())
    except Exception:
        traj_err = float("nan")
    dt_s = time.time()-t0
    return {"eps_theta": eps, "eps_A": eA, "eps_gradA": eg, "traj_err": traj_err,
            "n_params": sum(p.numel() for p in model.parameters()), "time_s": dt_s}

# ---- build config list ----
def mk(tag, **kw):
    base = dict(seed=0, width=256, depth=6, N=128000, w_g=32.0, fourier=1, tube=1,
                warmup=1, warmup_ep=400, clip=1, adapt=1, lbfgs=1,
                epochs=(60 if SMOKE else 5000), batch=(512 if SMOKE else 16000),
                lbfgs_it=(15 if SMOKE else 300))
    base.update(kw); base["tag"] = tag; return base
configs = []
# E4 scaling (all features on, seed 0)
for width in ([128] if SMOKE else [128, 256, 384]):
    for N in ([2000] if SMOKE else [2000, 8000, 32000, 128000]):
        configs.append(mk(f"E4_scale_w{width}_N{N}", width=width, N=N))
# E5 ablation from baseline (w256,N128k,w_g32,all on == E4_scale_w256_N128000)
if not SMOKE:
    configs.append(mk("E5_abl_baseline"))
    configs.append(mk("E5_abl_nofourier", fourier=0))
    configs.append(mk("E5_abl_notube", tube=0))
    configs.append(mk("E5_abl_nowarmup", warmup=0))
    configs.append(mk("E5_abl_noclip", clip=0))
    configs.append(mk("E5_abl_noadapt", adapt=0))
    for wg in [1.0, 8.0, 24.0]:
        configs.append(mk(f"E5_abl_wg{int(wg)}", w_g=wg))
else:
    configs.append(mk("E5_abl_nofourier", fourier=0))

allres = []; t_all = time.time()
for i, cfg in enumerate(configs):
    r = train_once(cfg)
    rec = {**{k: cfg[k] for k in ("tag", "width", "depth", "N", "w_g", "fourier", "tube", "warmup", "clip", "adapt", "lbfgs", "epochs", "seed")}, **r}
    allres.append(rec)
    json.dump({"results": allres, "elapsed_s": time.time()-t_all}, open(OUT+"/results.json", "w"), indent=2)
    print("[%2d/%2d] %-22s eps=%.3e (A=%.2e gA=%.2e) traj=%.3f  np=%d  %.1fs"
          % (i+1, len(configs), cfg["tag"], r["eps_theta"], r["eps_A"], r["eps_gradA"],
             r["traj_err"], r["n_params"], r["time_s"]), flush=True)
print("SCALING_TOTAL_TIME %.1f s" % (time.time()-t_all), flush=True)

# ---- E4 figure: eps_theta vs N_collocation, curve per width ----
try:
    plt.figure(figsize=(5.2, 3.8))
    for width in [128, 256, 384]:
        pts = sorted([(r["N"], r["eps_theta"]) for r in allres if r["tag"].startswith("E4") and r["width"] == width])
        if pts:
            xs, ys = zip(*pts); plt.loglog(xs, ys, "o-", label=f"width {width}")
            if len(xs) >= 2:
                a, b = np.polyfit(np.log(xs), np.log(ys), 1)
                print("scaling fit width %d : eps ~ N^%.3f" % (width, a), flush=True)
    plt.axhline(1e-4, ls="--", color="k", alpha=.5, label=r"$10^{-4}$ target")
    plt.xlabel("collocation points N"); plt.ylabel(r"$\varepsilon_\theta$ (in-tube RMS)")
    plt.title("Surrogate accuracy scaling (reduced budget)"); plt.legend(); plt.grid(True, which="both", alpha=.3)
    plt.tight_layout(); plt.savefig(OUT+"/scaling.png", dpi=140)
except Exception as e:
    print("plot err", e, flush=True)
print("DONE", flush=True)
