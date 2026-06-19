# E1 - GPU throughput benchmark: batched Boris / RK4 / Tao-symplectic pushers.
# Measures particle-steps/s vs ensemble size N (10^2..10^7) on GPU, for the
# uniform-magnetic-field test (isolates the INTEGRATOR arithmetic overhead --
# the same field/gradients feed all three pushers). Honest claim: the Tao map's
# per-step overhead vs Boris is a bounded, N-independent constant factor that is
# fully amortized in the batched GPU regime; it is NOT a scaling penalty.
#
# Local smoke test (CPU, ~20 s):  SMOKE=1 python bench.py
import os, sys, subprocess, time, json, math
ON_KAGGLE = os.path.isdir("/kaggle/working")
if ON_KAGGLE:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--force-reinstall",
                    "--no-deps", "torch", "--index-url",
                    "https://download.pytorch.org/whl/cu121"])
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
dev = "cuda" if torch.cuda.is_available() else "cpu"
SMOKE = bool(os.environ.get("SMOKE"))
OUT = "/kaggle/working" if ON_KAGGLE else "."
print("GPU_AVAILABLE", torch.cuda.is_available(),
      "DEVICE", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
      "SMOKE", SMOKE, flush=True)

B0 = 1.0; ch = -1.0; omega = 20.0; dt = 0.05

# ---- batched magnetic field ops (symmetric gauge), r,P shape (N,3) ----
def A_of(r):
    x = r[..., 0]; y = r[..., 1]
    return torch.stack([-0.5*B0*y, 0.5*B0*x, torch.zeros_like(x)], -1)
def gamma_p(p):
    return torch.sqrt(1.0 + (p*p).sum(-1, keepdim=True))
def gradH_P(r, P):
    p = P - ch*A_of(r); return p/gamma_p(p)
def gradH_r(r, P):
    p = P - ch*A_of(r); g = gamma_p(p)[..., 0]; px = p[..., 0]; py = p[..., 1]
    dHdx = (1.0/g)*(py*(-ch*0.5*B0)); dHdy = (1.0/g)*(px*(ch*0.5*B0))
    return torch.stack([dHdx, dHdy, torch.zeros_like(dHdx)], -1)
def E_of(r): return torch.zeros_like(r)
def B_of(r):
    b = torch.zeros_like(r); b[..., 2] = B0; return b

# ---- pushers ----
def boris(r, p, dt):
    E = E_of(r); B = B_of(r)
    pm = p + ch*E*dt/2; gm = torch.sqrt(1.0 + (pm*pm).sum(-1, keepdim=True))
    tt = ch*B*dt/(2*gm); t2 = (tt*tt).sum(-1, keepdim=True); s = 2*tt/(1+t2)
    pp = pm + torch.cross(pm, tt, dim=-1); pl = pm + torch.cross(pp, s, dim=-1)
    pn = pl + ch*E*dt/2; gn = torch.sqrt(1.0 + (pn*pn).sum(-1, keepdim=True))
    return r + dt*pn/gn, pn
def deriv_kin(r, p):
    g = torch.sqrt(1.0 + (p*p).sum(-1, keepdim=True)); v = p/g
    return v, ch*(E_of(r) + torch.cross(v, B_of(r), dim=-1))
def rk4(r, p, dt):
    k1r, k1p = deriv_kin(r, p)
    k2r, k2p = deriv_kin(r+.5*dt*k1r, p+.5*dt*k1p)
    k3r, k3p = deriv_kin(r+.5*dt*k2r, p+.5*dt*k2p)
    k4r, k4p = deriv_kin(r+dt*k3r, p+dt*k3p)
    return r+dt/6*(k1r+2*k2r+2*k3r+k4r), p+dt/6*(k1p+2*k2p+2*k3p+k4p)
w1 = 1.0/(2.0-2.0**(1.0/3.0)); w0 = 1.0-2.0*w1; cset = (w1, w0, w1)
def phiA(r, P, x, y, dt): return r, P-dt*gradH_r(r, y), x+dt*gradH_P(r, y), y
def phiB(r, P, x, y, dt): return r+dt*gradH_P(x, P), P, x, y-dt*gradH_r(x, P)
def phiC(r, P, x, y, dt):
    c = math.cos(2*omega*dt); s = math.sin(2*omega*dt)
    R = r+x; S = P+y; u = r-x; v = P-y
    un = u*c+v*s; vn = -u*s+v*c
    return 0.5*(R+un), 0.5*(S+vn), 0.5*(R-un), 0.5*(S-vn)
def strang(r, P, x, y, dt):
    r, P, x, y = phiA(r, P, x, y, dt/2); r, P, x, y = phiB(r, P, x, y, dt/2)
    r, P, x, y = phiC(r, P, x, y, dt)
    r, P, x, y = phiB(r, P, x, y, dt/2); r, P, x, y = phiA(r, P, x, y, dt/2)
    return r, P, x, y
def tao(r, P, x, y, dt):
    for c in cset: r, P, x, y = strang(r, P, x, y, c*dt)
    return r, P, x, y

def time_pusher(kind, N, steps, dtype, device):
    torch.manual_seed(0)
    r = torch.randn(N, 3, device=device, dtype=dtype)*0.1
    p = torch.randn(N, 3, device=device, dtype=dtype)*1.0
    P = p + ch*A_of(r); x = r.clone(); y = P.clone()
    def one():
        nonlocal r, p, P, x, y
        if kind == "boris": r, p = boris(r, p, dt)
        elif kind == "rk4": r, p = rk4(r, p, dt)
        else: r, P, x, y = tao(r, P, x, y, dt)
    for _ in range(3): one()                      # warmup
    if device == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps): one()
    if device == "cuda": torch.cuda.synchronize()
    return (time.perf_counter()-t0)/steps         # sec/step for N particles

Ns = [100, 1000] if SMOKE else [100, 1000, 10_000, 100_000, 1_000_000, 3_000_000, 10_000_000]
steps_for = lambda N: (5 if SMOKE else (200 if N <= 1e5 else (60 if N <= 1e6 else 25)))
res = {"device": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
       "dt": dt, "omega": omega, "runs": {}}
t_start = time.time()
for dtype, dname in ([(torch.float64, "f64"), (torch.float32, "f32")] if not SMOKE else [(torch.float64, "f64")]):
    for kind in ["boris", "rk4", "tao"]:
        for N in Ns:
            steps = steps_for(N)
            try:
                sps = time_pusher(kind, N, steps, dtype, dev)
                thr = N/sps
                res["runs"].setdefault(dname, {}).setdefault(kind, {})[str(N)] = {
                    "sec_per_step": sps, "throughput_psps": thr, "steps": steps}
                print("%s %-5s N=%9d  %.3e s/step  thr=%.3e p-steps/s" % (dname, kind, N, sps, thr), flush=True)
            except RuntimeError as e:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                res["runs"].setdefault(dname, {}).setdefault(kind, {})[str(N)] = {"oom": str(e)[:80]}
                print("%s %-5s N=%9d  OOM/skip" % (dname, kind, N), flush=True)
            if torch.cuda.is_available(): torch.cuda.empty_cache()
res["wall_time_s"] = time.time()-t_start
print("BENCH_TIME %.1f s" % res["wall_time_s"], flush=True)

# overhead ratio Tao/Boris at the largest common N (f64)
def thr_at(dname, kind, N):
    d = res["runs"].get(dname, {}).get(kind, {}).get(str(N), {}); return d.get("throughput_psps")
for dname in res["runs"]:
    big = [N for N in Ns if thr_at(dname, "tao", N) and thr_at(dname, "boris", N)]
    if big:
        Nb = max(big)
        ratio = thr_at(dname, "boris", Nb)/thr_at(dname, "tao", Nb)
        res.setdefault("overhead_tao_over_boris", {})[dname] = {"N": Nb, "ratio": ratio}
        print("%s overhead Tao/Boris at N=%d : %.2fx" % (dname, Nb, ratio), flush=True)

json.dump(res, open(OUT+"/results.json", "w"), indent=2)

# ---- figure: throughput vs N (f64), 3 curves ----
plt.figure(figsize=(5.2, 3.6))
for kind, mk in [("boris", "o-"), ("rk4", "s-"), ("tao", "^-")]:
    xs = [N for N in Ns if thr_at("f64", kind, N)]
    ys = [thr_at("f64", kind, N) for N in xs]
    if xs: plt.loglog(xs, ys, mk, label=kind.upper() if kind != "tao" else "Tao (SP-PINN Stage-2)")
plt.xlabel("ensemble size N"); plt.ylabel("throughput  (particle-steps / s)")
plt.title("GPU throughput (float64), uniform B"); plt.legend(); plt.grid(True, which="both", alpha=.3)
plt.tight_layout(); plt.savefig(OUT+"/throughput.png", dpi=140)
print("DONE", flush=True)
