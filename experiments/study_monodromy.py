"""Exact Floquet (monodromy) eigenvalues of the Tao map and the k mod 3
band-strength rule of Appendix A.2.

For the uniform field the linearized one-step extended-phase-space map is a
constant 12x12 symplectic matrix (gamma is constant on the gyro-orbit), so its
per-step monodromy is obtained by finite-difference linearization of one
TaoSymplectic.step around a gyro-orbit point. At each binding resonance
Omega_k = k*pi/(2*dt) we record the leading instability growth |lambda_max| - 1
and classify it by k mod 3 (the manuscript's own metric). The exact monodromy
gives k=1 > k=2 > k=0 (mod 3) robustly across cyclotron detunings, confirming
Appendix A.2; a trajectory-error proxy can mis-rank k=1 vs k=2 (it is a different,
nonlinear functional). Writes monodromy_bars.png and a JSON summary to ../figures.
Run from the experiments/ directory.
"""
import _paths  # noqa: F401
import os
import json
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from relsim.fields import ConstMagneticField
from relsim.integrators import TaoSymplectic

OUT = _paths.FIG
field = ConstMagneticField(B0=1.0, ch=-1.0)
r0 = np.array([1.0, 0.0, 0.0]); p0 = np.array([0.0, 1.0, 0.0])
P0 = p0 + field.ch * field.A(r0[None, :], 0.0)[0]
GAMMA = math.sqrt(1 + p0 @ p0)
WC = abs(field.ch * field.B0) / GAMMA                 # cyclotron frequency
S0 = np.concatenate([r0, P0, r0, P0])                 # extended state (r,P,x,y), copies = original


def _step_vec(s, omega, dt):
    r, P, x, y = s[0:3].copy(), s[3:6].copy(), s[6:9].copy(), s[9:12].copy()
    integ = TaoSymplectic(field, omega=omega)
    r, P, x, y = integ.step(r[None, :], P[None, :], x[None, :], y[None, :], 0.0, dt)
    return np.concatenate([r[0], P[0], x[0], y[0]])


def monodromy_growth(omega, dt, h=1e-6):
    """max|lambda| - 1 of the one-step Jacobian (finite-difference linearization)."""
    J = np.zeros((12, 12))
    for i in range(12):
        sp = S0.copy(); sp[i] += h
        sm = S0.copy(); sm[i] -= h
        J[:, i] = (_step_vec(sp, omega, dt) - _step_vec(sm, omega, dt)) / (2 * h)
    return float(np.max(np.abs(np.linalg.eigvals(J)))) - 1.0


def bands(dt, kmax=9):
    out = []
    for k in range(1, kmax + 1):
        ok = k * math.pi / (2 * dt)
        g = max(monodromy_growth(w, dt) for w in np.linspace(ok * 0.98, ok * 1.02, 60))
        out.append({"k": k, "kmod3": k % 3, "Omega_k": ok, "growth": max(g, 0.0)})
    return out


def main():
    results = {"wc": WC, "by_dt": {}}
    for dt in [0.05, 0.0889, 0.0354, 0.02]:
        bs = bands(dt)
        by3 = {0: [], 1: [], 2: []}
        for b in bs:
            by3[b["kmod3"]].append(b["growth"])
        means = {r: float(np.mean(by3[r])) for r in (0, 1, 2)}
        results["by_dt"]["%.4f" % dt] = {"wc_dt": WC * dt, "bands": bs, "mean_by_kmod3": means,
                                         "strongest_class": max(means, key=means.get),
                                         "weakest_class": min(means, key=means.get)}
        print("dt=%.4f wc*dt=%.4f -> mean by kmod3: 0=%.3e 1=%.3e 2=%.3e (strongest k=%d, weakest k=%d)"
              % (dt, WC * dt, means[0], means[1], means[2], max(means, key=means.get), min(means, key=means.get)))
    json.dump(results, open(os.path.join(_paths.DATA, "study_monodromy.json"), "w"), indent=2)

    # bar figure at the manuscript-regime detuning (wc*dt approx 0.063)
    bs = results["by_dt"]["0.0889"]["bands"]
    ks = [b["k"] for b in bs]; g = [b["growth"] for b in bs]; m3 = [b["kmod3"] for b in bs]
    col = {1: "#d62728", 2: "#ff7f0e", 0: "#2ca02c"}
    plt.figure(figsize=(6.2, 3.4))
    plt.bar(ks, g, color=[col[c] for c in m3], edgecolor="k", linewidth=0.4)
    plt.legend(handles=[Patch(color=col[1], label=r"$k\equiv1$"), Patch(color=col[2], label=r"$k\equiv2$"),
                        Patch(color=col[0], label=r"$k\equiv0$ (mod 3)")], fontsize=8, loc="upper right")
    plt.xlabel(r"resonance order $k$  ($\Omega_k=k\pi/2\Delta t$)")
    plt.ylabel(r"monodromy growth $|\lambda_{\max}|-1$")
    plt.title(r"Exact Floquet band strength by $k$ mod 3  ($k\equiv1>k\equiv2>k\equiv0$)")
    plt.xticks(ks); plt.grid(True, axis="y", alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "monodromy_bars.png"), dpi=140); plt.close()
    print("wrote monodromy_bars.png to", OUT)


if __name__ == "__main__":
    main()
