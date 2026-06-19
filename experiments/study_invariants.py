"""Direct measurement of the first Poincare--Cartan loop invariant delta I_1(t)
and the energy drift delta H(t), with a structure-preserving comparator panel
{Boris, Higuera--Cary, RK4, Tao} on three test cases (uniform B, focused laser,
chaotic magnetic trap).

Two complementary diagnostics:
  * delta I_1(t): drift of the SYMPLECTIC two-form invariant I_1 = oint P.dr.
    The symplectic Tao map keeps it bounded (and high-order); the volume-preserving
    but non-symplectic Boris/Higuera--Cary drift at O(dt); RK4 is small on smooth
    integrable orbits but secular.
  * delta H(t): energy drift for the autonomous cases (uniform B, chaotic trap),
    where H is an exact constant of motion. In the chaotic trap the symplectic map
    stays bounded while Boris/HC drift secularly.

Writes deltaI1.png, energy_drift.png, loop_convergence.png and a JSON summary to
../figures. Run from the experiments/ directory.
"""
import _paths  # noqa: F401  (puts repo root on sys.path; creates ../figures, ../data)
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from relsim.fields import ConstMagneticField, GaussianLaserPulse, MagneticWell
from relsim.integrators import boris_step, rk4_step, higuera_cary_step, TaoSymplectic
from relsim.diagnostics import poincare_loop_invariant

OUT = _paths.FIG
STEP = {"boris": boris_step, "hc": higuera_cary_step, "rk4": rk4_step}
LAB = {"boris": "Boris", "hc": "Higuera-Cary", "rk4": "RK4", "tao": "Tao (symplectic)"}
METHODS = ["boris", "hc", "rk4", "tao"]


def make_loop(r0, P0, M, eps):
    """Closed loop of M points on a small circle in the canonical (x, P_x) plane."""
    th = np.linspace(0, 2 * np.pi, M, endpoint=False)
    r = np.tile(np.asarray(r0, float), (M, 1))
    P = np.tile(np.asarray(P0, float), (M, 1))
    r[:, 0] = r0[0] + eps * np.cos(th)
    P[:, 0] = P0[0] + eps * np.sin(th)
    return r, P


def evolve_loop(field, r0loop, P0loop, t0, dt, nsteps, method, omega, rec):
    times, I1 = [], []
    if method == "tao":
        r, P = r0loop.copy(), P0loop.copy()
        x, y = r.copy(), P.copy()
        integ = TaoSymplectic(field, omega=omega)
        for k in range(nsteps + 1):
            t = t0 + k * dt
            if k % rec == 0:
                times.append(t); I1.append(poincare_loop_invariant(r, P))
            if k < nsteps:
                r, P, x, y = integ.step(r, P, x, y, t, dt)
    else:
        step = STEP[method]
        r = r0loop.copy()
        p = P0loop - field.ch * field.A(r0loop, t0)          # canonical -> kinetic
        for k in range(nsteps + 1):
            t = t0 + k * dt
            if k % rec == 0:
                Pc = p + field.ch * field.A(r, t)             # kinetic -> canonical
                times.append(t); I1.append(poincare_loop_invariant(r, Pc))
            if k < nsteps:
                r, p = step(field, r, p, t, dt)
    return np.array(times), np.array(I1)


def evolve_energy(field, r0, p0, t0, dt, nsteps, method, omega, rec):
    """Track H(t) of the base orbit (single particle)."""
    r0 = np.asarray(r0, float)[None, :]
    p0 = np.asarray(p0, float)[None, :]
    P0 = p0 + field.ch * field.A(r0, t0)
    times, H = [], []
    if method == "tao":
        r, P = r0.copy(), P0.copy()
        x, y = r.copy(), P.copy()
        integ = TaoSymplectic(field, omega=omega)
        for k in range(nsteps + 1):
            t = t0 + k * dt
            if k % rec == 0:
                times.append(t); H.append(float(field.H(r, P, t)[0]))
            if k < nsteps:
                r, P, x, y = integ.step(r, P, x, y, t, dt)
    else:
        step = STEP[method]
        r, p = r0.copy(), p0.copy()
        for k in range(nsteps + 1):
            t = t0 + k * dt
            if k % rec == 0:
                Pc = p + field.ch * field.A(r, t)
                times.append(t); H.append(float(field.H(r, Pc, t)[0]))
            if k < nsteps:
                r, p = step(field, r, p, t, dt)
    return np.array(times), np.array(H)


def main():
    cases = {
        "magnetic": dict(field=ConstMagneticField(B0=1.0, ch=-1.0), r0=[1.0, 0.0, 0.0], p0=[0.0, 1.0, 0.0],
                         t0=0.0, dt=0.05, nsteps=4000, eps=0.02, omega=20.0, e_nsteps=20000,
                         title="(a) uniform B (regular)"),
        "laser": dict(field=GaussianLaserPulse(a0=5.0), r0=[0.0, 0.0, 0.0], p0=[0.0, 0.0, 0.0],
                      t0=-90.0, dt=0.05, nsteps=3600, eps=0.01, omega=20.0, e_nsteps=0,
                      title="(b) focused laser pulse"),
        "chaotic": dict(field=MagneticWell(B0=1.0, kperp=1.0, eps=0.30, ch=1.0), r0=[1.0, 0.5, 0.0], p0=[0.0, 0.0, 0.0],
                        t0=0.0, dt=0.05, nsteps=4000, eps=0.02, omega=20.0, e_nsteps=40000,
                        title="(c) chaotic magnetic trap"),
    }
    M = 256
    results = {}

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ci, (cname, cfg) in enumerate(cases.items()):
        field = cfg["field"]
        r0 = np.array(cfg["r0"], float); p0 = np.array(cfg["p0"], float)
        P0 = p0 + field.ch * field.A(r0[None, :], cfg["t0"])[0]
        r0loop, P0loop = make_loop(r0, P0, M, cfg["eps"])
        results[cname] = {}
        ax = axes[ci]
        for m in METHODS:
            ts, I1 = evolve_loop(field, r0loop, P0loop, cfg["t0"], cfg["dt"], cfg["nsteps"], m, cfg["omega"], 20)
            dI = np.abs(I1 - I1[0]) / abs(I1[0])
            half = len(ts) // 2
            slope = float(np.polyfit(ts[half:], np.maximum(dI[half:], 1e-16), 1)[0])
            results[cname][m] = {"dI1_final": float(dI[-1]), "dI1_max": float(dI.max()), "late_slope": slope}
            ax.semilogy(ts, np.maximum(dI, 1e-16), label=LAB[m], lw=1.4)
        ax.set_title(cfg["title"]); ax.set_xlabel("t"); ax.grid(True, which="both", alpha=.3)
        if ci == 0:
            ax.set_ylabel(r"$\delta I_1(t)$ (rel.)")
        if ci == 2:
            ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "deltaI1.png"), dpi=140); plt.close()

    ecases = [c for c in cases if cases[c]["e_nsteps"] > 0]
    fig, axes = plt.subplots(1, len(ecases), figsize=(4.4 * len(ecases), 3.6))
    if len(ecases) == 1:
        axes = [axes]
    for ci, cname in enumerate(ecases):
        cfg = cases[cname]; field = cfg["field"]; ax = axes[ci]; results[cname]["energy"] = {}
        for m in METHODS:
            ts, H = evolve_energy(field, cfg["r0"], cfg["p0"], cfg["t0"], cfg["dt"], cfg["e_nsteps"], m, cfg["omega"], 50)
            dH = np.abs(H - H[0]) / abs(H[0])
            half = len(ts) // 2
            slope = float(np.polyfit(ts[half:], np.maximum(dH[half:], 1e-16), 1)[0])
            results[cname]["energy"][m] = {"dH_final": float(dH[-1]), "dH_max": float(dH.max()), "late_slope": slope}
            ax.semilogy(ts, np.maximum(dH, 1e-16), label=LAB[m], lw=1.4)
        ax.set_title(cfg["title"]); ax.set_xlabel("t"); ax.grid(True, which="both", alpha=.3)
        if ci == 0:
            ax.set_ylabel(r"$\delta H(t)$ (rel. energy drift)")
        ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "energy_drift.png"), dpi=140); plt.close()

    # spectral convergence of the loop quadrature (magnetic, t0)
    cfg = cases["magnetic"]; field = cfg["field"]
    r0 = np.array(cfg["r0"], float); p0 = np.array(cfg["p0"], float)
    P0 = p0 + field.ch * field.A(r0[None, :], 0.0)[0]
    Ms = [8, 16, 32, 64, 128, 256, 512]
    I1M = np.array([poincare_loop_invariant(*make_loop(r0, P0, mm, cfg["eps"])) for mm in Ms])
    conv = np.abs(I1M[:-1] - I1M[-1]) / abs(I1M[-1])
    plt.figure(figsize=(4.2, 3.2)); plt.loglog(Ms[:-1], np.maximum(conv, 1e-16), "o-")
    plt.xlabel("loop sample count M"); plt.ylabel(r"$|I_1(M)-I_1(512)|/|I_1|$")
    plt.title("loop-quadrature convergence"); plt.grid(True, which="both", alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "loop_convergence.png"), dpi=140); plt.close()
    results["loop_convergence"] = {"M": Ms, "rel_err_to_M512": [float(x) for x in conv] + [0.0]}

    json.dump(results, open(os.path.join(_paths.DATA, "study_invariants.json"), "w"), indent=2)
    for cname in cases:
        print("==", cname, "==")
        for m in METHODS:
            d = results[cname][m]
            print("  %-13s dI1_final=%.3e dI1_max=%.3e slope=%.2e" % (m, d["dI1_final"], d["dI1_max"], d["late_slope"]))
    print("wrote deltaI1.png, energy_drift.png, loop_convergence.png to", OUT)


if __name__ == "__main__":
    main()
