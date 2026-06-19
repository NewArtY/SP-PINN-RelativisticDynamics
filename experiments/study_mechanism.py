"""Realized order of the Tao map and the shadow (modified) Hamiltonian.

Fits the trajectory error against a DOP853 reference as a power law in dt for the
uniform-field orbit, at fixed Omega=20 and in the coupled limit Omega=1/dt where
the paper claims realized order two; and the amplitude of the bounded energy
oscillation, whose O(dt^2) scaling in the coupled limit is the numerical signature
of a conserved shadow Hamiltonian Htilde = H + O(dt^2). Writes mechanism.png and a
JSON summary to ../figures. Run from the experiments/ directory.
"""
import _paths  # noqa: F401
import os
import json
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from relsim.fields import ConstMagneticField
from relsim.integrators import boris_step, rk4_step, TaoSymplectic, rk8_reference

OUT = _paths.FIG


def traj_err_and_energy_amp(field, r0, P0, p0, H0, rT_ref, T, method, dt, omega=20.0):
    n = int(T / dt)
    if method == "tao":
        integ = TaoSymplectic(field, omega=omega)
        r, P = r0[None, :].copy(), P0[None, :].copy()
        x, y = r.copy(), P.copy()
        dHmax = 0.0
        for k in range(n):
            r, P, x, y = integ.step(r, P, x, y, k * dt, dt)
            if not np.all(np.isfinite(r)):
                return np.inf, np.inf
            dHmax = max(dHmax, abs(float(field.H(r, P, (k + 1) * dt)[0]) - H0) / abs(H0))
        return float(np.linalg.norm(r[0] - rT_ref)), dHmax
    step = {"boris": boris_step, "rk4": rk4_step}[method]
    r, p = r0[None, :].copy(), p0[None, :].copy()
    dHmax = 0.0
    for k in range(n):
        r, p = step(field, r, p, k * dt, dt)
        Pc = p + field.ch * field.A(r, (k + 1) * dt)
        dHmax = max(dHmax, abs(float(field.H(r, Pc, (k + 1) * dt)[0]) - H0) / abs(H0))
    return float(np.linalg.norm(r[0] - rT_ref)), dHmax


def main():
    field = ConstMagneticField(B0=1.0, ch=-1.0)
    T = 40.0
    r0 = np.array([1.0, 0.0, 0.0]); p0 = np.array([0.0, 1.0, 0.0])
    P0 = p0 + field.ch * field.A(r0[None, :], 0.0)[0]
    H0 = float(field.H(r0[None, :], P0[None, :], 0.0)[0])
    Rref, _ = rk8_reference(field, r0, p0, np.array([0.0, T]))
    rT_ref = Rref[-1]

    dts = [0.1, 0.05, 0.025, 0.0125, 0.00625]
    series = {}
    for label, method, omega_fn in [("Tao (Omega=20 fixed)", "tao", lambda dt: 20.0),
                                     ("Tao (Omega=1/dt coupled)", "tao", lambda dt: 1.0 / dt),
                                     ("RK4", "rk4", None), ("Boris", "boris", None)]:
        errs, amps = [], []
        for dt in dts:
            om = omega_fn(dt) if omega_fn else 20.0
            e, a = traj_err_and_energy_amp(field, r0, P0, p0, H0, rT_ref, T, method, dt, om)
            errs.append(e); amps.append(a)
        errs = np.array(errs); amps = np.array(amps)
        ok = np.isfinite(errs) & (errs > 0)
        p_traj = float(np.polyfit(np.log(np.array(dts)[ok]), np.log(errs[ok]), 1)[0]) if ok.sum() >= 2 else float("nan")
        oka = np.isfinite(amps) & (amps > 0)
        p_amp = float(np.polyfit(np.log(np.array(dts)[oka]), np.log(amps[oka]), 1)[0]) if oka.sum() >= 2 else float("nan")
        series[label] = {"traj_err": [float(x) for x in errs], "energy_amp": [float(x) for x in amps],
                         "order_traj": p_traj, "order_energy_amp": p_amp}
        print("%-26s order(traj)=%.2f order(energy-amp)=%.2f" % (label, p_traj, p_amp))

    json.dump({"dts": dts, "series": series}, open(os.path.join(_paths.DATA, "study_mechanism.json"), "w"), indent=2)
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for label, d in series.items():
        ax[0].loglog(dts, np.maximum(d["traj_err"], 1e-16), "o-", label="%s (p=%.2f)" % (label, d["order_traj"]), lw=1.3, ms=4)
        ax[1].loglog(dts, np.maximum(d["energy_amp"], 1e-18), "o-", label="%s (q=%.2f)" % (label, d["order_energy_amp"]), lw=1.3, ms=4)
    ax[0].set_title("realized order: trajectory error"); ax[0].set_xlabel(r"$\Delta t$"); ax[0].set_ylabel("error at T")
    ax[0].legend(fontsize=7); ax[0].grid(True, which="both", alpha=.3)
    ax[1].set_title("shadow Hamiltonian: energy-oscillation amp"); ax[1].set_xlabel(r"$\Delta t$"); ax[1].set_ylabel(r"max $\delta H$")
    ax[1].legend(fontsize=7); ax[1].grid(True, which="both", alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "mechanism.png"), dpi=140); plt.close()
    print("wrote mechanism.png to", OUT)


if __name__ == "__main__":
    main()
