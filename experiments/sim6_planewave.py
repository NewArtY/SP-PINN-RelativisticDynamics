"""
Simulation 6 -- Plane electromagnetic wave: time-as-coordinate symplecticity
and transverse-momentum conservation (Figure A1; addresses review points 1 and 7).

An electron initially at rest is driven by a linearly polarised plane wave
A = a0 cos(k0(z - t)) x_hat.  Two exact invariants exist:
  * the light-front quantity   K = gamma - p_z = 1  (sensitive to the explicit
    time dependence of the field);
  * the transverse canonical momentum  P_x = p_x + ch A_x  (Noether invariant of
    the exact transverse translational symmetry of a *plane* wave).

We compare four schemes over many wave periods:
  RK4, Boris, the frozen-in-time Tao symplectic map ("SP-PINN, frozen"),
  and the autonomized Tao map ("SP-PINN, time-as-coordinate") in which time is a
  canonical coordinate so the extended Hamiltonian is autonomous and the map is
  exactly symplectic for the time-dependent field.

Result: autonomizing time restores bounded conservation of K (point 1); and for
a true plane wave P_x is exactly conserved, so its deviation is purely numerical
-- the Boris pusher shows the largest violation (point 7).
"""
import _paths  # noqa: F401
import numpy as np
import csv
import os

from relsim.fields import PlaneWave, AutonomizedField
from relsim.integrators import boris_step, rk4_step, TaoSymplectic
from relsim.diagnostics import (gamma_of_p, kinetic_to_canonical,
                                canonical_to_kinetic)
import matplotlib.pyplot as plt
from relsim.plotstyle import apply_style, COLORS


def run(a0=2.0, k0=1.0, n_periods=1000, steps_per_period=100, omega=20.0):
    pw = PlaneWave(a0=a0, k0=k0, ch=-1.0)
    T = 2 * np.pi / k0
    dt = T / steps_per_period
    nsteps = n_periods * steps_per_period
    r0 = np.array([[0.0, 0.0, 0.0]])
    p0 = np.array([[0.0, 0.0, 0.0]])
    Px0 = kinetic_to_canonical(pw, r0, p0, 0.0)[0, 0]

    schemes = ["RK4", "Boris", "SP-PINN (frozen)", "SP-PINN (time-coord)"]
    dK = {s: np.zeros(n_periods) for s in schemes}
    dPx = {s: np.zeros(n_periods) for s in schemes}

    for s in schemes:
        if s == "SP-PINN (time-coord)":
            auto = AutonomizedField(pw)
            tao = TaoSymplectic(auto, omega=omega)
            r4 = np.zeros((1, 4)); P4 = np.zeros((1, 4))
            P4[0, :3] = kinetic_to_canonical(pw, r0, p0, 0.0)[0]
            xx, yy = tao.init_copies(r4, P4)
        elif s == "SP-PINN (frozen)":
            tao = TaoSymplectic(pw, omega=omega)
            r = r0.copy(); P = kinetic_to_canonical(pw, r0, p0, 0.0)
            xx, yy = tao.init_copies(r, P)
        else:
            r, p = r0.copy(), p0.copy()
        m = 0
        for n in range(nsteps):
            t = n * dt
            if s == "RK4":
                r, p = rk4_step(pw, r, p, t, dt)
                pk, Px = p, p[0, 0] + pw.ch * pw.A(r, t + dt)[0, 0]
            elif s == "Boris":
                r, p = boris_step(pw, r, p, t, dt)
                pk, Px = p, p[0, 0] + pw.ch * pw.A(r, t + dt)[0, 0]
            elif s == "SP-PINN (frozen)":
                r, P, xx, yy = tao.step(r, P, xx, yy, t, dt)
                pk = canonical_to_kinetic(pw, r, P, t + dt); Px = P[0, 0]
            else:
                r4, P4, xx, yy = tao.step(r4, P4, xx, yy, 0.0, dt)
                tt = r4[0, 3]
                pk = canonical_to_kinetic(pw, r4[:, :3], P4[:, :3], tt)
                Px = P4[0, 0]
            if (n + 1) % steps_per_period == 0:
                K = gamma_of_p(pk)[0] - pk[0, 2]
                dK[s][m] = abs(K - 1.0)
                dPx[s][m] = abs(Px - Px0)
                m += 1
    return dict(periods=np.arange(1, n_periods + 1), dK=dK, dPx=dPx,
                params=dict(a0=a0, dt=dt, n_periods=n_periods, omega=omega))


def make_figure(res):
    apply_style()
    per = res["periods"]
    # Keep green = SP-PINN throughout; the frozen vs. autonomized (time-coord)
    # variants are distinguished by line style, not by a new colour.  The
    # highlighted bounded trace (time-coord) is drawn last, on top, opaque.
    order = ["Boris", "RK4", "SP-PINN (frozen)", "SP-PINN (time-coord)"]
    col = {"RK4": COLORS["RK4"], "Boris": COLORS["Boris"],
           "SP-PINN (frozen)": COLORS["SP-PINN"], "SP-PINN (time-coord)": COLORS["SP-PINN"]}
    ls = {"RK4": "-", "Boris": "--", "SP-PINN (frozen)": ":", "SP-PINN (time-coord)": "-"}
    zo = {"RK4": 2, "Boris": 2, "SP-PINN (frozen)": 1, "SP-PINN (time-coord)": 5}
    al = {"RK4": 0.9, "Boris": 0.9, "SP-PINN (frozen)": 0.7, "SP-PINN (time-coord)": 1.0}
    lw = {"RK4": 1.6, "Boris": 1.6, "SP-PINN (frozen)": 1.6, "SP-PINN (time-coord)": 2.2}
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7))
    for s in order:
        axes[0].loglog(per, np.maximum(res["dK"][s], 1e-17), color=col[s], ls=ls[s],
                       lw=lw[s], alpha=al[s], zorder=zo[s], label=s)
        axes[1].loglog(per, np.maximum(res["dPx"][s], 1e-17), color=col[s], ls=ls[s],
                       lw=lw[s], alpha=al[s], zorder=zo[s], label=s)
    axes[0].set_xlabel("Wave periods $n$")
    axes[0].set_ylabel(r"$|K-1|$,  $K=\gamma-p_z$ (light-front)")
    axes[0].set_title("(a) time-related invariant")
    axes[1].set_xlabel("Wave periods $n$")
    axes[1].set_ylabel(r"$|P_x(t)-P_x(0)|$ (transverse Noether)")
    axes[1].set_title("(b) transverse canonical momentum")
    for ax in axes:
        ax.set_ylim(1e-17, 1e1)               # extra top decade as clear legend space
        ax.legend(loc="upper right", fontsize=8)
    fig.savefig(os.path.join(_paths.FIG, "figureA1.pdf"))
    fig.savefig(os.path.join(_paths.FIG, "figureA1.png"))
    plt.close(fig)


def main():
    res = run()
    make_figure(res)
    with open(os.path.join(_paths.DATA, "sim6_planewave.csv"), "w", newline="") as f:
        w = csv.writer(f)
        cols = ["period"]
        for s in res["dK"]:
            cols += [f"dK_{s}", f"dPx_{s}"]
        w.writerow(cols)
        per = res["periods"]
        for i in range(0, len(per), max(1, len(per) // 300)):
            row = [per[i]]
            for s in res["dK"]:
                row += [res["dK"][s][i], res["dPx"][s][i]]
            w.writerow(row)
    print("=== Simulation 6: plane wave (points 1 & 7) ===")
    for s in res["dK"]:
        print(f"  {s:24s}: max|K-1|={res['dK'][s].max():.3e}  "
              f"max|dPx|={res['dPx'][s].max():.3e}")
    print("  params:", res["params"])
    return res


if __name__ == "__main__":
    main()
