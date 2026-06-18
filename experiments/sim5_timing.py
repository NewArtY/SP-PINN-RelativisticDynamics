"""
Simulation 5 -- Computational cost (Figure 5, Table 3).

Wall-clock time per integration step for Boris, RK4 and the symplectic
SP-PINN (Tao) map, as a function of the number of particles integrated in
parallel (batched NumPy), measured on the constant-magnetic-field problem.
"""
import _paths  # noqa: F401
import numpy as np
import csv
import os
import time

from relsim.fields import ConstMagneticField
from relsim.integrators import boris_step, rk4_step, TaoSymplectic
from relsim.diagnostics import kinetic_to_canonical, canonical_to_kinetic
import matplotlib.pyplot as plt
from relsim.plotstyle import apply_style, COLORS


def time_scheme(field, scheme, Np, nsteps=2000, dt=0.05):
    rng = np.random.default_rng(0)
    r = rng.standard_normal((Np, 3)) * 0.1
    p = np.zeros((Np, 3)); p[:, 1] = 4.899
    if scheme == "SP-PINN":
        tao = TaoSymplectic(field, omega=30.0)
        P = kinetic_to_canonical(field, r, p, 0.0)
        xx, yy = tao.init_copies(r, P)
    # warm-up
    t0 = time.perf_counter()
    for n in range(nsteps):
        t = n * dt
        if scheme == "Boris":
            r, p = boris_step(field, r, p, t, dt)
        elif scheme == "RK4":
            r, p = rk4_step(field, r, p, t, dt)
        else:
            r, P, xx, yy = tao.step(r, P, xx, yy, t, dt)
    t1 = time.perf_counter()
    return (t1 - t0) / nsteps * 1e3            # ms per step (whole batch)


def run():
    field = ConstMagneticField(B0=1.0, ch=-1.0)
    Nps = [1, 10, 100, 1000]
    res = {s: [] for s in ["Boris", "RK4", "SP-PINN"]}
    for Np in Nps:
        for s in res:
            res[s].append(time_scheme(field, s, Np))
    return Nps, res


def make_figure(Nps, res):
    apply_style()
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for s in ["Boris", "RK4", "SP-PINN"]:
        ax.loglog(Nps, res[s], "o-", color=COLORS[s], label=s)
    ax.set_xlabel(r"Number of particles $N_p$")
    ax.set_ylabel(r"Wall-clock time per step [ms]")
    ax.legend(loc="best")
    fig.savefig(os.path.join(_paths.FIG, "figure6.pdf"))  # manuscript Figure 6 (computational cost)
    fig.savefig(os.path.join(_paths.FIG, "figure6.png"))
    plt.close(fig)


def main():
    Nps, res = run()
    make_figure(Nps, res)
    with open(os.path.join(_paths.DATA, "sim5_timing.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["Np", "t_Boris_ms", "t_RK4_ms", "t_SPPINN_ms"])
        for i, Np in enumerate(Nps):
            w.writerow([Np, res["Boris"][i], res["RK4"][i], res["SP-PINN"][i]])
    print("=== Simulation 5: timing (ms per step, whole batch) ===")
    print(f"  {'Np':>6} {'Boris':>10} {'RK4':>10} {'SP-PINN':>10}")
    for i, Np in enumerate(Nps):
        print(f"  {Np:>6} {res['Boris'][i]:>10.4f} {res['RK4'][i]:>10.4f} {res['SP-PINN'][i]:>10.4f}")
    # per-particle at Np=1
    print(f"  per-step @Np=1: Boris {res['Boris'][0]:.4f} ms, RK4 {res['RK4'][0]:.4f} ms, "
          f"SP-PINN {res['SP-PINN'][0]:.4f} ms")
    return Nps, res


if __name__ == "__main__":
    main()
