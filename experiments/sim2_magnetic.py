"""
Simulation 2 -- Relativistic gyration in a uniform magnetic field (Figure 2).

A relativistic particle (gamma0 = 5) gyrates in B = B0 z, an integrable
relativistic Hamiltonian system in which the magnetic force does no work, so
the Lorentz factor and the Larmor radius are exact constants of motion.  We
integrate for many cyclotron periods and monitor the relative error of the
Larmor radius and of the Lorentz factor for Boris, RK4 and the symplectic
SP-PINN (Tao) map.

The result is the textbook contrast between BOUNDED and SECULAR error growth:
RK4 (neither symplectic nor volume preserving) drifts secularly; the Boris
pusher is exactly volume preserving for pure gyration and conserves both
invariants to machine precision; and the symplectic SP-PINN map keeps the
error BOUNDED for all time, so that at long integration times it overtakes
RK4 while remaining a structure-preserving scheme applicable to the general
non-separable relativistic Hamiltonian.
"""
import _paths  # noqa: F401
import numpy as np
import csv
import os

from relsim.fields import ConstMagneticField
from relsim.integrators import boris_step, rk4_step, TaoSymplectic
from relsim.diagnostics import (gamma_of_p, kinetic_to_canonical,
                                canonical_to_kinetic)
import matplotlib.pyplot as plt
from relsim.plotstyle import apply_style, COLORS


def run(gamma0=5.0, B0=1.0, n_periods=4000, steps_per_period=100, omega=15.0):
    field = ConstMagneticField(B0=B0, ch=-1.0)
    p_perp = np.sqrt(gamma0 ** 2 - 1.0)
    omega_c = abs(field.ch * B0) / gamma0
    Tc = 2 * np.pi / omega_c
    dt = Tc / steps_per_period
    nsteps = n_periods * steps_per_period
    rL0 = p_perp / abs(field.ch * B0)

    r0 = np.array([[0.0, 0.0, 0.0]])
    p0 = np.array([[0.0, p_perp, 0.0]])

    schemes = ["Boris", "RK4", "SP-PINN"]
    drL = {s: np.zeros(n_periods) for s in schemes}
    dgam = {s: np.zeros(n_periods) for s in schemes}

    for s in schemes:
        r, p = r0.copy(), p0.copy()
        if s == "SP-PINN":
            tao = TaoSymplectic(field, omega=omega)
            P = kinetic_to_canonical(field, r, p, 0.0)
            xx, yy = tao.init_copies(r, P)
        m = 0
        for n in range(nsteps):
            t = n * dt
            if s == "Boris":
                r, p = boris_step(field, r, p, t, dt)
            elif s == "RK4":
                r, p = rk4_step(field, r, p, t, dt)
            else:
                r, P, xx, yy = tao.step(r, P, xx, yy, t, dt)
                p = canonical_to_kinetic(field, r, P, t + dt)
            if (n + 1) % steps_per_period == 0:
                pp = np.sqrt(p[0, 0] ** 2 + p[0, 1] ** 2)
                rL = pp / abs(field.ch * B0)
                drL[s][m] = abs(rL - rL0) / rL0
                dgam[s][m] = abs(gamma_of_p(p)[0] - gamma0) / gamma0
                m += 1

    periods = np.arange(1, n_periods + 1)
    return dict(periods=periods, drL=drL, dgam=dgam,
                params=dict(gamma0=gamma0, B0=B0, dt=dt, Tc=Tc,
                            n_periods=n_periods, omega=omega))


def make_figure(res):
    apply_style()
    per = res["periods"]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7))
    floor = 1e-16
    for s in ["RK4", "Boris", "SP-PINN"]:
        axes[0].loglog(per, np.maximum(res["drL"][s], floor), color=COLORS[s], label=s)
        axes[1].loglog(per, np.maximum(res["dgam"][s], floor), color=COLORS[s], label=s)
    axes[0].set_xlabel("Cyclotron periods $n$")
    axes[0].set_ylabel(r"Relative Larmor-radius error $\Delta r_L/r_L^{(0)}$")
    axes[0].set_title("(a)")
    axes[1].set_xlabel("Cyclotron periods $n$")
    axes[1].set_ylabel(r"Relative Lorentz-factor error $\Delta\gamma/\gamma_0$")
    axes[1].set_title("(b)")
    for ax in axes:
        ax.legend(loc="upper left")
        ax.set_ylim(1e-16, 1e-1)
    fig.savefig(os.path.join(_paths.FIG, "figure2.pdf"))
    fig.savefig(os.path.join(_paths.FIG, "figure2.png"))
    plt.close(fig)


def main():
    res = run()
    make_figure(res)
    per = res["periods"]
    with open(os.path.join(_paths.DATA, "sim2_magnetic.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["period", "drL_Boris", "drL_RK4", "drL_SPPINN",
                    "dgamma_Boris", "dgamma_RK4", "dgamma_SPPINN"])
        for i in range(0, len(per), max(1, len(per) // 400)):
            w.writerow([per[i], res["drL"]["Boris"][i], res["drL"]["RK4"][i],
                        res["drL"]["SP-PINN"][i], res["dgam"]["Boris"][i],
                        res["dgam"]["RK4"][i], res["dgam"]["SP-PINN"][i]])
    print("=== Simulation 2: magnetic field (long-time) ===")
    for s in ["Boris", "RK4", "SP-PINN"]:
        print(f"  {s:8s}: drL(final)={res['drL'][s][-1]:.3e}  "
              f"dgamma(final)={res['dgam'][s][-1]:.3e}  "
              f"drL(max)={res['drL'][s].max():.3e}")
    # find RK4 crossover over SP-PINN floor
    rk4 = res["drL"]["RK4"]; spp = res["drL"]["SP-PINN"]
    cross = np.where(rk4 > spp)[0]
    if len(cross):
        print(f"  RK4 Larmor error exceeds SP-PINN beyond ~{res['periods'][cross[0]]} periods")
    print("  params:", res["params"])
    return res


if __name__ == "__main__":
    main()
