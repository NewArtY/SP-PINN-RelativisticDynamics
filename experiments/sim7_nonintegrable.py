"""
Simulation 7 -- Non-integrable autonomous system: where symplecticity is
decisive (Figure 6; addresses review point 6).

A charged particle moves in a uniform magnetic field B = B0 z plus a static
anharmonic electrostatic well phi = 0.5*kperp*(x^2+y^2) + eps*x^2*y^2.  The
coupling eps*x^2*y^2 makes the system NON-INTEGRABLE (chaotic at moderate
energy), while the total energy H = gamma + ch*phi is an exact constant of
motion.  Unlike the uniform-field gyration of Test Case 2, there is no exact
volume-preserving rotation to exploit here, so this is the regime in which a
symplectic integrator is genuinely advantageous: it keeps the energy error
BOUNDED for all time, whereas RK4 and the Boris pusher drift secularly.
"""
import _paths  # noqa: F401
import numpy as np
import csv
import os

from relsim.fields import MagneticWell
from relsim.integrators import boris_step, rk4_step, TaoSymplectic
from relsim.diagnostics import kinetic_to_canonical, canonical_to_kinetic
import matplotlib.pyplot as plt
from relsim.plotstyle import apply_style, COLORS, LINESTYLES


def run(B0=1.0, kperp=1.0, eps=0.30, n_steps=300000, dt=0.05, omega=10.0):
    field = MagneticWell(B0=B0, kperp=kperp, eps=eps, ch=1.0)
    r0 = np.array([[1.5, 1.0, 0.0]])
    p0 = np.array([[0.2, 0.0, 0.0]])          # kinetic momentum
    H0 = field.H(r0, kinetic_to_canonical(field, r0, p0, 0.0), 0.0)[0]

    schemes = ["RK4", "Boris", "SP-PINN"]
    sample = max(1, n_steps // 2000)
    dH = {s: [] for s in schemes}
    times = []
    orbit = None

    for s in schemes:
        if s == "SP-PINN":
            tao = TaoSymplectic(field, omega=omega)
            r = r0.copy(); P = kinetic_to_canonical(field, r0, p0, 0.0)
            xx, yy = tao.init_copies(r, P)
            traj = [r0[0, :2].copy()]
        else:
            r, p = r0.copy(), p0.copy()
        for n in range(n_steps):
            t = n * dt
            if s == "RK4":
                r, p = rk4_step(field, r, p, t, dt)
                Hn = field.H(r, kinetic_to_canonical(field, r, p, t + dt), t + dt)[0]
            elif s == "Boris":
                r, p = boris_step(field, r, p, t, dt)
                Hn = field.H(r, kinetic_to_canonical(field, r, p, t + dt), t + dt)[0]
            else:
                r, P, xx, yy = tao.step(r, P, xx, yy, t, dt)
                Hn = field.H(r, P, t + dt)[0]
                if n % 20 == 0 and n * dt < 400:
                    traj.append(r[0, :2].copy())
            if (n + 1) % sample == 0:
                dH[s].append(abs(Hn - H0) / abs(H0))
                if s == schemes[0]:
                    times.append((n + 1) * dt)
        if s == "SP-PINN":
            orbit = np.array(traj)
    return dict(times=np.array(times), dH=dH, orbit=orbit,
                params=dict(B0=B0, kperp=kperp, eps=eps, dt=dt, H0=float(H0),
                            n_steps=n_steps, omega=omega))


def make_figure(res):
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7))
    # (a) chaotic orbit -- neutral colour: this is an illustrative trajectory of
    # the dynamics, not a method comparison (avoid implying it is "the SP-PINN" run)
    ob = res["orbit"]
    axes[0].plot(ob[:, 0], ob[:, 1], color="0.30", lw=0.6)
    axes[0].set_xlabel("$x$"); axes[0].set_ylabel("$y$")
    axes[0].set_title("(a) representative orbit"); axes[0].set_aspect("equal", "box")
    axes[0].grid(alpha=0.3)
    # (b) energy error
    t = res["times"]
    for s in ["RK4", "Boris", "SP-PINN"]:
        axes[1].loglog(t, np.maximum(res["dH"][s], 1e-17), color=COLORS[s], ls=LINESTYLES[s], label=s)
    axes[1].set_xlabel("time $t$")
    axes[1].set_ylabel(r"Relative energy error $|H(t)-H_0|/|H_0|$")
    axes[1].set_title("(b) energy conservation"); axes[1].legend(loc="best")
    fig.savefig(os.path.join(_paths.FIG, "figure6.pdf"))
    fig.savefig(os.path.join(_paths.FIG, "figure6.png"))
    plt.close(fig)


def main():
    res = run()
    make_figure(res)
    with open(os.path.join(_paths.DATA, "sim7_nonintegrable.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["time", "dH_RK4", "dH_Boris", "dH_SPPINN"])
        t = res["times"]
        for i in range(len(t)):
            w.writerow([t[i], res["dH"]["RK4"][i], res["dH"]["Boris"][i], res["dH"]["SP-PINN"][i]])
    print("=== Simulation 7: non-integrable B + anharmonic well (point 6) ===")
    for s in ["RK4", "Boris", "SP-PINN"]:
        print(f"  {s:8s}: dH(final)={res['dH'][s][-1]:.3e}  dH(max)={max(res['dH'][s]):.3e}")
    print("  params:", res["params"])
    return res


if __name__ == "__main__":
    main()
