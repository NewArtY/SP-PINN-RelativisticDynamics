"""
Simulation 3 -- Single electron in a focused Gaussian laser pulse (Figure 3).

The pulse propagates in +z with its envelope centred at z = t (code units
c = 1, k0 = omega0 = 1).  The electron starts at rest at the origin while the
pulse is still far behind it; the pulse overtakes the electron around t = 0.
We compare z(t), gamma(t) and the relative energy error against a DOP853
(RK8) reference for Boris, RK4 and the symplectic SP-PINN (Tao) map.
"""
import _paths  # noqa: F401
import numpy as np
import csv
import os

from relsim.fields import GaussianLaserPulse
from relsim.integrators import boris_step, rk4_step, TaoSymplectic, rk8_reference
from relsim.diagnostics import (gamma_of_p, kinetic_to_canonical,
                                canonical_to_kinetic)
import matplotlib.pyplot as plt
from relsim.plotstyle import apply_style, COLORS


def run(a0=5.0, tau=30.0, dt=0.01):
    field = GaussianLaserPulse(a0=a0, w0=5 * 2 * np.pi, tau=tau, ch=-1.0)
    t0 = -3 * tau
    t1 = 3 * tau
    nsteps = int(round((t1 - t0) / dt))
    times = t0 + dt * np.arange(nsteps + 1)

    r0 = np.array([[0.0, 0.0, 0.0]])
    p0 = np.array([[0.0, 0.0, 0.0]])

    def integrate(scheme):
        z = np.zeros(nsteps + 1); g = np.zeros(nsteps + 1)
        if scheme == "SP-PINN":
            tao = TaoSymplectic(field, omega=40.0)
            r = r0.copy(); P = kinetic_to_canonical(field, r0, p0, t0)
            xx, yy = tao.init_copies(r, P)
            p = canonical_to_kinetic(field, r, P, t0)
        else:
            r, p = r0.copy(), p0.copy()
        z[0] = r[0, 2]; g[0] = gamma_of_p(p)[0]
        for n in range(nsteps):
            t = times[n]
            if scheme == "Boris":
                r, p = boris_step(field, r, p, t, dt)
            elif scheme == "RK4":
                r, p = rk4_step(field, r, p, t, dt)
            else:
                r, P, xx, yy = tao.step(r, P, xx, yy, t, dt)
                p = canonical_to_kinetic(field, r, P, t + dt)
            z[n + 1] = r[0, 2]; g[n + 1] = gamma_of_p(p)[0]
        return z, g

    out = {}
    for s in ["Boris", "RK4", "SP-PINN"]:
        out[s] = integrate(s)

    # RK8 reference
    Rref, Pref = rk8_reference(field, r0[0], p0[0], times)
    g_ref = np.sqrt(1.0 + np.sum(Pref * Pref, axis=1))
    z_ref = Rref[:, 2]
    out["RK8"] = (z_ref, g_ref)
    return times, out, dict(a0=a0, tau=tau, dt=dt, nsteps=nsteps)


def make_figure(times, out, params):
    apply_style()
    plt.rcParams.update({"font.size": 13, "axes.labelsize": 15,
                         "axes.titlesize": 14, "legend.fontsize": 12,
                         "xtick.labelsize": 12, "ytick.labelsize": 12,
                         "lines.linewidth": 2.0})
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5))
    tau = params["tau"]
    tt = times / tau
    # (a) z(t)
    for s in ["RK4", "Boris", "SP-PINN"]:
        axes[0].plot(tt, out[s][0], color=COLORS[s], label=s)
    axes[0].plot(tt, out["RK8"][0], "k--", lw=1.2, label="RK8 ref")
    axes[0].set_xlabel(r"$t/\tau_L$"); axes[0].set_ylabel(r"$z(t)$  [$c/\omega_0$]")
    axes[0].set_title("(a)"); axes[0].legend(loc="best")
    # (b) gamma(t)
    for s in ["RK4", "Boris", "SP-PINN"]:
        axes[1].plot(tt, out[s][1], color=COLORS[s], label=s)
    axes[1].plot(tt, out["RK8"][1], "k--", lw=1.2, label="RK8 ref")
    axes[1].set_xlabel(r"$t/\tau_L$"); axes[1].set_ylabel(r"$\gamma(t)$")
    axes[1].set_title("(b)"); axes[1].legend(loc="best")
    # (c) energy error after the pulse
    g_ref = out["RK8"][1]
    mask = times > tau           # post-interaction
    for s in ["RK4", "Boris", "SP-PINN"]:
        err = np.abs(out[s][1] - g_ref) / np.abs(g_ref)
        axes[2].semilogy(tt[mask], np.maximum(err[mask], 1e-17),
                         color=COLORS[s], label=s)
    axes[2].set_xlabel(r"$t/\tau_L$")
    axes[2].set_ylabel(r"Relative energy error $\Delta\mathcal{E}/\mathcal{E}_0$")
    axes[2].set_title("(c)"); axes[2].legend(loc="best")
    fig.savefig(os.path.join(_paths.FIG, "figure3.pdf"))
    fig.savefig(os.path.join(_paths.FIG, "figure3.png"))
    plt.close(fig)


def main():
    times, out, params = run()
    make_figure(times, out, params)
    g_ref = out["RK8"][1]
    mask = times > params["tau"]
    path = os.path.join(_paths.DATA, "sim3_energy_error.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["t", "dE_Boris", "dE_RK4", "dE_SPPINN"])
        idx = np.where(mask)[0]
        for i in idx[::20]:
            row = [times[i]]
            for s in ["Boris", "RK4", "SP-PINN"]:
                row.append(abs(out[s][1][i] - g_ref[i]) / abs(g_ref[i]))
            w.writerow(row)
    print("=== Simulation 3: Gaussian laser pulse, single electron ===")
    print(f"  peak gamma (RK8 ref): {g_ref.max():.4f}")
    for s in ["Boris", "RK4", "SP-PINN"]:
        err = np.abs(out[s][1][mask] - g_ref[mask]) / np.abs(g_ref[mask])
        print(f"  {s:8s}: max post-pulse energy error = {err.max():.3e}")
    return times, out, params


if __name__ == "__main__":
    main()
