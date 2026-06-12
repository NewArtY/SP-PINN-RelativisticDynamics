"""
Simulation 4 -- Electron ensemble in a Gaussian laser pulse (Figure 4).

(a) Kinetic-energy spectrum dN/dgamma of an ensemble of electrons with
    uniformly distributed initial transverse positions, after the pulse has
    passed, for Boris, RK4, SP-PINN and a fine-step reference.
(b) Numerical violation of the transverse canonical momentum
    P_x = p_x + ch*A_x, the Noether invariant of the transverse translational
    symmetry of the field (exactly conserved in the plane-wave limit).  We
    plot |P_x^scheme(t) - P_x^ref(t)| for a single on-axis electron against a
    DOP853 (RK8) reference -- a direct measure of how well each integrator
    preserves the field's translational symmetry.
"""
import _paths  # noqa: F401
import numpy as np
import csv
import os

from relsim.fields import GaussianLaserPulse
from relsim.integrators import boris_step, rk4_step, TaoSymplectic, rk8_reference
from relsim.diagnostics import gamma_of_p, kinetic_to_canonical, canonical_to_kinetic
import matplotlib.pyplot as plt
from relsim.plotstyle import apply_style, COLORS, LINESTYLES


def integrate_ensemble(field, r0, p0, times, dt, scheme):
    n = len(times) - 1
    r = r0.copy(); p = p0.copy()
    if scheme == "SP-PINN":
        tao = TaoSymplectic(field, omega=40.0)
        P = kinetic_to_canonical(field, r, p, times[0])
        xx, yy = tao.init_copies(r, P)
    for k in range(n):
        t = times[k]
        if scheme == "Boris":
            r, p = boris_step(field, r, p, t, dt)
        elif scheme == "RK4":
            r, p = rk4_step(field, r, p, t, dt)
        else:
            r, P, xx, yy = tao.step(r, P, xx, yy, t, dt)
            p = canonical_to_kinetic(field, r, P, t + dt)
    return r, p


def run(a0=5.0, tau=30.0, Np=256, dt=0.02):
    field = GaussianLaserPulse(a0=a0, w0=5 * 2 * np.pi, tau=tau, ch=-1.0)
    w0 = field.w0
    t0 = -3 * tau; t1 = 3 * tau
    nsteps = int(round((t1 - t0) / dt))
    times = t0 + dt * np.arange(nsteps + 1)

    rng = np.random.default_rng(1)
    xy = (rng.random((Np, 2)) * 2 - 1) * w0
    r0 = np.zeros((Np, 3)); r0[:, 0] = xy[:, 0]; r0[:, 1] = xy[:, 1]
    p0 = np.zeros((Np, 3))

    spectra = {}
    for s in ["Boris", "RK4", "SP-PINN"]:
        _, p = integrate_ensemble(field, r0.copy(), p0.copy(), times, dt, s)
        spectra[s] = gamma_of_p(p)

    # reference: fine-step RK4 (dt/5)
    dt_ref = dt / 5.0
    nref = int(round((t1 - t0) / dt_ref))
    times_ref = t0 + dt_ref * np.arange(nref + 1)
    _, p_ref = integrate_ensemble(field, r0.copy(), p0.copy(), times_ref, dt_ref, "RK4")
    spectra["RK8"] = gamma_of_p(p_ref)

    # ---- panel (b): single on-axis electron, transverse canonical momentum
    r0s = np.array([[0.0, 0.0, 0.0]]); p0s = np.array([[0.0, 0.0, 0.0]])
    # RK8 reference trajectory (kinetic), then canonical P_x = p_x + ch A_x at
    # each sampled time
    Rref, Pkin_ref = rk8_reference(field, r0s[0], p0s[0], times)
    Px_ref = np.array([Pkin_ref[i, 0] + field.ch *
                       field.A(Rref[i][None, :], times[i])[0, 0]
                       for i in range(len(times))])

    Pxerr = {}
    for s in ["Boris", "RK4", "SP-PINN"]:
        r = r0s.copy(); p = p0s.copy()
        if s == "SP-PINN":
            tao = TaoSymplectic(field, omega=40.0)
            P = kinetic_to_canonical(field, r, p, times[0])
            xx, yy = tao.init_copies(r, P)
        err = np.zeros(len(times))
        err[0] = 0.0
        for k in range(nsteps):
            t = times[k]
            if s == "Boris":
                r, p = boris_step(field, r, p, t, dt)
            elif s == "RK4":
                r, p = rk4_step(field, r, p, t, dt)
            else:
                r, P, xx, yy = tao.step(r, P, xx, yy, t, dt)
                p = canonical_to_kinetic(field, r, P, t + dt)
            Px = p[0, 0] + field.ch * field.A(r, t + dt)[0, 0]
            err[k + 1] = abs(Px - Px_ref[k + 1])
        Pxerr[s] = err

    return times, tau, spectra, Pxerr


def make_figure(times, tau, spectra, Pxerr):
    apply_style()
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 13,
                         "legend.fontsize": 11, "xtick.labelsize": 11,
                         "ytick.labelsize": 11})
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    # (a) spectrum -- draw histograms, then build a clean Line2D legend so the
    # reference handle is a dashed line rather than a patch box
    from matplotlib.lines import Line2D
    gmax = max(spectra["RK8"].max(), 1.0)
    bins = np.linspace(1.0, gmax * 1.05, 30)
    handles = []
    for s in ["RK8", "RK4", "Boris", "SP-PINN"]:
        if s == "RK8":
            axes[0].hist(spectra[s], bins=bins, histtype="step",
                         color="k", ls=LINESTYLES["RK8"], lw=1.3)
            handles.append(Line2D([0], [0], color="k", ls=LINESTYLES["RK8"], lw=1.3,
                                  label="reference (curves overlap)"))
        else:
            axes[0].hist(spectra[s], bins=bins, histtype="step",
                         color=COLORS[s], ls=LINESTYLES[s], lw=1.6)
            handles.append(Line2D([0], [0], color=COLORS[s], ls=LINESTYLES[s], lw=1.6, label=s))
    axes[0].set_xlabel(r"$\gamma_{\rm final}$")
    axes[0].set_ylabel(r"$dN/d\gamma$  (counts)")
    axes[0].set_title("(a)"); axes[0].legend(handles=handles, loc="upper right")
    # (b) transverse canonical momentum error
    tt = times / tau
    for s in ["RK4", "Boris", "SP-PINN"]:
        axes[1].semilogy(tt, np.maximum(Pxerr[s], 1e-17), color=COLORS[s], ls=LINESTYLES[s], label=s)
    axes[1].set_xlabel(r"$t/\tau_L$")
    axes[1].set_ylabel(r"$|\,P_x(t)-P_x^{\rm ref}(t)\,|$")
    axes[1].set_title("(b)"); axes[1].legend(loc="lower right")
    fig.savefig(os.path.join(_paths.FIG, "figure4.pdf"))
    fig.savefig(os.path.join(_paths.FIG, "figure4.png"))
    plt.close(fig)


def main():
    times, tau, spectra, Pxerr = run()
    make_figure(times, tau, spectra, Pxerr)
    with open(os.path.join(_paths.DATA, "sim4_Px_violation.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["t", "Pxerr_Boris", "Pxerr_RK4", "Pxerr_SPPINN"])
        for i in range(0, len(times), 20):
            w.writerow([times[i], Pxerr["Boris"][i], Pxerr["RK4"][i], Pxerr["SP-PINN"][i]])
    print("=== Simulation 4: ensemble spectrum + transverse-momentum symmetry ===")
    print(f"  reference <gamma> = {spectra['RK8'].mean():.4f}, peak = {spectra['RK8'].max():.3f}")
    for s in ["Boris", "RK4", "SP-PINN"]:
        d = abs(spectra[s].mean() - spectra["RK8"].mean()) / spectra["RK8"].mean()
        print(f"  {s:8s}: <gamma> rel. error = {d:.3e}, "
              f"max Px error = {Pxerr[s].max():.3e}")
    return times, tau, spectra, Pxerr


if __name__ == "__main__":
    main()
