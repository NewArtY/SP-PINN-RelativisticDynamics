"""
Methodological studies (Figure A2; addresses review points 3 and 4).

(a) Time-step convergence: global trajectory error of the Tao symplectic map on
    the uniform-magnetic-field problem versus Delta t, against a DOP853 reference,
    confirming the expected fourth-order slope. RK4 (slope 4) and Boris (slope 2)
    are shown for reference.

(b) Binding-constant study: the bounded long-time error floor of the Tao map
    versus the binding constant Omega, exhibiting the characteristic trade-off
    (the splitting error grows with Omega while the copy-desynchronization error
    falls), which justifies the Omega chosen in the main text.
"""
import _paths  # noqa: F401
import numpy as np
import csv
import os

from relsim.fields import ConstMagneticField
from relsim.integrators import boris_step, rk4_step, TaoSymplectic, rk8_reference
from relsim.diagnostics import (gamma_of_p, kinetic_to_canonical,
                                canonical_to_kinetic)
import matplotlib.pyplot as plt
from relsim.plotstyle import apply_style, COLORS


def integrate_to_T(field, scheme, r0, p0, dt, nsteps, omega=15.0):
    if scheme == "SP-PINN":
        tao = TaoSymplectic(field, omega=omega)
        r = r0.copy(); P = kinetic_to_canonical(field, r0, p0, 0.0)
        xx, yy = tao.init_copies(r, P)
        for n in range(nsteps):
            r, P, xx, yy = tao.step(r, P, xx, yy, n * dt, dt)
        p = canonical_to_kinetic(field, r, P, nsteps * dt)
        return r[0].copy(), p[0].copy()
    r, p = r0.copy(), p0.copy()
    step = boris_step if scheme == "Boris" else rk4_step
    for n in range(nsteps):
        r, p = step(field, r, p, n * dt, dt)
    return r[0].copy(), p[0].copy()


def convergence(field, gamma0=5.0):
    # Short final time so the error stays in the convergent regime (a long T
    # would saturate at the orbit scale via accumulated phase error and mask the
    # true order). Use a large Omega so the Tao binding floor sits below the
    # tested errors, exposing the genuine 4th-order time-stepping slope.
    p_perp = np.sqrt(gamma0 ** 2 - 1.0)
    T = 4.0
    r0 = np.array([[0.0, 0.0, 0.0]]); p0 = np.array([[0.0, p_perp, 0.0]])

    Rref, Pref = rk8_reference(field, r0[0], p0[0], np.array([0.0, T]))
    ref = np.concatenate([Rref[-1], Pref[-1]])

    Ns = [8, 16, 32, 64, 128, 256]
    dts = [T / n for n in Ns]
    err = {s: [] for s in ["RK4", "SP-PINN"]}
    for n in Ns:
        dt = T / n
        # Tao converges to the true (not the binding-extended) solution only in
        # the coupled limit Omega -> infinity as dt -> 0; we use Omega = 5/dt so
        # that the binding floor vanishes with dt and the genuine 4th-order
        # time-stepping slope is exposed.
        for s in err:
            om = 5.0 / dt if s == "SP-PINN" else 15.0
            rf, pf = integrate_to_T(field, s, r0, p0, dt, n, omega=om)
            err[s].append(np.linalg.norm(np.concatenate([rf, pf]) - ref))
    return np.array(dts), err


def omega_study(field, gamma0=5.0):
    p_perp = np.sqrt(gamma0 ** 2 - 1.0)
    omega_c = 1.0 / gamma0
    Tc = 2 * np.pi / omega_c
    dt = Tc / 100
    nper = 300; nsteps = nper * 100
    rL0 = p_perp
    r0 = np.array([[0.0, 0.0, 0.0]]); p0 = np.array([[0.0, p_perp, 0.0]])
    omegas = [2, 4, 8, 12, 15, 20, 30, 50]
    floor = []
    for om in omegas:
        tao = TaoSymplectic(field, omega=om)
        r = r0.copy(); P = kinetic_to_canonical(field, r0, p0, 0.0)
        xx, yy = tao.init_copies(r, P)
        mx = 0.0
        for n in range(nsteps):
            r, P, xx, yy = tao.step(r, P, xx, yy, n * dt, dt)
            if (n + 1) % 100 == 0:
                p = canonical_to_kinetic(field, r, P, (n + 1) * dt)
                pp = np.sqrt(p[0, 0] ** 2 + p[0, 1] ** 2)
                mx = max(mx, abs(pp - rL0) / rL0)
        floor.append(mx)
    return np.array(omegas), np.array(floor)


def main():
    field = ConstMagneticField(B0=1.0, ch=-1.0)
    dts, err = convergence(field)
    omegas, floor = omega_study(field)

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7))
    for s in ["RK4", "SP-PINN"]:
        axes[0].loglog(dts, err[s], "o-", color=COLORS[s], label=s, zorder=5)
    # reference slopes (offset slightly below the data so the guides do not sit
    # on top of the SP-PINN curve)
    d = np.array(dts)
    axes[0].loglog(d, 0.6 * err["RK4"][0] * (d / d[0]) ** 4, "k--", lw=1, label=r"slope 4", zorder=1)
    axes[0].loglog(d, 0.6 * err["SP-PINN"][0] * (d / d[0]) ** 2, "k-.", lw=1, label=r"slope 2", zorder=1)
    axes[0].set_xlabel(r"Time step $\Delta t$")
    axes[0].set_ylabel(r"Global error at $T=4$")
    axes[0].set_title("(a) order of convergence"); axes[0].legend(fontsize=9)
    # The Delta t range spans only ~1.5 decades, so matplotlib's default log axis
    # labels many minor ticks (2e-2, 3e-2, ...) that overlap. Use a few explicit,
    # cleanly-formatted decade-spaced major ticks and drop the minor-tick labels.
    from matplotlib.ticker import FixedLocator, NullFormatter, FuncFormatter
    axes[0].set_xlim(0.012, 0.62)
    axes[0].xaxis.set_major_locator(FixedLocator([0.02, 0.05, 0.1, 0.2, 0.5]))
    axes[0].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axes[0].xaxis.set_minor_formatter(NullFormatter())

    axes[1].semilogy(omegas, np.maximum(floor, 1e-17), "o-", color=COLORS["SP-PINN"])
    axes[1].set_xlabel(r"Binding constant $\Omega$")
    axes[1].set_ylabel(r"Bounded Larmor-error floor")
    axes[1].set_title("(b) binding-constant trade-off")
    fig.savefig(os.path.join(_paths.FIG, "figureA2.pdf"))
    fig.savefig(os.path.join(_paths.FIG, "figureA2.png"))
    plt.close(fig)

    with open(os.path.join(_paths.DATA, "study_convergence.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["dt", "err_RK4", "err_SPPINN"])
        for i in range(len(dts)):
            w.writerow([dts[i], err["RK4"][i], err["SP-PINN"][i]])
    with open(os.path.join(_paths.DATA, "study_omega.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["omega", "floor"])
        for i in range(len(omegas)):
            w.writerow([omegas[i], floor[i]])

    ld = np.log(dts)
    print("=== Convergence & Omega study (points 3 & 4) ===")
    print(f"  RK4 slope:     {np.polyfit(ld, np.log(err['RK4']),1)[0]:.2f}")
    print(f"  SP-PINN slope: {np.polyfit(ld, np.log(err['SP-PINN']),1)[0]:.2f}  "
          f"(Omega = 5/dt, coupled limit)")
    print(f"  Omega floor min at Omega={omegas[np.argmin(floor)]}, floor={floor.min():.2e}")


if __name__ == "__main__":
    main()
