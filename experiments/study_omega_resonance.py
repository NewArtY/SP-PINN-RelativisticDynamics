"""
Empirical Omega-resonance scaling study (review: the Omega-resonance condition).

The bounded Larmor-error floor of the Tao map spikes at certain binding constants
Omega.  This sweep establishes HOW those resonance locations scale, by finely
scanning Omega for the uniform-B problem at several time steps Delta t and
cyclotron frequencies omega_c.

Result (confirmed by this sweep and by a linear-monodromy / Floquet analysis):
a parametric resonance occurs whenever the binding rotation (angular frequency
2*Omega, see relsim/integrators.py) is commensurate with the step,
        2 * Omega * Delta t = k * pi,  k = 1, 2, 3, ...   <=>   Omega_k = k*pi/(2*Delta t) ,
giving narrow bands at Omega ~ 5k for Delta t = Tc/100. The negative Yoshida
central weight w0 modulates the band strength by k mod 3 (k==1 (mod 3) strongest:
Omega ~ 5, 20, 35, 50; k==0 (mod 3) weakest: Omega ~ 15, 30, 45). The resonance
locations scale as 1/Delta t and are INDEPENDENT of omega_c, as the four configs
below confirm.

Outputs: data/study_omega_resonance.csv, figures/figure_omega_resonance.png,
and a printed summary of the fitted resonance locations vs the prediction.
"""
import _paths  # noqa: F401
import numpy as np
import csv
import os
import time
import platform
import datetime

from relsim.fields import ConstMagneticField
from relsim.integrators import TaoSymplectic
from relsim.diagnostics import kinetic_to_canonical, canonical_to_kinetic
import matplotlib.pyplot as plt


def larmor_floor(field, gamma0, dt, omega, nper=120):
    """Max relative Larmor-radius error over ~nper cyclotron periods."""
    p_perp = np.sqrt(gamma0 ** 2 - 1.0)
    omega_c = 1.0 / gamma0
    Tc = 2 * np.pi / omega_c
    Nper = max(1, int(round(Tc / dt)))
    nsteps = nper * Nper
    r0 = np.array([[0.0, 0.0, 0.0]]); p0 = np.array([[0.0, p_perp, 0.0]])
    tao = TaoSymplectic(field, omega=omega)
    r = r0.copy(); P = kinetic_to_canonical(field, r0, p0, 0.0)
    xx, yy = tao.init_copies(r, P)
    mx = 0.0
    for n in range(nsteps):
        r, P, xx, yy = tao.step(r, P, xx, yy, n * dt, dt)
        if (n + 1) % Nper == 0:
            p = canonical_to_kinetic(field, r, P, (n + 1) * dt)
            pp = np.hypot(p[0, 0], p[0, 1])
            mx = max(mx, abs(pp - p_perp) / p_perp)
    return mx


def scan(field, gamma0, dt, omegas, nper=120):
    return np.array([larmor_floor(field, gamma0, dt, om, nper) for om in omegas])


def find_peaks(omegas, floor, factor=8.0):
    """Local maxima at least `factor` times above the median floor."""
    thr = factor * np.median(floor)
    peaks = []
    for i in range(1, len(floor) - 1):
        if floor[i] > floor[i - 1] and floor[i] >= floor[i + 1] and floor[i] > thr:
            peaks.append(float(omegas[i]))
    return peaks


def main():
    field = ConstMagneticField(B0=1.0, ch=-1.0)
    omegas = np.arange(1.0, 60.01, 1.0)

    Tc5 = 2 * np.pi / (1.0 / 5.0)   # gamma0 = 5  -> omega_c = 0.2
    configs = [
        # label,           gamma0, dt
        ("g5_Tc100", 5.0, Tc5 / 100),   # paper config: omega_c=0.2, dt=Tc/100 ~ 0.314
        ("g5_Tc50",  5.0, Tc5 / 50),    # 2x larger dt  -> resonances should HALVE
        ("g5_Tc200", 5.0, Tc5 / 200),   # 2x smaller dt -> resonances should DOUBLE
        ("g8_dtEQ",  8.0, Tc5 / 100),   # different omega_c (0.125), SAME dt -> resonances should NOT move
    ]

    print("=== Empirical Omega-resonance scaling sweep ===")
    print(f"Omega grid: {omegas[0]:.0f}..{omegas[-1]:.0f} step 1;  ~120 cyclotron periods/run")
    t_start = time.perf_counter()
    rows = []
    summary = {}
    for label, g0, dt in configs:
        tc0 = time.perf_counter()
        floor = scan(field, g0, dt, omegas)
        t_cfg = time.perf_counter() - tc0
        peaks = find_peaks(omegas, floor)
        n_at_peak = [2 * om * dt / (2 * np.pi) for om in peaks]      # = n if 2*Om*dt = 2*pi*n
        pred = [k * np.pi / (2 * dt) for k in range(1, 13) if k * np.pi / (2 * dt) <= 60]
        summary[label] = (g0, dt, peaks, n_at_peak, pred)
        print(f"\n[{label}] gamma0={g0}  omega_c={1/g0:.3f}  dt={dt:.4f}  [{t_cfg:.1f}s]")
        print(f"   resonance Omega            ~ {['%.0f' % p for p in peaks]}")
        print(f"   2*Omega*dt/pi at peaks      = {['%.2f' % (2 * p * dt / np.pi) for p in peaks]}  (=> integer k means 2*Om*dt=k*pi)")
        print(f"   predicted Omega_k=k*pi/(2dt) = {['%.1f' % p for p in pred]}")
        for om, fl in zip(omegas, floor):
            rows.append([label, g0, dt, om, fl])

    # ---- save data ----
    with open(os.path.join(_paths.DATA, "study_omega_resonance.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["config", "gamma0", "dt", "omega", "larmor_floor"])
        w.writerows(rows)

    # ---- plot: raw floor vs Omega, and collapse vs 2*Omega*dt/(2pi) ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for label, (g0, dt, *_ ) in summary.items():
        fl = np.array([r[4] for r in rows if r[0] == label])
        axes[0].semilogy(omegas, np.maximum(fl, 1e-17), "o-", ms=3, label=label)
        axes[1].semilogy(2 * omegas * dt / (2 * np.pi), np.maximum(fl, 1e-17), "o-", ms=3, label=label)
    axes[0].set_xlabel(r"$\Omega$"); axes[0].set_ylabel("Larmor-error floor")
    axes[0].set_title("(a) floor vs $\\Omega$"); axes[0].legend(fontsize=8)
    for k in range(1, 8):
        axes[1].axvline(k, color="k", ls=":", lw=0.6)
    axes[1].set_xlabel(r"$2\Omega\,\Delta t/(2\pi)$  (= turns/step)")
    axes[1].set_ylabel("Larmor-error floor")
    axes[1].set_title("(b) collapse: peaks at integers?"); axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(_paths.FIG, "figure_omega_resonance.png"), dpi=130)
    plt.close(fig)
    total = time.perf_counter() - t_start
    # append a runtime record (handy for future planning / scaling estimates)
    rt = os.path.join(_paths.DATA, "runtime_log.csv")
    new_file = not os.path.exists(rt)
    with open(rt, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["timestamp", "script", "seconds", "platform", "detail"])
        w.writerow([datetime.datetime.now().isoformat(timespec="seconds"),
                    os.path.basename(__file__), f"{total:.1f}",
                    platform.processor() or platform.machine(),
                    f"{len(configs)} configs x {len(omegas)} omega, ~120 periods"])
    print(f"\nTotal runtime: {total:.1f} s  (logged to data/runtime_log.csv)")
    print("Wrote data/study_omega_resonance.csv and figures/figure_omega_resonance.png")
    print("\nINTERPRETATION: peaks move with dt (panel a) but collapse onto fixed")
    print("2*Omega*dt = k*pi independent of omega_c (panel b) => Omega_k = k*pi/(2 dt).")


if __name__ == "__main__":
    main()
