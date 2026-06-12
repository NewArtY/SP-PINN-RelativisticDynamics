"""
SP-PINN Stage 1 demonstrator: train the unsupervised PINN Hamiltonian
surrogate for the constant-magnetic-field problem, report the training error
eps_theta, then run the symplectic Stage-2 (Tao) map driven by the *learned*
Hamiltonian and compare the Larmor-radius drift against the analytic-H
symplectic run.  This quantifies the realistic eps_theta error floor of the
fully learned integrator.
"""
import _paths  # noqa: F401
import numpy as np
import time
import os
import csv

from relsim.fields import ConstMagneticField
from relsim.integrators import TaoSymplectic
from relsim.diagnostics import gamma_of_p, kinetic_to_canonical, canonical_to_kinetic


def main():
    field = ConstMagneticField(B0=1.0, ch=-1.0)
    try:
        from relsim.pinn import train_pinn, PINNField
    except RuntimeError as e:
        print("PINN demo skipped:", e)
        return

    domain = {"r": 6.0, "P": 6.0}
    t0 = time.perf_counter()
    model, history, eps = train_pinn(field, domain, n_coll=8000, epochs=8000,
                                     verbose=True)
    t_train = time.perf_counter() - t0
    print(f"\nPINN training time: {t_train:.1f} s")
    print(f"Training error on a fresh test set over the domain:")
    print(f"   RMS |H - gamma|     = {eps['rmsH']:.3e}   (max {eps['H']:.3e})")
    print(f"   RMS |grad_P H - v|  = {eps['rmsgradP']:.3e}   (max {eps['gradP']:.3e})")
    print(f"   RMS |grad_r H - f_r|= {eps['rmsgradr']:.3e}   (max {eps['gradr']:.3e})")
    print(f"   eps_theta (RMS)     = {eps['eps_theta']:.3e}")
    print(f"   eps_theta (max)     = {eps['eps_theta_max']:.3e}")

    # write eps file immediately (independent of the slow Stage-2 below)
    with open(os.path.join(_paths.DATA, "pinn_eps_theta.txt"), "w") as f:
        f.write(f"training_time_s {t_train:.2f}\n")
        for k, v in eps.items():
            f.write(f"{k} {v:.6e}\n")

    if eps["eps_theta"] > 0.1:
        print(f"\n[skip] eps_theta = {eps['eps_theta']:.2e} > 0.1: the learned "
              f"Hamiltonian is too inaccurate for a stable Stage-2 integration "
              f"on this domain. Reduce the domain or train longer. "
              f"This is the documented current bottleneck of the learned pipeline.")
        return

    # Stage 2 with learned Hamiltonian
    pf = PINNField(model, field)
    gamma0 = 5.0
    p_perp = np.sqrt(gamma0 ** 2 - 1.0)
    r = np.array([[0.0, 0.0, 0.0]])
    p = np.array([[0.0, p_perp, 0.0]])
    rL0 = p_perp
    n_periods = 60
    omega_c = 1.0 / gamma0
    Tc = 2 * np.pi / omega_c
    dt = Tc / 100
    nsteps = n_periods * 100

    tao = TaoSymplectic(pf, omega=30.0)
    P = kinetic_to_canonical(field, r, p, 0.0)
    xx, yy = tao.init_copies(r, P)
    drift = []
    for n in range(nsteps):
        r, P, xx, yy = tao.step(r, P, xx, yy, n * dt, dt)
        if (n + 1) % 100 == 0:
            pk = canonical_to_kinetic(field, r, P, (n + 1) * dt)
            pperp = np.sqrt(pk[0, 0] ** 2 + pk[0, 1] ** 2)
            drift.append(abs(pperp / 1.0 - rL0) / rL0)
    print(f"\nSP-PINN (learned H) Larmor drift after {n_periods} periods: "
          f"{drift[-1]:.3e}  (floor set by eps_theta ~ {eps['eps_theta']:.1e})")

    with open(os.path.join(_paths.DATA, "pinn_training.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["epoch", "loss", "L_constraint", "L_eqs"])
        for row in history:
            w.writerow(row)
    with open(os.path.join(_paths.DATA, "pinn_eps_theta.txt"), "w") as f:
        f.write(f"training_time_s {t_train:.2f}\n")
        for k, v in eps.items():
            f.write(f"{k} {v:.6e}\n")
        f.write(f"larmor_drift_200periods {drift[-1]:.6e}\n")


if __name__ == "__main__":
    main()
