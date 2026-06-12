"""
Simulation 1 -- Free relativistic particle (Table 2).

A free particle (E = B = 0) moves with constant Lorentz factor.  We integrate
for N steps with Boris, RK4 and the symplectic SP-PINN (Tao) map and record
the drift of gamma and of |p|^2.  Because H = sqrt(|p|^2+1) depends only on
the (conserved) momentum, the symplectic map keeps gamma constant to machine
precision; RK4 drifts and Boris keeps a bounded error.
"""
import _paths  # noqa: F401
import numpy as np
import csv
import os

from relsim.fields import FreeField
from relsim.integrators import boris_step, rk4_step, TaoSymplectic
from relsim.diagnostics import gamma_of_p, kinetic_to_canonical, canonical_to_kinetic


def run(gamma0=10.0, dt=0.01, nsteps=10000):
    field = FreeField(ch=-1.0)
    pz = np.sqrt(gamma0 ** 2 - 1.0)
    r0 = np.array([[0.0, 0.0, 0.0]])
    p0 = np.array([[0.0, 0.0, pz]])      # kinetic momentum
    p2_0 = pz ** 2

    # --- Boris ---
    r, p = r0.copy(), p0.copy()
    dg_boris = np.zeros(nsteps); dp_boris = np.zeros(nsteps)
    for n in range(nsteps):
        r, p = boris_step(field, r, p, n * dt, dt)
        dg_boris[n] = abs(gamma_of_p(p)[0] - gamma0)
        dp_boris[n] = abs(np.sum(p * p) - p2_0) / p2_0

    # --- RK4 ---
    r, p = r0.copy(), p0.copy()
    dg_rk4 = np.zeros(nsteps); dp_rk4 = np.zeros(nsteps)
    for n in range(nsteps):
        r, p = rk4_step(field, r, p, n * dt, dt)
        dg_rk4[n] = abs(gamma_of_p(p)[0] - gamma0)
        dp_rk4[n] = abs(np.sum(p * p) - p2_0) / p2_0

    # --- SP-PINN (Tao symplectic on analytic H) ---
    tao = TaoSymplectic(field, omega=20.0)
    r = r0.copy(); P = kinetic_to_canonical(field, r0, p0, 0.0)
    x, y = tao.init_copies(r, P)
    dg_sp = np.zeros(nsteps); dp_sp = np.zeros(nsteps)
    for n in range(nsteps):
        r, P, x, y = tao.step(r, P, x, y, n * dt, dt)
        p = canonical_to_kinetic(field, r, P, (n + 1) * dt)
        dg_sp[n] = abs(gamma_of_p(p)[0] - gamma0)
        dp_sp[n] = abs(np.sum(p * p) - p2_0) / p2_0

    return dict(dg_boris=dg_boris, dg_rk4=dg_rk4, dg_sp=dg_sp,
                dp_boris=dp_boris, dp_rk4=dp_rk4, dp_sp=dp_sp)


def main():
    res = run()
    n = len(res["dg_boris"])
    path = os.path.join(_paths.DATA, "sim1_free_particle.csv")
    step_idx = np.arange(1, n + 1)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "dgamma_Boris", "dgamma_RK4", "dgamma_SPPINN",
                    "dp2_Boris", "dp2_RK4", "dp2_SPPINN"])
        for i in range(0, n, 50):
            w.writerow([step_idx[i], res["dg_boris"][i], res["dg_rk4"][i],
                        res["dg_sp"][i], res["dp_boris"][i], res["dp_rk4"][i],
                        res["dp_sp"][i]])
    print("=== Simulation 1: free particle, N =", n, "===")
    print(f"  Boris   : dgamma = {res['dg_boris'][-1]:.3e}  d|p|^2/p0^2 = {res['dp_boris'][-1]:.3e}")
    print(f"  RK4     : dgamma = {res['dg_rk4'][-1]:.3e}  d|p|^2/p0^2 = {res['dp_rk4'][-1]:.3e}")
    print(f"  SP-PINN : dgamma = {res['dg_sp'][-1]:.3e}  d|p|^2/p0^2 = {res['dp_sp'][-1]:.3e}")
    print("  wrote", path)
    return res


if __name__ == "__main__":
    main()
