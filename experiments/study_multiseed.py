"""
Multi-seed statistics (addresses review point 5).

Trains the static magnetic-field PINN surrogate over several random seeds and
reports the mean +/- standard deviation of the training error eps_theta (RMS),
and measures the per-step wall-clock time of each integrator over repeated
trials.  This quantifies the run-to-run variability of the single-seed numbers
reported in the main text.
"""
import _paths  # noqa: F401
import numpy as np
import time
import os

from relsim.fields import ConstMagneticField
from relsim.integrators import boris_step, rk4_step, TaoSymplectic
from relsim.diagnostics import kinetic_to_canonical


def pinn_seeds(n_seeds=5, epochs=2500, lbfgs_iters=80):
    from relsim.pinn import train_pinn
    field = ConstMagneticField(B0=1.0, ch=-1.0)
    domain = {"r": 6.0, "P": 6.0}
    eps_list = []
    for seed in range(n_seeds):
        t0 = time.perf_counter()
        _, _, eps = train_pinn(field, domain, n_coll=4000, epochs=epochs,
                               lbfgs_iters=lbfgs_iters, seed=seed)
        eps_list.append(eps["eps_theta"])
        print(f"  seed {seed}: eps_theta(RMS)={eps['eps_theta']:.3e} "
              f"(max {eps['eps_theta_max']:.2e})  [{time.perf_counter()-t0:.0f}s]",
              flush=True)
    return np.array(eps_list)


def timing_repeats(n_rep=5, nsteps=3000, dt=0.05):
    field = ConstMagneticField(B0=1.0, ch=-1.0)
    res = {s: [] for s in ["Boris", "RK4", "SP-PINN"]}
    for _ in range(n_rep):
        for s in res:
            r = np.zeros((1, 3)); p = np.zeros((1, 3)); p[0, 1] = 4.899
            if s == "SP-PINN":
                tao = TaoSymplectic(field, omega=15.0)
                P = kinetic_to_canonical(field, r, p, 0.0)
                xx, yy = tao.init_copies(r, P)
            t0 = time.perf_counter()
            for n in range(nsteps):
                t = n * dt
                if s == "Boris":
                    r, p = boris_step(field, r, p, t, dt)
                elif s == "RK4":
                    r, p = rk4_step(field, r, p, t, dt)
                else:
                    r, P, xx, yy = tao.step(r, P, xx, yy, t, dt)
            res[s].append((time.perf_counter() - t0) / nsteps * 1e3)
    return res


def main():
    print("=== Multi-seed PINN eps_theta (point 5) ===")
    eps = pinn_seeds()
    print(f"  eps_theta (RMS): mean={eps.mean():.3e}  std={eps.std():.2e}  "
          f"min={eps.min():.2e}  max={eps.max():.2e}  (n={len(eps)})")
    print("=== Timing repeats (ms/step, single particle) ===")
    tm = timing_repeats()
    for s in ["Boris", "RK4", "SP-PINN"]:
        a = np.array(tm[s])
        print(f"  {s:8s}: {a.mean():.3f} +/- {a.std():.3f} ms")
    with open(os.path.join(_paths.DATA, "study_multiseed.txt"), "w") as f:
        f.write(f"eps_theta_rms_mean {eps.mean():.6e}\n")
        f.write(f"eps_theta_rms_std {eps.std():.6e}\n")
        f.write(f"eps_theta_rms_values {','.join('%.4e'%e for e in eps)}\n")
        for s in ["Boris", "RK4", "SP-PINN"]:
            a = np.array(tm[s])
            f.write(f"time_{s}_mean_ms {a.mean():.4f}\n")
            f.write(f"time_{s}_std_ms {a.std():.4f}\n")
    print("  wrote data/study_multiseed.txt")


if __name__ == "__main__":
    main()
