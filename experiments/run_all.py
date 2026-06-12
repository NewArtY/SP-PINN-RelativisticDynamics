"""Run every simulation and regenerate all figures and data files."""
import _paths  # noqa: F401
import time


def main():
    import sim1_free_particle, sim2_magnetic, sim3_laser_trajectory
    import sim4_ensemble_spectrum, sim5_timing
    import sim6_planewave, sim7_nonintegrable, study_convergence_omega
    t0 = time.perf_counter()
    print("\n>>> Simulation 1 (free particle, Table 2)")
    sim1_free_particle.main()
    print("\n>>> Simulation 2 (magnetic field, Figure 2)")
    sim2_magnetic.main()
    print("\n>>> Simulation 3 (laser trajectory, Figure 3)")
    sim3_laser_trajectory.main()
    print("\n>>> Simulation 4 (ensemble spectrum, Figure 4)")
    sim4_ensemble_spectrum.main()
    print("\n>>> Simulation 5 (timing, Figure 5)")
    sim5_timing.main()
    print("\n>>> Simulation 6 (plane wave, Figure A1)")
    sim6_planewave.main()
    print("\n>>> Simulation 7 (non-integrable, Figure 6)")
    sim7_nonintegrable.main()
    print("\n>>> Study (convergence + Omega, Figure A2)")
    study_convergence_omega.main()
    print(f"\nAll simulations finished in {time.perf_counter() - t0:.1f} s.")
    print("Run  python make_fig1_schematic.py  for Figure 1,")
    print("    python pinn_demo.py  for the Stage-1 PINN demonstrator,")
    print("    python study_multiseed.py  for multi-seed statistics.")


if __name__ == "__main__":
    main()
