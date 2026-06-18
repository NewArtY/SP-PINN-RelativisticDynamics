# SP-PINN: Symmetry-Preserving Integration of Relativistic Charged-Particle Dynamics

Reference implementation and reproducibility package for the paper

> N. S. Akintsov, A. P. Nevecheria, G. Yuan, V. S. Igumnov, S. N. Andreev, Q.-H. Qin,
> *Symmetry-Preserving Physics-Informed Neural Network Framework for
> Relativistic Charged-Particle Dynamics in 3+1 Dimensions*, submitted to
> **Symmetry** (MDPI), 2026.

The code integrates the relativistic equations of motion of a charged particle
in prescribed electromagnetic fields (field-free space, a uniform magnetic
field, and a focused Gaussian laser pulse) with four schemes and compares their
long-time structure-preservation properties:

| Scheme | Order | Structure | Notes |
|---|---|---|---|
| **Boris** | 2 | volume-preserving | standard PIC pusher; exact for pure gyration |
| **RK4** | 4 | none | high short-time accuracy, secular long-time drift |
| **RK8 (DOP853)** | 8 | none | tight-tolerance reference trajectory |
| **SP-PINN** | 4 (realized 2) | **symplectic** | Stage-2 explicit symplectic map (Tao 2016) driven by a Hamiltonian surrogate |

Units throughout are `c = 1`, `m = 1`; the particle charge is `ch` (`-1` for an
electron). Positions `r` and canonical momenta `P` are 3-vectors related to the
kinetic momentum by `p = P - ch*A`.

## What "SP-PINN" means here

The method has two stages (see paper, Section 4):

* **Stage 1 (learning).** An unsupervised physics-informed neural network
  learns a surrogate relativistic Hamiltonian `H_θ(r,P)` directly from the
  governing relations, with a Lorentz-invariant loss enforcing the mass-shell
  constraint `H = mc²γ`. See [`relsim/pinn.py`](relsim/pinn.py) and
  [`experiments/pinn_demo.py`](experiments/pinn_demo.py).
* **Stage 2 (integration).** The (learned or analytic) Hamiltonian is advanced
  with an explicit fourth-order symplectic map built on Tao's extended phase
  space, which is valid for the *non-separable* relativistic Hamiltonian. See
  [`relsim/integrators.py`](relsim/integrators.py) (`TaoSymplectic`).

> **Important (reproducibility note).** The headline comparison figures
> (Figures 2–6) use the **analytic** Hamiltonian inside the Stage-2 symplectic
> map, i.e. the `ε_θ → 0` limit of a perfectly learned surrogate. This isolates
> the *geometric* properties of the integrator from neural-network
> approximation error. The realistic training-error floor `ε_θ` of the Stage-1
> PINN is measured separately by `experiments/pinn_demo.py` and is the error
> floor of the fully learned integrator. Closing the gap between `ε_θ` and
> machine precision for the full 3+1D surrogate is ongoing work.

## Honest summary of the numerical results

On the integrable magnetic-field benchmark the comparison is the textbook
contrast between **bounded** and **secular** error growth:

* **RK4** drifts secularly (`∝ n`) in the Lorentz factor and Larmor radius.
* **Boris** conserves both to machine precision — it is *optimal* for pure
  gyration, being an exact volume-preserving rotation.
* **SP-PINN** keeps the error **bounded** for all time and overtakes RK4 within
  a few hundred gyrations, while remaining a structure-preserving scheme for
  the general non-separable Hamiltonian (where Boris-type exactness is not
  available).

> **Note on time-dependent fields.** For the static (free, magnetic) cases the
> Stage-2 map is exactly symplectic and 4th order. For the time-dependent laser
> field the present implementation freezes the field within each step, so it is
> a high-order *structure-aware* scheme there rather than strictly symplectic;
> promoting time to a canonical coordinate restores exact symplecticity — this
> *autonomized* map is implemented (`AutonomizedField` in `relsim/fields.py`) and
> validated on a plane wave by `sim6_planewave.py` (see below), where it conserves
> the light-front invariant `gamma - p_z` about three orders of magnitude better
> than the frozen-in-time map.

We do **not** claim that SP-PINN beats a high-order method like RK4 in absolute
short-time accuracy on these benign benchmarks; its value is the absence of
secular drift plus applicability to general fields. The symplectic advantage
becomes decisive for long-time and non-integrable problems.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Reproducing the figures

```bash
cd experiments
python run_all.py        # Simulations 1–7 -> figures/figure2..7.{pdf,png}, data/*.csv
python make_fig1_schematic.py   # Figure 1 (architecture schematic)
python pinn_demo.py      # Stage-1 PINN demonstrator: reports ε_θ
```

Each script can also be run individually; all write to `../figures` and
`../data`.

| Script | Produces |
|---|---|
| `sim1_free_particle.py` | Table A1 (free particle, all schemes exact; now in Appendix A) |
| `sim2_magnetic.py` | Figure 2 (Larmor & Lorentz-factor error, long time) |
| `sim3_laser_trajectory.py` | Figure 3 (laser trajectory & energy error) |
| `sim4_ensemble_spectrum.py` | Figure 4 (energy spectrum & transverse-momentum symmetry) |
| `sim5_timing.py` | Figure 6 / Table 2 (computational cost) |
| `sim6_planewave.py` | Figure A1 (autonomized time-as-coordinate symplecticity; plane-wave P_x control) |
| `sim7_nonintegrable.py` | Figure 5 (non-integrable B + anharmonic well; symplectic advantage) |
| `study_convergence_omega.py` | Figure A2 (Δt-convergence order; Ω binding-constant study) |
| `study_omega_resonance.py` | Ω-resonance scaling sweep verifying the Floquet condition 2ΩΔt = kπ, Ω_k = kπ/(2Δt) (data + figure; underpins Appendix A.2) |
| `study_multiseed.py` | multi-seed ε_θ and timing statistics |
| `make_fig1_schematic.py` | Figure 1 (architecture schematic) |
| `train_laser_A_residual.py` | **GPU-recommended** Stage-1 training of the A-residual light-cone laser surrogate (the formulation that works; reaches ε_θ ≈ 3.0e-4, 3-seed mean). Writes the checkpoint to `../data`. |
| `make_fig_laser_surrogate.py` | Figure 7 (the *learned* surrogate in action: γ(t) of the learned-Hamiltonian trajectory vs the analytic reference) |
| `notebooks/SP_PINN_3plus1D_surrogate_colab.ipynb` | **GPU** training of the time-dependent 3+1D laser surrogate (Google Colab) |

### Time-dependent fields and the autonomized symplectic map

For a time-dependent field the Stage-2 map can be made genuinely symplectic by
adjoining time as a canonical coordinate (`AutonomizedField` in
[`relsim/fields.py`](relsim/fields.py)); on a plane wave this conserves the
light-front invariant `gamma - p_z` about three orders of magnitude better than
the frozen-in-time map (`sim6_planewave.py`).

### Realized order of the symplectic map (honest note)

The convergence study (`study_convergence_omega.py`) shows that, while the
Yoshida triple-jump is a 4th-order composition, the realized order of the Tao
map for the true trajectory is **second order** (the extended-phase-space
binding error caps it). The map's value is the *bounded, non-secular* error, not
a higher formal order — clearest on the non-integrable test (`sim7`).

## Repository layout

```
relsim/            core library
  fields.py        EM fields + analytic relativistic Hamiltonian and gradients
  integrators.py   Boris, RK4, RK8 reference, Tao explicit symplectic map
  pinn.py          Stage-1 PINN Hamiltonian learning (+ PINNField adapter)
  diagnostics.py   gamma, Larmor radius, Poincaré loop invariant, etc.
  plotstyle.py     shared publication-quality Matplotlib styling
experiments/       runnable scripts producing every figure/table
figures/  data/    outputs (created on first run)
```

## License

MIT — see [LICENSE](LICENSE).
