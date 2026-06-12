"""relsim: symmetry-preserving relativistic charged-particle integrators.

Modules
-------
fields       -- EM field configurations and analytic Hamiltonian.
integrators  -- Boris, RK4, RK8 reference, Tao explicit symplectic (SP-PINN Stage 2).
pinn         -- PINN Hamiltonian learning (SP-PINN Stage 1).
diagnostics  -- gamma, Larmor radius, Poincare loop invariant, Lorentz violation.
plotstyle    -- shared Matplotlib styling for publication-quality figures.
"""
from . import fields, integrators, diagnostics  # noqa: F401
