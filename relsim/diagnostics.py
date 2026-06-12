"""Diagnostics: Lorentz factor, kinetic/canonical conversion, Larmor radius,
and the first Poincare loop invariant."""

from __future__ import annotations
import numpy as np


def gamma_of_p(p):
    return np.sqrt(1.0 + np.sum(p * p, axis=-1))


def kinetic_to_canonical(field, r, p, t):
    return p + field.ch * field.A(r, t)


def canonical_to_kinetic(field, r, P, t):
    return P - field.ch * field.A(r, t)


def larmor_radius(p_perp, B0, ch=1.0):
    # r_L = p_perp / |ch B0|  (units c=1, m=1)
    return p_perp / abs(ch * B0)


def poincare_loop_invariant(q_loop, P_loop):
    """
    First Poincare integral invariant I1 = oint P . dq evaluated on a closed
    loop of M phase-space points.  q_loop, P_loop have shape (M, d); the loop
    is closed (point M wraps to 0).  Uses the trapezoidal midpoint rule.
    """
    M = q_loop.shape[0]
    qn = np.roll(q_loop, -1, axis=0)
    Pn = np.roll(P_loop, -1, axis=0)
    return np.sum(0.5 * (P_loop + Pn) * (qn - q_loop))


def lorentz_violation(p):
    """V = |gamma - cosh(arcsinh(p_z))|; identically zero analytically,
    so it measures purely numerical drift of the gamma(theta)=cosh(theta)
    relation along the trajectory."""
    g = gamma_of_p(p)
    theta = np.arcsinh(p[..., 2])
    return np.abs(g - np.cosh(theta))
