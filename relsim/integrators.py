"""
Time integrators for the relativistic equations of motion of a charged
particle (units c = 1, m = 1).

Implemented schemes
-------------------
* ``boris_push``      -- relativistic Boris pusher (2nd order, volume
                         preserving, not symplectic).
* ``rk4_push``        -- classical explicit Runge--Kutta (4th order,
                         neither symplectic nor volume preserving).
* ``rk8_reference``   -- high-accuracy reference trajectory (DOP853).
* ``TaoSymplectic``   -- explicit 4th-order symplectic integrator in the
                         extended phase space of Tao (J. Comput. Phys. 327,
                         245, 2016).  This is the "Stage 2" map of the
                         SP-PINN method, here driven by the analytic
                         Hamiltonian (the eps_theta -> 0 limit of a learned
                         surrogate).

Boris and RK4 advance the *kinetic* momentum ``p``; the Tao integrator
advances the *canonical* momentum ``P = p + ch A``.  Helper routines convert
between the two so diagnostics are computed consistently.
"""

from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp


# ----------------------------------------------------------------------
# Boris pusher (kinetic momentum p)
# ----------------------------------------------------------------------
def boris_step(field, r, p, t, dt):
    ch = field.ch
    E = field.E(r, t)
    B = field.B(r, t)
    p_minus = p + ch * E * dt / 2.0
    g_minus = np.sqrt(1.0 + np.sum(p_minus * p_minus, axis=-1, keepdims=True))
    tt = ch * B * dt / (2.0 * g_minus)
    t2 = np.sum(tt * tt, axis=-1, keepdims=True)
    s = 2.0 * tt / (1.0 + t2)
    p_prime = p_minus + np.cross(p_minus, tt)
    p_plus = p_minus + np.cross(p_prime, s)
    p_new = p_plus + ch * E * dt / 2.0
    g_new = np.sqrt(1.0 + np.sum(p_new * p_new, axis=-1, keepdims=True))
    r_new = r + dt * p_new / g_new
    return r_new, p_new


# ----------------------------------------------------------------------
# RK4 on (r, p_kin)
# ----------------------------------------------------------------------
def _deriv_kinetic(field, r, p, t):
    g = np.sqrt(1.0 + np.sum(p * p, axis=-1, keepdims=True))
    v = p / g
    E = field.E(r, t)
    B = field.B(r, t)
    drdt = v
    dpdt = field.ch * (E + np.cross(v, B))
    return drdt, dpdt


def rk4_step(field, r, p, t, dt):
    k1r, k1p = _deriv_kinetic(field, r, p, t)
    k2r, k2p = _deriv_kinetic(field, r + 0.5 * dt * k1r, p + 0.5 * dt * k1p, t + 0.5 * dt)
    k3r, k3p = _deriv_kinetic(field, r + 0.5 * dt * k2r, p + 0.5 * dt * k2p, t + 0.5 * dt)
    k4r, k4p = _deriv_kinetic(field, r + dt * k3r, p + dt * k3p, t + dt)
    r_new = r + dt / 6.0 * (k1r + 2 * k2r + 2 * k3r + k4r)
    p_new = p + dt / 6.0 * (k1p + 2 * k2p + 2 * k3p + k4p)
    return r_new, p_new


# ----------------------------------------------------------------------
# Higuera--Cary pusher (kinetic momentum p; volume preserving, 2nd order)
# ----------------------------------------------------------------------
def higuera_cary_step(field, r, p, t, dt):
    """Higuera--Cary relativistic pusher (Phys. Plasmas 24, 052104, 2017): the
    Boris rotation performed with the future-consistent Lorentz factor
    ``gamma^+`` obtained from the Higuera--Cary quadratic.  Volume preserving and
    second order, like Boris, but with the correct E x B drift.  Same kinetic
    ``(r, p)`` interface as :func:`boris_step`, so it is a drop-in comparator."""
    ch = field.ch
    E = field.E(r, t)
    B = field.B(r, t)
    u_minus = p + ch * E * dt / 2.0
    gm2 = 1.0 + np.sum(u_minus * u_minus, axis=-1, keepdims=True)
    tau = ch * B * dt / 2.0
    tau2 = np.sum(tau * tau, axis=-1, keepdims=True)
    u_star = np.sum(u_minus * tau, axis=-1, keepdims=True)
    sigma = gm2 - tau2
    g_plus = np.sqrt((sigma + np.sqrt(sigma * sigma + 4.0 * (tau2 + u_star * u_star))) / 2.0)
    tvec = tau / g_plus
    t2 = np.sum(tvec * tvec, axis=-1, keepdims=True)
    s = 2.0 * tvec / (1.0 + t2)
    u_prime = u_minus + np.cross(u_minus, tvec)
    u_plus = u_minus + np.cross(u_prime, s)
    p_new = u_plus + ch * E * dt / 2.0
    g_new = np.sqrt(1.0 + np.sum(p_new * p_new, axis=-1, keepdims=True))
    r_new = r + dt * p_new / g_new
    return r_new, p_new


# ----------------------------------------------------------------------
# RK8 reference trajectory (single particle), DOP853
# ----------------------------------------------------------------------
def rk8_reference(field, r0, p0, t_eval):
    r0 = np.asarray(r0, dtype=float)
    p0 = np.asarray(p0, dtype=float)

    def rhs(t, y):
        r = y[:3][None, :]
        p = y[3:][None, :]
        drdt, dpdt = _deriv_kinetic(field, r, p, t)
        return np.concatenate([drdt[0], dpdt[0]])

    y0 = np.concatenate([r0, p0])
    sol = solve_ivp(rhs, (t_eval[0], t_eval[-1]), y0, t_eval=t_eval,
                    method="DOP853", rtol=1e-12, atol=1e-13, max_step=np.inf)
    R = sol.y[:3].T
    P = sol.y[3:].T
    return R, P


# ----------------------------------------------------------------------
# Tao explicit symplectic integrator (canonical r, P)
# ----------------------------------------------------------------------
class TaoSymplectic:
    """
    Explicit 4th-order symplectic integrator using the extended phase space
    (r, P, x, y) with binding constant ``omega``.  The Hamiltonian is taken
    from ``field`` (methods H, gradH_r, gradH_P).
    """

    def __init__(self, field, omega=20.0):
        self.field = field
        self.omega = omega
        # Yoshida 4th-order composition of a symmetric 2nd-order map
        w1 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
        w0 = 1.0 - 2.0 * w1
        self.cset = (w1, w0, w1)

    # --- elementary maps -------------------------------------------------
    def _phiA(self, r, P, x, y, t, dt):
        # uses H(r, y): P += -dt dH/dr(r,y); x += dt dH/dP(r,y)
        gr = self.field.gradH_r(r, y, t)
        gP = self.field.gradH_P(r, y, t)
        P = P - dt * gr
        x = x + dt * gP
        return r, P, x, y

    def _phiB(self, r, P, x, y, t, dt):
        # uses H(x, P): r += dt dH/dP(x,P); y += -dt dH/dr(x,P)
        gP = self.field.gradH_P(x, P, t)
        gr = self.field.gradH_r(x, P, t)
        r = r + dt * gP
        y = y - dt * gr
        return r, P, x, y

    def _phiC(self, r, P, x, y, t, dt):
        # binding rotation with angular frequency 2 omega
        w = self.omega
        c = np.cos(2 * w * dt)
        s = np.sin(2 * w * dt)
        R = r + x
        S = P + y
        u = r - x
        v = P - y
        u_new = u * c + v * s
        v_new = -u * s + v * c
        r = 0.5 * (R + u_new)
        x = 0.5 * (R - u_new)
        P = 0.5 * (S + v_new)
        y = 0.5 * (S - v_new)
        return r, P, x, y

    def _strang(self, r, P, x, y, t, dt):
        r, P, x, y = self._phiA(r, P, x, y, t, dt / 2)
        r, P, x, y = self._phiB(r, P, x, y, t, dt / 2)
        r, P, x, y = self._phiC(r, P, x, y, t, dt)
        r, P, x, y = self._phiB(r, P, x, y, t, dt / 2)
        r, P, x, y = self._phiA(r, P, x, y, t, dt / 2)
        return r, P, x, y

    def step(self, r, P, x, y, t, dt):
        for c in self.cset:
            r, P, x, y = self._strang(r, P, x, y, t, c * dt)
        return r, P, x, y

    def init_copies(self, r, P):
        return r.copy(), P.copy()
