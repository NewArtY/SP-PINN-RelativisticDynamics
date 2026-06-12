"""
Electromagnetic field configurations and the associated relativistic
Hamiltonian for a charged particle.

Units: c = 1, m = 1.  The particle charge ``ch`` is dimensionless
(``ch = -1`` for an electron).  Position ``r`` and canonical momentum
``P`` are 3-vectors.  The canonical (gauge) relation between the kinetic
momentum ``p`` and the canonical momentum ``P`` is

    p = P - ch * A(r, t).

The relativistic Hamiltonian is

    H(r, P, t) = sqrt(|P - ch A|^2 + 1) + ch * phi(r, t),

with Lorentz factor gamma = sqrt(1 + |p|^2) = H - ch*phi.

Every field exposes
    E(r, t), B(r, t)             -- used by the Boris pusher,
    A(r, t)                      -- vector potential (radiation gauge, phi = 0),
    H(r, P, t)                   -- Hamiltonian,
    gradH_r(r, P, t)             -- dH/dr   (canonical force, -dP/dt),
    gradH_P(r, P, t)             -- dH/dP   (= velocity v = dr/dt).

All routines are written with NumPy and broadcast over a leading particle
axis, i.e. ``r`` and ``P`` may have shape (3,) or (N, 3).
"""

from __future__ import annotations
import numpy as np


def _gamma_from_p(p):
    return np.sqrt(1.0 + np.sum(p * p, axis=-1, keepdims=True))


class FreeField:
    """Field-free space: E = B = A = 0."""

    name = "free"

    def __init__(self, ch=-1.0):
        self.ch = ch

    def A(self, r, t):
        return np.zeros_like(r)

    def E(self, r, t):
        return np.zeros_like(r)

    def B(self, r, t):
        return np.zeros_like(r)

    def H(self, r, P, t):
        p = P - self.ch * self.A(r, t)
        return _gamma_from_p(p)[..., 0]

    def gradH_P(self, r, P, t):
        p = P - self.ch * self.A(r, t)
        g = _gamma_from_p(p)
        return p / g

    def gradH_r(self, r, P, t):
        return np.zeros_like(r)


class ConstMagneticField:
    """Uniform magnetic field B = B0 z, symmetric gauge A = 0.5 B x r."""

    name = "magnetic"

    def __init__(self, B0=1.0, ch=-1.0):
        self.B0 = B0
        self.ch = ch

    def A(self, r, t):
        x = r[..., 0]
        y = r[..., 1]
        Ax = -0.5 * self.B0 * y
        Ay = 0.5 * self.B0 * x
        Az = np.zeros_like(x)
        return np.stack([Ax, Ay, Az], axis=-1)

    def E(self, r, t):
        return np.zeros_like(r)

    def B(self, r, t):
        b = np.zeros_like(r)
        b[..., 2] = self.B0
        return b

    def H(self, r, P, t):
        p = P - self.ch * self.A(r, t)
        return _gamma_from_p(p)[..., 0]

    def gradH_P(self, r, P, t):
        p = P - self.ch * self.A(r, t)
        g = _gamma_from_p(p)
        return p / g

    def gradH_r(self, r, P, t):
        # dH/dr = (1/gamma) * (p . d(p)/dr); p = P - ch A, dp/dr = -ch dA/dr
        p = P - self.ch * self.A(r, t)
        g = _gamma_from_p(p)  # (...,1)
        # dA/dr for symmetric gauge: dAx/dy = -0.5 B0, dAy/dx = 0.5 B0
        px = p[..., 0]
        py = p[..., 1]
        # dH/dx = (1/g)(px*dpx/dx + py*dpy/dx); dpx/dx=0, dpy/dx=-ch*0.5*B0
        dHdx = (1.0 / g[..., 0]) * (py * (-self.ch * 0.5 * self.B0))
        # dH/dy = (1/g)(px*dpx/dy + py*dpy/dy); dpx/dy=-ch*(-0.5*B0)=ch*0.5*B0
        dHdy = (1.0 / g[..., 0]) * (px * (-self.ch * (-0.5 * self.B0)))
        dHdz = np.zeros_like(dHdx)
        return np.stack([dHdx, dHdy, dHdz], axis=-1)


class GaussianLaserPulse:
    """
    Linearly polarised (along x) focused Gaussian laser pulse propagating
    along +z, in the paraxial approximation.  All lengths in units of the
    laser wavenumber k0 = omega0/c (so k0 = omega0 = 1 in code units), times
    in units of 1/omega0.

    Vector potential (radiation gauge, phi = 0):

        A_x = a0 * (w0/w(z)) * exp[-(x^2+y^2)/w(z)^2 - (z-t)^2/(c tau)^2]
                  * cos[ (z - t) + psi ],

    with the Gouy phase and wavefront curvature collected in ``psi``.
    Fields are obtained from E = -dA/dt, B = curl A by 2nd-order central
    finite differences (step h=1e-4), which is adequate for trajectory tests
    (field error ~ O(h^2) ~ 1e-8, well below the integrator differences).
    """

    name = "laser"

    def __init__(self, a0=5.0, w0=5.0 * 2 * np.pi, tau=30.0, ch=-1.0,
                 lam=2 * np.pi):
        self.a0 = a0
        self.w0 = w0
        self.tau = tau            # pulse duration (code units, 1/omega0)
        self.ch = ch
        self.lam = lam            # wavelength = 2 pi in code units (k0 = 1)
        self.k0 = 2 * np.pi / lam  # = 1
        self.zR = self.k0 * w0 ** 2 / 2.0

    # ---- vector potential -------------------------------------------------
    def _Ax_scalar(self, x, y, z, t):
        zR = self.zR
        wz = self.w0 * np.sqrt(1.0 + (z / zR) ** 2)
        Rinv = z / (z ** 2 + zR ** 2)              # 1/R(z)
        gouy = np.arctan2(z, zR)
        env_t = np.exp(-((z - t) ** 2) / (self.tau ** 2))
        env_r = np.exp(-(x ** 2 + y ** 2) / wz ** 2)
        psi = self.k0 * (x ** 2 + y ** 2) * 0.5 * Rinv - gouy
        phase = self.k0 * (z - t) + psi
        return self.a0 * (self.w0 / wz) * env_r * env_t * np.cos(phase)

    def A(self, r, t):
        x = r[..., 0]; y = r[..., 1]; z = r[..., 2]
        Ax = self._Ax_scalar(x, y, z, t)
        out = np.zeros_like(r)
        out[..., 0] = Ax
        return out

    def _grad_A(self, r, t, h=1e-4):
        """Return dAx/dx, dAx/dy, dAx/dz, dAx/dt via 2nd-order central FD."""
        x = r[..., 0]; y = r[..., 1]; z = r[..., 2]
        f = self._Ax_scalar
        dAx_dx = (f(x + h, y, z, t) - f(x - h, y, z, t)) / (2 * h)
        dAx_dy = (f(x, y + h, z, t) - f(x, y - h, z, t)) / (2 * h)
        dAx_dz = (f(x, y, z + h, t) - f(x, y, z - h, t)) / (2 * h)
        dAx_dt = (f(x, y, z, t + h) - f(x, y, z, t - h)) / (2 * h)
        return dAx_dx, dAx_dy, dAx_dz, dAx_dt

    def E(self, r, t):
        # E = -dA/dt (code units, c=1); only Ax non-zero
        _, _, _, dAx_dt = self._grad_A(r, t)
        out = np.zeros_like(r)
        out[..., 0] = -dAx_dt
        return out

    def B(self, r, t):
        # B = curl A, A = (Ax, 0, 0):
        # Bx = dAz/dy - dAy/dz = 0
        # By = dAx/dz - dAz/dx = dAx/dz
        # Bz = dAy/dx - dAx/dy = -dAx/dy
        _, dAx_dy, dAx_dz, _ = self._grad_A(r, t)
        out = np.zeros_like(r)
        out[..., 1] = dAx_dz
        out[..., 2] = -dAx_dy
        return out

    def H(self, r, P, t):
        p = P - self.ch * self.A(r, t)
        return _gamma_from_p(p)[..., 0]

    def gradH_P(self, r, P, t):
        p = P - self.ch * self.A(r, t)
        g = _gamma_from_p(p)
        return p / g

    def gradH_r(self, r, P, t):
        # dH/dr = (1/gamma) p . dp/dr, dp/dr = -ch dA/dr (only Ax non-zero)
        p = P - self.ch * self.A(r, t)
        g = _gamma_from_p(p)[..., 0]
        px = p[..., 0]
        dAx_dx, dAx_dy, dAx_dz, _ = self._grad_A(r, t)
        # dpx/dx = -ch dAx/dx, etc.; only px couples (Ay=Az=0)
        dHdx = (1.0 / g) * px * (-self.ch * dAx_dx)
        dHdy = (1.0 / g) * px * (-self.ch * dAx_dy)
        dHdz = (1.0 / g) * px * (-self.ch * dAx_dz)
        return np.stack([dHdx, dHdy, dHdz], axis=-1)

    def gradH_t(self, r, P, t):
        # dH/dt = (1/gamma) px * (-ch dAx/dt)
        p = P - self.ch * self.A(r, t)
        g = _gamma_from_p(p)[..., 0]
        px = p[..., 0]
        _, _, _, dAx_dt = self._grad_A(r, t)
        return (1.0 / g) * px * (-self.ch * dAx_dt)


class PlaneWave:
    """Linearly polarised plane electromagnetic wave A = a0 cos(k0(z - t)) x_hat,
    propagating along +z (code units c = 1).  Exact invariants: the transverse
    canonical momentum P_x and the light-front quantity K = gamma - p_z."""

    name = "planewave"

    def __init__(self, a0=1.0, k0=1.0, ch=-1.0):
        self.a0 = a0
        self.k0 = k0
        self.ch = ch

    def _phase(self, r, t):
        return self.k0 * (r[..., 2] - t)

    def A(self, r, t):
        out = np.zeros_like(r)
        out[..., 0] = self.a0 * np.cos(self._phase(r, t))
        return out

    def _dAx(self, r, t):
        # dAx/dz and dAx/dt of a0 cos(k0(z-t))
        s = np.sin(self._phase(r, t))
        dAx_dz = -self.a0 * self.k0 * s
        dAx_dt = self.a0 * self.k0 * s
        return dAx_dz, dAx_dt

    def E(self, r, t):
        _, dAx_dt = self._dAx(r, t)
        out = np.zeros_like(r)
        out[..., 0] = -dAx_dt
        return out

    def B(self, r, t):
        dAx_dz, _ = self._dAx(r, t)
        out = np.zeros_like(r)
        out[..., 1] = dAx_dz                      # By = dAx/dz
        return out

    def H(self, r, P, t):
        p = P - self.ch * self.A(r, t)
        return _gamma_from_p(p)[..., 0]

    def gradH_P(self, r, P, t):
        p = P - self.ch * self.A(r, t)
        return p / _gamma_from_p(p)

    def gradH_r(self, r, P, t):
        p = P - self.ch * self.A(r, t)
        g = _gamma_from_p(p)[..., 0]
        px = p[..., 0]
        dAx_dz, _ = self._dAx(r, t)
        out = np.zeros_like(r)
        out[..., 2] = (1.0 / g) * px * (-self.ch * dAx_dz)
        return out

    def gradH_t(self, r, P, t):
        p = P - self.ch * self.A(r, t)
        g = _gamma_from_p(p)[..., 0]
        px = p[..., 0]
        _, dAx_dt = self._dAx(r, t)
        return (1.0 / g) * px * (-self.ch * dAx_dt)


class MagneticWell:
    """Autonomous, NON-INTEGRABLE relativistic system: a uniform magnetic field
    B = B0 z plus a static anharmonic electrostatic potential

        phi(r) = 0.5*kperp*(x^2 + y^2) + eps*x^2*y^2 ,

    confining the particle in the transverse plane.  The coupling term eps*x^2*y^2
    breaks integrability (Henon-Heiles-type), so the dynamics are chaotic at
    moderate energy, while the total energy H = gamma + ch*phi is an exact constant
    of motion (autonomous Hamiltonian).  This is the test where symplecticity is
    genuinely decisive: a symplectic map keeps H bounded for all time, whereas
    non-symplectic schemes drift secularly even though no scheme can exploit an
    exact volume-preserving rotation here.
    """

    name = "magneticwell"

    def __init__(self, B0=1.0, kperp=1.0, eps=0.30, ch=1.0):
        self.B0 = B0
        self.kperp = kperp
        self.eps = eps
        self.ch = ch

    def A(self, r, t):
        x = r[..., 0]; y = r[..., 1]
        Ax = -0.5 * self.B0 * y
        Ay = 0.5 * self.B0 * x
        return np.stack([Ax, Ay, np.zeros_like(x)], axis=-1)

    def phi(self, r):
        x = r[..., 0]; y = r[..., 1]
        return 0.5 * self.kperp * (x ** 2 + y ** 2) + self.eps * x ** 2 * y ** 2

    def grad_phi(self, r):
        x = r[..., 0]; y = r[..., 1]
        dx = self.kperp * x + 2 * self.eps * x * y ** 2
        dy = self.kperp * y + 2 * self.eps * x ** 2 * y
        return np.stack([dx, dy, np.zeros_like(x)], axis=-1)

    def E(self, r, t):
        return -self.grad_phi(r)

    def B(self, r, t):
        b = np.zeros_like(r)
        b[..., 2] = self.B0
        return b

    def H(self, r, P, t):
        p = P - self.ch * self.A(r, t)
        return _gamma_from_p(p)[..., 0] + self.ch * self.phi(r)

    def gradH_P(self, r, P, t):
        p = P - self.ch * self.A(r, t)
        return p / _gamma_from_p(p)

    def gradH_r(self, r, P, t):
        p = P - self.ch * self.A(r, t)
        g = _gamma_from_p(p)[..., 0]
        px = p[..., 0]; py = p[..., 1]
        # magnetic part (symmetric gauge): see ConstMagneticField
        dHdx = (1.0 / g) * (py * (-self.ch * 0.5 * self.B0))
        dHdy = (1.0 / g) * (px * (-self.ch * (-0.5 * self.B0)))
        dHdz = np.zeros_like(dHdx)
        gphi = self.grad_phi(r)
        return np.stack([dHdx, dHdy, dHdz], axis=-1) + self.ch * gphi


class AutonomizedField:
    """Promote a time-dependent field to an AUTONOMOUS one by adjoining time as a
    canonical coordinate.  The extended position is q = (x, y, z, t) and the
    extended momentum is P = (Px, Py, Pz, p_t), with extended Hamiltonian

        Hbar(q, P) = H(r, P_spatial, q_t) + p_t .

    Then dq_t/dtau = dHbar/dp_t = 1 (time advances with the integration parameter)
    and dp_t/dtau = -dHbar/dq_t = -dH/dt, so the field is no longer frozen within a
    step and the Tao symplectic map applies exactly.  ``base`` must expose
    gradH_t.  All arrays carry a trailing axis of length 4."""

    name = "autonomized"

    def __init__(self, base):
        self.base = base
        self.ch = base.ch

    def A(self, r4, t):
        a = np.zeros_like(r4)
        a[..., :3] = self.base.A(r4[..., :3], r4[..., 3])
        return a

    def H(self, r4, P4, t):
        r = r4[..., :3]; tt = r4[..., 3]
        return self.base.H(r, P4[..., :3], tt) + P4[..., 3]

    def gradH_P(self, r4, P4, t):
        r = r4[..., :3]; tt = r4[..., 3]
        gP = self.base.gradH_P(r, P4[..., :3], tt)
        out = np.zeros_like(P4)
        out[..., :3] = gP
        out[..., 3] = 1.0                           # dHbar/dp_t = 1
        return out

    def gradH_r(self, r4, P4, t):
        r = r4[..., :3]; tt = r4[..., 3]
        gr = self.base.gradH_r(r, P4[..., :3], tt)
        gt = self.base.gradH_t(r, P4[..., :3], tt)
        out = np.zeros_like(r4)
        out[..., :3] = gr
        out[..., 3] = gt                            # dHbar/dt-coordinate = dH/dt
        return out
