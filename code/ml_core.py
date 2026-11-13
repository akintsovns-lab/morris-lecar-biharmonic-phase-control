"""
Core numerics for phase-controlled Morris-Lecar under high-frequency biharmonic forcing.

Model (Morris-Lecar, class II / Hopf parameter set):
    C dV/dt = I0 - gCa m_inf(V)(V-ECa) - gK w (V-EK) - gL (V-EL) + I_ac(t)
    dw/dt   = gam cosh((V-V3)/(2 V4)) [w_inf(V) - w]
    I_ac(t) = A [cos(Om t) + cos(2 Om t + phi)],  s = A/(C Om)

Exact fast displacement: xi(theta) = s [ sin(theta) + 1/2 sin(2 theta + phi) ],  theta = Om t.
Averaged (slow) system: Fbar(u,w) = <f_auto(u + xi(theta), w)>_theta.
"""
import numpy as np
from numba import njit, prange

# parameter vector order
# 0 C, 1 gCa, 2 gK, 3 gL, 4 ECa, 5 EK, 6 EL, 7 V1, 8 V2, 9 V3, 10 V4, 11 gam, 12 I0
P0 = np.array([20.0, 4.4, 8.0, 2.0, 120.0, -84.0, -60.0, -1.2, 18.0, 2.0, 30.0, 0.04, 100.0])

NQ = 256  # quadrature nodes for phase averaging (trapezoid on periodic function = spectral)
THETA = 2.0 * np.pi * np.arange(NQ) / NQ


def xi_nodes(s, phi, ratio=1.0, nq=NQ):
    """Fast voltage displacement at quadrature nodes. ratio = A2/A1 ; s = A1/(C Om)."""
    th = 2.0 * np.pi * np.arange(nq) / nq
    return s * (np.sin(th) + 0.5 * ratio * np.sin(2.0 * th + phi))


@njit(cache=True)
def f_auto(V, w, p):
    """Autonomous ML right-hand side (no ac current). Returns (dV, dw)."""
    C, gCa, gK, gL, ECa, EK, EL, V1, V2, V3, V4, gam, I0 = p
    minf = 0.5 * (1.0 + np.tanh((V - V1) / V2))
    winf = 0.5 * (1.0 + np.tanh((V - V3) / V4))
    lam = gam * np.cosh((V - V3) / (2.0 * V4))
    dV = (I0 - gCa * minf * (V - ECa) - gK * w * (V - EK) - gL * (V - EL)) / C
    dw = lam * (winf - w)
    return dV, dw


@njit(cache=True)
def jac_auto(V, w, p):
    """Analytic Jacobian of the autonomous ML field."""
    C, gCa, gK, gL, ECa, EK, EL, V1, V2, V3, V4, gam, I0 = p
    t1 = np.tanh((V - V1) / V2)
    t3 = np.tanh((V - V3) / V4)
    minf = 0.5 * (1.0 + t1)
    dminf = 0.5 * (1.0 - t1 * t1) / V2
    winf = 0.5 * (1.0 + t3)
    dwinf = 0.5 * (1.0 - t3 * t3) / V4
    lam = gam * np.cosh((V - V3) / (2.0 * V4))
    dlam = gam * np.sinh((V - V3) / (2.0 * V4)) / (2.0 * V4)
    a11 = (-gCa * (dminf * (V - ECa) + minf) - gK * w - gL) / C
    a12 = -gK * (V - EK) / C
    a21 = dlam * (winf - w) + lam * dwinf
    a22 = -lam
    return a11, a12, a21, a22


@njit(cache=True)
def f_full(t, V, w, p, A, Om, phi, ratio):
    dV, dw = f_auto(V, w, p)
    Iac = A * (np.cos(Om * t) + ratio * np.cos(2.0 * Om * t + phi))
    return dV + Iac / p[0], dw


# ---------------------------------------------------------------- averaged field
@njit(cache=True)
def fbar(u, w, p, xi):
    n = xi.shape[0]
    su = 0.0
    sw = 0.0
    for k in range(n):
        dV, dw = f_auto(u + xi[k], w, p)
        su += dV
        sw += dw
    return su / n, sw / n


@njit(cache=True)
def jbar(u, w, p, xi):
    n = xi.shape[0]
    s11 = 0.0; s12 = 0.0; s21 = 0.0; s22 = 0.0
    for k in range(n):
        a11, a12, a21, a22 = jac_auto(u + xi[k], w, p)
        s11 += a11; s12 += a12; s21 += a21; s22 += a22
    return s11 / n, s12 / n, s21 / n, s22 / n


@njit(cache=True)
def wstar_of_u(u, p, xi):
    """Averaged w-nullcline: w = <lam w_inf>/<lam>."""
    C, gCa, gK, gL, ECa, EK, EL, V1, V2, V3, V4, gam, I0 = p
    n = xi.shape[0]
    a = 0.0; b = 0.0
    for k in range(n):
        V = u + xi[k]
        lam = gam * np.cosh((V - V3) / (2.0 * V4))
        winf = 0.5 * (1.0 + np.tanh((V - V3) / V4))
        a += lam * winf
        b += lam
    return a / b


@njit(cache=True)
def g_equil(u, p, xi):
    """Scalar equilibrium residual: averaged dV/dt on the averaged w-nullcline."""
    w = wstar_of_u(u, p, xi)
    du, dw = fbar(u, w, p, xi)
    return du


def equilibria(p, xi, umin=-90.0, umax=80.0, n=400):
    """All equilibria of the averaged system by sign scan + bisection."""
    us = np.linspace(umin, umax, n)
    gs = np.array([g_equil(u, p, xi) for u in us])
    roots = []
    for i in range(n - 1):
        if gs[i] == 0.0:
            roots.append(us[i])
        elif gs[i] * gs[i + 1] < 0:
            a, b = us[i], us[i + 1]
            ga = gs[i]
            for _ in range(80):
                m = 0.5 * (a + b)
                gm = g_equil(m, p, xi)
                if ga * gm <= 0:
                    b = m
                else:
                    a, ga = m, gm
            roots.append(0.5 * (a + b))
    out = []
    for u in roots:
        w = wstar_of_u(u, p, xi)
        out.append((u, w))
    return out


def eig_equil(p, xi, u, w):
    a11, a12, a21, a22 = jbar(u, w, p, xi)
    J = np.array([[a11, a12], [a21, a22]])
    return np.linalg.eigvals(J), J


def trace_at_equil(s, phi, p, ratio=1.0, nq=NQ):
    xi = xi_nodes(s, phi, ratio, nq)
    eq = equilibria(p, xi)
    assert len(eq) == 1, (s, phi, eq)
    u, w = eq[0]
    a11, a12, a21, a22 = jbar(u, w, p, xi)
    return a11 + a22, a11 * a22 - a12 * a21, u, w


def hopf_threshold(phi, p, ratio=1.0, smin=0.0, smax=60.0, nq=NQ, tol=1e-10):
    """Critical s at which the trace of the averaged Jacobian vanishes (det>0 checked)."""
    from scipy.optimize import brentq
    f = lambda s: trace_at_equil(s, phi, p, ratio, nq)[0]
    # bracket: trace >0 (unstable) at s=0 for I0=100 ; scan upward
    ss = np.linspace(smin, smax, 61)
    ts = [f(s) for s in ss]
    for i in range(60):
        if ts[i] > 0 and ts[i + 1] <= 0:
            sc = brentq(f, ss[i], ss[i + 1], xtol=tol)
            tr, det, u, w = trace_at_equil(sc, phi, p, ratio, nq)
            return sc, det, u, w
    return np.nan, np.nan, np.nan, np.nan


# ---------------------------------------------------------------- averaged flow, RK4 + Poincare map
@njit(cache=True)
def rk4_avg(u, w, p, xi, dt, nsteps):
    for _ in range(nsteps):
        k1u, k1w = fbar(u, w, p, xi)
        k2u, k2w = fbar(u + 0.5 * dt * k1u, w + 0.5 * dt * k1w, p, xi)
        k3u, k3w = fbar(u + 0.5 * dt * k2u, w + 0.5 * dt * k2w, p, xi)
        k4u, k4w = fbar(u + dt * k3u, w + dt * k3w, p, xi)
        u += dt * (k1u + 2 * k2u + 2 * k3u + k4u) / 6.0
        w += dt * (k1w + 2 * k2w + 2 * k3w + k4w) / 6.0
    return u, w


@njit(cache=True)
def traj_avg(u, w, p, xi, dt, nsteps, nsave):
    """Return saved trajectory (every nsave steps)."""
    m = nsteps // nsave + 1
    out = np.empty((m, 3))
    out[0, 0] = 0.0; out[0, 1] = u; out[0, 2] = w
    j = 1
    for i in range(1, nsteps + 1):
        k1u, k1w = fbar(u, w, p, xi)
        k2u, k2w = fbar(u + 0.5 * dt * k1u, w + 0.5 * dt * k1w, p, xi)
        k3u, k3w = fbar(u + 0.5 * dt * k2u, w + 0.5 * dt * k2w, p, xi)
        k4u, k4w = fbar(u + dt * k3u, w + dt * k3w, p, xi)
        u += dt * (k1u + 2 * k2u + 2 * k3u + k4u) / 6.0
        w += dt * (k1w + 2 * k2w + 2 * k3w + k4w) / 6.0
        if i % nsave == 0 and j < m:
            out[j, 0] = i * dt; out[j, 1] = u; out[j, 2] = w
            j += 1
    return out[:j]


@njit(cache=True)
def poincare_avg(u0, ustar, wstar, p, xi, dt, tmax, direction):
    """Return map on section w = wstar, u > ustar, crossing with dw/dt>0 (direction=+1 forward
    time, -1 backward time). Starts at (u0, wstar). Uses Henon's trick for the final step.
    Returns (u_return, T_return); u_return = nan if no return within tmax."""
    u = u0
    w = wstar
    t = 0.0
    sgn = 1.0 if direction > 0 else -1.0
    h = sgn * dt
    # leave the section first: integrate until w - wstar changes sign to negative then positive
    left = False
    nmax = int(tmax / dt) + 1
    for i in range(nmax):
        k1u, k1w = fbar(u, w, p, xi)
        k2u, k2w = fbar(u + 0.5 * h * k1u, w + 0.5 * h * k1w, p, xi)
        k3u, k3w = fbar(u + 0.5 * h * k2u, w + 0.5 * h * k2w, p, xi)
        k4u, k4w = fbar(u + h * k3u, w + h * k3w, p, xi)
        un = u + h * (k1u + 2 * k2u + 2 * k3u + k4u) / 6.0
        wn = w + h * (k1w + 2 * k2w + 2 * k3w + k4w) / 6.0
        t += dt
        if not left:
            if (wn - wstar) < -1e-9 * (1.0 + abs(wstar)):
                left = True
        else:
            if (w - wstar) < 0.0 and (wn - wstar) >= 0.0 and un > ustar:
                # Henon step: integrate with w as independent variable from w to wstar
                dw_needed = wstar - w
                # du/dw = fu/fw ; dt/dw = 1/fw   (in the direction of time 'sgn')
                f1u, f1w = fbar(u, w, p, xi)
                a1u = f1u / f1w; a1t = 1.0 / f1w
                f2u, f2w = fbar(u + 0.5 * dw_needed * a1u, w + 0.5 * dw_needed, p, xi)
                a2u = f2u / f2w; a2t = 1.0 / f2w
                f3u, f3w = fbar(u + 0.5 * dw_needed * a2u, w + 0.5 * dw_needed, p, xi)
                a3u = f3u / f3w; a3t = 1.0 / f3w
                f4u, f4w = fbar(u + dw_needed * a3u, w + dw_needed, p, xi)
                a4u = f4u / f4w; a4t = 1.0 / f4w
                uc = u + dw_needed * (a1u + 2 * a2u + 2 * a3u + a4u) / 6.0
                tc = t - dt + sgn * dw_needed * (a1t + 2 * a2t + 2 * a3t + a4t) / 6.0
                return uc, tc
        u = un
        w = wn
        if abs(u) > 1e3 or abs(w) > 1e3:
            return np.nan, np.nan
    return np.nan, np.nan


# ---------------------------------------------------------------- full system, RK4
@njit(cache=True)
def rk4_full_step(t, V, w, p, A, Om, phi, ratio, dt):
    k1V, k1w = f_full(t, V, w, p, A, Om, phi, ratio)
    k2V, k2w = f_full(t + 0.5 * dt, V + 0.5 * dt * k1V, w + 0.5 * dt * k1w, p, A, Om, phi, ratio)
    k3V, k3w = f_full(t + 0.5 * dt, V + 0.5 * dt * k2V, w + 0.5 * dt * k2w, p, A, Om, phi, ratio)
    k4V, k4w = f_full(t + dt, V + dt * k3V, w + dt * k3w, p, A, Om, phi, ratio)
    return V + dt * (k1V + 2 * k2V + 2 * k3V + k4V) / 6.0, w + dt * (k1w + 2 * k2w + 2 * k3w + k4w) / 6.0


@njit(cache=True)
def strobe_full(V, w, p, A, Om, phi, ratio, nper, nsub, t0):
    """Integrate nper forcing periods from time t0 with nsub RK4 steps/period.
    Returns stroboscopic samples (V_k, w_k) at t0 + k T, k=0..nper, and final time."""
    T = 2.0 * np.pi / Om
    dt = T / nsub
    out = np.empty((nper + 1, 2))
    out[0, 0] = V; out[0, 1] = w
    t = t0
    for k in range(nper):
        for j in range(nsub):
            V, w = rk4_full_step(t, V, w, p, A, Om, phi, ratio, dt)
            t += dt
        out[k + 1, 0] = V; out[k + 1, 1] = w
    return out, t


@njit(cache=True)
def dense_full(V, w, p, A, Om, phi, ratio, nper, nsub, t0, every):
    """Dense output every 'every' RK4 steps."""
    T = 2.0 * np.pi / Om
    dt = T / nsub
    ntot = nper * nsub
    m = ntot // every + 1
    out = np.empty((m, 3))
    out[0, 0] = t0; out[0, 1] = V; out[0, 2] = w
    t = t0
    j = 1
    for i in range(1, ntot + 1):
        V, w = rk4_full_step(t, V, w, p, A, Om, phi, ratio, dt)
        t += dt
        if i % every == 0 and j < m:
            out[j, 0] = t; out[j, 1] = V; out[j, 2] = w
            j += 1
    return out[:j], V, w, t


@njit(cache=True)
def strobo_map_var(V, w, p, A, Om, phi, ratio, nsub):
    """One-period stroboscopic map from t=0 together with the monodromy matrix (variational eqs)."""
    T = 2.0 * np.pi / Om
    dt = T / nsub
    # state y = (V, w, m11, m12, m21, m22)
    y = np.array([V, w, 1.0, 0.0, 0.0, 1.0])
    t = 0.0
    k = np.empty((4, 6))
    for j in range(nsub):
        yy = y.copy()
        for st in range(4):
            if st == 0:
                tt = t; yy = y
            elif st == 1:
                tt = t + 0.5 * dt; yy = y + 0.5 * dt * k[0]
            elif st == 2:
                tt = t + 0.5 * dt; yy = y + 0.5 * dt * k[1]
            else:
                tt = t + dt; yy = y + dt * k[2]
            dV, dw = f_full(tt, yy[0], yy[1], p, A, Om, phi, ratio)
            a11, a12, a21, a22 = jac_auto(yy[0], yy[1], p)
            k[st, 0] = dV
            k[st, 1] = dw
            k[st, 2] = a11 * yy[2] + a12 * yy[4]
            k[st, 3] = a11 * yy[3] + a12 * yy[5]
            k[st, 4] = a21 * yy[2] + a22 * yy[4]
            k[st, 5] = a21 * yy[3] + a22 * yy[5]
        y = y + dt * (k[0] + 2 * k[1] + 2 * k[2] + k[3]) / 6.0
        t += dt
    return y


def small_response_floquet(p, A, Om, phi, ratio, nsub, V0, w0, maxit=50):
    """Newton iteration for the fixed point of the stroboscopic map (small forced response)
    and its Floquet multipliers."""
    V, w = V0, w0
    for it in range(maxit):
        y = strobo_map_var(V, w, p, A, Om, phi, ratio, nsub)
        r = np.array([y[0] - V, y[1] - w])
        M = np.array([[y[2], y[3]], [y[4], y[5]]])
        d = np.linalg.solve(M - np.eye(2), -r)
        V += d[0]; w += d[1]
        if np.max(np.abs(d)) < 1e-12:
            break
    y = strobo_map_var(V, w, p, A, Om, phi, ratio, nsub)
    M = np.array([[y[2], y[3]], [y[4], y[5]]])
    mu = np.linalg.eigvals(M)
    return V, w, mu, M


def floquet_threshold(phi, Om, p, ratio=1.0, nsub=64, sguess=(15.0, 40.0), tol=1e-9):
    """s at which the small periodic response loses stability (max|mu| = 1)."""
    from scipy.optimize import brentq
    C = p[0]
    state = {"V": None, "w": None}

    def h(s):
        A = s * C * Om
        xi = xi_nodes(s, phi, ratio)
        eq = equilibria(p, xi)
        u, w = eq[0]
        # initial guess: averaged equilibrium shifted by xi(0)
        V0 = u + xi[0]
        V, ww, mu, M = small_response_floquet(p, A, Om, phi, ratio, nsub, V0, w)
        return np.max(np.abs(mu)) - 1.0

    sc = brentq(h, sguess[0], sguess[1], xtol=tol)
    A = sc * C * Om
    xi = xi_nodes(sc, phi, ratio)
    u, w = equilibria(p, xi)[0]
    V, ww, mu, M = small_response_floquet(p, A, Om, phi, ratio, nsub, u + xi[0], w)
    return sc, mu


# ---------------------------------------------------------------- Lyapunov coefficient (Kuznetsov)
def lyapunov_coefficient(p, xi, u, w, h=0.25):
    """First Lyapunov coefficient l1 at a Hopf point of the averaged 2D system via the
    invariant formula (Kuznetsov, Elements of Applied Bifurcation Theory, Eq. 3.20 / 5.39):
        l1 = (1/(2 w0)) Re[ <p,C(q,q,qb)> - 2<p,B(q,A^{-1}B(q,qb))> + <p,B(qb,(2 i w0 I - A)^{-1} B(q,q))> ]
    with <p,q> = conj(p).q = 1. Multilinear forms from finite differences of the analytic Jacobian."""
    x0 = np.array([u, w])

    def J(x):
        a11, a12, a21, a22 = jbar(x[0], x[1], p, xi)
        return np.array([[a11, a12], [a21, a22]])

    def F(x):
        return np.array(fbar(x[0], x[1], p, xi))

    A = J(x0)
    # Hessian tensor H[i,j,k] = d2F_i/dx_j dx_k via central differences of J
    hs = np.array([h, h * 0.01])  # w scale is ~0.01 of u scale
    H = np.zeros((2, 2, 2))
    for k in range(2):
        e = np.zeros(2); e[k] = hs[k]
        H[:, :, k] = (J(x0 + e) - J(x0 - e)) / (2 * hs[k])
    # Third-derivative tensor T[i,j,k,l] via second differences of J
    T = np.zeros((2, 2, 2, 2))
    for k in range(2):
        for l in range(2):
            ek = np.zeros(2); ek[k] = hs[k]
            el = np.zeros(2); el[l] = hs[l]
            T[:, :, k, l] = (J(x0 + ek + el) - J(x0 + ek - el) - J(x0 - ek + el) + J(x0 - ek - el)) / (4 * hs[k] * hs[l])

    def B(a, b):
        return np.einsum('ijk,j,k->i', H, a, b)

    def Cf(a, b, c):
        return np.einsum('ijkl,j,k,l->i', T, a, b, c)

    ev, vec = np.linalg.eig(A)
    i = np.argmax(ev.imag)
    lam = ev[i]
    w0 = lam.imag
    q = vec[:, i]
    evl, vecl = np.linalg.eig(A.T)
    j = np.argmin(np.abs(evl - np.conj(lam)))
    pp = vecl[:, j]
    pp = pp / np.conj(np.vdot(pp, q))  # <p,q> = conj(p).q = 1
    qb = np.conj(q)
    term1 = np.vdot(pp, Cf(q, q, qb))
    term2 = -2 * np.vdot(pp, B(q, np.linalg.solve(A, B(q, qb))))
    term3 = np.vdot(pp, B(qb, np.linalg.solve(2j * w0 * np.eye(2) - A, B(q, q))))
    l1 = (term1 + term2 + term3).real / (2 * w0)
    return l1, w0, lam


# ---------------------------------------------------------------- batched full-system integration (basins)
@njit(parallel=True, cache=True)
def batch_strobo_amp(V0s, w0s, p, A, Om, phi, ratio, nper, nsub, nlast):
    """For each initial condition integrate nper periods (forcing starts at theta=0) and return
    the stroboscopic peak-to-peak of V over the last nlast periods and final (V,w)."""
    n = V0s.shape[0]
    amp = np.empty(n)
    Vf = np.empty(n)
    wf = np.empty(n)
    T = 2.0 * np.pi / Om
    dt = T / nsub
    for i in prange(n):
        V = V0s[i]; w = w0s[i]
        t = 0.0
        vmin = 1e9; vmax = -1e9
        for k in range(nper):
            for j in range(nsub):
                V, w = rk4_full_step(t, V, w, p, A, Om, phi, ratio, dt)
                t += dt
            if k >= nper - nlast:
                if V < vmin: vmin = V
                if V > vmax: vmax = V
        amp[i] = vmax - vmin
        Vf[i] = V; wf[i] = w
    return amp, Vf, wf


# ---------------------------------------------------------------- tabulated averaged system (fast)
# Averaged field structure (exact, since <xi>=0):
#   C dU/dt = a(u) - gK w (u - EK) ,   dw/dt = b(u) - c(u) w
#   a(u) = < I0 - gCa m_inf(u+xi)(u+xi-ECa) - gL (u+xi-EL) >,  b = <lam w_inf>,  c = <lam>
# Tabulated on a uniform grid with analytic derivatives -> cubic Hermite interpolation.
UGRID_MIN, UGRID_MAX, UGRID_H = -160.0, 140.0, 0.02


def build_table(p, xi):
    C, gCa, gK, gL, ECa, EK, EL, V1, V2, V3, V4, gam, I0 = p
    u = np.arange(UGRID_MIN, UGRID_MAX + 0.5 * UGRID_H, UGRID_H)
    V = u[:, None] + xi[None, :]
    t1 = np.tanh((V - V1) / V2); t3 = np.tanh((V - V3) / V4)
    minf = 0.5 * (1 + t1); dminf = 0.5 * (1 - t1 ** 2) / V2
    winf = 0.5 * (1 + t3); dwinf = 0.5 * (1 - t3 ** 2) / V4
    lam = gam * np.cosh((V - V3) / (2 * V4)); dlam = gam * np.sinh((V - V3) / (2 * V4)) / (2 * V4)
    a = np.mean(I0 - gCa * minf * (V - ECa) - gL * (V - EL), axis=1)
    da = np.mean(-gCa * (dminf * (V - ECa) + minf) - gL, axis=1)
    b = np.mean(lam * winf, axis=1)
    db = np.mean(dlam * winf + lam * dwinf, axis=1)
    c = np.mean(lam, axis=1)
    dc = np.mean(dlam, axis=1)
    tab = np.empty((6, u.size))
    tab[0] = a; tab[1] = da; tab[2] = b; tab[3] = db; tab[4] = c; tab[5] = dc
    return tab


@njit(cache=True, inline='always')
def _hermite(tab, row, u):
    x = (u - UGRID_MIN) / UGRID_H
    i = int(x)
    n = tab.shape[1]
    if i < 0:
        i = 0
    elif i > n - 2:
        i = n - 2
    t = x - i
    f0 = tab[row, i]; f1 = tab[row, i + 1]
    d0 = tab[row + 1, i] * UGRID_H; d1 = tab[row + 1, i + 1] * UGRID_H
    h00 = (1 + 2 * t) * (1 - t) ** 2; h10 = t * (1 - t) ** 2
    h01 = t * t * (3 - 2 * t); h11 = t * t * (t - 1)
    return h00 * f0 + h10 * d0 + h01 * f1 + h11 * d1


@njit(cache=True)
def fbar_tab(u, w, p, tab):
    C = p[0]; gK = p[2]; EK = p[5]
    a = _hermite(tab, 0, u)
    b = _hermite(tab, 2, u)
    c = _hermite(tab, 4, u)
    return (a - gK * w * (u - EK)) / C, b - c * w


@njit(cache=True)
def rk4_tab(u, w, p, tab, dt, nsteps):
    for _ in range(nsteps):
        k1u, k1w = fbar_tab(u, w, p, tab)
        k2u, k2w = fbar_tab(u + 0.5 * dt * k1u, w + 0.5 * dt * k1w, p, tab)
        k3u, k3w = fbar_tab(u + 0.5 * dt * k2u, w + 0.5 * dt * k2w, p, tab)
        k4u, k4w = fbar_tab(u + dt * k3u, w + dt * k3w, p, tab)
        u += dt * (k1u + 2 * k2u + 2 * k3u + k4u) / 6.0
        w += dt * (k1w + 2 * k2w + 2 * k3w + k4w) / 6.0
    return u, w


@njit(cache=True)
def traj_tab(u, w, p, tab, dt, nsteps, nsave):
    m = nsteps // nsave + 1
    out = np.empty((m, 3))
    out[0, 0] = 0.0; out[0, 1] = u; out[0, 2] = w
    j = 1
    for i in range(1, nsteps + 1):
        k1u, k1w = fbar_tab(u, w, p, tab)
        k2u, k2w = fbar_tab(u + 0.5 * dt * k1u, w + 0.5 * dt * k1w, p, tab)
        k3u, k3w = fbar_tab(u + 0.5 * dt * k2u, w + 0.5 * dt * k2w, p, tab)
        k4u, k4w = fbar_tab(u + dt * k3u, w + dt * k3w, p, tab)
        u += dt * (k1u + 2 * k2u + 2 * k3u + k4u) / 6.0
        w += dt * (k1w + 2 * k2w + 2 * k3w + k4w) / 6.0
        if i % nsave == 0 and j < m:
            out[j, 0] = i * dt; out[j, 1] = u; out[j, 2] = w
            j += 1
    return out[:j]


@njit(cache=True)
def poincare_tab(u0, ustar, wstar, p, tab, dt, tmax):
    """Forward-time return map on the section w = wstar, u > ustar, dw/dt > 0.
    Returns (u_return, T_return, umax, umin) ; nan if no return."""
    u = u0; w = wstar; t = 0.0
    left = False
    umax = u0; umin = u0
    nmax = int(tmax / dt) + 1
    for i in range(nmax):
        k1u, k1w = fbar_tab(u, w, p, tab)
        k2u, k2w = fbar_tab(u + 0.5 * dt * k1u, w + 0.5 * dt * k1w, p, tab)
        k3u, k3w = fbar_tab(u + 0.5 * dt * k2u, w + 0.5 * dt * k2w, p, tab)
        k4u, k4w = fbar_tab(u + dt * k3u, w + dt * k3w, p, tab)
        un = u + dt * (k1u + 2 * k2u + 2 * k3u + k4u) / 6.0
        wn = w + dt * (k1w + 2 * k2w + 2 * k3w + k4w) / 6.0
        t += dt
        if un > umax: umax = un
        if un < umin: umin = un
        if not left:
            if (wn - wstar) < -1e-7 * (1.0 + abs(wstar)):
                left = True
        else:
            if (w - wstar) < 0.0 and (wn - wstar) >= 0.0 and un > ustar:
                dwn = wstar - w
                f1u, f1w = fbar_tab(u, w, p, tab); a1u = f1u / f1w; a1t = 1.0 / f1w
                f2u, f2w = fbar_tab(u + 0.5 * dwn * a1u, w + 0.5 * dwn, p, tab); a2u = f2u / f2w; a2t = 1.0 / f2w
                f3u, f3w = fbar_tab(u + 0.5 * dwn * a2u, w + 0.5 * dwn, p, tab); a3u = f3u / f3w; a3t = 1.0 / f3w
                f4u, f4w = fbar_tab(u + dwn * a3u, w + dwn, p, tab); a4u = f4u / f4w; a4t = 1.0 / f4w
                uc = u + dwn * (a1u + 2 * a2u + 2 * a3u + a4u) / 6.0
                tc = t - dt + dwn * (a1t + 2 * a2t + 2 * a3t + a4t) / 6.0
                return uc, tc, umax, umin
        u = un; w = wn
        if abs(u) > 1e3 or abs(w) > 1e3:
            return np.nan, np.nan, np.nan, np.nan
    return np.nan, np.nan, np.nan, np.nan


def cycles_at(p, tab, ustar, wstar, dt=0.02, tmax=2000.0, du=0.25, umax_scan=90.0, nscan=None):
    """Find all limit cycles crossing the section (roots of g(u0)=P(u0)-u0) with stability.
    Returns list of dicts: u0, T, umax, umin, mult (=1+g')."""
    us = np.arange(ustar + du, ustar + umax_scan, du)
    g = np.full(us.size, np.nan)
    for i, u0 in enumerate(us):
        ur, T, a, b = poincare_tab(u0, ustar, wstar, p, tab, dt, tmax)
        g[i] = ur - u0
    out = []
    for i in range(us.size - 1):
        if np.isfinite(g[i]) and np.isfinite(g[i + 1]) and g[i] * g[i + 1] < 0:
            a, b = us[i], us[i + 1]; ga = g[i]
            for _ in range(60):
                m = 0.5 * (a + b)
                gm = poincare_tab(m, ustar, wstar, p, tab, dt, tmax)[0] - m
                if ga * gm <= 0:
                    b = m
                else:
                    a, ga = m, gm
                if b - a < 1e-11:
                    break
            u0 = 0.5 * (a + b)
            ur, T, umx, umn = poincare_tab(u0, ustar, wstar, p, tab, dt, tmax)
            h = 1e-4
            gp = (poincare_tab(u0 + h, ustar, wstar, p, tab, dt, tmax)[0] - (u0 + h)
                  - poincare_tab(u0 - h, ustar, wstar, p, tab, dt, tmax)[0] + (u0 - h)) / (2 * h)
            out.append(dict(u0=u0, T=T, umax=umx, umin=umn, mult=1.0 + gp, stable=gp < 0))
    return out, us, g
