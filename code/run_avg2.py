"""Averaged-system computations: Hopf curve, Lyapunov coefficient, fold of large cycles,
one-parameter diagrams at selected phases, nullclines, harmonic-ratio and parameter sensitivity."""
import numpy as np, time, json, sys
from ml_core import *
from ml_core import _hermite

p = P0.copy()
t0 = time.time()
res = {}

# ---------- A. Hopf curve and first Lyapunov coefficient
import os
if os.path.exists("data_hopf.npz"):
    d = np.load("data_hopf.npz"); phis = d['phis']; sH = d['sH']; l1 = d['l1']; bautin = d['bautin']
    print("A loaded", flush=True)
else:
  phis = np.linspace(0, 2 * np.pi, 145)
  sH = np.empty_like(phis); l1 = np.empty_like(phis); w0 = np.empty_like(phis); detH = np.empty_like(phis)
  ueq = np.empty_like(phis); weq = np.empty_like(phis)
  for i, ph in enumerate(phis):
      sc, det, u, w = hopf_threshold(ph, p)
      xi = xi_nodes(sc, ph)
      l, om, lam = lyapunov_coefficient(p, xi, u, w)
      sH[i] = sc; l1[i] = l; w0[i] = om; detH[i] = det; ueq[i] = u; weq[i] = w
  print("A done", time.time() - t0, flush=True)
  # Bautin points: zeros of l1(phi)
  from scipy.optimize import brentq
  def l1_of_phi(ph):
      sc, det, u, w = hopf_threshold(ph, p)
      return lyapunov_coefficient(p, xi_nodes(sc, ph), u, w)[0]
  bautin = []
  for i in range(len(phis) - 1):
      if l1[i] * l1[i + 1] < 0:
          phb = brentq(l1_of_phi, phis[i], phis[i + 1], xtol=1e-8)
          bautin.append((phb, hopf_threshold(phb, p)[0]))
  print("Bautin points", bautin, flush=True)
  np.savez("data_hopf.npz", phis=phis, sH=sH, l1=l1, w0=w0, detH=detH, ueq=ueq, weq=weq, bautin=np.array(bautin))


def large_cycle_exists(p, tab, ustar, wstar, dt=0.02, u_start=70.0, nit=300, amp_min=0.5):
    u0 = ustar + u_start
    last = None
    for k in range(nit):
        ur, T, umx, umn = poincare_tab(u0, ustar, wstar, p, tab, dt, 3000.0)
        if not np.isfinite(ur):
            return False, None
        if abs(ur - u0) < 1e-10:
            return (umx - umn) > amp_min, dict(u0=ur, T=T, umax=umx, umin=umn)
        u0 = ur
        last = dict(u0=ur, T=T, umax=umx, umin=umn)
    return (last['umax'] - last['umin']) > amp_min, last

def fold_s(ph, p, s_lo, s_hi, tol=2e-4):
    """bisection on s for disappearance of the large cycle; s_lo must have it, s_hi must not"""
    def exists(s):
        xi = xi_nodes(s, ph); tab = build_table(p, xi); us, ws = equilibria(p, xi)[0]
        return large_cycle_exists(p, tab, us, ws)[0]
    assert exists(s_lo), (ph, s_lo)
    while exists(s_hi):
        s_hi += 2.0
        if s_hi > 80: return np.nan
    while s_hi - s_lo > tol:
        m = 0.5 * (s_lo + s_hi)
        if exists(m): s_lo = m
        else: s_hi = m
    return 0.5 * (s_lo + s_hi)


phis2 = np.load('data_fold.npz')['phis2']; sSN = np.load('data_fold.npz')['sSN']
# ---------- D. averaged nullclines / effective currents for Fig 2
ugrid = np.linspace(-80, 60, 561)
nullc = {}
for name, s, ph in [("s0", 0.0, 0.0), ("pi2", 24.0, np.pi / 2), ("3pi2", 24.0, 3 * np.pi / 2), ("0", 24.0, 0.0)]:
    xi = xi_nodes(s, ph); tab = build_table(p, xi)
    a = np.array([_hermite(tab, 0, u) for u in ugrid])
    b = np.array([_hermite(tab, 2, u) for u in ugrid])
    c = np.array([_hermite(tab, 4, u) for u in ugrid])
    nullc[name] = np.vstack([ugrid, a, b, c])  # u-nullcline: w = a/(gK (u-EK)); w-nullcline: w=b/c
np.savez("data_nullclines.npz", **nullc)

# ---------- E. harmonic-ratio comparison at equal RMS
ratios = [0.0, 0.25, 0.5, 1.0, 2.0]
phg = np.linspace(0, 2 * np.pi, 49)
ratio_res = {}
for r in ratios:
    fac = np.sqrt(2.0 / (1.0 + r * r))  # s = s_rms * fac
    if r == 0.0:
        sc = hopf_threshold(0.0, p, ratio=0.0)[0]
        ratio_res[str(r)] = np.array([[0.0, sc / fac]])
    else:
        rows = []
        for ph in phg:
            sc = hopf_threshold(ph, p, ratio=r)[0]
            rows.append([ph, sc / fac])
        ratio_res[str(r)] = np.array(rows)
    print("ratio", r, ratio_res[str(r)][:, 1].min(), ratio_res[str(r)][:, 1].max(), flush=True)
np.savez("data_ratio.npz", **{k.replace('.', '_'): v for k, v in ratio_res.items()})

# ---------- F. parameter sensitivity: s_H, l1 and s_SN at pi/2 and 3pi/2
variants = [("base", {}), ("I0=90", {12: 90.0}), ("I0=110", {12: 110.0}), ("gCa=3.96", {1: 3.96}), ("gCa=4.84", {1: 4.84}),
            ("gam=0.036", {11: 0.036}), ("gam=0.044", {11: 0.044})]
sens = []
for name, ch in variants:
    pp = P0.copy()
    for k, v in ch.items(): pp[k] = v
    row = [name]
    for ph in [np.pi / 2, 3 * np.pi / 2]:
        sc, det, u, w = hopf_threshold(ph, pp)
        l = lyapunov_coefficient(pp, xi_nodes(sc, ph), u, w)[0]
        ssn = fold_s(ph, pp, max(sc - 0.5, 1.0), sc + 1.0) if np.isfinite(sc) else np.nan
        row += [sc, l, ssn]
    sens.append(row)
    print("sens", row, flush=True)
json.dump(sens, open("data_sens.json", "w"))
print("ALL DONE", time.time() - t0, flush=True)
