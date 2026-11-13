import numpy as np
from ml_core import *
p = P0.copy()
hop = np.load("data_hopf.npz"); fold = np.load("data_fold.npz")
out = {}
for name, ph in [("pi2", np.pi/2), ("3pi2", 3*np.pi/2), ("0", 0.0)]:
    sh = np.interp(ph, hop['phis'], hop['sH']); ssn = np.interp(ph, fold['phis2'], fold['sSN'])
    s_vals = np.arange(sh - 0.6, max(sh, ssn) + 0.12, 0.02)
    rows = []
    for s in s_vals:
        xi = xi_nodes(s, ph); tab = build_table(p, xi); us, ws = equilibria(p, xi)[0]
        tr, det, _, _ = trace_at_equil(s, ph, p)
        cyc, _, _ = cycles_at(p, tab, us, ws, dt=0.02, du=0.2, tmax=4000.0)
        for c in cyc:
            rows.append([s, us, ws, tr, c['u0'], c['T'], c['umax'], c['umin'], c['mult'], 1.0 if c['stable'] else 0.0])
        if not cyc:
            rows.append([s, us, ws, tr] + [np.nan]*6)
    out[name] = np.array(rows); print(name, "done", flush=True)
np.savez("data_oneparam_zoom.npz", **out)
