import numpy as np, time
from ml_core import *
p = P0.copy(); C = p[0]; Om = 5.0
bs = np.load("data_basins.npz")
out = {}
t0 = time.time()
for k, (Vlo, Vhi, wlo, whi) in enumerate([(-60, 10, 0.08, 0.48), (-30, 35, 0.18, 0.58)]):
    s = float(bs[f's{k}']); ph = float(bs[f'phi{k}']); A = s * C * Om
    nV = nw = 71
    Vg = np.linspace(Vlo, Vhi, nV); wg = np.linspace(wlo, whi, nw); VV, WW = np.meshgrid(Vg, wg)
    nper = int(10000.0 / (2 * np.pi / Om))
    amp, Vf, wf = batch_strobo_amp(VV.ravel(), WW.ravel(), p, A, Om, ph, 1.0, nper, 64, 1500)
    xi = xi_nodes(s, ph); tab = build_table(p, xi)
    amp_avg = np.empty(VV.size)
    for i, (V0, w0) in enumerate(zip(VV.ravel(), WW.ravel())):
        u, w = rk4_tab(V0 - xi[0], w0, p, tab, 0.05, int(10000.0 / 0.05))
        tr = traj_tab(u, w, p, tab, 0.05, int(400.0 / 0.05), 10)
        amp_avg[i] = tr[:, 1].max() - tr[:, 1].min()
    out[f"Vg{k}"] = Vg; out[f"wg{k}"] = wg; out[f"amp{k}"] = amp.reshape(nw, nV); out[f"ampavg{k}"] = amp_avg.reshape(nw, nV)
    print(f"zoom case {k}: frac_sup_full={np.mean(amp<=10):.4f} frac_sup_avg={np.mean(amp_avg<=10):.4f} agree={np.mean((amp>10)==(amp_avg>10)):.4f}", time.time() - t0, flush=True)
np.savez("data_basins_zoom.npz", **out)
