"""Full (non-autonomous) system computations: Floquet thresholds, finite-frequency convergence,
fold of the large attractor, basins, switching protocol, hysteresis sweep."""
import numpy as np, time, json, sys
from ml_core import *

p = P0.copy()
C = p[0]
t0 = time.time()
stage = sys.argv[1] if len(sys.argv) > 1 else "all"

# ---------- A. Floquet thresholds vs phase at Omega = 5, and convergence in Omega at selected phases
if stage in ("all", "floquet"):
    phg = np.linspace(0, 2 * np.pi, 49)
    rows = []
    for Om in [2.0, 5.0]:
        for ph in phg:
            sc, mu = floquet_threshold(ph, Om, p, nsub=64, sguess=(10.0, 45.0))
            rows.append([Om, ph, sc, np.angle(mu[0])])
            print(f"Om={Om} phi={ph:.3f} sF={sc:.6f} arg={abs(np.angle(mu[0])):.4f}", flush=True)
    np.savez("data_floquet_phi.npz", rows=np.array(rows))
    conv = []
    for ph in [np.pi / 2, 3 * np.pi / 2, np.pi / 4, 0.0]:
        for Om in [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
            nsub = 64 if Om >= 5 else int(64 * 5 / Om)
            sc, mu = floquet_threshold(ph, Om, p, nsub=nsub, sguess=(10.0, 45.0))
            sc2, mu2 = floquet_threshold(ph, Om, p, nsub=2 * nsub, sguess=(10.0, 45.0))
            conv.append([ph, Om, sc, sc2, abs(np.angle(mu[0]))])
            print(f"conv phi={ph:.3f} Om={Om} sF={sc:.7f} sF(2x steps)={sc2:.7f}", flush=True)
    np.savez("data_floquet_conv.npz", conv=np.array(conv))
    print("A done", time.time() - t0, flush=True)


# ---------- helpers for large-attractor detection in the full system
def large_attractor_full(s, ph, Om, V0, w0, nper, nsub=64, nlast=3000, amp_min=10.0):
    A = s * C * Om
    out, t = strobe_full(V0, w0, p, A, Om, ph, 1.0, nper, nsub, 0.0)
    V = out[-nlast:, 0]
    return (V.max() - V.min()) > amp_min, out[-1, 0], out[-1, 1], V.max() - V.min()


# ---------- B. fold of the large attractor in the full system at Omega=5 (and 2) for selected phases
if stage in ("all", "fold"):
    hop = np.load("data_hopf.npz"); fold = np.load("data_fold.npz")
    res = []
    for Om in [5.0, 2.0]:
        Tper = 2 * np.pi / Om
        nper = int(40000.0 / Tper)
        ph_list = [np.pi / 2, 3 * np.pi / 2] if Om == 2.0 else [np.pi / 2, 3 * np.pi / 2, 1.25, 1.9, 4.0, 5.5, 3.6, 2.0]
        for ph in ph_list:
            sH = np.interp(ph, hop['phis'], hop['sH']); sSN = np.interp(ph, fold['phis2'], fold['sSN'])
            # start from the unforced-cycle point on the large attractor, continue upward in s
            V0, w0 = -6.73, 0.158
            s_lo = max(sH - 0.3, 1.0); s_hi = sSN + 0.6
            ok, V0, w0, amp = large_attractor_full(s_lo, ph, Om, V0, w0, nper)
            assert ok, (ph, Om, s_lo, amp)
            Vl, wl = V0, w0
            while True:
                ok, _, _, amp = large_attractor_full(s_hi, ph, Om, Vl, wl, nper)
                if not ok: break
                s_hi += 1.0
            while s_hi - s_lo > 2e-3:
                m = 0.5 * (s_lo + s_hi)
                ok, Vm, wm, amp = large_attractor_full(m, ph, Om, Vl, wl, nper)
                if ok:
                    s_lo = m; Vl, wl = Vm, wm
                else:
                    s_hi = m
            res.append([Om, ph, sH, sSN, 0.5 * (s_lo + s_hi)])
            print(f"fold full: Om={Om} phi={ph:.3f} sH_avg={sH:.4f} sSN_avg={sSN:.4f} sSN_full={0.5*(s_lo+s_hi):.4f}", flush=True)
    np.savez("data_fold_full.npz", res=np.array(res))
    print("B done", time.time() - t0, flush=True)


# ---------- C. basins of attraction in the full system (and averaged boundary) inside the bistable band
if stage in ("all", "basins"):
    hop = np.load("data_hopf.npz"); fold = np.load("data_fold.npz")
    Om = 5.0
    cases = []
    for ph in [3 * np.pi / 2, np.pi / 2]:
        sH = np.interp(ph, hop['phis'], hop['sH']); sSN = np.interp(ph, fold['phis2'], fold['sSN'])
        s = sH + 0.5 * (sSN - sH)  # middle of the bistable band
        cases.append((ph, s))
    nV, nw = 81, 81
    Vg = np.linspace(-80, 60, nV); wg = np.linspace(0.0, 0.8, nw)
    VV, WW = np.meshgrid(Vg, wg)
    nper = int(15000.0 / (2 * np.pi / Om))
    out = {}
    for k, (ph, s) in enumerate(cases):
        A = s * C * Om
        amp, Vf, wf = batch_strobo_amp(VV.ravel(), WW.ravel(), p, A, Om, ph, 1.0, nper, 64, 2000)
        out[f"amp{k}"] = amp.reshape(nw, nV); out[f"s{k}"] = s; out[f"phi{k}"] = ph
        # averaged system: unstable cycle = basin boundary, stable cycle, equilibrium
        xi = xi_nodes(s, ph); tab = build_table(p, xi); us, ws = equilibria(p, xi)[0]
        cyc, _, _ = cycles_at(p, tab, us, ws, dt=0.02, du=0.25)
        curves = []
        for c in cyc:
            tr = traj_tab(c['u0'], ws, p, tab, 0.02, int(c['T'] / 0.02) + 2, 5)
            curves.append(tr)
        out[f"eq{k}"] = np.array([us, ws])
        for j, c in enumerate(cyc):
            out[f"cyc{k}_{j}"] = curves[j]; out[f"cycstable{k}_{j}"] = float(c['stable'])
        # averaged-system basin classification on the same grid (interpret (V,w) at t=0 as (u + xi(0), w))
        amp_avg = np.empty(VV.size)
        for i, (V0, w0) in enumerate(zip(VV.ravel(), WW.ravel())):
            u0 = V0 - xi[0]
            u, w = rk4_tab(u0, w0, p, tab, 0.05, int(15000.0 / 0.05))
            tr = traj_tab(u, w, p, tab, 0.05, int(400.0 / 0.05), 10)
            amp_avg[i] = tr[:, 1].max() - tr[:, 1].min()
        out[f"ampavg{k}"] = amp_avg.reshape(nw, nV)
        print(f"basins case {k}: phi={ph:.3f} s={s:.4f} frac_osc_full={np.mean(amp>10):.4f} frac_osc_avg={np.mean(amp_avg>10):.4f} agreement={np.mean((amp>10)==(amp_avg>10)):.4f}", flush=True)
    np.savez("data_basins.npz", Vg=Vg, wg=wg, **out)
    print("C done", time.time() - t0, flush=True)


# ---------- D. switching protocol at s = 24, Omega = 5: dense trace and dependence on switching moment
if stage in ("all", "switch"):
    Om = 5.0; Tper = 2 * np.pi / Om; nsub = 64
    s = 24.0; A = s * C * Om
    # dense trajectory: phase pi/2 (0-1500 ms) -> 3pi/2 (1500-3000) -> pi/2 (3000-4500), state continuous
    segs = []
    V, w, t = -6.73, 0.158, 0.0
    for ph, dur in [(np.pi / 2, 1500.0), (3 * np.pi / 2, 1500.0), (np.pi / 2, 1500.0)]:
        nper = int(round(dur / Tper))
        dens, V, w, t = dense_full(V, w, p, A, Om, ph, 1.0, nper, nsub, t, 4)
        segs.append(dens)
    np.savez("data_switch_trace.npz", seg0=segs[0], seg1=segs[1], seg2=segs[2], s=s, Om=Om)
    # also the stroboscopic (slow) component u = V - xi(theta) with theta = Om t mod 2pi
    # dependence on the switching moment: start on the slow attractor at phi=pi/2, switch to 3pi/2 at N different periods
    # success = strobo amplitude < 1 mV in the window [2500, 3000] ms after the switch; also time to suppression (< 2 mV)
    out0, t = strobe_full(-6.73, 0.158, p, A, Om, np.pi / 2, 1.0, int(2000.0 / Tper), nsub, 0.0)
    V0, w0 = out0[-1]
    nsw = 80
    res_sw = []
    slow_T_periods = int(round(75.0 / Tper))  # ~ one slow period
    for j in range(nsw):
        # advance j/nsw of a slow period on the oscillatory attractor
        o, _ = strobe_full(V0, w0, p, A, Om, np.pi / 2, 1.0, int(j * slow_T_periods / nsw), nsub, 0.0)
        Vs, ws = o[-1]
        o2, _ = strobe_full(Vs, ws, p, A, Om, 3 * np.pi / 2, 1.0, int(3000.0 / Tper), nsub, 0.0)
        Vst = o2[:, 0]
        # time to suppression: first period after which the running strobo amplitude over 100 ms stays < 2 mV
        win = int(100.0 / Tper)
        tsup = np.nan
        for k in range(0, len(Vst) - win):
            seg = Vst[k:k + win]
            if seg.max() - seg.min() < 2.0:
                tsup = k * Tper; break
        amp_end = Vst[-int(500 / Tper):].max() - Vst[-int(500 / Tper):].min()
        res_sw.append([j / nsw, Vs, ws, tsup, amp_end])
    np.savez("data_switch_moment.npz", res=np.array(res_sw))
    print("switching moments done", time.time() - t0, flush=True)
    # success fraction vs s inside and above the bistable band at phi = 3pi/2
    hop = np.load("data_hopf.npz"); fold = np.load("data_fold.npz")
    sH = np.interp(3 * np.pi / 2, hop['phis'], hop['sH']); sSN = np.interp(3 * np.pi / 2, fold['phis2'], fold['sSN'])
    s_vals = np.concatenate([np.linspace(sH - 0.2, sSN + 0.4, 25), [24.0]])
    rows = []
    for s in s_vals:
        A = s * C * Om
        out0, t = strobe_full(-6.73, 0.158, p, A, Om, np.pi / 2, 1.0, int(2000.0 / Tper), nsub, 0.0)
        V0, w0 = out0[-1]
        nsw2 = 40
        Vs = np.empty(nsw2); ws = np.empty(nsw2)
        for j in range(nsw2):
            o, _ = strobe_full(V0, w0, p, A, Om, np.pi / 2, 1.0, int(j * slow_T_periods / nsw2), nsub, 0.0)
            Vs[j], ws[j] = o[-1]
        amp, Vf, wf = batch_strobo_amp(Vs, ws, p, A, Om, 3 * np.pi / 2, 1.0, int(6000.0 / Tper), nsub, int(500 / Tper))
        rows.append([s, np.mean(amp < 2.0)])
        print(f"success s={s:.4f}: {np.mean(amp<2.0):.3f}", flush=True)
    np.savez("data_switch_success.npz", rows=np.array(rows), sH=sH, sSN=sSN)
    print("D done", time.time() - t0, flush=True)


# ---------- E. quasi-static phase sweep (hysteresis) in the full system at fixed s
if stage in ("all", "sweep"):
    Om = 5.0; Tper = 2 * np.pi / Om; nsub = 64
    hop = np.load("data_hopf.npz"); fold = np.load("data_fold.npz")
    out = {}
    for k, s in enumerate([22.0, 27.0]):
        A = s * C * Om
        nph = 360; hold = int(2000.0 / Tper); meas = int(600.0 / Tper)
        for direction in ["up", "down"]:
            phs = np.linspace(0, 2 * np.pi, nph + 1)
            if direction == "down": phs = phs[::-1]
            # initial state: oscillatory attractor at phi = phs[0]
            V, w = -6.73, 0.158
            o, t = strobe_full(V, w, p, A, Om, phs[0], 1.0, int(3000 / Tper), nsub, 0.0); V, w = o[-1]
            amps = []
            for ph in phs:
                o, t = strobe_full(V, w, p, A, Om, ph, 1.0, hold, nsub, 0.0); V, w = o[-1]
                Vst = o[-meas:, 0]; amps.append(Vst.max() - Vst.min())
            out[f"phs_{k}_{direction}"] = phs; out[f"amp_{k}_{direction}"] = np.array(amps)
            print(f"sweep s={s} {direction} done", time.time() - t0, flush=True)
        out[f"s_{k}"] = s
    np.savez("data_sweep.npz", **out)
    print("E done", time.time() - t0, flush=True)

print("STAGE", stage, "FINISHED", time.time() - t0, flush=True)
