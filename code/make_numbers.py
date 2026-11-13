import numpy as np, json, os
PI = np.pi
hop = np.load("data_hopf.npz"); phis, sH, l1 = hop['phis'], hop['sH'], hop['l1']
fold = np.load("data_fold.npz"); phis2, sSN = fold['phis2'], fold['sSN']
conv = np.load("data_floquet_conv.npz")['conv']
flq = np.load("data_floquet_phi.npz")['rows']
out = []
def cmd(name, val): out.append(f"\\newcommand{{\\{name}}}{{{val}}}")

# convergence table
phs = [PI / 2, 3 * PI / 2, PI / 4, 0.0]
rows = ["averaged ($\\Omega\\to\\infty$) & " + " & ".join(f"{np.interp(ph, phis, sH):.4f}" for ph in phs) + " \\\\"]
for Om in [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
    vals = []
    for ph in phs:
        r = conv[(np.abs(conv[:, 0] - ph) < 1e-9) & (conv[:, 1] == Om)]
        vals.append(f"{r[0, 2]:.4f}")
    rows.append(f"$\\Omega={Om:g}$ rad/ms & " + " & ".join(vals) + " \\\\")
cmd("CONVTABLE", "\n".join(rows))

# asymmetry amplitudes
for Om, name in [(5.0, "ASYMfive"), (2.0, "ASYMtwo")]:
    r = flq[flq[:, 0] == Om]; phv = r[:, 1]; sv = r[:, 2]
    mirror = np.interp(np.mod(PI - phv, 2 * PI), phv, sv)
    cmd(name, f"{np.max(np.abs(sv - mirror)):.2f}")

# fold full at Omega=2
ff = np.load("data_fold_full.npz")['res']
f2 = ff[ff[:, 0] == 2.0]
cmd("FOLDtwoA", f"{f2[np.abs(f2[:,1]-PI/2)<1e-6][0,4]:.3f}")
cmd("FOLDtwoB", f"{f2[np.abs(f2[:,1]-3*PI/2)<1e-6][0,4]:.3f}")

# sensitivity table
sens = json.load(open("data_sens.json"))
names = {"base": "reference", "I0=90": "$I_0=90~\\mu$A/cm$^2$", "I0=110": "$I_0=110~\\mu$A/cm$^2$",
         "gCa=3.96": "$g_{\\mathrm{Ca}}=3.96$ mS/cm$^2$", "gCa=4.84": "$g_{\\mathrm{Ca}}=4.84$ mS/cm$^2$",
         "gam=0.036": "$\\gamma=0.036$ ms$^{-1}$", "gam=0.044": "$\\gamma=0.044$ ms$^{-1}$"}
rows = []; widths = []
for r in sens:
    nm, sh1, l11, ssn1, sh2, l12, ssn2 = r
    def fmt_sn(sh, l, ssn):
        return f"{ssn:.3f}" if l > 0 else "--"
    rows.append(f"{names[nm]} & {sh1:.3f} & {1e3*l11:+.2f} & {fmt_sn(sh1,l11,ssn1)} & {sh2:.3f} & {1e3*l12:+.2f} & {fmt_sn(sh2,l12,ssn2)} \\\\")
    if l11 > 0: widths.append(ssn1 - sh1)
    if l12 > 0: widths.append(ssn2 - sh2)
cmd("SENSTABLE", "\n".join(rows))
cmd("SENSWMIN", f"{min(widths):.2f}"); cmd("SENSWMAX", f"{max(widths):.2f}")

# ratio table
d = np.load("data_ratio.npz")
rows = []
def phstr(x):
    k = x / PI
    if abs(k) < 1e-9: return "--"
    frac = {0.5: "\\pi/2", 1.5: "3\\pi/2", 1.0: "\\pi"}
    for kk, ss in frac.items():
        if abs(k - kk) < 0.02: return f"${ss}$"
    return f"${k:.2f}\\pi$"
for key, r in [("0_0", 0.0), ("0_25", 0.25), ("0_5", 0.5), ("1_0", 1.0), ("2_0", 2.0)]:
    a = d[key]; i = np.argmin(a[:, 1]); j = np.argmax(a[:, 1])
    if r == 0.0:
        rows.append(f"0 (sinusoid) & {a[i,1]:.3f} & -- & {a[j,1]:.3f} & -- \\\\")
    else:
        rows.append(f"{r:g} & {a[i,1]:.3f} & {phstr(a[i,0])} & {a[j,1]:.3f} & {phstr(a[j,0])} \\\\")
cmd("RATIOTABLE", "\n".join(rows))
a = d["0_5"]; s0 = d["0_0"][0, 1]
cmd("RATIOHALF", f"{a[:,1].min():.2f} mV ({100*(s0-a[:,1].min())/s0:.1f}\\% below)")

# basins
if os.path.exists("data_basins.npz"):
    bs = np.load("data_basins.npz")
    amp0 = bs['amp0']; ampa0 = bs['ampavg0']
    cmd("BASINAGREE", f"{100*np.mean((amp0>10)==(ampa0>10)):.2f}")
    cmd("BASINFRACA", f"{100*np.mean(amp0<=10):.1f}")
    if 'amp1' in bs.files:
        amp1 = bs['amp1']; cmd("BASINFRACB", f"{100*np.mean(amp1<=10):.1f}")
        cmd("BASINAGREEB", f"{100*np.mean((amp1>10)==(bs['ampavg1']>10)):.2f}")
    else:
        cmd("BASINFRACB", "??"); cmd("BASINAGREEB", "??")
    cmd("BASINSA", f"{float(bs['s0']):.2f}"); cmd("BASINSB", f"{float(bs['s1']):.2f}" if 's1' in bs.files else "??")

# switching
if os.path.exists("data_switch_moment.npz"):
    sm = np.load("data_switch_moment.npz")['res']
    cmd("TSUPMIN", f"{np.nanmin(sm[:,3]):.0f}"); cmd("TSUPMAX", f"{np.nanmax(sm[:,3]):.0f}"); cmd("TSUPMEAN", f"{np.nanmean(sm[:,3]):.0f}")
    ss = np.load("data_switch_success.npz"); rows = ss['rows']
    inband = rows[(rows[:, 0] > float(ss['sH'])) & (rows[:, 0] < float(ss['sSN']))]
    cmd("SUCCMAXBAND", f"{100*inband[:,1].max():.0f}")

# sweep
if os.path.exists("data_sweep.npz"):
    sw = np.load("data_sweep.npz")
    res = {}
    for k in [0, 1]:
        tr = {}
        for dname in ["up", "down"]:
            ph = sw[f"phs_{k}_{dname}"]; a = sw[f"amp_{k}_{dname}"]; osc = a > 5
            idx = np.where(np.diff(osc.astype(int)) != 0)[0]
            tr[dname] = [(0.5 * (ph[i] + ph[i + 1]), 'os' if osc[i] else 'so') for i in idx]
        res[k] = tr
    # hysteresis width at s=22: difference between up and down transition phases on each side
    tr = res[0]
    up_os = [x for x, t in tr['up'] if t == 'os']; dn_so = [x for x, t in tr['down'] if t == 'so']
    up_so = [x for x, t in tr['up'] if t == 'so']; dn_os = [x for x, t in tr['down'] if t == 'os']
    wL = up_os[0] - dn_so[0] if up_os and dn_so else np.nan
    wR = up_so[0] - dn_os[0] if up_so and dn_os else np.nan
    cmd("HYSTL", f"{wL:.2f}"); cmd("HYSTR", f"{wR:.2f}")
    cmd("HYSTUPOS", f"{up_os[0]:.2f}"); cmd("HYSTDNSO", f"{dn_so[0]:.2f}"); cmd("HYSTUPSO", f"{up_so[0]:.2f}"); cmd("HYSTDNOS", f"{dn_os[0]:.2f}")
    tr = res[1]
    up_os = [x for x, t in tr['up'] if t == 'os']; dn_so = [x for x, t in tr['down'] if t == 'so']
    up_so = [x for x, t in tr['up'] if t == 'so']; dn_os = [x for x, t in tr['down'] if t == 'os']
    cmd("HYSTBL", f"{abs(up_os[0]-dn_so[0]):.2f}" if up_os and dn_so else "??"); cmd("HYSTBR", f"{abs(up_so[0]-dn_os[0]):.2f}" if up_so and dn_os else "??")
    cmd("SWEEPSB", f"{float(sw['s_1']):.0f}")
    # predicted band crossing phases at s=22
    s22 = 22.0
    sH2 = np.interp(phis2, phis, sH); l12 = np.interp(phis2, phis, l1); sosc = np.where(l12 > 0, sSN, sH2)
    band = phis2[(sH2 < s22) & (sosc > s22)]
    cmd("BANDLO", f"{band.min():.2f}"); cmd("BANDHI", f"{band.max():.2f}")

open("paper/numbers.tex", "w").write("\n".join(out) + "\n")
print("\n".join(out))
