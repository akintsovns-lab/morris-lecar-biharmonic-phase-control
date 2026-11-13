import numpy as np, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from ml_core import *

rcParams.update({"font.size": 9, "font.family": "serif", "mathtext.fontset": "stix",
                 "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                 "figure.dpi": 150, "savefig.dpi": 300, "axes.linewidth": 0.7})
os.makedirs("figs", exist_ok=True)
p = P0.copy(); C = p[0]; gK = p[2]; EK = p[5]
hop = np.load("data_hopf.npz"); fold = np.load("data_fold.npz")
phis, sH, l1 = hop['phis'], hop['sH'], hop['l1']; bautin = hop['bautin']
phis2, sSN = fold['phis2'], fold['sSN']
# oscillation boundary: s_osc = s_SN where subcritical (l1>0), else = s_H
l1_on2 = np.interp(phis2, phis, l1)
sH_on2 = np.interp(phis2, phis, sH)
sOSC = np.where(l1_on2 > 0, sSN, sH_on2)
PI = np.pi
def pi_axis(ax, which='x'):
    ticks = [0, PI / 2, PI, 3 * PI / 2, 2 * PI]
    labels = ['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$']
    if which == 'x':
        ax.set_xticks(ticks); ax.set_xticklabels(labels)
    else:
        ax.set_yticks(ticks); ax.set_yticklabels(labels)

# ------------------------------------------------------------------ Fig 1: forcing waveforms, xi, moments
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.2))
th = np.linspace(0, 2 * PI, 400)
for ph, c, lab in [(PI / 2, 'C0', r'$\varphi=\pi/2$'), (3 * PI / 2, 'C3', r'$\varphi=3\pi/2$'), (0.0, 'C2', r'$\varphi=0$')]:
    axs[0].plot(th / (2 * PI), np.cos(th) + np.cos(2 * th + ph), color=c, lw=1.2, label=lab)
    axs[1].plot(th / (2 * PI), np.sin(th) + 0.5 * np.sin(2 * th + ph), color=c, lw=1.2)
axs[0].set_xlabel(r'$t/T$'); axs[0].set_ylabel(r'$I_{\rm ac}/A$'); axs[0].legend(frameon=False, loc='lower left', ncol=1, fontsize=7); axs[0].set_ylim(-2.3, 2.3)
axs[0].axhline(0, color='k', lw=0.4)
axs[1].set_xlabel(r'$t/T$'); axs[1].set_ylabel(r'$\xi/s$'); axs[1].axhline(0, color='k', lw=0.4)
phg = np.linspace(0, 2 * PI, 400)
xi3 = -0.375 * np.sin(phg)  # <xi^3>/s^3 with a=1, b=1/2
peak = np.array([np.max(np.abs(np.cos(th) + np.cos(2 * th + ph))) for ph in phg])
axs[2].plot(phg, xi3, 'k', lw=1.2)
axs[2].set_ylabel(r'$\langle\xi^3\rangle/s^3$'); axs[2].set_ylim(-0.45, 0.45)
axs[2].axhline(0, color='k', lw=0.4)
ax2 = axs[2].twinx()
ax2.plot(phg, peak, color='C1', ls='--', lw=1.2)
ax2.set_ylabel(r'$I_{\rm peak}/A$', color='C1'); ax2.tick_params(axis='y', colors='C1'); ax2.set_ylim(1.4, 2.1)
axs[2].set_xlabel(r'$\varphi$'); pi_axis(axs[2])
for a, t in zip(axs, 'abc'):
    a.text(-0.22, 1.02, f'({t})', transform=a.transAxes, fontsize=9, fontweight='bold')
fig.tight_layout(); fig.savefig("figs/fig1_forcing.pdf"); plt.close(fig)

# ------------------------------------------------------------------ Fig 2: averaged nullclines and mechanism
nc = np.load("data_nullclines.npz")
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.4))
ax = axs[0]
for key, c, lab in [("s0", 'k', r'$s=0$'), ("pi2", 'C0', r'$s=24$, $\varphi=\pi/2$'), ("3pi2", 'C3', r'$s=24$, $\varphi=3\pi/2$')]:
    u, a, b, cc = nc[key]
    wu = a / (gK * (u - EK))
    ax.plot(u, wu, color=c, lw=1.2, label=lab)
    ax.plot(u, b / cc, color=c, lw=0.8, ls='--')
ax.set_xlim(-70, 50); ax.set_ylim(-0.05, 0.9); ax.set_xlabel(r'$u$ (mV)'); ax.set_ylabel(r'$w$')
ax.legend(frameon=False, fontsize=7, loc='upper left')
ax.text(0.97, 0.05, 'solid: $u$-nullcline\ndashed: $w$-nullcline', transform=ax.transAxes, ha='right', fontsize=7)
# (b) effective averaged calcium current  <I_Ca>(u) for the three cases
ax = axs[1]
for key, c, lab in [("s0", 'k', r'$s=0$'), ("pi2", 'C0', r'$\pi/2$'), ("3pi2", 'C3', r'$3\pi/2$')]:
    u, a, b, cc = nc[key]
    ICa = p[12] - a - p[3] * (u - p[6])  # a = I0 - <I_Ca> - gL<(u+xi-EL)> ; <xi>=0
    ax.plot(u, ICa, color=c, lw=1.2, label=lab)
ax.set_xlabel(r'$u$ (mV)'); ax.set_ylabel(r'$\langle I_{\rm Ca}\rangle$ ($\mu$A/cm$^2$)'); ax.set_xlim(-70, 50)
ax.legend(frameon=False, fontsize=7)
# (c) trace of the averaged Jacobian at the equilibrium vs s for several phases
ax = axs[2]
sv = np.linspace(0, 40, 161)
for ph, c, lab in [(PI / 2, 'C0', r'$\pi/2$'), (3 * PI / 2, 'C3', r'$3\pi/2$'), (0.0, 'C2', r'$0$'), (PI / 4, 'C1', r'$\pi/4$')]:
    tr = [trace_at_equil(s, ph, p)[0] for s in sv]
    ax.plot(sv, tr, color=c, lw=1.2, label=lab)
ax.axhline(0, color='k', lw=0.5); ax.set_xlabel(r'$s$ (mV)'); ax.set_ylabel(r'tr $\bar J$ (ms$^{-1}$)')
ax.legend(frameon=False, fontsize=7, title=r'$\varphi$', title_fontsize=7, loc='upper right')
for a, t in zip(axs, 'abc'):
    a.text(-0.25, 1.02, f'({t})', transform=a.transAxes, fontsize=9, fontweight='bold')
fig.tight_layout(); fig.savefig("figs/fig2_averaged.pdf"); plt.close(fig)

# ------------------------------------------------------------------ Fig 3: one-parameter diagrams
op = np.load("data_oneparam.npz")
zoom = np.load("data_oneparam_zoom.npz") if os.path.exists("data_oneparam_zoom.npz") else None
flq = np.load("data_floquet_phi.npz")['rows']
ff = np.load("data_fold_full.npz")['res'] if os.path.exists("data_fold_full.npz") else None
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
def draw_branches(ax, d, ms_st=2.0, ms_un=2.4, every=1):
    sw = np.unique(d[:, 0])
    us_u = np.array([d[d[:, 0] == x, 1][0] for x in sw]); tr_u = np.array([d[d[:, 0] == x, 3][0] for x in sw])
    ax.plot(sw, np.where(tr_u > 0, us_u, np.nan), 'k--', lw=0.9)
    ax.plot(sw, np.where(tr_u <= 0, us_u, np.nan), 'k-', lw=1.2)
    m = np.isfinite(d[:, 4])
    st = d[m][d[m, 9] == 1][::every]; un = d[m][d[m, 9] == 0][::every]
    ax.plot(st[:, 0], st[:, 6], 'o', color='C0', ms=ms_st, mec='none'); ax.plot(st[:, 0], st[:, 7], 'o', color='C0', ms=ms_st, mec='none')
    ax.plot(un[:, 0], un[:, 6], 'o', color='C3', ms=ms_un, mfc='none', mew=0.7); ax.plot(un[:, 0], un[:, 7], 'o', color='C3', ms=ms_un, mfc='none', mew=0.7)
    return sw, us_u
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.7))
for ax, name, ph, title in zip(axs, ["3pi2", "pi2", "0"], [3 * PI / 2, PI / 2, 0.0], [r'$\varphi=3\pi/2$', r'$\varphi=\pi/2$', r'$\varphi=0$']):
    d = op[name]
    sw, us_u = draw_branches(ax, d, every=2)
    shh = np.interp(ph, phis, sH); l1h = np.interp(ph, phis, l1)
    sosc_h = np.interp(ph, phis2, sOSC)
    ax.axvline(shh, color='0.6', lw=0.6, ls=':')
    ax.set_xlim(0, sosc_h + 1.5); ax.set_ylim(-110, 48)
    ax.set_xlabel(r'$s$ (mV)'); ax.set_title(title, fontsize=9)
    ax.text(shh, 40, 'H', ha='center', fontsize=8)
    ax.set_yticks([-100, -75, -50, -25, 0, 25])
    # inset zoom around the band
    if zoom is not None and name in zoom.files:
        dz = zoom[name]
        axin = inset_axes(ax, width="50%", height="36%", loc='lower left', borderpad=1.0)
        draw_branches(axin, dz, ms_st=1.6, ms_un=2.0)
        axin.axvline(shh, color='0.6', lw=0.6, ls=':')
        if l1h > 0:
            ssn = np.interp(ph, phis2, sSN); axin.axvline(ssn, color='0.6', lw=0.6, ls=':')
            axin.axvspan(shh, ssn, color='C1', alpha=0.18, lw=0)
            axin.text(ssn, 1.02, 'SNPO', ha='center', fontsize=6, transform=axin.get_xaxis_transform())
        axin.text(shh, 1.02, 'H', ha='center', fontsize=6, transform=axin.get_xaxis_transform())
        r5 = flq[(flq[:, 0] == 5.0)]
        sF = np.interp(ph, r5[:, 1], r5[:, 2])
        axin.plot([sF], [np.interp(sF, np.unique(dz[:, 0]), [dz[dz[:, 0] == x, 1][0] for x in np.unique(dz[:, 0])])], 'kx', ms=5, mew=1.0)
        if ff is not None:
            row = [r for r in ff if r[0] == 5.0 and abs(r[1] - ph) < 1e-6]
            if row:
                axin.plot([row[0][4]], [dz[:, 7][np.isfinite(dz[:, 7])].min() - 3], 'k^', ms=4, mfc='none')
        axin.set_xlim(dz[:, 0].min(), max(dz[:, 0].max(), sF + 0.05))
        lo = np.nanmin(dz[:, 7]) - 8; hi = np.nanmax(dz[:, 6]) + 6
        axin.set_ylim(lo, hi)
        axin.tick_params(labelsize=5.5, pad=1.5, length=2)
        for sp in axin.spines.values(): sp.set_linewidth(0.5)
axs[0].set_ylabel(r'$u$ (mV)')
for a, t in zip(axs, 'abc'):
    a.text(-0.2, 1.02, f'({t})', transform=a.transAxes, fontsize=9, fontweight='bold')
fig.tight_layout(); fig.savefig("figs/fig3_oneparam.pdf"); plt.close(fig)

# ------------------------------------------------------------------ Fig 4: two-parameter diagram
fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.3), gridspec_kw={'width_ratios': [2.2, 1]})
ax = axs[0]
ax.fill_between(phis2, 0, sH_on2, color='C0', alpha=0.12, lw=0)
ax.fill_between(phis2, sH_on2, sOSC, color='C1', alpha=0.35, lw=0)
ax.fill_between(phis2, sOSC, 40, color='C2', alpha=0.10, lw=0)
sub = l1 > 0
ax.plot(phis, np.where(sub, sH, np.nan), 'C3-', lw=1.4, label=r'Hopf (subcritical, $l_1>0$)')
ax.plot(phis, np.where(~sub, sH, np.nan), 'C0-', lw=1.4, label=r'Hopf (supercritical, $l_1<0$)')
ax.plot(phis2, np.where(l1_on2 > 0, sSN, np.nan), 'k-', lw=1.2, label='fold of cycles (SNPO)')
for pb, sb in bautin:
    ax.plot(pb, sb, 'o', color='gold', mec='k', ms=6, zorder=5)
ax.text(bautin[0][0], bautin[0][1] + 1.3, 'GH', ha='center', fontsize=8)
r5 = flq[flq[:, 0] == 5.0]; r2 = flq[flq[:, 0] == 2.0]
ax.plot(r5[:, 1], r5[:, 2], 'kx', ms=4, mew=0.8, label=r'full system, $\Omega=5$ (NS)')
ax.plot(r2[:, 1], r2[:, 2], 'k+', ms=5, mew=0.8, label=r'full system, $\Omega=2$ (NS)')
if ff is not None:
    f5 = ff[ff[:, 0] == 5.0]
    ax.plot(f5[:, 1], f5[:, 4], 'k^', ms=5, mfc='none', label=r'full system fold, $\Omega=5$')
# guaranteed switching window
s_on_max = sH.max(); s_off_min = sOSC.min()
ax.axhline(s_off_min, color='0.3', lw=0.6, ls='--'); ax.axhline(s_on_max, color='0.3', lw=0.6, ls='--')
ax.annotate('', xy=(3 * PI / 2, s_off_min), xytext=(3 * PI / 2, s_on_max), arrowprops=dict(arrowstyle='<->', lw=0.8))
ax.text(3 * PI / 2 + 0.12, 0.5 * (s_off_min + s_on_max) + 1.0, 'switching\nwindow', fontsize=7, va='center')
ax.text(PI / 2 + 0.3, 10, 'slow oscillation only', ha='center', fontsize=8)
ax.text(PI / 2, 36.5, 'suppressed state only', ha='center', fontsize=8)
ax.annotate('bistable band', xy=(4.15, 22.5), xytext=(2.5, 18.5), fontsize=7, color='C1', arrowprops=dict(arrowstyle='->', lw=0.6, color='C1'))
ax.set_xlim(0, 2 * PI); ax.set_ylim(5, 40); pi_axis(ax)
ax.set_xlabel(r'relative phase $\varphi$'); ax.set_ylabel(r'$s=A/(C\Omega)$ (mV)')
ax.legend(frameon=False, fontsize=6.5, loc='upper center', ncol=3, handlelength=1.6, bbox_to_anchor=(0.5, -0.2), columnspacing=1.2)
# inset: zoom on the band around 3pi/2
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
axin = inset_axes(ax, width="30%", height="34%", loc='lower right', borderpad=0.9)
mk = (phis2 > 4.2) & (phis2 < 5.3)
axin.fill_between(phis2[mk], sH_on2[mk], sOSC[mk], color='C1', alpha=0.35, lw=0)
axin.plot(phis, sH, 'C3-', lw=1.2); axin.plot(phis2, np.where(l1_on2 > 0, sSN, np.nan), 'k-', lw=1.0)
axin.plot(r5[:, 1], r5[:, 2], 'kx', ms=4, mew=0.8)
if ff is not None:
    axin.plot(f5[:, 1], f5[:, 4], 'k^', ms=4, mfc='none')
axin.set_xlim(4.2, 5.3); axin.set_ylim(21.0, 23.2)
axin.set_xticks([3 * PI / 2]); axin.set_xticklabels([r'$3\pi/2$']); axin.tick_params(labelsize=6, pad=1.5, length=2)
axin.text(0.05, 0.9, 'bistable', transform=axin.transAxes, fontsize=6, color='C1', va='top')
for sp in axin.spines.values(): sp.set_linewidth(0.5)
ax = axs[1]
ax.plot(phis, l1 * 1e3, 'k', lw=1.2); ax.axhline(0, color='0.5', lw=0.6)
for pb, sb in bautin:
    ax.plot(pb, 0, 'o', color='gold', mec='k', ms=6, zorder=5)
ax.set_xlim(0, 2 * PI); pi_axis(ax); ax.set_xlabel(r'$\varphi$'); ax.set_ylabel(r'$l_1 \times 10^{3}$')
ax.fill_between(phis, 0, l1 * 1e3, where=l1 > 0, color='C3', alpha=0.2, lw=0)
ax.fill_between(phis, 0, l1 * 1e3, where=l1 < 0, color='C0', alpha=0.2, lw=0)
for a, t in zip(axs, 'ab'):
    a.text(-0.12 if t == 'a' else -0.3, 1.02, f'({t})', transform=a.transAxes, fontsize=9, fontweight='bold')
fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.3, wspace=0.3)
fig.savefig("figs/fig4_twoparam.pdf"); plt.close(fig)

# ------------------------------------------------------------------ Fig 5: basins
if os.path.exists("data_basins.npz"):
    bs = np.load("data_basins.npz")
    bz = np.load("data_basins_zoom.npz") if os.path.exists("data_basins_zoom.npz") else None
    from matplotlib.colors import ListedColormap
    cmap2 = ListedColormap(['#f3e6c8', '#3b6ea8'])
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.55))
    def overlay(ax, k, s, ph):
        xi = xi_nodes(s, ph)
        j = 0
        while f'cyc{k}_{j}' in bs.files:
            tr = bs[f'cyc{k}_{j}']; stb = bs[f'cycstable{k}_{j}'] > 0.5
            ax.plot(tr[:, 1] + xi[0], tr[:, 2], color='C3' if not stb else 'k', lw=1.0 if not stb else 1.3)
            j += 1
        eq = bs[f'eq{k}']; ax.plot(eq[0] + xi[0], eq[1], 'o', color='w', mec='k', ms=5)
    # (a) full physical range, case 0
    ax = axs[0]; k = 0
    s = float(bs[f's{k}']); ph = float(bs[f'phi{k}'])
    ax.pcolormesh(bs['Vg'], bs['wg'], (bs[f'amp{k}'] > 10).astype(float), cmap=cmap2, vmin=0, vmax=1, shading='nearest', rasterized=True)
    overlay(ax, k, s, ph)
    if bz is not None:
        Vz, wz = bz['Vg0'], bz['wg0']
        ax.add_patch(plt.Rectangle((Vz[0], wz[0]), Vz[-1] - Vz[0], wz[-1] - wz[0], fill=False, ec='k', lw=0.6, ls='--'))
    ax.set_xlabel(r'$V(0)$ (mV)'); ax.set_ylabel(r'$w(0)$'); ax.set_title(r'$\varphi=3\pi/2$, $s=%.2f$ mV' % s, fontsize=9)
    ax.set_xlim(bs['Vg'][0], bs['Vg'][-1]); ax.set_ylim(bs['wg'][0], bs['wg'][-1])
    # (b),(c) zooms
    if bz is not None:
        for ax, k, ttl in zip(axs[1:], [0, 1], [r'$\varphi=3\pi/2$ (zoom)', r'$\varphi=\pi/2$ (zoom)']):
            s = float(bs[f's{k}']); ph = float(bs[f'phi{k}'])
            Vz, wz = bz[f'Vg{k}'], bz[f'wg{k}']
            ax.pcolormesh(Vz, wz, (bz[f'amp{k}'] > 10).astype(float), cmap=cmap2, vmin=0, vmax=1, shading='nearest', rasterized=True)
            # disagreement with averaged classification
            dis = (bz[f'amp{k}'] > 10) != (bz[f'ampavg{k}'] > 10)
            VV, WW = np.meshgrid(Vz, wz)
            ax.plot(VV[dis], WW[dis], 's', color='lime', ms=1.6, mec='none')
            overlay(ax, k, s, ph)
            ax.set_xlim(Vz[0], Vz[-1]); ax.set_ylim(wz[0], wz[-1])
            ax.set_xlabel(r'$V(0)$ (mV)'); ax.set_title(ttl + r', $s=%.2f$ mV' % s, fontsize=9)
    for a, t in zip(axs, 'abc'):
        a.text(-0.2, 1.04, f'({t})', transform=a.transAxes, fontsize=9, fontweight='bold')
    fig.tight_layout(); fig.savefig("figs/fig5_basins.pdf"); plt.close(fig)

# ------------------------------------------------------------------ Fig 6: switching protocol
if os.path.exists("data_switch_trace.npz"):
    st = np.load("data_switch_trace.npz"); Om = float(st['Om']); s = float(st['s'])
    fig = plt.figure(figsize=(7.0, 4.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1], hspace=0.5, wspace=0.35, left=0.1, right=0.98, top=0.95, bottom=0.1)
    ax = fig.add_subplot(gs[0, :])
    for seg, ph in zip([st['seg0'], st['seg1'], st['seg2']], [PI / 2, 3 * PI / 2, PI / 2]):
        t = seg[:, 0]; V = seg[:, 1]
        xi = s * (np.sin(Om * t) + 0.5 * np.sin(2 * Om * t + ph))
        ax.plot(t, V, color='0.75', lw=0.3, rasterized=True)
        ax.plot(t, V - xi, 'k', lw=0.9)
    ax.axvspan(1500, 3000, color='C3', alpha=0.08, lw=0)
    ax.text(750, 70, r'$\varphi=\pi/2$', ha='center', fontsize=8); ax.text(2250, 70, r'$\varphi=3\pi/2$', ha='center', fontsize=8); ax.text(3750, 70, r'$\varphi=\pi/2$', ha='center', fontsize=8)
    ax.set_xlim(0, 4500); ax.set_ylim(-110, 85); ax.set_xlabel(r'$t$ (ms)'); ax.set_ylabel(r'$V$, $V-\xi$ (mV)')
    ax.text(-0.07, 1.04, '(a)', transform=ax.transAxes, fontsize=9, fontweight='bold')
    ax = fig.add_subplot(gs[1, 0])
    sm = np.load("data_switch_moment.npz")['res']
    ax.plot(sm[:, 0], sm[:, 3], 'ko-', ms=2.5, lw=0.8)
    ax.set_xlabel('switching moment (fraction of slow period)'); ax.set_ylabel('time to suppression (ms)')
    ax.text(-0.2, 1.04, '(b)', transform=ax.transAxes, fontsize=9, fontweight='bold')
    ax = fig.add_subplot(gs[1, 1])
    ss = np.load("data_switch_success.npz"); rows = ss['rows']; rows = rows[np.argsort(rows[:, 0])]
    ax.plot(rows[:, 0], rows[:, 1], 'ko-', ms=2.5, lw=0.8)
    ax.axvline(float(ss['sH']), color='0.5', lw=0.6, ls=':'); ax.axvline(float(ss['sSN']), color='0.5', lw=0.6, ls=':')
    ax.text(float(ss['sH']), 1.04, 'H', ha='center', fontsize=8); ax.text(float(ss['sSN']), 1.04, 'SNPO', ha='center', fontsize=8)
    ax.axvspan(float(ss['sH']), float(ss['sSN']), color='C1', alpha=0.15, lw=0)
    ax.set_xlim(rows[0, 0] - 0.1, rows[-2, 0] + 0.1); ax.set_ylim(-0.05, 1.12)
    ax.set_xlabel(r'$s$ (mV) at $\varphi=3\pi/2$'); ax.set_ylabel('fraction suppressed')
    ax.text(-0.2, 1.04, '(c)', transform=ax.transAxes, fontsize=9, fontweight='bold')
    fig.savefig("figs/fig6_switching.pdf"); plt.close(fig)

# ------------------------------------------------------------------ Fig 7: validation
conv = np.load("data_floquet_conv.npz")['conv']
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.4))
ax = axs[0]
for ph, c, lab in [(PI / 2, 'C0', r'$\pi/2$'), (3 * PI / 2, 'C3', r'$3\pi/2$'), (PI / 4, 'C1', r'$\pi/4$'), (0.0, 'C2', r'$0$')]:
    d = conv[np.abs(conv[:, 0] - ph) < 1e-9]
    sa = np.interp(ph, phis, sH)
    ax.loglog(d[:, 1], np.abs(d[:, 2] - sa), 'o-', color=c, ms=3, lw=0.9, label=lab)
Omr = np.array([1, 50.0])
ax.loglog(Omr, 0.008 * Omr ** -2, 'k:', lw=0.8); ax.text(2.2, 0.008 * 2.2 ** -2 * 0.3, r'$\propto\Omega^{-2}$', fontsize=7)
ax.loglog(Omr, 0.25 * Omr ** -1, 'k--', lw=0.6); ax.text(3.5, 0.25 * 3.5 ** -1 * 0.45, r'$\propto\Omega^{-1}$', fontsize=7)
ax.set_xlabel(r'$\Omega$ (rad/ms)'); ax.set_ylabel(r'$|s_{\rm NS}(\Omega)-s_{\rm H}|$ (mV)'); ax.legend(frameon=False, fontsize=6.5, title=r'$\varphi$', title_fontsize=6.5, loc='lower left', ncol=2, columnspacing=0.8, handlelength=1.5)
ax = axs[1]
ax.plot(phis, sH, 'k-', lw=1.0, label='averaged')
ax.plot(r5[:, 1], r5[:, 2], 'C0x', ms=4, mew=0.8, label=r'$\Omega=5$')
ax.plot(r2[:, 1], r2[:, 2], 'C3+', ms=5, mew=0.8, label=r'$\Omega=2$')
ax.set_xlim(0, 2 * PI); pi_axis(ax); ax.set_xlabel(r'$\varphi$'); ax.set_ylabel(r'threshold $s$ (mV)'); ax.legend(frameon=False, fontsize=7)
ax = axs[2]
# mirror asymmetry s(phi) - s(pi - phi) for finite Omega (zero for the averaged system)
for rows, c, lab in [(r5, 'C0', r'$\Omega=5$'), (r2, 'C3', r'$\Omega=2$')]:
    phv = rows[:, 1]; sv_ = rows[:, 2]
    mirror = np.interp(np.mod(PI - phv, 2 * PI), phv, sv_)
    ax.plot(phv, sv_ - mirror, 'o-', color=c, ms=2.5, lw=0.8, label=lab)
ax.axhline(0, color='k', lw=0.5); ax.set_xlim(0, 2 * PI); pi_axis(ax)
ax.set_xlabel(r'$\varphi$'); ax.set_ylabel(r'$s_{\rm NS}(\varphi)-s_{\rm NS}(\pi-\varphi)$ (mV)'); ax.legend(frameon=False, fontsize=7)
for a, t in zip(axs, 'abc'):
    a.text(-0.28, 1.02, f'({t})', transform=a.transAxes, fontsize=9, fontweight='bold')
fig.tight_layout(); fig.savefig("figs/fig7_validation.pdf"); plt.close(fig)

# ------------------------------------------------------------------ Fig 8: hysteresis sweep (optional)
if os.path.exists("data_sweep.npz"):
    sw = np.load("data_sweep.npz")
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.3))
    for k, ax in enumerate(axs):
        s = float(sw[f's_{k}'])
        ax.plot(sw[f'phs_{k}_up'], sw[f'amp_{k}_up'], color='C0', ls='-', lw=1.0, label=r'$\varphi$ increasing')
        ax.plot(sw[f'phs_{k}_down'], sw[f'amp_{k}_down'], color='C3', ls='--', lw=1.0, label=r'$\varphi$ decreasing')
        ax.set_xlim(0, 2 * PI); pi_axis(ax); ax.set_xlabel(r'$\varphi$'); ax.set_title(fr'$s={s:.0f}$ mV', fontsize=9)
        # averaged prediction of the bistable intervals at this s (root finding on the Hopf and fold curves)
        from scipy.interpolate import interp1d
        fH = interp1d(phis, sH, kind='cubic'); fS = interp1d(phis2, sSN, kind='cubic')
        pg = np.linspace(0, 2 * PI, 4001); l1g = np.interp(pg, phis, l1)
        inband = (fH(pg) < s) & (np.where(l1g > 0, fS(pg), fH(pg)) > s)
        edges = np.where(np.diff(inband.astype(int)) != 0)[0]
        for i in range(0, len(edges) - 1, 2):
            ax.axvspan(pg[edges[i]], pg[edges[i + 1]], color='C1', alpha=0.3, lw=0, zorder=0)
    axs[0].set_ylabel('strobosc. amplitude (mV)'); axs[0].legend(frameon=False, fontsize=7, loc='lower left')
    for a, t in zip(axs, 'ab'):
        a.text(-0.24, 1.04, f'({t})', transform=a.transAxes, fontsize=9, fontweight='bold')
    fig.tight_layout(); fig.savefig("figs/fig8_sweep.pdf"); plt.close(fig)
print("figures done")
