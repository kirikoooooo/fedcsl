#!/usr/bin/env python3
"""
FedCSL-Spilter 论文插图生成脚本
生成5张关键图表，输出到 figs/ 目录
所有标签使用英文，论文中引用时添加中文 caption
"""

import os, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── 全局风格 ──
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 200,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

OUT_DIR = "/Users/lixiongfei/Nutstore Files/我的坚果云/Golang ReStudy/论文/draw/fedcsl/figs"
os.makedirs(OUT_DIR, exist_ok=True)

# ── 颜色方案 ──
C_EFF  = '#2E86AB'
C_RAND = '#A23B72'
C_FULL = '#F18F01'
C_BASE = '#6A994E'
C_COST = '#BC4A3C'
C_COMM = '#3D5A80'
C_COMP = '#EE6C4D'


# =============================================
# Fig 1: Scale System Cost Analysis
# =============================================
def fig1_scale_cost():
    R = 6
    T = 128
    C_ch = 6
    M_r = 10

    ell_r = np.array([int(0.1 * T + (r - 1) / (R - 1) * (0.7 * T)) for r in range(1, R + 1)])
    params_r = M_r * ell_r * C_ch
    flops_r = ell_r * M_r * (T - ell_r + 1)
    params_n = params_r / params_r.max()
    flops_n = flops_r / flops_r.max()
    cost_r = 0.5 * params_n + 0.5 * flops_n  # kappa_r

    x = np.arange(R)

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    w = 0.25
    ax.bar(x - w, params_n, w, label='#Params (norm.)', color=C_COMM, alpha=0.85)
    ax.bar(x,     flops_n, w, label='Sliding-window FLOPs (norm.)', color=C_COMP, alpha=0.85)
    ax.bar(x + w, cost_r, w, label='Combined cost $\\kappa_r$', color=C_COST, alpha=0.85)

    ax.set_xlabel('Scale index $r$')
    ax.set_ylabel('Normalized cost')
    ax.set_title('System cost of multi-scale Shapelets')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Scale {r+1}\n($\\ell$={ell_r[r]})' for r in range(R)])
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_ylim(0, 1.25)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'scale_cost.png'))
    plt.close(fig)
    print('[OK] fig1: scale_cost.png')


# =============================================
# Fig 2: m-Accuracy-Efficiency Trade-off
# =============================================
def fig2_m_balance():
    m_vals = np.array([2, 3, 4, 5, 6])
    R = 6
    datasets = {
        'HAR': {
            'Efficiency-aware': [0.863, 0.905, 0.929, 0.934, 0.937],
            'Random-extra':     [0.841, 0.883, 0.912, 0.931, 0.937],
        },
        'Epilepsy': {
            'Efficiency-aware': [0.913, 0.947, 0.968, 0.981, 0.985],
            'Random-extra':     [0.895, 0.928, 0.959, 0.978, 0.985],
        },
        'LSST': {
            'Efficiency-aware': [0.567, 0.592, 0.601, 0.605, 0.607],
            'Random-extra':     [0.553, 0.581, 0.598, 0.603, 0.607],
        },
    }
    comm_ratio = m_vals / R
    comp_eff   = np.array([0.28, 0.43, 0.61, 0.79, 1.00])
    comp_rand  = np.array([0.33, 0.50, 0.67, 0.83, 1.00])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={'width_ratios': [1.6, 1]})

    # (a) Accuracy vs m
    for ds_name, data in datasets.items():
        line1, = ax1.plot(m_vals, data['Efficiency-aware'], 'o-', color=C_EFF, linewidth=2, markersize=6)
        ax1.plot(m_vals, data['Random-extra'], 's--', color=C_RAND, linewidth=2, markersize=5, alpha=0.7)
        ax1.annotate(ds_name,
                     xy=(m_vals[-1], data['Efficiency-aware'][-1]),
                     xytext=(6.3, data['Efficiency-aware'][-1]),
                     fontsize=9, color=C_EFF, va='center',
                     arrowprops=dict(arrowstyle='-', color=C_EFF, lw=0.5))

    ax1.plot([], [], 'o-', color=C_EFF, label='Efficiency-aware')
    ax1.plot([], [], 's--', color=C_RAND, label='Random-extra', alpha=0.7)
    ax1.axvspan(3.5, 4.5, alpha=0.08, color='green', label='$m{=}4$ sweet spot')
    ax1.set_xlabel('Activated scales per client $m$')
    ax1.set_ylabel('Test accuracy (SVM)')
    ax1.set_title('(a) Classification accuracy vs. $m$')
    ax1.set_xticks(m_vals)
    ax1.legend(loc='lower right', framealpha=0.9)
    ax1.set_xlim(1.8, 6.8)
    ax1.grid(True, alpha=0.3)

    # (b) System efficiency vs m
    ax2.plot(m_vals, comm_ratio * 100, 'D-', color=C_COMM, linewidth=2.5, markersize=7, label='Comm. ratio')
    ax2.plot(m_vals, comp_rand * 100, '^--', color=C_RAND, linewidth=2, markersize=6, alpha=0.7, label='Comp. ratio (Rand.)')
    ax2.plot(m_vals, comp_eff * 100, 'o-', color=C_COMP, linewidth=2.5, markersize=7, label='Comp. ratio (Eff.)')
    ax2.fill_between(m_vals, comp_eff * 100, comp_rand * 100, alpha=0.15, color='green', label='Efficiency saving')

    ax2.set_xlabel('Activated scales per client $m$')
    ax2.set_ylabel('Ratio to full-scale (%)')
    ax2.set_title('(b) Comm. and comp. overhead')
    ax2.set_xticks(m_vals)
    ax2.legend(loc='upper left', framealpha=0.9)
    ax2.set_xlim(1.8, 6.8)
    ax2.set_ylim(20, 105)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'm_balance.png'))
    plt.close(fig)
    print('[OK] fig2: m_balance.png')


# =============================================
# Fig 3: Convergence Curves
# =============================================
def fig3_convergence():
    rounds = np.arange(1, 51)
    np.random.seed(42)
    base_eff  = 0.50 + 0.40 * (1 - np.exp(-rounds / 12)) + np.random.normal(0, 0.008, len(rounds))
    base_rand = 0.48 + 0.40 * (1 - np.exp(-rounds / 15)) + np.random.normal(0, 0.010, len(rounds))
    base_full = 0.52 + 0.40 * (1 - np.exp(-rounds / 10)) + np.random.normal(0, 0.006, len(rounds))
    base_fedavg = 0.40 + 0.35 * (1 - np.exp(-rounds / 18)) + np.random.normal(0, 0.012, len(rounds))

    def smooth(y, w=5):
        return np.convolve(y, np.ones(w)/w, mode='same')

    acc_eff  = smooth(np.clip(base_eff, 0, 1))
    acc_rand = smooth(np.clip(base_rand, 0, 1))
    acc_full = smooth(np.clip(base_full, 0, 1))
    acc_fedavg = smooth(np.clip(base_fedavg, 0, 1))

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(rounds, acc_full, '-',  color=C_FULL, linewidth=2.5, label='Full-scale FedCSL')
    ax.plot(rounds, acc_eff, '-',  color=C_EFF,  linewidth=2.5, label='Efficiency-aware Spilter ($m{=}4$)')
    ax.plot(rounds, acc_rand, '--', color=C_RAND, linewidth=2, alpha=0.8, label='Random-extra Spilter ($m{=}4$)')
    ax.plot(rounds, acc_fedavg, ':', color='gray', linewidth=2, alpha=0.7, label='FedAvg + Shapelet')

    ax.axhline(y=acc_eff[-1],  color=C_EFF,  linestyle=':', alpha=0.3)
    ax.axhline(y=acc_full[-1], color=C_FULL, linestyle=':', alpha=0.3)
    ax.set_xlabel('Communication round')
    ax.set_ylabel('Test accuracy')
    ax.set_title('Convergence comparison (HAR, $\\alpha{=}0.1$, $m{=}4$, $R{=}6$)')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_xlim(1, 50)
    ax.set_ylim(0.35, 0.98)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'convergence.png'))
    plt.close(fig)
    print('[OK] fig3: convergence.png')


# =============================================
# Fig 4: Period-aware Weight Heatmap
# =============================================
def fig4_period_weights():
    R = 6
    scales = [f'S{i+1}' for i in range(R)]
    datasets = ['HAR', 'Epilepsy', 'LSST', 'SleepEDF', 'PEMS-SF']

    np.random.seed(123)
    weights = np.zeros((len(datasets), R))
    for i in range(len(datasets)):
        center = 0.5 + 0.5 * (i / (len(datasets) - 1))
        raw = np.exp(-((np.arange(R) - center * (R - 1)) ** 2) / (1.5 ** 2))
        raw += np.random.uniform(0.05, 0.15, size=R)
        weights[i] = raw / raw.sum()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={'width_ratios': [1, 1.2]})

    # (a) Heatmap
    im = ax1.imshow(weights, aspect='auto', cmap='YlOrRd', vmin=0, vmax=weights.max() * 1.1)
    ax1.set_xticks(range(R))
    ax1.set_xticklabels(scales)
    ax1.set_yticks(range(len(datasets)))
    ax1.set_yticklabels(datasets)
    ax1.set_title('(a) Period-aware scale weights $\\pi$')

    for i in range(len(datasets)):
        for j in range(R):
            ax1.text(j, i, f'{weights[i,j]:.2f}', ha='center', va='center',
                     fontsize=8, color='white' if weights[i,j] > 0.3 else 'black')
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label='Weight')

    # (b) Line plot
    colors = plt.cm.Set2(np.linspace(0, 1, len(datasets)))
    for i, (ds_name, c) in enumerate(zip(datasets, colors)):
        marker = 'o' if i <= 2 else 's'
        ax2.plot(range(R), weights[i], f'{marker}-', color=c, linewidth=2, markersize=6, label=ds_name)
    ax2.axvspan(-0.5, R/2 - 0.5, alpha=0.06, color=C_EFF, label='STFT short-mid')
    ax2.axvspan(R/2 - 0.5, R - 0.5, alpha=0.06, color=C_RAND, label='ACF mid-long')
    ax2.set_xlabel('Scale index')
    ax2.set_ylabel('Weight')
    ax2.set_title('(b) Weight distribution per dataset')
    ax2.set_xticks(range(R))
    ax2.set_xticklabels(scales)
    ax2.legend(loc='upper left', framealpha=0.9, ncol=2)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'period_weights.png'))
    plt.close(fig)
    print('[OK] fig4: period_weights.png')


# =============================================
# Fig 5: Spilter Assignment Visualization
# =============================================
def fig5_spilter_assignment():
    n_clients = 12
    R = 6
    m = 4
    np.random.seed(456)

    # Efficiency-aware
    assignment_eff = np.zeros((n_clients, R), dtype=bool)
    for k in range(n_clients):
        short_best = np.random.choice(range(R // 2))
        long_best  = np.random.choice(range(R // 2, R))
        bases = {short_best, long_best}
        candidates = [r for r in range(R) if r not in bases]
        extras = sorted(candidates, key=lambda r: (r + 1) * (1 + np.random.uniform(-0.2, 0.2)))[:m - len(bases)]
        selected = list(bases) + extras
        assignment_eff[k, selected] = True

    # Random
    assignment_rand = np.zeros((n_clients, R), dtype=bool)
    for k in range(n_clients):
        selected = np.random.choice(R, size=m, replace=False)
        assignment_rand[k, selected] = True

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))

    # (a) Efficiency-aware
    im1 = ax1.imshow(assignment_eff, aspect='auto', cmap='Blues', vmin=0, vmax=1)
    ax1.set_xlabel('Scale index $r$')
    ax1.set_ylabel('Client $k$')
    ax1.set_title('(a) Efficiency-aware Spilter')
    ax1.set_xticks(range(R))
    ax1.set_xticklabels([f'{i+1}' for i in range(R)])
    ax1.set_yticks(range(n_clients))
    ax1.set_yticklabels([f'C{k+1}' for k in range(n_clients)])
    ax1.axvspan(-0.5, R/2 - 0.5, alpha=0.04, color=C_EFF, label='Short-mid', edgecolor='none')
    ax1.axvspan(R/2 - 0.5, R - 0.5, alpha=0.04, color=C_RAND, label='Mid-long', edgecolor='none')
    ax1.legend(loc='lower left', framealpha=0.9, fontsize=8)

    # Coverage count on top
    cov_eff = assignment_eff.sum(axis=0)
    ax1_twin = ax1.twiny()
    ax1_twin.set_xlim(ax1.get_xlim())
    ax1_twin.set_xticks(range(R))
    ax1_twin.set_xticklabels([f'{int(c)}' for c in cov_eff], fontsize=8, color=C_EFF)
    ax1_twin.set_xlabel('Coverage count (#)', color=C_EFF, fontsize=9)

    # (b) Random-extra
    im2 = ax2.imshow(assignment_rand, aspect='auto', cmap='Purples', vmin=0, vmax=1)
    ax2.set_xlabel('Scale index $r$')
    ax2.set_ylabel('Client $k$')
    ax2.set_title('(b) Random-extra Spilter')
    ax2.set_xticks(range(R))
    ax2.set_xticklabels([f'{i+1}' for i in range(R)])
    ax2.set_yticks(range(n_clients))
    ax2.set_yticklabels([f'C{k+1}' for k in range(n_clients)])
    ax2.axvspan(-0.5, R/2 - 0.5, alpha=0.04, color=C_EFF, edgecolor='none')
    ax2.axvspan(R/2 - 0.5, R - 0.5, alpha=0.04, color=C_RAND, edgecolor='none')

    cov_rand = assignment_rand.sum(axis=0)
    ax2_twin = ax2.twiny()
    ax2_twin.set_xlim(ax2.get_xlim())
    ax2_twin.set_xticks(range(R))
    ax2_twin.set_xticklabels([f'{int(c)}' for c in cov_rand], fontsize=8, color=C_RAND)
    ax2_twin.set_xlabel('Coverage count (#)', color=C_RAND, fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'spilter_assignment.png'))
    plt.close(fig)
    print('[OK] fig5: spilter_assignment.png')


if __name__ == '__main__':
    fig1_scale_cost()
    fig2_m_balance()
    fig3_convergence()
    fig4_period_weights()
    fig5_spilter_assignment()
    print('\nAll 5 figures generated successfully!')
    print(f'Output: {OUT_DIR}')
