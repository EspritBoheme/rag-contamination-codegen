"""
Nature-style figures for paper_output/paper.tex (4 figures).
Matches the system architecture and lifecycle narrative of the rewritten paper.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json, csv, os
from collections import defaultdict

# ============================================================
# Global Style
# ============================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.6,
    'grid.linewidth': 0.3,
    'lines.linewidth': 1.3,
    'lines.markersize': 5,
    'legend.frameon': False,
})

# ============================================================
# Palette
# ============================================================
C = {
    'blue_main':      '#0F4D92',
    'blue_secondary': '#3775BA',
    'blue_light':     '#9EC5E8',
    'green_3':        '#8BCF8B',
    'red_strong':     '#B64342',
    'teal':           '#42949E',
    'violet':         '#9A4D8E',
    'gold':           '#C4A43E',
    'neutral_light':  '#CFCECE',
    'neutral_mid':    '#767676',
    'neutral_dark':   '#4D4D4D',
    'neutral_black':  '#272727',
    'full':     '#0F4D92',
    'no_temp':  '#E8A87C',
    'no_feed':  '#B64342',
    'no_div':   '#8BCF8B',
    'baseline': '#767676',
    'helped':   '#8BCF8B',
    'harmed':   '#B64342',
    'neutral':  '#CFCECE',
}

BASE = 'D:/模型微调加蒸馏'
RESULTS = os.path.join(BASE, 'adaptive_memory_rag', 'results')
OUT_DIR = os.path.join(BASE, 'paper_output', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

def save(fig, name):
    for fmt in ['pdf', 'png']:
        path = os.path.join(OUT_DIR, f'{name}.{fmt}')
        fig.savefig(path, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f'  Saved: {name}.pdf + {name}.png')

def panel_label(ax, label, x=-0.06, y=1.02):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=10,
            fontweight='bold', ha='left', va='bottom', color=C['neutral_black'])


# ============================================================
# Fig 1: Pass@1 Ablation Bar Chart
# ============================================================
def fig1_ablation():
    data = json.load(open(os.path.join(RESULTS, 'summary_full.json'), 'r', encoding='utf-8'))

    labels_map = {
        'full': 'Full\nSystem', 'no-temporal': 'No\nTemporal',
        'no-feedback': 'No\nFeedback', 'no-diversity': 'No\nDiversity',
        'baseline': 'Baseline\nChromaDB',
    }
    order = ['baseline', 'no-feedback', 'no-temporal', 'full', 'no-diversity']

    configs = []
    pass_rates = []
    colors = []
    for exp in order:
        for d in data:
            if d['experiment'] == exp:
                configs.append(labels_map[exp])
                pass_rates.append(d['pass_at_1'])
                colors.append(C.get(exp, C['neutral_mid']))
                break

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    bars = ax.bar(configs, pass_rates, color=colors, width=0.55, edgecolor='white', linewidth=0.6)

    for bar, val in zip(bars, pass_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold',
                color=C['neutral_dark'])

    ax.set_ylabel('Pass@1')
    ax.set_ylim(0.64, 0.88)
    ax.axhline(y=0.765, color=C['neutral_mid'], linestyle='--', alpha=0.4, linewidth=0.8)
    ax.text(4.5, 0.768, 'Full System = 0.765', fontsize=7, color=C['neutral_mid'], alpha=0.7)

    # Annotation
    ax.annotate('Moderate diversity,\nhighest quality',
                xy=(4, 0.794), xytext=(4.3, 0.85),
                arrowprops=dict(arrowstyle='->', color=C['no_div'], lw=1.0),
                fontsize=7.5, color=C['no_div'], ha='center', fontweight='bold')

    ax.grid(True, alpha=0.2, linestyle='--', axis='y')
    ax.set_axisbelow(True)

    save(fig, 'fig1_pass_at1_ablation')
    print('  [OK] fig1: Pass@1 ablation')


# ============================================================
# Fig 2: Diversity vs Pass@1
# ============================================================
def fig2_diversity():
    data = json.load(open(os.path.join(RESULTS, 'summary_full.json'), 'r', encoding='utf-8'))
    contam = list(csv.DictReader(open(os.path.join(RESULTS, 'retrieval_contamination.csv'), 'r', encoding='utf-8')))

    configs, diversities, pass_rates = [], [], []
    for d in data:
        configs.append(d['experiment'])
        diversities.append(d['avg_diversity'])
        pass_rates.append(d['pass_at_1'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    # Left: config-level scatter with quadratic trend
    labels_map = {
        'full': 'Full', 'no-temporal': 'No Temp', 'no-feedback': 'No FB',
        'no-diversity': 'No Div', 'baseline': 'Baseline'
    }
    for i, (c, div, p) in enumerate(zip(configs, diversities, pass_rates)):
        ax1.scatter(div, p, color=C.get(c, C['neutral_mid']), s=100, zorder=5,
                   edgecolors='white', linewidth=1)
        off_x = 0.012 if c != 'no-diversity' else -0.06
        off_y = 0.008 if c != 'no-diversity' else -0.016
        ax1.annotate(labels_map[c], (div, p), xytext=(div + off_x, p + off_y),
                    fontsize=7.5, fontweight='bold', color=C.get(c, C['neutral_mid']))

    # Quadratic trend
    z = np.polyfit(diversities, pass_rates, 2)
    x_smooth = np.linspace(min(diversities) - 0.05, max(diversities) + 0.05, 100)
    y_smooth = np.polyval(z, x_smooth)
    ax1.plot(x_smooth, y_smooth, '--', color=C['neutral_mid'], alpha=0.35, linewidth=1.0)

    # Optimal range shading
    ax1.axvspan(0.68, 0.77, alpha=0.06, color=C['green_3'], zorder=0)
    ax1.text(0.725, 0.665, 'Optimal\nRange', fontsize=7, ha='center', style='italic', color=C['green_3'])

    ax1.set_xlabel('Mean Retrieval Diversity')
    ax1.set_ylabel('Pass@1')
    ax1.set_xlim(0.45, 1.0)
    ax1.set_ylim(0.66, 0.85)
    ax1.grid(True, alpha=0.2, linestyle='--')
    ax1.set_axisbelow(True)
    panel_label(ax1, 'a')

    # Right: per-problem delta scatter
    for row in contam:
        div = float(row.get('retrieval_diversity_full', 0.5))
        delta = float(row.get('delta_full_vs_baseline', 0))
        ctype = row.get('contamination_type', 'neutral')
        color = C.get(ctype, C['neutral'])
        marker = '^' if delta > 0 else ('v' if delta < 0 else 'o')
        size = 50 if delta != 0 else 30
        ax2.scatter(div, delta, color=color, s=size, alpha=0.7, marker=marker,
                   edgecolors='white', linewidth=0.5)

    ax2.axhline(y=0, color=C['neutral_dark'], linewidth=0.5, alpha=0.3)
    ax2.set_xlabel('Retrieval Diversity (per problem)')
    ax2.set_ylabel(r'$\Delta$ Pass@1 (RAG $-$ No-RAG)')
    panel_label(ax2, 'b')

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='^', color='w', markerfacecolor=C['helped'], markersize=7,
               label='RAG helped'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=C['neutral'], markersize=7,
               label='No effect'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor=C['harmed'], markersize=7,
               label='RAG harmed'),
    ]
    ax2.legend(handles=legend_elements, fontsize=7, loc='lower left', handlelength=1.0)

    fig.tight_layout(pad=0.8)
    save(fig, 'fig2_diversity_vs_pass')
    print('  [OK] fig2: diversity vs Pass@1')


# ============================================================
# Fig 3: Retrieval Impact Characterization
# ============================================================
def fig3_impact():
    contam = list(csv.DictReader(open(os.path.join(RESULTS, 'retrieval_contamination.csv'), 'r', encoding='utf-8')))
    prompt_cases = json.load(open(os.path.join(RESULTS, 'prompt_contamination_cases.json'), 'r', encoding='utf-8'))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4.0))

    # Panel a: Horizontal bar — impact distribution
    type_counts = defaultdict(int)
    for row in contam:
        type_counts[row['contamination_type']] += 1

    categories = ['No Effect\n(Neutral)', 'RAG Harmed', 'RAG Helped']
    counts = [type_counts.get('neutral', 0), type_counts.get('rag_harmed', 0), type_counts.get('rag_helped', 0)]
    bar_colors = [C['neutral_light'], C['harmed'], C['helped']]
    y_pos = [2, 1, 0]

    bars = ax1.barh(y_pos, counts, height=0.55, color=bar_colors, edgecolor='white', linewidth=0.5)
    for bar, cat, cnt in zip(bars, categories, counts):
        pct = f'{100*cnt/34:.0f}%'
        if cnt > 5:
            ax1.text(bar.get_width() / 2, bar.get_y() + bar.get_height() / 2,
                    f'{cat}\n{cnt} ({pct})', ha='center', va='center',
                    fontsize=7.5, fontweight='bold', color=C['neutral_black'])
        else:
            ax1.text(bar.get_width() + 0.4, bar.get_y() + bar.get_height() / 2,
                    f'{cat}: {cnt} ({pct})', ha='left', va='center',
                    fontsize=7, fontweight='bold', color=C['neutral_black'])

    ax1.set_yticks([])
    ax1.set_xlabel('Number of Problems')
    ax1.set_xlim(0, 40)
    ax1.grid(True, alpha=0.2, linestyle='--', axis='x')
    ax1.set_axisbelow(True)
    panel_label(ax1, 'a')

    # Panel b: Impact by difficulty
    diff_counts = defaultdict(lambda: {'helped': 0, 'harmed': 0, 'neutral': 0})
    for row in contam:
        d = row['difficulty']
        t = row['contamination_type']
        if t == 'rag_helped':
            diff_counts[d]['helped'] += 1
        elif t == 'rag_harmed':
            diff_counts[d]['harmed'] += 1
        else:
            diff_counts[d]['neutral'] += 1

    difficulties = ['Easy', 'Medium', 'Hard', 'Expert']
    x = np.arange(len(difficulties))
    w = 0.23
    helped_vals = [diff_counts[d.lower()]['helped'] for d in difficulties]
    harmed_vals = [diff_counts[d.lower()]['harmed'] for d in difficulties]
    neutral_vals = [diff_counts[d.lower()]['neutral'] for d in difficulties]

    ax2.bar(x - w, helped_vals, w, label='Helped', color=C['helped'], edgecolor='white', linewidth=0.4)
    ax2.bar(x, harmed_vals, w, label='Harmed', color=C['harmed'], edgecolor='white', linewidth=0.4)
    ax2.bar(x + w, neutral_vals, w, label='No Effect', color=C['neutral_light'], edgecolor='white', linewidth=0.4)
    ax2.set_xticks(x)
    ax2.set_xticklabels(difficulties)
    ax2.set_ylabel('Problem Count')
    ax2.legend(fontsize=6.5, handlelength=1.0, handletextpad=0.4)
    ax2.grid(True, alpha=0.2, linestyle='--', axis='y')
    ax2.set_axisbelow(True)
    panel_label(ax2, 'b')

    # Panel c: Instruction strategy impact
    strategy_labels = {'A_current': 'Standard', 'B_defensive': 'Memory\nPriority',
                       'C_minimal': 'Minimal', 'D_conflict_aware': 'Conflict\nAware'}
    strategy_order = ['A_current', 'B_defensive', 'C_minimal', 'D_conflict_aware']
    s_helped = defaultdict(int)
    s_harmed = defaultdict(int)
    for case in prompt_cases:
        s_helped[case['strategy']] += 1 if case['effect'] == 'helped' else 0
        s_harmed[case['strategy']] += 1 if case['effect'] == 'harmed' else 0

    x2 = np.arange(len(strategy_order))
    w2 = 0.28
    h_vals = [s_helped[s] for s in strategy_order]
    hm_vals = [s_harmed[s] for s in strategy_order]

    ax3.bar(x2 - w2/2, h_vals, w2, label='Helped', color=C['helped'], edgecolor='white', linewidth=0.4)
    ax3.bar(x2 + w2/2, hm_vals, w2, label='Harmed', color=C['harmed'], edgecolor='white', linewidth=0.4)
    ax3.set_xticks(x2)
    ax3.set_xticklabels([strategy_labels[s] for s in strategy_order], fontsize=6.5)
    ax3.set_ylabel('Cases')
    ax3.legend(fontsize=6.5, handlelength=1.0, handletextpad=0.4)
    ax3.grid(True, alpha=0.2, linestyle='--', axis='y')
    ax3.set_axisbelow(True)
    panel_label(ax3, 'c')

    fig.tight_layout(pad=1.0)
    save(fig, 'fig3_contamination_stats')
    print('  [OK] fig3: retrieval impact characterization')


# ============================================================
# Fig 4: Latency vs Performance
# ============================================================
def fig4_latency():
    data = json.load(open(os.path.join(RESULTS, 'summary_full.json'), 'r', encoding='utf-8'))
    prompt_data = json.load(open(os.path.join(RESULTS, 'prompt_optimization_summary.json'), 'r', encoding='utf-8'))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    # Left: Ablation configs
    labels_map = {
        'full': 'Full', 'no-temporal': 'No Temp', 'no-feedback': 'No FB',
        'no-diversity': 'No Div', 'baseline': 'Baseline'
    }
    for d in data:
        exp = d['experiment']
        ax1.scatter(d['avg_latency_s'], d['pass_at_1'], color=C.get(exp, C['neutral_mid']),
                   s=120, zorder=5, edgecolors='white', linewidth=1.0)
        ax1.annotate(labels_map[exp], (d['avg_latency_s'], d['pass_at_1']),
                    xytext=(d['avg_latency_s'] + 0.02, d['pass_at_1'] + 0.004),
                    fontsize=7.5, fontweight='bold', color=C.get(exp, C['neutral_mid']))

    ax1.set_xlabel('Average Latency (s)')
    ax1.set_ylabel('Pass@1')
    ax1.grid(True, alpha=0.2, linestyle='--')
    ax1.set_axisbelow(True)
    panel_label(ax1, 'a')

    # Right: Prompt strategies
    prompt_colors = {
        'A_current': C['neutral_mid'],
        'B_defensive': C['blue_main'],
        'C_minimal': C['no_temp'],
        'D_conflict_aware': C['red_strong'],
        'E_no_rag': C['teal'],
    }
    prompt_labels = {
        'A_current': 'Standard', 'B_defensive': 'Memory Priority',
        'C_minimal': 'Minimal', 'D_conflict_aware': 'Conflict Aware',
        'E_no_rag': 'No Retrieval',
    }
    for d in prompt_data:
        s = d['strategy']
        ax2.scatter(d['avg_latency_s'], d['pass_at_1'], color=prompt_colors[s],
                   s=120, zorder=5, edgecolors='white', linewidth=1.0)
        ax2.annotate(prompt_labels[s], (d['avg_latency_s'], d['pass_at_1']),
                    xytext=(d['avg_latency_s'] + 0.02, d['pass_at_1'] + 0.004),
                    fontsize=7, fontweight='bold', color=prompt_colors[s])

    ax2.set_xlabel('Average Latency (s)')
    ax2.set_ylabel('Pass@1')
    ax2.grid(True, alpha=0.2, linestyle='--')
    ax2.set_axisbelow(True)
    panel_label(ax2, 'b')

    fig.tight_layout(pad=0.8)
    save(fig, 'fig4_latency_vs_performance')
    print('  [OK] fig4: latency vs performance')


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print('=' * 55)
    print('Nature-Figure Generation — paper_output/figures')
    print('=' * 55)
    fig1_ablation()
    fig2_diversity()
    fig3_impact()
    fig4_latency()
    print('=' * 55)
    print('All 4 figures regenerated: PDF (vector) + PNG (300dpi)')
    print(f'Output: {OUT_DIR}/')
    print('=' * 55)
