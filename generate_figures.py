# -*- coding: utf-8 -*-
"""
Generate two combined figures for the paper
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as FancyBboxPatch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

# Set style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2

# Color scheme
BLUE = '#2b6cb0'
DARK_BLUE = '#1a365d'
GREEN = '#276749'
RED = '#c53030'
ORANGE = '#dd6b20'
GRAY = '#4a5568'
LIGHT_GRAY = '#f7fafc'
BORDER_GRAY = '#e2e8f0'

# ============================================================
# FIGURE 1: System Architecture & Mechanisms
# ============================================================

def create_figure1():
    fig, axes = plt.subplots(3, 1, figsize=(14, 16))
    fig.suptitle('Figure 1: System Architecture and Mechanisms', fontsize=16, fontweight='bold', color=DARK_BLUE, y=0.98)

    # ---- Section A: System Pipeline ----
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 3)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('(a) System Pipeline Architecture', fontsize=13, fontweight='bold', color=DARK_BLUE, loc='left', pad=10)

    # Pipeline components
    boxes = [
        (1.0, 1.5, 'User\nQuery', BLUE),
        (3.0, 1.5, 'ChromaDB\nVector Search', BLUE),
        (5.0, 1.5, 'Memory\nWeight\nScoring', BLUE),
        (7.0, 1.5, 'MMR\nRe-ranking\nλ=0.7', BLUE),
        (9.0, 1.5, 'Qwen2.5-3B\nDistilled', GREEN),
    ]

    for x, y, text, color in boxes:
        box = FancyBboxPatch((x-0.55, y-0.55), 1.1, 1.1,
                            boxstyle="round,pad=0.1",
                            facecolor=color, alpha=0.15,
                            edgecolor=color, linewidth=2)
        ax1.add_patch(box)
        ax1.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold', color=color)

    # Arrows
    for i in range(4):
        ax1.annotate('', xy=(boxes[i+1][0]-0.6, 1.5),
                    xytext=(boxes[i][0]+0.6, 1.5),
                    arrowprops=dict(arrowstyle='->', color=GRAY, lw=2))

    # Formula annotation
    ax1.text(5.0, 0.4, r'$W_{total} = \alpha \cdot W_t + \beta \cdot W_f + \gamma \cdot W_r$',
            ha='center', va='center', fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor=GRAY, alpha=0.8))

    # Separator line
    axes[0].axhline(y=-0.1, color=BORDER_GRAY, linewidth=1, linestyle='--')

    # ---- Section B: Retrieval Contamination ----
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 3.5)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title('(b) Retrieval Contamination Mechanism', fontsize=13, fontweight='bold', color=DARK_BLUE, loc='left', pad=10)

    # Clean path (top)
    clean_boxes = [
        (1.5, 2.5, 'Distilled\nWeights'),
        (4.0, 2.5, 'Internal\nPattern'),
        (6.5, 2.5, 'Correct\nOutput ✓'),
    ]

    for x, y, text in clean_boxes:
        box = FancyBboxPatch((x-0.5, y-0.4), 1.0, 0.8,
                            boxstyle="round,pad=0.1",
                            facecolor=GREEN, alpha=0.2,
                            edgecolor=GREEN, linewidth=2)
        ax2.add_patch(box)
        ax2.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold', color=GREEN)

    # Arrow for clean path
    ax2.annotate('', xy=(3.5, 2.5), xytext=(2.0, 2.5),
                arrowprops=dict(arrowstyle='->', color=GREEN, lw=2))
    ax2.annotate('', xy=(6.0, 2.5), xytext=(4.5, 2.5),
                arrowprops=dict(arrowstyle='->', color=GREEN, lw=2))

    ax2.text(4.0, 3.2, 'Clean Knowledge Path', ha='center', fontsize=10, color=GREEN, fontweight='bold')

    # Contaminated path (bottom)
    contam_boxes = [
        (1.5, 1.0, 'Vector DB\nRetrieval'),
        (4.0, 1.0, 'Conflicting\nSnippet'),
        (6.5, 1.0, 'Pattern\nConflict'),
        (8.5, 1.0, 'Wrong\nOutput ✗'),
    ]

    for x, y, text in contam_boxes:
        box = FancyBboxPatch((x-0.5, y-0.4), 1.0, 0.8,
                            boxstyle="round,pad=0.1",
                            facecolor=RED, alpha=0.2,
                            edgecolor=RED, linewidth=2)
        ax2.add_patch(box)
        ax2.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold', color=RED)

    # Arrows for contaminated path
    for i in range(3):
        ax2.annotate('', xy=(contam_boxes[i+1][0]-0.55, 1.0),
                    xytext=(contam_boxes[i][0]+0.55, 1.0),
                    arrowprops=dict(arrowstyle='->', color=RED, lw=2))

    ax2.text(4.0, 0.2, 'Contaminated Retrieval Path', ha='center', fontsize=10, color=RED, fontweight='bold')

    # Lightning bolt (conflict zone)
    ax2.text(8.8, 2.5, '⚡', fontsize=24, ha='center', va='center')
    ax2.text(8.8, 2.0, 'Contamination', ha='center', fontsize=9, color=RED, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=RED, alpha=0.9))

    # Separator line
    axes[1].axhline(y=-0.2, color=BORDER_GRAY, linewidth=1, linestyle='--')

    # ---- Section C: Memory Lifecycle ----
    ax3 = axes[2]
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 4)
    ax3.set_aspect('equal')
    ax3.axis('off')
    ax3.set_title('(c) Memory Lifecycle States', fontsize=13, fontweight='bold', color=DARK_BLUE, loc='left', pad=10)

    # Weight components (left side)
    weight_boxes = [
        (1.5, 3.2, r'$W_t = e^{-\lambda \cdot \Delta t}$', r'$\alpha=0.5$'),
        (1.5, 2.2, r'$W_f = \frac{s + p \cdot k}{t + k}$', r'$\beta=0.3$'),
        (1.5, 1.2, r'$W_r = 1 - sim_{max}$', r'$\gamma=0.2$'),
    ]

    for x, y, formula, param in weight_boxes:
        box = FancyBboxPatch((x-0.8, y-0.35), 1.6, 0.7,
                            boxstyle="round,pad=0.1",
                            facecolor=BLUE, alpha=0.1,
                            edgecolor=BLUE, linewidth=1.5)
        ax3.add_patch(box)
        ax3.text(x-0.2, y, formula, ha='center', va='center', fontsize=9)
        ax3.text(x+0.65, y, param, ha='center', va='center', fontsize=8, color=GRAY)

    # Arrow to aggregation
    ax3.annotate('', xy=(3.0, 2.2), xytext=(2.3, 2.2),
                arrowprops=dict(arrowstyle='->', color=GRAY, lw=2))
    ax3.text(3.5, 2.2, r'$W_{total}$', fontsize=12, fontweight='bold', color=DARK_BLUE, va='center')

    # Lifecycle states (right side)
    states = [
        (6.0, 3.3, 'ACTIVE', 'W > 0.8', '#48bb78'),
        (6.0, 2.6, 'WARM', '0.5 < W ≤ 0.8', '#9ae6b4'),
        (6.0, 1.9, 'COLD', '0.2 < W ≤ 0.5', '#63b3ed'),
        (6.0, 1.2, 'ARCHIVED', 'W ≤ 0.2', '#a0aec0'),
        (6.0, 0.5, 'SUPPRESSED', 'W ≤ 0.05', '#718096'),
    ]

    for x, y, name, cond, color in states:
        circle = plt.Circle((x, y), 0.25, facecolor=color, alpha=0.6, edgecolor=color, linewidth=2)
        ax3.add_patch(circle)
        ax3.text(x, y, name, ha='center', va='center', fontsize=7, fontweight='bold', color='white')
        ax3.text(x+0.9, y, cond, ha='left', va='center', fontsize=9, color=GRAY)

    # Arrows between states
    for i in range(4):
        ax3.annotate('', xy=(6.0, states[i+1][1]+0.28), xytext=(6.0, states[i][1]-0.28),
                    arrowprops=dict(arrowstyle='->', color=GRAY, lw=1.5))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('paper_output/figure1_architecture.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print('Figure 1 saved: paper_output/figure1_architecture.png')


# ============================================================
# FIGURE 2: Experimental Results & Analysis
# ============================================================

def create_figure2():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.suptitle('Figure 2: Experimental Results and Analysis', fontsize=16, fontweight='bold', color=DARK_BLUE, y=1.02)

    # ---- Panel A: Ablation Study ----
    ax1 = axes[0]

    configs = ['No-Div', 'Full\nSystem', 'Baseline', 'No-Temp', 'No-Feed']
    pass1 = [0.794, 0.765, 0.765, 0.735, 0.706]
    diversity = [0.725, 0.905, 0.533, 0.905, 0.905]

    x = np.arange(len(configs))
    width = 0.35

    bars1 = ax1.bar(x - width/2, pass1, width, label='Pass@1', color=BLUE, alpha=0.85, edgecolor=BLUE, linewidth=1.2)
    bars2 = ax1.bar(x + width/2, diversity, width, label='Diversity', color=ORANGE, alpha=0.85, edgecolor=ORANGE, linewidth=1.2)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    # Star on best Pass@1
    ax1.plot(0 - width/2, 0.794, 'r*', markersize=15)

    # Reference lines
    ax1.axhline(y=0.765, color=GREEN, linestyle='--', linewidth=1.5, alpha=0.7, label='No-RAG Baseline')
    ax1.axhline(y=0.676, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, label='Original Model')

    ax1.set_xlabel('Configuration', fontsize=10)
    ax1.set_ylabel('Score', fontsize=10)
    ax1.set_title('(a) Ablation Study', fontsize=12, fontweight='bold', color=DARK_BLUE, loc='left')
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, fontsize=9)
    ax1.set_ylim(0.6, 0.95)
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(axis='y', alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # ---- Panel B: Diversity-Performance ----
    ax2 = axes[1]

    diversity_points = [0.533, 0.725, 0.905]
    pass1_points = [0.765, 0.794, 0.765]
    labels = ['Baseline', 'Optimal', 'Full System']
    colors = [GRAY, GREEN, ORANGE]
    sizes = [80, 200, 80]

    # Smooth curve
    x_smooth = np.linspace(0.45, 0.95, 100)
    # Fit a quadratic
    coeffs = np.polyfit(diversity_points, pass1_points, 2)
    y_smooth = np.polyval(coeffs, x_smooth)

    # Optimal zone
    ax2.axvspan(0.70, 0.75, alpha=0.15, color=GREEN, label='Optimal Range')

    # Plot curve and points
    ax2.plot(x_smooth, y_smooth, '-', color=BLUE, linewidth=2, alpha=0.7)

    for i, (x, y, label, color, size) in enumerate(zip(diversity_points, pass1_points, labels, colors, sizes)):
        ax2.scatter(x, y, c=color, s=size, zorder=5, edgecolors='white', linewidth=2)
        offset_y = 0.015 if i != 1 else 0.02
        offset_x = 0.02 if i == 0 else (-0.05 if i == 2 else 0)
        ax2.annotate(label, (x, y), xytext=(x+offset_x, y+offset_y),
                    fontsize=9, fontweight='bold', color=color, ha='center')

    # Star on optimal
    ax2.plot(0.725, 0.794, 'r*', markersize=20, zorder=6)

    # Arrows
    ax2.annotate('', xy=(0.82, 0.775), xytext=(0.76, 0.79),
                arrowprops=dict(arrowstyle='->', color=RED, lw=1.5))
    ax2.text(0.82, 0.77, 'Too diverse\n→ Conflicts', fontsize=8, color=RED, ha='center')

    ax2.annotate('', xy=(0.60, 0.775), xytext=(0.66, 0.79),
                arrowprops=dict(arrowstyle='->', color=RED, lw=1.5))
    ax2.text(0.60, 0.77, 'Too redundant\n→ Waste', fontsize=8, color=RED, ha='center')

    ax2.set_xlabel('Retrieval Diversity', fontsize=10)
    ax2.set_ylabel('Pass@1 Score', fontsize=10)
    ax2.set_title('(b) Diversity-Performance', fontsize=12, fontweight='bold', color=DARK_BLUE, loc='left')
    ax2.set_xlim(0.45, 0.95)
    ax2.set_ylim(0.72, 0.83)
    ax2.grid(alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # ---- Panel C: Prompt Strategy ----
    ax3 = axes[2]

    strategies = ['Conflict-\nAware', 'Minimal', 'Current', 'No RAG\n(Control)', 'Defensive']
    scores = [0.706, 0.735, 0.735, 0.765, 0.765]
    bar_colors = [RED, BLUE, BLUE, GRAY, GREEN]

    bars = ax3.barh(strategies, scores, color=bar_colors, alpha=0.8, edgecolor='white', linewidth=1.2, height=0.6)

    # Value labels
    for bar, score in zip(bars, scores):
        width = bar.get_width()
        ax3.text(width + 0.005, bar.get_y() + bar.get_height()/2.,
                f'{score:.3f}', ha='left', va='center', fontsize=9, fontweight='bold')

    # Checkmark and X
    ax3.text(0.77, 4, '✓', fontsize=16, color=GREEN, fontweight='bold', va='center')
    ax3.text(0.77, 0, '✗', fontsize=16, color=RED, fontweight='bold', va='center')

    # Baseline line
    ax3.axvline(x=0.765, color=GREEN, linestyle='--', linewidth=1.5, alpha=0.7)

    # Bracket for equivalent
    ax3.annotate('', xy=(0.765, 3.7), xytext=(0.765, 4.3),
                arrowprops=dict(arrowstyle='-', color=DARK_BLUE, lw=1.5))
    ax3.text(0.78, 4.0, 'Equivalent', fontsize=8, color=DARK_BLUE, rotation=90, va='center')

    ax3.set_xlabel('Pass@1 Score', fontsize=10)
    ax3.set_title('(c) Prompt Strategies', fontsize=12, fontweight='bold', color=DARK_BLUE, loc='left')
    ax3.set_xlim(0.65, 0.82)
    ax3.grid(axis='x', alpha=0.3)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('paper_output/figure2_results.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print('Figure 2 saved: paper_output/figure2_results.png')


if __name__ == '__main__':
    create_figure1()
    create_figure2()
    print('\nDone! Both figures saved to paper_output/')
