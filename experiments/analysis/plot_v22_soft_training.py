"""Training curves of a V22 MJX run from its Slurm log: exploration, optimizer, evaluation. Writes PNGs and a JSON of the curves.

    python experiments/analysis/plot_v22_soft_training.py --log LOG --output DIR
"""
import argparse
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BG, FG, MUTED, GRID, ACCENT, WARM = '#11171e', '#dce4ec', '#a3b1bf', '#2a3540', '#84d4c0', '#edbe85'
ROW = re.compile(r'\[(\d\d-\d\d [\d:]+)\] \[ppo\.py:\d+\] INFO - │ (\S+)\s*│ ([-\d.e+]+)\s*│')


def parse(path):
    blocks, current = [], {}
    for line in open(path, errors='ignore'):
        if 'ppo.py' not in line:
            continue
        match = ROW.search(line)
        if match:
            current[match[2]] = float(match[3])
        elif '└' in line and current:
            blocks.append(current)
            current = {}
    return blocks


def series(blocks, key):
    pairs = [(b['steps/nr_env_steps'] / 1e6, b[key]) for b in blocks if key in b and 'steps/nr_env_steps' in b]
    return np.array(pairs).T


def style(ax, title, ylabel):
    ax.set_facecolor(BG)
    ax.set_title(title, color=FG, fontsize=11, loc='left', pad=8)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.set_xlabel('mln kroków środowiska', color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color=GRID, linewidth=.6)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)


def smooth(y, width=9):
    if len(y) < width:
        return y
    pad = np.pad(y, width // 2, mode='edge')
    return np.convolve(pad, np.ones(width) / width, mode='valid')


def figure(panels, path, rows, cols, blocks, height=3.2):
    fig, axes = plt.subplots(rows, cols, figsize=(5.4 * cols, height * rows), facecolor=BG)
    for ax, (key, title, ylabel, color, kwargs) in zip(np.atleast_1d(axes).ravel(), panels):
        x, y = series(blocks, key)
        style(ax, title, ylabel)
        if kwargs.get('raw_too'):
            ax.plot(x, y, color=color, alpha=.25, linewidth=1)
        ax.plot(x, smooth(y) if kwargs.get('raw_too') else y, color=color, linewidth=2, marker='o' if kwargs.get('markers') else None, markersize=4)
        for line, label in kwargs.get('refs', []):
            ax.axhline(line, color=WARM, linewidth=1, linestyle=(0, (4, 3)))
            ax.annotate(label, (x[0], line), textcoords='offset points', xytext=(4, 4), color=WARM, fontsize=8)
        if 'ylim' in kwargs:
            ax.set_ylim(*kwargs['ylim'])
        ax.annotate(f'{y[-1]:.4g}', (x[-1], y[-1]), textcoords='offset points', xytext=(-4, 7), color=FG, fontsize=9, ha='right')
    fig.tight_layout(h_pad=2, w_pad=2)
    fig.savefig(path, dpi=130, facecolor=BG)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    blocks = parse(args.log)
    keys = ('policy/std_dev', 'lr/learning_rate', 'policy_ratio/approx_kl', 'policy_ratio/clip_fraction', 'loss/entropy_loss',
            'loss/critic_loss', 'v_value/explained_variance', 'gradients/policy_grad_norm', 'gradients/critic_grad_norm',
            'eval/landing_success_rate', 'eval/worst_cell_success_rate', 'eval/episode_return', 'eval/worst_height_success_rate',
            'env_info/metrics/loaded_slip_integral', 'time/sps')
    figure([('policy/std_dev', 'Odchylenie standardowe akcji (std)', 'std', ACCENT, {}),
            ('policy_ratio/clip_fraction', 'Ułamek przyciętych próbek PPO', 'clip fraction', ACCENT, dict(raw_too=True, refs=[(.2, 'clip_range 0,2')])),
            ('policy_ratio/approx_kl', 'Przybliżone KL starej i nowej polityki', 'approx KL', ACCENT, dict(raw_too=True)),
            ('loss/entropy_loss', 'Entropia polityki (współczynnik 0)', 'entropia', ACCENT, {})],
           args.output / 'exploration.png', 2, 2, blocks)
    figure([('lr/learning_rate', 'Learning rate (liniowo 3e-5 → 1e-5)', 'learning rate', ACCENT, {}),
            ('gradients/policy_grad_norm', 'Norma gradientu polityki', 'norma', ACCENT, dict(raw_too=True)),
            ('loss/critic_loss', 'Strata krytyka', 'strata', ACCENT, dict(raw_too=True)),
            ('v_value/explained_variance', 'Wyjaśniona wariancja krytyka', 'EV', ACCENT, dict(raw_too=True, ylim=(.8, 1)))],
           args.output / 'optimizer.png', 2, 2, blocks)
    figure([('eval/landing_success_rate', 'Sukces na całej siatce ewaluacji', 'udział', ACCENT, dict(markers=True, ylim=(.85, 1))),
            ('eval/worst_cell_success_rate', 'Najgorsza komórka siatki', 'udział', ACCENT, dict(markers=True, ylim=(.85, 1.01))),
            ('eval/episode_return', 'Zwrot z epizodu (ewaluacja)', 'zwrot', ACCENT, dict(markers=True)),
            ('env_info/metrics/loaded_slip_integral', 'Całka poślizgu pod obciążeniem (trening)', 'całka', ACCENT, dict(raw_too=True))],
           args.output / 'evaluation.png', 2, 2, blocks)
    sample = lambda k: [[round(float(a), 3), round(float(b), 5)] for a, b in zip(*series(blocks, k))][::4]
    (args.output / 'training_curves.json').write_text(json.dumps(
        dict(source=args.log.name, rollouts=len(blocks), curves={k: sample(k) for k in keys}), indent=0))
    final = {k: series(blocks, k)[1][-1] for k in keys}
    print(len(blocks), json.dumps(final, indent=1))
    std = series(blocks, 'policy/std_dev')[1]
    kl = series(blocks, 'policy_ratio/approx_kl')[1]
    clip = series(blocks, 'policy_ratio/clip_fraction')[1]
    print('std first/min/last', std[0], std.min(), std[-1], 'kl mean', kl.mean(), 'clip first/mean/last', clip[0], clip.mean(), clip[-1])


if __name__ == '__main__':
    main()
