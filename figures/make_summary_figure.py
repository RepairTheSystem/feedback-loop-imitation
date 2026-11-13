"""
Главный график сводки: насколько модель, обученная на всех событиях, лучше модели на органике —
отдельно на органических и на рекомендованных целях, по трём стендам. Читает results/*/metrics.csv.

  python figures/make_summary_figure.py      # → figures/imitation_gain.png
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
ORG, REC = "#2a78d6", "#eb6834"          # категориальные слоты 1 и 2 эталонной палитры (светлая тема)


def gains():
    """Δ Recall@10 (all / organic − 1), %, для органических и рекомендованных целей."""
    out = []
    m = pd.read_csv(os.path.join(ROOT, "results/kaggle_stage2/main/metrics.csv"))
    syn = []
    for _, g in m.groupby(["policy", "lam"]):
        a = g[g.variant == "all"].mean(numeric_only=True)
        o = g[g.variant == "organic"].mean(numeric_only=True)
        syn.append((100 * (a["Recall@10 organic"] / o["Recall@10 organic"] - 1),
                    100 * (a["Recall@10 rec"] / o["Recall@10 rec"] - 1)))
    syn = np.array(syn)
    out.append(("Синтетика\n(6 конфигураций π0 × λ)", syn.mean(0), syn.min(0), syn.max(0)))
    for name, path in (("Yambda-50M\n(музыка)", "results/kaggle_yambda_v2/yambda/metrics.csv"),
                       ("FINN.no\n(объявления)", "results/kaggle_finn/finn/metrics.csv")):
        g = pd.read_csv(os.path.join(ROOT, path)).groupby("variant").mean(numeric_only=True)
        v = np.array([100 * (g.loc["all", "Recall@10 organic"] / g.loc["organic", "Recall@10 organic"] - 1),
                      100 * (g.loc["all", "Recall@10 rec"] / g.loc["organic", "Recall@10 rec"] - 1)])
        out.append((name, v, None, None))
    return out


def main():
    data = gains()
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    x = np.arange(len(data))
    w = 0.34
    for j, (label, color) in enumerate((("цели из органики", ORG), ("цели из рекомендаций", REC))):
        vals = np.array([d[1][j] for d in data])
        pos = x + (j - 0.5) * (w + 0.02)
        ax.bar(pos, vals, width=w, color=color, label=label, zorder=2)
        for i, d in enumerate(data):
            if d[2] is not None:                                    # разброс по конфигурациям синтетики
                ax.plot([pos[i], pos[i]], [d[2][j], d[3][j]], color=INK2, linewidth=1, zorder=3)
            v = vals[i]
            y = v if d[2] is None else (d[3][j] if v >= 0 else d[2][j])   # подпись — за концом уса
            ax.annotate(f"{v:+.0f}%" if abs(v) >= 1 else f"{v:+.1f}%", (pos[i], y),
                        xytext=(0, 4 if v >= 0 else -12), textcoords="offset points",
                        ha="center", fontsize=9, color=INK)
    ax.axhline(0, color=INK2, linewidth=0.8, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([d[0] for d in data], fontsize=9, color=INK)
    ax.set_ylabel("Recall@10: all vs organic, %", fontsize=9, color=INK2)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.set_title("Обучение на всех событиях выигрывает почти только на целях, порождённых старой системой",
                 fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    out = os.path.join(ROOT, "figures", "imitation_gain.png")
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print("written", out)


if __name__ == "__main__":
    main()
