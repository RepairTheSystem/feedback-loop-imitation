"""
Собирает самодостаточный ноутбук этапа 2 (синтетика, DESIGN.md §10) для Kaggle/Colab:
исходники feedback_loop и нужные модули repeat_switch вставляются в ячейки %%writefile.

  python feedback_loop/kaggle/build_notebook.py
  python feedback_loop/kaggle/build_notebook.py --quick --no-gpu --out <path>   # для локальной проверки

Пересобирай после любых правок в .py.
"""
import argparse
import os

import nbformat as nbf

FL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RS = os.path.join(FL, "rs_core")   # копия модулей repeat_switch
SOURCES = [(RS, f) for f in ("data.py", "metrics.py", "mixing.py", "baselines.py", "heuristics.py", "model.py",
                             "train.py")] + \
          [(FL, f) for f in ("common.py", "sim.py", "fl_data.py", "fl_train.py", "sanity.py")]

INTRO = """# Петля обратной связи — этап 2: синтетика

Копирует ли модель старый рекомендатель? Симулятор маркетплейса со старой системой π0 и силой давления показа λ;
одна и та же модель учится с разными весами событий (`all`, `organic`, `all-matched`, `alpha-X`, `ips-true`,
`ips-est`, `imitator`) и оценивается на органических целях, на истинной полезности и в симулированном онлайне.
Дизайн — `feedback_loop/DESIGN.md` в репозитории.

**Запуск на Kaggle**

1. *File → Import Notebook* — этот `.ipynb`; *Settings → Accelerator → GPU*. Интернет не нужен.
2. Сначала `QUICK = True` и *Run All* (несколько минут) — убедиться, что всё работает, и посмотреть время на конфигурацию.
3. Полный прогон: `QUICK = False`, *Save Version → Save & Run All (Commit)* — считается в фоне.
4. Результаты — `results.zip` во вкладке *Output* (`summary.md`, CSV, графики, лог).

**Время (оценка, не замер):** симуляция идёт на CPU (~3–4 мин на конфигурацию с калибровкой), обучение 8 моделей —
на GPU (~1 мин каждая). Полная сетка 2 политики × 3 λ × 3 сида = 18 конфигураций ≈ 3–4.5 ч — в пределах 12 ч сессии.
"""

SETUP = """import os, sys, time, subprocess, torch

REQUIRE_GPU = {require_gpu}
print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "нет GPU")
if REQUIRE_GPU:
    assert torch.cuda.is_available(), "Включи GPU: Settings → Accelerator"

BASE = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
WORK = os.path.join(BASE, "feedback_loop")
os.makedirs(WORK, exist_ok=True)
os.chdir(WORK)
print("работаем в", WORK)
try:
    import tabulate  # noqa: F401
except ImportError:
    os.system(f"{{sys.executable}} -m pip install -q tabulate")


def run(args, log):
    \"\"\"Запускает python-скрипт, печатает вывод по ходу и пишет его в лог.\"\"\"
    print("$ python", args, flush=True)
    env = {{**os.environ, "PYTHONIOENCODING": "utf-8"}}
    with open(log, "w", encoding="utf-8") as f, subprocess.Popen(
            f'"{{sys.executable}}" -u {{args}}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", env=env) as p:
        for line in p.stdout:
            print(line, end="", flush=True)
            f.write(line)
    assert p.returncode == 0, f"процесс упал (код {{p.returncode}}), см. {{log}}"
"""

CONFIG = """# ---- настройки прогона ----
QUICK = {quick}                 # True — дымовой прогон, False — полный
POLICIES = "good skewed"        # π0: good — близка к правде, skewed — продвигает сегмент; можно добавить pop
LAMS = "0 0.5 1"                # сила давления показа
SEEDS = "0 1 2"
VARIANTS = "all organic all-matched alpha-0.25 alpha-0.5 ips-true ips-est imitator"
COMMON = "--n-users 4000 --max-epochs 40"
RUN_SANITY = False              # проверки на вменяемость (уже пройдены локально, см. README)

if QUICK:
    POLICIES, LAMS, SEEDS = "skewed", "1", "0"
    COMMON = "--n-users 500 --max-epochs 2"
os.makedirs("results", exist_ok=True)
print(f"QUICK={{QUICK}} POLICIES={{POLICIES}} LAMS={{LAMS}} SEEDS={{SEEDS}}")
print(f"VARIANTS={{VARIANTS}} COMMON={{COMMON}}")
"""

RUN_MAIN = """t0 = time.time()
run(f"fl_train.py --policy {POLICIES} --lam {LAMS} --seeds {SEEDS} --variants {VARIANTS} {COMMON} --out results/main",
    "results/main.log")
print(f"основной прогон: {(time.time() - t0) / 60:.1f} мин")
"""

RUN_SANITY = """if RUN_SANITY:
    run("sanity.py --part data --out results/sanity", "results/sanity_data.log")
    run(f"sanity.py --part models --device {'cuda' if torch.cuda.is_available() else 'cpu'} --out results/sanity",
        "results/sanity_models.log")
else:
    print("пропущено (RUN_SANITY = False)")
"""

SHOW = """from IPython.display import Markdown, display
display(Markdown(open("results/main/summary.md", encoding="utf-8").read()))
"""

# Графики: кривые по α, одна линия на λ. λ — порядковая величина → одна синяя шкала светлое→тёмное
# (шаги 250/450/650 эталонной палитры — проходят 2:1 на светлом фоне для порядковых меток),
# плюс разные маркеры и подписи у концов линий, чтобы λ не кодировалось только цветом.
PLOT = """import numpy as np, pandas as pd, matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
LAM_STYLE = {0.0: ("#86b6ef", "o"), 0.5: ("#2a78d6", "s"), 1.0: ("#104281", "^")}
METRICS = [("true NDCG@10", "true NDCG@10 (правда)"), ("online util/user", "онлайн: полезность на пользователя"),
           ("Recall@10 organic", "Recall@10 на органике"), ("promo share", "доля promo в выдаче"),
           ("overlap@10 imitator", "сходство с имитатором π0")]
ALPHA = {"organic": 0.0, "alpha-0.25": 0.25, "alpha-0.5": 0.5, "all": 1.0}

m = pd.read_csv("results/main/metrics.csv")
c = m[m.variant.isin(list(ALPHA))].assign(alpha=lambda d: d.variant.map(ALPHA))
metrics = [x for x in METRICS if x[0] in c]
policies = list(dict.fromkeys(c.policy))
fig, axes = plt.subplots(len(metrics), len(policies), figsize=(4.6 * len(policies), 2.9 * len(metrics)),
                         squeeze=False, sharex=True)
fig.patch.set_facecolor(SURFACE)
for i, (col, title) in enumerate(metrics):
    for j, pol in enumerate(policies):
        ax = axes[i][j]
        ax.set_facecolor(SURFACE)
        d = c[c.policy == pol]
        for lam, g in d.groupby("lam"):
            color, marker = LAM_STYLE.get(float(lam), ("#2a78d6", "o"))
            s = g.groupby("alpha")[col].agg(["mean", "std"]).reset_index()
            ax.errorbar(s.alpha, s["mean"], yerr=s["std"].fillna(0), color=color, marker=marker, markersize=6,
                        linewidth=2, capsize=0, elinewidth=1, label=f"λ = {lam:g}")
            ax.annotate(f"λ={lam:g}", (s.alpha.iloc[-1], s["mean"].iloc[-1]), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=8, color=INK2)
        ax.set_title(f"{title} — π0 = {pol}" if i == 0 else title, fontsize=9, color=INK, loc="left")
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=8)
        if i == len(metrics) - 1:
            ax.set_xlabel("α — вес рекомендованных событий (0 = organic, 1 = all)", fontsize=8, color=INK2)
        ax.set_xticks([0, 0.25, 0.5, 1.0])
        ax.set_xlim(-0.05, 1.18)
if c.lam.nunique() > 1:                     # одна линия — легенда не нужна, её называет подпись
    axes[0][-1].legend(frameon=False, fontsize=8, loc="best")
fig.tight_layout()
fig.savefig("results/main/alpha_curves.png", dpi=150, facecolor=SURFACE)
plt.show()

# таблица к графику (тот же набор чисел — для чтения без цвета)
display(c.groupby(["policy", "lam", "alpha"])[[x[0] for x in metrics]].mean().round(3))
"""

ZIP = """import shutil
archive = shutil.make_archive(os.path.join(BASE, "results"), "zip", "results")
print("сохранено:", archive)
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(FL, "kaggle", "feedback_loop_stage2.ipynb"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-gpu", action="store_true")
    a = ap.parse_args()

    nb = nbf.v4.new_notebook()
    cells = [nbf.v4.new_markdown_cell(INTRO),
             nbf.v4.new_code_cell(SETUP.format(require_gpu=not a.no_gpu)),
             nbf.v4.new_markdown_cell("## Код\nСгенерировано `feedback_loop/kaggle/build_notebook.py`: "
                                      "модули `repeat_switch` (модель, метрики) и `feedback_loop`.")]
    for root, name in SOURCES:
        with open(os.path.join(root, name), encoding="utf-8") as f:
            cells.append(nbf.v4.new_code_cell(f"%%writefile {name}\n" + f.read()))
    cells += [nbf.v4.new_markdown_cell("## Прогон"),
              nbf.v4.new_code_cell(CONFIG.format(quick=a.quick)),
              nbf.v4.new_code_cell(RUN_SANITY),
              nbf.v4.new_code_cell(RUN_MAIN),
              nbf.v4.new_markdown_cell("## Результаты"),
              nbf.v4.new_code_cell(SHOW),
              nbf.v4.new_code_cell(PLOT),
              nbf.v4.new_code_cell(ZIP)]
    nb["cells"] = cells
    nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                      "language_info": {"name": "python"},
                      "kaggle": {"accelerator": "gpu", "isGpuEnabled": True, "isInternetEnabled": False}}
    nbf.validate(nb)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    nbf.write(nb, a.out)
    print("written", a.out)


if __name__ == "__main__":
    main()
