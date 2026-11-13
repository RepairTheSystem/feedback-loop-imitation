"""
Собирает самодостаточный ноутбук этапа 3 (Yambda-50M, DESIGN.md §10) для Kaggle/Colab.
Общие ячейки (настройка окружения, run) — из build_notebook.py.

  python feedback_loop/kaggle/build_yambda_notebook.py
  python feedback_loop/kaggle/build_yambda_notebook.py --quick --no-gpu --data <stub.parquet> --out <path>
"""
import argparse
import os
import sys

import nbformat as nbf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_notebook import FL, RS, SETUP, ZIP  # noqa: E402

SOURCES = [(RS, f) for f in ("data.py", "metrics.py", "mixing.py", "baselines.py", "heuristics.py", "model.py",
                             "train.py")] + \
          [(FL, f) for f in ("common.py", "sim.py", "fl_data.py", "fl_train.py", "yambda.py", "yambda_train.py")]

INTRO = """# Петля обратной связи — этап 3: Yambda-50M

Копирует ли модель старый рекомендатель на реальных логах Яндекс Музыки? В Yambda у каждого события есть флаг
`is_organic`; около половины прослушиваний пришли из рекомендаций. Одна и та же модель учится с разными весами
рекомендованных событий (`all`, `organic`, `all-matched`, `alpha-X`, `ips-est`, `imitator`) и оценивается на
органических и рекомендованных целях тестового дня. Дизайн и результаты синтетики — `feedback_loop/` в репозитории.

**Данные:** [yandex/yambda](https://huggingface.co/datasets/yandex/yambda), `flat/50m/listens.parquet`
(10 тыс. пользователей, ~46.5 млн прослушиваний), лицензия Apache 2.0, только для исследований.

**Запуск на Kaggle**

1. *File → Import Notebook* — этот `.ipynb`; *Settings → Accelerator → GPU*.
2. Данные — один из двух способов:
   * *Settings → Internet → On* (нужен подтверждённый телефон) — ноутбук скачает файл с Hugging Face сам;
   * или добавить `listens.parquet` как Kaggle Dataset (*Add Input*) — ноутбук найдёт его в `/kaggle/input`.
3. Сначала `QUICK = True` и *Run All* (1000 пользователей, ~10 мин) — проверить и посмотреть время эпохи.
4. Полный прогон: `QUICK = False`, *Save Version → Save & Run All (Commit)*.

**Время (оценка, не замер):** ~15 мин на вариант, 7 вариантов ≈ 1.7 ч на сид; по умолчанию 2 сида ≈ 3.5 ч.
Сиды меняют только инициализацию — неопределённость по пользователям покрывает парный бутстрап.
"""

CONFIG = """# ---- настройки прогона ----
QUICK = {quick}
SEEDS = "0 1"
TOP_ITEMS = 20000              # каталог: самые частые треки обучения (полный softmax по 0.9 млн не влезет)
TRAIN_TAIL = 300               # последних позитивов на пользователя в обучении
VARIANTS = "all organic all-matched alpha-0.25 alpha-0.5 ips-est imitator"
EXTRA = ""
DATA_PATH = {data_path}        # None — найти в /kaggle/input или скачать с Hugging Face

if QUICK:
    SEEDS, TOP_ITEMS, EXTRA = "0", 5000, "--n-users 1000 --max-epochs 3"
os.makedirs("results", exist_ok=True)
print(f"QUICK={{QUICK}} SEEDS={{SEEDS}} TOP_ITEMS={{TOP_ITEMS}} TRAIN_TAIL={{TRAIN_TAIL}} EXTRA={{EXTRA}}")
"""

DATA = """import glob
path = DATA_PATH
if path is None:
    found = glob.glob("/kaggle/input/**/listens.parquet", recursive=True)
    if found:
        path = found[0]
    else:
        from yambda import download_listens
        path = download_listens()          # нужен интернет
print("данные:", path, f"{os.path.getsize(path) / 2**20:.0f} МБ")
"""

RUN = """t0 = time.time()
run(f'yambda_train.py --path "{path}" --seeds {SEEDS} --top-items {TOP_ITEMS} --train-tail {TRAIN_TAIL} '
    f'--variants {VARIANTS} {EXTRA} --out results/yambda', "results/yambda.log")
print(f"прогон: {(time.time() - t0) / 60:.1f} мин")
"""

SHOW = """from IPython.display import Markdown, display
display(Markdown(open("results/yambda/summary.md", encoding="utf-8").read()))
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(FL, "kaggle", "feedback_loop_yambda.ipynb"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--data", default=None, help="путь к listens.parquet (для локальной проверки)")
    a = ap.parse_args()

    nb = nbf.v4.new_notebook()
    cells = [nbf.v4.new_markdown_cell(INTRO),
             nbf.v4.new_code_cell(SETUP.format(require_gpu=not a.no_gpu).replace(
                 'os.path.join(BASE, "feedback_loop")', 'os.path.join(BASE, "feedback_loop_yambda")')),
             nbf.v4.new_markdown_cell("## Код\nСгенерировано `feedback_loop/kaggle/build_yambda_notebook.py`.")]
    for root, name in SOURCES:
        with open(os.path.join(root, name), encoding="utf-8") as f:
            cells.append(nbf.v4.new_code_cell(f"%%writefile {name}\n" + f.read()))
    cells += [nbf.v4.new_markdown_cell("## Прогон"),
              nbf.v4.new_code_cell(CONFIG.format(quick=a.quick, data_path=repr(a.data))),
              nbf.v4.new_code_cell(DATA),
              nbf.v4.new_code_cell(RUN),
              nbf.v4.new_markdown_cell("## Результаты"),
              nbf.v4.new_code_cell(SHOW),
              nbf.v4.new_code_cell(ZIP)]
    nb["cells"] = cells
    nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                      "language_info": {"name": "python"},
                      "kaggle": {"accelerator": "gpu", "isGpuEnabled": True, "isInternetEnabled": True}}
    nbf.validate(nb)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    nbf.write(nb, a.out)
    print("written", a.out)


if __name__ == "__main__":
    main()
