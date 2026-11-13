"""
Собирает самодостаточный ноутбук этапа 4 (FINN.no Slates, DESIGN.md §10) для Kaggle/Colab.

  python feedback_loop/kaggle/build_finn_notebook.py
  python feedback_loop/kaggle/build_finn_notebook.py --quick --no-gpu --data <stub.npz> --out <path>
"""
import argparse
import os
import sys

import nbformat as nbf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_notebook import FL, RS, SETUP, ZIP  # noqa: E402

SOURCES = [(RS, f) for f in ("data.py", "metrics.py", "mixing.py", "baselines.py", "heuristics.py", "model.py",
                             "train.py")] + \
          [(FL, f) for f in ("common.py", "sim.py", "fl_data.py", "fl_train.py", "yambda.py", "yambda_train.py",
                             "finn.py", "finn_train.py")]

INTRO = """# Петля обратной связи — этап 4: FINN.no Slates

Маркетплейс объявлений FINN.no: 2.3 млн пользователей, по 20 первых взаимодействий, у каждого — **весь показанный
слейт** и источник: поиск (~70%) или рекомендации (~30%). «Органика» здесь — клики из поиска (оговорка: поиск —
тоже ранжирующая система, но выдача обусловлена запросом). Показы дают то, чего не было в Yambda:
IPS по реальной частоте показа (`ips-true`), имитатор старой системы на её показах (`imitator-imp`) и прямую меру
имитации — долю топ-10 модели, попавшую в реально показанный слейт (`agreement`).

**Данные:** [simeneide/recsys_slates_dataset](https://huggingface.co/datasets/simeneide/recsys_slates_dataset),
`data.npz`, Apache 2.0.

**Запуск на Kaggle**

1. *File → Import Notebook*, *Settings → Accelerator → GPU*.
2. Данные: *Settings → Internet → On* (ноутбук скачает `data.npz` с Hugging Face или через pip-пакет авторов)
   или добавь `data.npz` как Kaggle Dataset (*Add Input*).
3. Сначала `QUICK = True` и *Run All* (20 тыс. пользователей, ~10 мин): проверить загрузку и посмотреть покрытие
   каталога и время эпохи. Путь к файлу на Hugging Face я не проверял — если скачивание упадёт, сообщение покажет почему.
4. Полный прогон: `QUICK = False`, *Save Version → Save & Run All (Commit)*.

**Время (оценка, не замер):** 300 тыс. пользователей, 9 вариантов, ~1.5 ч на сид; по умолчанию 2 сида ≈ 3 ч.
Оценка — на выборке 30 тыс. пользователей теста (матрица скоров U × N иначе не влезет в память).
"""

CONFIG = """# ---- настройки прогона ----
QUICK = {quick}
SEEDS = "0 1"
N_USERS = 300000               # пользователей в обучении (из 2.3 млн)
TOP_ITEMS = 20000              # каталог: самые кликабельные объявления обучения
VARIANTS = "all organic all-matched alpha-0.25 alpha-0.5 ips-true ips-est imitator imitator-imp"
EXTRA = ""
DATA_PATH = {data_path}        # None — найти в /kaggle/input или скачать

if QUICK:
    SEEDS, N_USERS, TOP_ITEMS, EXTRA = "0", 20000, 5000, "--max-epochs 3 --n-eval 5000 5000"
os.makedirs("results", exist_ok=True)
print(f"QUICK={{QUICK}} SEEDS={{SEEDS}} N_USERS={{N_USERS}} TOP_ITEMS={{TOP_ITEMS}} EXTRA={{EXTRA}}")
"""

DATA = """import glob
path = DATA_PATH
if path is None:
    found = glob.glob("/kaggle/input/**/data.npz", recursive=True)
    if found:
        path = found[0]
    else:
        from finn import download
        path = download()                  # нужен интернет
print("данные:", path, f"{os.path.getsize(path) / 2**20:.0f} МБ")
"""

RUN = """t0 = time.time()
run(f'finn_train.py --path "{path}" --seeds {SEEDS} --n-users {N_USERS} --top-items {TOP_ITEMS} '
    f'--variants {VARIANTS} {EXTRA} --out results/finn', "results/finn.log")
print(f"прогон: {(time.time() - t0) / 60:.1f} мин")
"""

SHOW = """from IPython.display import Markdown, display
display(Markdown(open("results/finn/summary.md", encoding="utf-8").read()))
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(FL, "kaggle", "feedback_loop_finn.ipynb"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--data", default=None)
    a = ap.parse_args()

    nb = nbf.v4.new_notebook()
    cells = [nbf.v4.new_markdown_cell(INTRO),
             nbf.v4.new_code_cell(SETUP.format(require_gpu=not a.no_gpu).replace(
                 'os.path.join(BASE, "feedback_loop")', 'os.path.join(BASE, "feedback_loop_finn")')),
             nbf.v4.new_markdown_cell("## Код\nСгенерировано `feedback_loop/kaggle/build_finn_notebook.py`.")]
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
