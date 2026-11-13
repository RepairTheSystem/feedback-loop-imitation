"""
Стенд B — Yambda-50M: та же модель и те же веса событий, что на синтетике (fl_train.py), без `ips-true`
(показов в Yambda нет). Оценка на органических / рекомендованных / всех целях тестового дня, @10 и @100.

  python yambda_train.py --path listens.parquet --seeds 0 1 2 --out results/yambda
  python yambda_train.py --path listens.parquet --n-users 1000 --top-items 5000 --max-epochs 2   # быстро

Разрыв имитации (DESIGN.md §6): gap = Recall_rec − Recall_org; нормированный — gap / gap популярности.
"""
import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch

from common import bootstrap_diff, nn_outputs, topk
from fl_data import evaluate
from fl_train import build_sequences, event_weights, to_md, train_variant
from yambda import build, download_listens, load_positives

VARIANTS = ["all", "organic", "all-matched", "alpha-0.25", "alpha-0.5", "ips-est", "imitator"]
KS = (10, 100)


def eval_all_k(scores, split, items, ref):
    """Метрики при k = 10 и 100; у каждой метрики суффикс @k (если его ещё нет)."""
    out, per = {}, {}
    for k in [k for k in KS if k < scores.shape[1]]:
        o, p, _ = evaluate(scores, split, items, k, {n: r[:, :k] for n, r in ref.items()})
        out.update({(key if f"@{k}" in key else f"{key}@{k}"): val for key, val in o.items()})
        for kind in ("organic", "rec", "all"):
            if kind in p:
                per[f"{kind}@{k}"] = p[kind]
        for name in ref:
            per[f"overlap {name}@{k}"] = p[f"overlap {name}"]
    return out, per


def run_seed(seed, data, args):
    train, val, test, items = data
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    n_items = len(items)
    rng = np.random.default_rng(seed)
    variants = list(args.variants)
    if "imitator" in variants:
        variants = ["imitator"] + [v for v in variants if v != "imitator"]

    rows, per, ref = [], {}, {}
    # эталон без обучения: популярность в обучении
    pop_scores = np.tile(items["log_pop"].to_numpy(np.float32), (len(test.users), 1))
    out, per["pop"] = eval_all_k(pop_scores, test, items, {})
    rows.append({"seed": seed, "variant": "pop", **out})
    del pop_scores

    for v in variants:
        t0 = time.time()
        seqs = build_sequences(train, event_weights(train, v, rng), args.max_len)
        model, ep = train_variant(seqs, n_items, args, val, items, v)
        scores = nn_outputs(model, test, test.users, args)
        out, per[v] = eval_all_k(scores, test, items, ref)
        if v == "imitator":
            ref["imitator"] = topk(scores, max(KS))
        rows.append({"seed": seed, "variant": v, "best epoch": ep, "train sec": round(time.time() - t0), **out})
        print(f"  [{v}] " + ", ".join(f"{k} {x:.4f}" for k, x in out.items() if k.startswith("Recall@10 ")), flush=True)
        del scores, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    boot = []
    if "all" in per:
        for v in per:
            if v in ("all", "pop", "imitator"):
                continue
            for key, col in (("organic@10", "recall"), ("organic@100", "recall"), ("rec@10", "recall"),
                             ("all@10", "recall"), ("overlap imitator@10", "overlap")):
                a, b = per[v].get(key), per["all"].get(key)
                if a is not None and b is not None:
                    d, lo, hi = bootstrap_diff(a, b, col, seed=seed)
                    boot.append({"seed": seed, "variant": v, "metric": f"{key}:{col}", "diff vs all": d,
                                 "ci_lo": lo, "ci_hi": hi})
    return rows, boot


SHOW = ["Recall@10 organic", "Recall@10 rec", "Recall@10 all", "NDCG@10 organic", "Recall@100 organic",
        "Recall@100 rec", "org Recall repeat@10", "org Recall explore@10", "overlap@10 imitator", "mean log-pop@10",
        "coverage@10"]
ALPHA = {"organic": 0.0, "alpha-0.25": 0.25, "alpha-0.5": 0.5, "all": 1.0}


YAMBDA_HEADER = ("Позитив — прослушано > 50%. Тест — последний день, зазор 30 мин; ранняя остановка по органическим "
                 "целям val-дня. Среднее по сидам (сиды меняют только инициализацию — данные одни).\n")


def report(rows, boot, stats, args, title="Yambda-50M", header=YAMBDA_HEADER, show=None):
    show = SHOW if show is None else show
    os.makedirs(args.out, exist_ok=True)
    res, bt = pd.DataFrame(rows), pd.DataFrame(boot)
    res.to_csv(os.path.join(args.out, "metrics.csv"), index=False)
    bt.to_csv(os.path.join(args.out, "bootstrap.csv"), index=False)
    with open(os.path.join(args.out, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=1)
    with open(os.path.join(args.out, "stats.json"), "w") as f:
        json.dump({k: float(v) for k, v in stats.items()}, f, indent=1)

    g = res.groupby("variant", sort=False)
    cols = [c for c in show if c in res]
    mean = g[cols].mean()
    gap = mean["Recall@10 rec"] - mean["Recall@10 organic"]
    mean["imitation gap@10"] = gap
    if "pop" in mean.index and abs(gap["pop"]) > 1e-9:
        mean["gap / gap(pop)"] = gap / gap["pop"]
    lines = [f"# {title}: петля обратной связи, seeds={args.seeds}\n", header,
             "## Данные\n", to_md(pd.Series(stats, name="value").to_frame(), floatfmt=".4f"),
             "\n\n## Метрики\n", to_md(mean, floatfmt=".4f")]
    curve = res[res.variant.isin(list(ALPHA))].assign(alpha=lambda d: d.variant.map(ALPHA))
    if len(curve):
        pv = curve.groupby("alpha")[[c for c in ("Recall@10 organic", "Recall@10 rec", "overlap@10 imitator",
                                                 "mean log-pop@10") if c in curve]].mean()
        lines += ["\n\n## Кривая по α (вес рекомендованных событий)\n", to_md(pv, floatfmt=".4f")]
    if len(bt):
        b = bt.groupby(["variant", "metric"], sort=False).agg(
            diff=("diff vs all", "mean"), ci_lo=("ci_lo", "mean"), ci_hi=("ci_hi", "mean"),
            **{"CI>0": ("ci_lo", lambda x: f"{(x > 0).sum()}/{len(x)}"),
               "CI<0": ("ci_hi", lambda x: f"{(x < 0).sum()}/{len(x)}")})
        lines += ["\n\n## Разница с `all` (парный бутстрап по пользователям)\n", to_md(b, floatfmt=".4f")]
    text = "\n".join(lines)
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print("\n" + text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=None, help="listens.parquet; без него — скачать с Hugging Face")
    ap.add_argument("--n-users", type=int, default=None, help="подвыборка пользователей (для быстрой проверки)")
    ap.add_argument("--top-items", type=int, default=20000)
    ap.add_argument("--train-tail", type=int, default=500, help="последних позитивов на пользователя в обучении")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    ap.add_argument("--select", default="organic", choices=["organic", "all"])
    ap.add_argument("--k", type=int, default=10, help="k для ранней остановки")
    ap.add_argument("--max-len", type=int, default=50)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--max-epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="results/yambda")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    path = args.path or download_listens()
    pos = load_positives(path, n_users=args.n_users)
    train, val, test, items, stats = build(pos, args.top_items, args.train_tail)
    del pos
    print("данные:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in stats.items()}, flush=True)
    print(f"train {len(train):,} событий, val users {len(val.targets['organic'])}, test users {len(test.users)}",
          flush=True)
    rows, boot = [], []
    for seed in args.seeds:
        print(f"\n=== seed {seed}", flush=True)
        r, b = run_seed(seed, (train, val, test, items), args)
        rows += r
        boot += b
    report(rows, boot, stats, args)


if __name__ == "__main__":
    main()
