"""
Стенд C — FINN.no Slates: та же модель и веса событий, что на синтетике и Yambda, плюс то, что дают показы:
  ips-true      — пропенсити по реальным показам (частота объявления в рекомендательных слейтах);
  imitator-imp  — имитатор π0, обученный предсказывать верхнее объявление рекомендательного слейта (а не клик);
  agreement     — доля топ-10 модели, попавшая в реально показанный на шаге цели слейт.
«Органика» = поиск (см. оговорки в finn.py).

  python finn_train.py --seeds 0 1 --out results/finn
  python finn_train.py --path data.npz --n-users 3000 --top-items 3000 --max-epochs 2      # быстро
"""
import argparse
import random
import time

import numpy as np
import pandas as pd
import torch

from common import bootstrap_diff, nn_outputs, topk
from finn import build, download, load_raw, make_stub  # noqa: F401 — make_stub для локальной проверки
from fl_train import build_sequences, event_weights, train_variant
from yambda_train import KS, SHOW, eval_all_k, report

VARIANTS = ["all", "organic", "all-matched", "alpha-0.25", "alpha-0.5", "ips-true", "ips-est",
            "imitator", "imitator-imp"]
HEADER = ("«Органика» — клики из поиска, «рекомендованные» — из рекомендательных слейтов. Сплит по шагам: последний "
          "клик — тест, предпоследний — валидация. Ранняя остановка по органическим целям val. Сиды меняют только "
          "инициализацию.\n")
FINN_SHOW = SHOW + ["agreement@10 rec", "agreement@10 organic", "overlap@10 imitator-imp"]


def agreement(recs, split, k=10):
    """Доля топ-k, попавшая в показанный на шаге цели слейт — по типу шага (поиск / рекомендация)."""
    users = split.users
    out, per = {}, {}
    for kind in ("rec", "organic"):
        rows = [(r, u) for r, u in enumerate(users) if split.kind.get(u) == kind and len(split.shown.get(u, [])) > 0]
        if not rows:
            continue
        a = np.array([np.isin(recs[r, :k], split.shown[u]).mean() for r, u in rows])
        out[f"agreement@{k} {kind}"] = a.mean()
        per[kind] = pd.DataFrame({"user": [u for _, u in rows], "agreement": a})
    return out, per


def run_seed(seed, data, args):
    train, val, test, items = data
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n_items = len(items)
    variants = list(args.variants)
    first = [v for v in ("imitator", "imitator-imp") if v in variants]
    variants = first + [v for v in variants if v not in first]

    rows, per, ref = [], {}, {}
    pop_scores = np.tile(items["log_pop"].to_numpy(np.float32), (len(test.users), 1))
    out, per["pop"] = eval_all_k(pop_scores, test, items, {})
    a, _ = agreement(topk(pop_scores, 10), test)
    rows.append({"seed": seed, "variant": "pop", **out, **a})
    del pop_scores

    base_train = train.drop(columns=["target_top"])
    for v in variants:
        t0 = time.time()
        if v == "imitator-imp":                               # цель — верхнее объявление рекомендательного слейта
            tr = train.rename(columns={"target_top": "target"})
            w = (tr["target"].to_numpy() >= 0).astype(float)
        else:
            tr = base_train
            w = event_weights(tr, v, rng)
        seqs = build_sequences(tr, w, args.max_len)
        model, ep = train_variant(seqs, n_items, args, val, items, v)
        scores = nn_outputs(model, test, test.users, args)
        out, per[v] = eval_all_k(scores, test, items, {n: r for n, r in ref.items()})
        recs = topk(scores, max(KS))
        a, per_a = agreement(recs, test)
        for kind, df in per_a.items():
            per[v][f"agreement {kind}"] = df
        if v in ("imitator", "imitator-imp"):
            ref[v] = recs
        rows.append({"seed": seed, "variant": v, "best epoch": ep, "train sec": round(time.time() - t0), **out, **a})
        print(f"  [{v}] " + ", ".join(f"{k} {x:.4f}" for k, x in {**out, **a}.items()
                                     if k.startswith(("Recall@10 ", "agreement"))), flush=True)
        del scores, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    boot = []
    if "all" in per:
        for v in per:
            if v in ("all", "pop", "imitator", "imitator-imp"):
                continue
            for key, col in (("organic@10", "recall"), ("rec@10", "recall"), ("all@10", "recall"),
                             ("organic@100", "recall"), ("overlap imitator@10", "overlap"),
                             ("overlap imitator-imp@10", "overlap"), ("agreement rec", "agreement")):
                a_, b_ = per[v].get(key), per["all"].get(key)
                if a_ is not None and b_ is not None:
                    d, lo, hi = bootstrap_diff(a_, b_, col, seed=seed)
                    boot.append({"seed": seed, "variant": v, "metric": f"{key}:{col}", "diff vs all": d,
                                 "ci_lo": lo, "ci_hi": hi})
    return rows, boot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=None, help="data.npz; без него — скачать")
    ap.add_argument("--stub", action="store_true", help="сгенерировать заглушку вместо настоящих данных")
    ap.add_argument("--n-users", type=int, default=300000)
    ap.add_argument("--top-items", type=int, default=20000)
    ap.add_argument("--n-eval", type=int, nargs=2, default=[20000, 30000], help="пользователей в оценке: val test")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    ap.add_argument("--select", default="organic", choices=["organic", "all"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--max-len", type=int, default=20)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--max-epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="results/finn")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    path = make_stub("finn_stub.npz") if args.stub else (args.path or download())
    click, itype, slate = load_raw(path, args.n_users)
    train, val, test, items, stats = build(click, itype, slate, args.top_items, tuple(args.n_eval))
    del click, itype, slate
    print("данные:", {k: (round(float(v), 4) if isinstance(v, float) else v) for k, v in stats.items()}, flush=True)
    print(f"train {len(train):,} кликов, val users {len(val.users)}, test users {len(test.users)}", flush=True)
    rows, boot = [], []
    for seed in args.seeds:
        print(f"\n=== seed {seed}", flush=True)
        r, b = run_seed(seed, (train, val, test, items), args)
        rows += r
        boot += b
    report(rows, boot, stats, args, title="FINN.no Slates", header=HEADER, show=FINN_SHOW)


if __name__ == "__main__":
    main()
