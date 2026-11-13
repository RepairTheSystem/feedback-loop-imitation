"""
Обучение одной и той же модели (SASRec-подобная base из repeat_switch) с разными весами событий в лоссе
(DESIGN.md §5). На входе модель всегда видит полную историю; веса меняют только то, ЧТО она учится предсказывать.

  all          — все события, вес 1
  organic      — рекомендованные события как цели не учитываются (вес 0)
  all-matched  — все события, но случайная доля целей, равная числу органических (контроль объёма)
  alpha-X      — вес рекомендованных событий X
  ips-true     — вес рекомендованного события ∝ 1 / пропенсити показа (истинная, из π0), клиппинг q95, SNIPS
  ips-est      — то же с эвристической пропенсити: доля товара среди рекомендованных событий (как на Yambda)
  imitator     — только рекомендованные события: приближение π0, для метрики сходства

  python fl_train.py --policy skewed --lam 1.0 --seeds 0 --out results/skewed_lam1
"""
import argparse
import copy
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch

from common import OneStageRec, bootstrap_diff, nn_outputs, topk
from fl_data import evaluate, make_split, online_eval, pi0_recs
from sim import SimConfig, run

VARIANTS = ["all", "organic", "all-matched", "alpha-0.25", "alpha-0.5", "ips-true", "ips-est", "imitator"]


# ----------------------------------------------------------------------------- веса событий


def event_weights(train, variant, rng, clip_q=0.95):
    org = train["organic"].to_numpy(bool)
    w = np.ones(len(train))
    if variant == "all":
        pass
    elif variant == "organic":
        w[~org] = 0.0
    elif variant == "imitator":
        w[org] = 0.0
    elif variant == "all-matched":
        w[:] = 0.0
        w[rng.choice(len(train), int(org.sum()), replace=False)] = 1.0
    elif variant.startswith("alpha-"):
        w[~org] = float(variant.split("-", 1)[1])
    elif variant in ("ips-true", "ips-est"):
        if variant == "ips-true":
            p = train.loc[~org, "prop"].to_numpy()
        else:                                            # без показов: популярность среди рекомендованных
            items = train.loc[~org, "item"].to_numpy()
            p = np.bincount(items)[items] / len(items)
        raw = 1.0 / np.maximum(p, 1e-12)
        raw = np.minimum(raw, np.quantile(raw, clip_q))  # клиппинг хвоста
        w[~org] = raw / raw.mean()                       # SNIPS: средний вес рекомендованных = 1
    else:
        raise ValueError(variant)
    return w


# ----------------------------------------------------------------------------- обучение


def build_sequences(train, w, max_len):
    """Окна по max_len+1 покупок с конца истории + веса целей; окна без целей (все веса 0) выкидываются."""
    seqs = []
    items, ts = train["item"].to_numpy() + 1, train["ts"].to_numpy()
    # колонка target (если есть) — что предсказывать в этой позиции вместо самого события; −1 → нет цели
    tgt = train["target"].to_numpy() + 1 if "target" in train else None
    for _, idx in train.groupby("user").indices.items():
        end = len(idx)
        while end >= 2:
            start = max(0, end - (max_len + 1))
            sl = idx[start:end]
            if w[sl[1:]].sum() > 0:
                seqs.append((items[sl], ts[sl], w[sl]) if tgt is None else (items[sl], ts[sl], w[sl], tgt[sl]))
            if start == 0:
                break
            end = start + 1
    return seqs


def pad(batch, length):
    """
    Элемент батча — (items, times, weights) или (items, times, weights, targets).
    targets[j] — что модель должна предсказать в позиции j по истории до j (по умолчанию — сам items[j]);
    так имитатор учится предсказывать показанное π0, а не кликнутое (FINN.no).
    """
    B = len(batch)
    it = np.zeros((B, length), np.int64)
    tg = np.zeros((B, length), np.int64)
    tm = np.zeros((B, length), np.float32)
    ww = np.zeros((B, length), np.float32)
    for b, seq in enumerate(batch):
        i, t, w = seq[:3]
        n = len(i)
        it[b, :n], tm[b, :n], tm[b, n:], ww[b, :n] = i, t, t[-1], w
        tg[b, :n] = seq[3] if len(seq) > 3 else i
    return torch.from_numpy(it), torch.from_numpy(tm), torch.from_numpy(ww), torch.from_numpy(tg)


def val_score(model, val, items, args):
    """Метрика ранней остановки: Recall@k на органических (по умолчанию) целях val."""
    kind = "organic" if args.select == "organic" else "all"
    users = np.array(sorted(val.targets[kind]))
    from common import per_user
    recs = topk(nn_outputs(model, val, users, args), args.k)
    return per_user(recs, users, val.target_view(kind), items, args.k).recall.mean()


def train_variant(seqs, n_items, args, val, items, tag):
    model = OneStageRec(n_items, mode="base", d=args.dim, n_layers=args.layers, max_len=args.max_len).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best, best_state, best_epoch, bad = -1.0, None, 0, 0
    for epoch in range(args.max_epochs):
        model.train()
        random.shuffle(seqs)
        tot, cnt = 0.0, 0.0
        for s in range(0, len(seqs), args.batch):
            it, tm, ww, tg = pad(seqs[s:s + args.batch], args.max_len + 1)
            it, tm, ww, tg = it.to(args.device), tm.to(args.device), ww.to(args.device), tg.to(args.device)
            logp, *_ = model(it[:, :-1], tm[:, :-1], tm[:, 1:])
            lp = logp.gather(-1, tg[:, 1:].unsqueeze(-1)).squeeze(-1)
            W = ww[:, 1:] * (tg[:, 1:] != 0)
            loss = -(lp * W).sum() / W.sum().clamp_min(1e-9)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * W.sum().item()
            cnt += W.sum().item()
        score = val_score(model, val, items, args)
        print(f"    [{tag}] epoch {epoch + 1}  weighted NLL {tot / max(cnt, 1e-9):.4f}  val Recall {score:.4f}", flush=True)
        if score > best + 1e-4:
            best, best_state, best_epoch, bad = score, copy.deepcopy(model.state_dict()), epoch + 1, 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    if best_epoch == args.max_epochs:
        print(f"    [{tag}] ВНИМАНИЕ: лучшая эпоха = лимит {args.max_epochs}")
    model.load_state_dict(best_state)
    return model, best_epoch


# ----------------------------------------------------------------------------- прогон


def run_one(cfg, args):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    t0 = time.time()
    res = run(cfg, capture_at=cfg.snapshot_times[1] if args.online else None)
    items = res.world.items_frame()
    ev = res.events
    print(f"\n=== policy={cfg.policy} λ={cfg.lam} seed={cfg.seed}: events {len(ev)}, "
          f"rec share {(~ev.organic).mean():.3f}, offset {res.cfg.offset:.2f}, sim {time.time() - t0:.0f}s", flush=True)
    t_val, t_test = cfg.snapshot_times
    val, test = make_split(res, t_val, args.window), make_split(res, t_test, args.window)
    train = ev[ev.ts < t_val].reset_index(drop=True)
    n_items = len(items)

    variants = list(args.variants)
    if "imitator" in variants:                            # имитатор нужен первым — для overlap
        variants = ["imitator"] + [v for v in variants if v != "imitator"]
    rng = np.random.default_rng(cfg.seed)
    ref = {"pi0": pi0_recs(test, args.k)}
    rows, per, epochs = [], {}, {}
    for v in variants:
        seqs = build_sequences(train, event_weights(train, v, rng), args.max_len)
        model, epochs[v] = train_variant(seqs, n_items, args, val, items, v)
        scores = nn_outputs(model, test, test.users, args)
        out, per[v], recs = evaluate(scores, test, items, args.k, ref)
        if v == "imitator":
            ref["imitator"] = recs
        if args.online:
            on, per[v]["online"] = online_eval(res, test, scores, args.window)
            out.update(on)
        rows.append({"policy": cfg.policy, "lam": cfg.lam, "seed": cfg.seed, "variant": v,
                     "best epoch": epochs[v], **out})

    # эталоны: π0 (офлайн — топ по её скору; онлайн — настоящая стохастическая π0) и оракул по истинной полезности
    for name, sc in (("pi0", test.pi0[test.users]), ("oracle", test.util[test.users])):
        out, per[name], _ = evaluate(sc, test, items, args.k, {"imitator": ref.get("imitator", ref["pi0"])})
        if args.online:
            on, per[name]["online"] = online_eval(res, test, None if name == "pi0" else sc, args.window)
            out.update(on)
        rows.append({"policy": cfg.policy, "lam": cfg.lam, "seed": cfg.seed, "variant": name, **out})

    boot = []
    if "all" in per:
        for v in per:
            if v in ("all", "pi0", "oracle", "imitator"):
                continue
            for kind, col in (("organic", "recall"), ("truth", "true_ndcg"), ("overlap pi0", "overlap"),
                              ("online", "util_all"), ("online", "util_rec")):
                if kind in per[v] and kind in per["all"]:
                    d, lo, hi = bootstrap_diff(per[v][kind], per["all"][kind], col, seed=cfg.seed)
                    boot.append({"policy": cfg.policy, "lam": cfg.lam, "seed": cfg.seed, "variant": v,
                                 "metric": f"{kind}:{col}", "diff vs all": d, "ci_lo": lo, "ci_hi": hi})
    return rows, boot


def to_md(df, **kw):
    try:
        return df.to_markdown(**kw)
    except ImportError:
        return "```\n" + df.to_string() + "\n```"


SHOW = ["Recall@10 organic", "Recall@10 rec", "true NDCG@10", "true util@10", "online util/user",
        "online rec util/user", "online rec buys/user", "overlap@10 pi0", "overlap@10 imitator", "mean log-pop",
        "promo share", "org Recall repeat", "org Recall explore"]

# кривая по весу рекомендованных событий: вариант → α
ALPHA = {"organic": 0.0, "alpha-0.25": 0.25, "alpha-0.5": 0.5, "all": 1.0}
CURVE_METRICS = ["true NDCG@10", "online util/user", "Recall@10 organic", "promo share", "overlap@10 imitator"]


def report(rows, boot, args):
    os.makedirs(args.out, exist_ok=True)
    res, bt = pd.DataFrame(rows), pd.DataFrame(boot)
    res.to_csv(os.path.join(args.out, "metrics.csv"), index=False)
    bt.to_csv(os.path.join(args.out, "bootstrap.csv"), index=False)
    with open(os.path.join(args.out, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=1)
    cols = [c for c in SHOW if c in res]
    g = res.groupby(["policy", "lam", "variant"], sort=False)[cols].mean()
    lines = [f"# Петля обратной связи: синтетика, seeds={args.seeds}\n",
             "Тест; среднее по сидам. Ранняя остановка по "
             f"{'органическим' if args.select == 'organic' else 'всем'} целям val.\n", to_md(g, floatfmt=".3f")]
    curve = res[res.variant.isin(list(ALPHA))].assign(alpha=lambda d: d.variant.map(ALPHA))
    if len(curve):
        lines.append("\n\n## Кривые по α (вес рекомендованных событий; α = 0 — organic, α = 1 — all)\n")
        for m in [c for c in CURVE_METRICS if c in curve]:
            pv = curve.pivot_table(index=["policy", "lam"], columns="alpha", values=m, aggfunc="mean")
            lines += [f"\n### {m}\n", to_md(pv, floatfmt=".3f")]
        curve.to_csv(os.path.join(args.out, "alpha_curves.csv"), index=False)
    if len(bt):
        b = bt.groupby(["policy", "lam", "variant", "metric"], sort=False).agg(
            diff=("diff vs all", "mean"), ci_lo=("ci_lo", "mean"), ci_hi=("ci_hi", "mean"))
        lines += ["\n\n## Разница с `all` (парный бутстрап, 95% CI; среднее по сидам)\n", to_md(b, floatfmt=".4f")]
    text = "\n".join(lines)
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print("\n" + text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", nargs="+", default=["good"], choices=["pop", "good", "skewed"])
    ap.add_argument("--lam", type=float, nargs="+", default=[0.5])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--n-users", type=int, default=4000)
    ap.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    ap.add_argument("--select", default="organic", choices=["organic", "all"], help="по каким целям val ранняя остановка")
    ap.add_argument("--window", type=float, default=7.0)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--max-len", type=int, default=50)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--max-epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="results/main")
    ap.add_argument("--no-online", dest="online", action="store_false", help="без симулированного онлайна")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    rows, boot = [], []
    for pol in args.policy:
        for lam in args.lam:
            for seed in args.seeds:
                cfg = SimConfig(n_users=args.n_users, policy=pol, lam=lam, seed=seed)
                r, b = run_one(cfg, args)
                rows += r
                boot += b
    report(rows, boot, args)


if __name__ == "__main__":
    main()
