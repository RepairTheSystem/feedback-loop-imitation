"""
Проверки на вменяемость симулятора и методики (DESIGN.md §4.5) — до любых выводов.

  python sanity.py --part data              # быстро: свойства логов по λ и π0 + юнит-тест пропенсити
  python sanity.py --part models            # обучение моделей на двух конфигурациях (~20 мин на CPU)

Критерии — заранее зафиксированные ожидания; FAIL означает ошибку в симуляторе или методике.
"""
import argparse
import os

import numpy as np
import pandas as pd

from sim import SimConfig, _softmax, run


def check(name, ok, detail):
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}: {detail}", flush=True)
    return {"check": name, "result": mark, "detail": detail}


# ----------------------------------------------------------------------------- данные


def propensity_unit_test(n=50, K=10, trials=40000, seed=0):
    """Частота попадания в слейт при K сэмплах с возвращением и удалением дублей = 1 − (1 − p)^K."""
    rng = np.random.default_rng(seed)
    p = _softmax(rng.normal(0, 1.5, n))
    hits = np.zeros(n)
    for _ in range(trials):
        hits[np.unique(rng.choice(n, K, p=p))] += 1
    emp, theo = hits / trials, 1 - (1 - p) ** K
    return np.abs(emp - theo).max(), np.sqrt(theo * (1 - theo) / trials).max()


def data_part(n_users):
    rows, checks = [], []
    for pol in ("good", "pop", "skewed"):
        for lam in (0.0, 0.5, 1.0):
            r = run(SimConfig(n_users=n_users, policy=pol, lam=lam))
            ev, im = r.events, r.impressions
            rec, org = ev[~ev.organic], ev[ev.organic]
            promo = r.world.promo
            rows.append({"policy": pol, "lam": lam, "offset": r.cfg.offset, "events/user": len(ev) / n_users,
                         "rec share": len(rec) / len(ev), "util organic": org.util.mean(), "util rec": rec.util.mean(),
                         "promo share rec": promo[rec.item].mean(),
                         "promo share organic": promo[org.item].mean(), "prop median rec": rec.prop.median(),
                         "uniq per slate": len(im) / im.groupby(["user", "ts"]).ngroups})
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)

    for pol in ("good", "pop", "skewed"):
        u = df[df.policy == pol].sort_values("lam")["util rec"].to_numpy()
        checks.append(check(f"{pol}: полезность рекомендованных покупок падает с λ", bool(np.all(np.diff(u) < 0)),
                            " → ".join(f"{x:.3f}" for x in u)))
    rs = df["rec share"]
    checks.append(check("калибровка: доля рекомендованных 0.5 ± 0.03 везде", bool(((rs - 0.5).abs() < 0.03).all()),
                        f"{rs.min():.3f}…{rs.max():.3f}"))
    sk = df[df.policy == "skewed"].sort_values("lam")["promo share rec"].to_numpy()
    checks.append(check("skewed: доля promo в рекомендованных покупках растёт с λ", bool(np.all(np.diff(sk) > 0)),
                        " → ".join(f"{x:.3f}" for x in sk) + " (база 0.10)"))
    err, se = propensity_unit_test()
    checks.append(check("пропенсити 1 − (1 − p)^K совпадает с частотой показа", err < 5 * se,
                        f"max |ошибка| {err:.4f} при SE {se:.4f}"))
    return df, checks


# ----------------------------------------------------------------------------- модели


def models_part(n_users, max_epochs, device):
    from fl_train import run_one
    import argparse as ap_
    args = ap_.Namespace(variants=["imitator", "all", "organic", "all-matched", "ips-true"], select="organic",
                         window=7.0, k=10, max_len=50, dim=64, layers=2, max_epochs=max_epochs, patience=3,
                         batch=128, lr=1e-3, device=device)
    rows, boot = [], []
    for pol, lam in (("good", 0.0), ("skewed", 1.0)):
        r, b = run_one(SimConfig(n_users=n_users, policy=pol, lam=lam, seed=0), args)
        rows += r
        boot += b
    df, bt = pd.DataFrame(rows), pd.DataFrame(boot)
    m = df.set_index(["policy", "variant"])

    def ci(pol, var, metric):
        x = bt[(bt.policy == pol) & (bt.variant == var) & (bt.metric == metric)].iloc[0]
        return x["diff vs all"], x.ci_lo, x.ci_hi

    checks = []
    g = m.loc["good"]
    d, lo, hi = ci("good", "all-matched", "truth:true_ndcg")
    d2, lo2, hi2 = ci("good", "organic", "truth:true_ndcg")
    checks.append(check("good, λ=0: при равном объёме рекомендованные события не хуже органики на правде",
                        g.loc["all-matched", "true NDCG@10"] >= g.loc["organic", "true NDCG@10"] - 0.01,
                        f"true NDCG all-matched {g.loc['all-matched', 'true NDCG@10']:.3f} vs organic "
                        f"{g.loc['organic', 'true NDCG@10']:.3f}"))
    s = m.loc["skewed"]
    checks.append(check("skewed, λ=1: all имитирует π0 — promo share заметно выше, чем у organic",
                        s.loc["all", "promo share"] > s.loc["organic", "promo share"] + 0.05,
                        f"all {s.loc['all', 'promo share']:.3f} vs organic {s.loc['organic', 'promo share']:.3f} (база 0.10)"))
    d, lo, hi = ci("skewed", "organic", "truth:true_ndcg")
    checks.append(check("skewed, λ=1: organic лучше all на правде (CI > 0)", lo > 0,
                        f"Δ true NDCG {d:+.4f} [{lo:+.4f}, {hi:+.4f}]"))
    d, lo, hi = ci("skewed", "ips-true", "truth:true_ndcg")
    checks.append(check("skewed, λ=1: IPS с истинными пропенсити лучше all на правде (CI > 0)", lo > 0,
                        f"Δ true NDCG {d:+.4f} [{lo:+.4f}, {hi:+.4f}]"))
    for pol in ("good", "skewed"):
        x = m.loc[pol]
        checks.append(check(f"{pol}: имитатор ближе к π0, чем organic (overlap@10 с π0)",
                            x.loc["imitator", "overlap@10 pi0"] > x.loc["organic", "overlap@10 pi0"],
                            f"imitator {x.loc['imitator', 'overlap@10 pi0']:.3f} vs organic {x.loc['organic', 'overlap@10 pi0']:.3f}"))
    return df, bt, checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["data", "models"], default="data")
    ap.add_argument("--n-users", type=int, default=None)
    ap.add_argument("--max-epochs", type=int, default=30)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/sanity")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.part == "data":
        df, checks = data_part(a.n_users or 600)
        df.to_csv(os.path.join(a.out, "data.csv"), index=False)
    else:
        df, bt, checks = models_part(a.n_users or 2000, a.max_epochs, a.device)
        df.to_csv(os.path.join(a.out, "models.csv"), index=False)
        bt.to_csv(os.path.join(a.out, "models_bootstrap.csv"), index=False)
    pd.DataFrame(checks).to_csv(os.path.join(a.out, f"checks_{a.part}.csv"), index=False)
    n_fail = sum(c["result"] == "FAIL" for c in checks)
    print(f"\nитого: {len(checks) - n_fail} PASS, {n_fail} FAIL")


if __name__ == "__main__":
    main()
