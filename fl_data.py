"""
Сплит и метрики для эксперимента про петлю обратной связи (DESIGN.md §5–§7).

Запрос — в начале окна t0; история — все события (органические и рекомендованные) до t0:
на инференсе модель видит полную историю независимо от того, как её учили.
Цели в окне [t0, t0 + window) — три набора: organic, rec, all.
Правда — снимок истинной полезности в момент t0 (есть у каждого пользователя, даже без целей).
"""
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd

from common import per_user, topk


@dataclass
class FLSplit:
    hist: dict          # user -> (items, times)
    targets: dict       # "organic" / "rec" / "all" -> {user -> np.array товаров}
    t_query: dict       # user -> t0
    t0: float
    util: np.ndarray    # (n_users, N) истинная полезность в момент t0
    pi0: np.ndarray     # (n_users, N) скоры π0 в момент t0
    events: pd.DataFrame

    @property
    def users(self):
        """Все пользователи с историей — на них считаются метрики «на правде»."""
        return np.array(sorted(self.hist))

    def target_view(self, kind):
        """Объект в формате repeat_switch (hist, targets) для per_user."""
        return SimpleNamespace(hist=self.hist, targets=self.targets[kind])


def make_split(res, t0, window=7.0):
    ev = res.events
    before = ev[ev.ts < t0]
    hist = {u: (g["item"].to_numpy(), g["ts"].to_numpy()) for u, g in before.groupby("user")}
    w = ev[(ev.ts >= t0) & (ev.ts < t0 + window) & ev.user.isin(list(hist))]
    targets = {}
    for kind, part in (("organic", w[w.organic]), ("rec", w[~w.organic]), ("all", w)):
        targets[kind] = {u: np.unique(g["item"].to_numpy()) for u, g in part.groupby("user")}
    snap = res.snapshots[t0]
    return FLSplit(hist, targets, {u: t0 for u in hist}, t0, snap["util"], snap["pi0"],
                   before.reset_index(drop=True))


# ----------------------------------------------------------------------------- метрики


def truth_per_user(recs, users, util, k=10):
    """
    Метрики на истинной полезности (только синтетика):
      true_ndcg — DCG полезностей выдачи / DCG идеальной выдачи по полезности;
      true_util — средняя полезность товаров в выдаче.
    """
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    u = util[users]
    got = np.take_along_axis(u, recs[:, :k], 1)
    ideal = -np.sort(-u, axis=1)[:, :k]
    return pd.DataFrame({"user": users,
                         "true_ndcg": (got * disc).sum(1) / np.maximum((ideal * disc).sum(1), 1e-12),
                         "true_util": got.mean(1)})


def overlap(recs_a, recs_b, k=10):
    """Доля общих товаров в топ-k двух выдач, по пользователям."""
    a, b = recs_a[:, :k], recs_b[:, :k]
    return np.array([len(np.intersect1d(x, y)) / k for x, y in zip(a, b)])


def evaluate(scores, split, items, k=10, ref_recs=None):
    """
    scores: (len(split.users), N) — скоры модели для всех пользователей с историей.
    ref_recs: {имя: (U, k)} — выдачи для сравнения (π0, имитатор) → overlap@k.
    Возвращает (сводка, по-пользовательские таблицы для бутстрапа).
    """
    users = split.users
    recs = topk(scores, k)
    row = {u: r for r, u in enumerate(users)}
    out, per = {}, {}

    for kind in ("organic", "rec", "all"):
        tu = np.array(sorted(split.targets[kind]))
        if len(tu) == 0:
            continue
        df = per_user(recs[[row[u] for u in tu]], tu, split.target_view(kind), items, k)
        per[kind] = df
        out[f"Recall@{k} {kind}"] = df.recall.mean()
        out[f"NDCG@{k} {kind}"] = df.ndcg.mean()
        if kind == "organic":
            rep, exp_ = df.n_rep_t.sum(), (df.n_t - df.n_rep_t).sum()
            out["org Recall repeat"] = df.hits_rep.sum() / max(rep, 1)
            out["org Recall explore"] = df.hits_exp.sum() / max(exp_, 1)

    if split.util is not None:                       # правда есть только в синтетике
        tr = truth_per_user(recs, users, split.util, k)
        per["truth"] = tr
        out[f"true NDCG@{k}"] = tr.true_ndcg.mean()
        out[f"true util@{k}"] = tr.true_util.mean()

    for name, ref in (ref_recs or {}).items():
        ov = overlap(recs, ref, k)
        per[f"overlap {name}"] = pd.DataFrame({"user": users, "overlap": ov})
        out[f"overlap@{k} {name}"] = ov.mean()

    log_pop = items["log_pop"].to_numpy()
    out["mean log-pop"] = log_pop[recs[:, :k]].mean()
    out["coverage"] = len(np.unique(recs[:, :k])) / len(items)
    if "promo" in items:
        out["promo share"] = items["promo"].to_numpy()[recs[:, :k]].mean()
    return out, per, recs


def online_eval(res, split, scores=None, window=7.0):
    """
    Симулированный онлайн (DESIGN.md §4.4): в окне [t0, t0 + window) π0 заменяется политикой «топ-K по scores»
    (скоры заморожены на момент t0); scores=None — оставить π0. Пользователи и их ГСЧ — те же, что в логе.
    Возвращает (сводка, по-пользовательская таблица) по всем пользователям с историей.
    """
    from sim import simulate_window
    users = split.users
    full = None
    if scores is not None:
        full = split.pi0.copy()                      # у пользователей без истории — π0
        full[users] = scores
    ev = simulate_window(res, window, full)
    ev = ev[ev.user.isin(users)]
    g = ev.groupby("user")
    rec = ev[~ev.organic].groupby("user")
    per = pd.DataFrame({"user": users})
    per["n_all"] = per.user.map(g.size()).fillna(0)
    per["n_rec"] = per.user.map(rec.size()).fillna(0)
    per["util_rec"] = per.user.map(rec["util"].sum()).fillna(0)
    per["util_all"] = per.user.map(g["util"].sum()).fillna(0)
    out = {"online rec buys/user": per.n_rec.mean(), "online rec util/user": per.util_rec.mean(),
           "online util/user": per.util_all.mean(), "online buys/user": per.n_all.mean()}
    return out, per


def pi0_recs(split, k=10):
    """Детерминированная выдача π0: топ-k по её скору (π0 сэмплирует, но топ — её «намерение»)."""
    return topk(split.pi0[split.users], k)
