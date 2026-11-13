"""
Симулятор маркетплейса с петлей обратной связи — стенд A (DESIGN.md §4).

Мир (категории, товары, персоны, расходники / длительного пользования) устроен как в repeat_switch/data.py.
Поверх органического процесса старая система π0 при каждом визите показывает слейт, и пользователь покупает
показанное с вероятностью, зависящей от истинной полезности и силы петли λ.

Каждый пользователь — дискретно-событийная симуляция (куча событий по времени):
  cons  — органическая покупка расходника категории c (личный интервал, любимый товар);
  dur   — органическая покупка товара длительного пользования (если ещё не куплен);
  visit — визит: π0 показывает слейт, пользователь, возможно, что-то покупает из него;
  snap  — снимок истинной полезности и скоров π0 (начала val/test-окон) для оценки «на правде».
Покупка из рекомендации меняет состояние так же, как органическая: купленный холодильник больше не нужен,
купленный расходник сдвигает следующую органическую покупку этой категории.

Истинная полезность u(user, item, t) ≥ 0:
  длительного пользования — аффинность к категории × качество; купленный товар — 0;
                             уже есть что-то в категории — × 0.3 (потребность закрыта);
  расходник, любимый       — аффинность × «созрел ли повтор»: exp(−|log(возраст / интервал)|);
  расходник, не любимый    — аффинность × качество × 0.3 (заменитель);
  нелюбимые категории      — аффинность 0.05.
Полезность в [0, 1] (у большей части каталога ≈ 0: нелюбимые категории), масштаб задают κ (π0) и β (отклик).

π0 (политика логирования), слейт = K независимых сэмплов из softmax(score / τ), без повторов:
  pop     — популярность, одна для всех;
  good    — κ·u + стойкая личная ошибка N(0, policy_noise) (π0 хорошая, но не идеальная);
  skewed  — 0.5·κ·u + promo_boost·[товар продвигается] + ошибка (продвигает сегмент независимо от пользы).
Вероятность попасть в слейт (пропенсити) известна точно: 1 − (1 − p)^K.

Отклик на показ: P(купить) = σ(β·(1 − λ)·u + offset), уже купленный товар длительного пользования — 0.
  λ = 0 — решение определяет полезность; λ = 1 — только сам показ (покупают что показали, вслепую).
  offset калибруется так, чтобы доля рекомендованных покупок была ≈ rec_share при любом λ —
  тогда λ меняет, ЧТО покупают из рекомендаций, а не СКОЛЬКО.
"""
import copy
import heapq
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

SNAP, VISIT, CONS, DUR = 0, 1, 2, 3        # порядок при равном времени: снимок раньше событий


@dataclass
class SimConfig:
    n_users: int = 4000
    n_items: int = 1500
    n_cats: int = 40
    n_personas: int = 8
    consumable_share: float = 0.4
    durable_rate: float = 0.5
    days: float = 365.0
    brand_switch: float = 0.03
    visit_rate: float = 0.25          # визитов в день (≈ раз в 4 дня)
    slate: int = 10                   # K — число сэмплов в слейт
    policy: str = "good"              # pop | good | skewed
    tau: float = 1.0                  # температура π0
    policy_noise: float = 1.0
    promo_share: float = 0.1
    promo_boost: float = 2.0
    lam: float = 0.5                  # сила петли λ ∈ [0, 1]
    beta: float = 6.0                 # чувствительность отклика к полезности
    kappa: float = 5.0                # масштаб полезности в скоре π0
    rec_share: float = 0.5            # целевая доля рекомендованных покупок
    offset: float = None              # None — откалибровать
    snapshot_times: tuple = (351.0, 358.0)
    seed: int = 0


@dataclass
class SimResult:
    cfg: SimConfig
    world: "World"
    events: pd.DataFrame              # user, item, ts, organic, prop, util
    impressions: pd.DataFrame         # user, ts, item, prop, bought
    snapshots: dict = field(default_factory=dict)   # t → {"util": (U, N), "pi0": (U, N)}


class World:
    """Статика мира — общая для всех пользователей (как в repeat_switch/data.make_synthetic)."""

    def __init__(self, cfg):
        rng = np.random.default_rng(cfg.seed)
        self.cfg = cfg
        self.cat_is_cons = rng.random(cfg.n_cats) < cfg.consumable_share
        self.cat_interval = rng.uniform(7, 45, cfg.n_cats)
        self.item_cat = rng.integers(0, cfg.n_cats, cfg.n_items)
        self.item_pop = np.minimum(rng.zipf(1.6, cfg.n_items), 1000).astype(float)
        self.items_by_cat = [np.where(self.item_cat == c)[0] for c in range(cfg.n_cats)]
        self.persona_w = rng.dirichlet(np.full(cfg.n_cats, 0.3), cfg.n_personas)
        self.quality = np.log1p(self.item_pop) / np.log1p(self.item_pop).max()
        self.is_cons_item = self.cat_is_cons[self.item_cat]
        self.promo = rng.random(cfg.n_items) < cfg.promo_share
        self.log_pop = np.log(self.item_pop)

    def items_frame(self):
        return pd.DataFrame({"item": np.arange(self.cfg.n_items), "cat": self.item_cat,
                             "is_consumable": self.is_cons_item, "promo": self.promo,
                             "log_pop": self.log_pop})


def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


class _User:
    """Состояние одного пользователя и его истинная полезность."""

    def __init__(self, w, u, rng):
        cfg = w.cfg
        self.w, self.rng = w, rng
        persona = rng.integers(cfg.n_personas)
        n_pref = rng.integers(6, 14)
        self.pref = rng.choice(cfg.n_cats, n_pref, replace=False, p=w.persona_w[persona])
        aff = np.full(cfg.n_cats, 0.05)
        aff[self.pref] = rng.uniform(0.5, 1.0, n_pref)
        self.aff_item = aff[w.item_cat]
        self.base = self.aff_item * w.quality
        pref_item = np.isin(w.item_cat, self.pref)
        self.sub_mask = w.is_cons_item & pref_item          # расходники любимых категорий → заменители
        self.owned = np.zeros(cfg.n_items, dtype=bool)
        self.owned_cat = np.zeros(cfg.n_cats, dtype=bool)
        self.last = np.full(cfg.n_items, np.nan)
        self.fav, self.interval, self.version = {}, {}, {}
        self.durables = []                                   # (t, item) — органический план
        for c in self.pref:
            cand = w.items_by_cat[c]
            if len(cand) == 0:
                continue
            p = w.item_pop[cand] / w.item_pop[cand].sum()
            if w.cat_is_cons[c]:
                self.fav[c] = list(rng.choice(cand, size=min(2, len(cand)), replace=False, p=p))
                self.interval[c] = w.cat_interval[c] * rng.uniform(0.6, 1.6)
                self.version[c] = 0
            else:
                for _ in range(1 + rng.poisson(cfg.durable_rate)):
                    self.durables.append((rng.uniform(0, cfg.days), int(rng.choice(cand, p=p))))
        self._rebuild_fav()
        self.noise = rng.normal(0, cfg.policy_noise, cfg.n_items) if cfg.policy != "pop" else 0.0

    def _rebuild_fav(self):
        items, ints = [], []
        for c, f in self.fav.items():
            items += f
            ints += [self.interval[c]] * len(f)
        self.fav_items = np.array(items, dtype=np.int64)
        self.fav_int = np.array(ints, dtype=float)

    def utility(self, t):
        w = self.w
        u = self.base.copy()
        u[~w.is_cons_item & self.owned_cat[w.item_cat]] *= 0.3
        u[self.owned] = 0.0
        u[self.sub_mask] *= 0.3
        if self.fav_items.size:
            age = t - self.last[self.fav_items]
            due = np.where(np.isnan(age), 0.5,
                           np.exp(-np.abs(np.log(np.maximum(np.nan_to_num(age, nan=1.0), 1e-3) / self.fav_int))))
            u[self.fav_items] = self.aff_item[self.fav_items] * due
        return u

    def pi0_score(self, util):
        cfg, w = self.w.cfg, self.w
        if cfg.policy == "pop":
            return w.log_pop
        if cfg.policy == "good":
            return cfg.kappa * util + self.noise
        if cfg.policy == "skewed":
            return 0.5 * cfg.kappa * util + cfg.promo_boost * w.promo + self.noise
        raise ValueError(cfg.policy)

    def buy(self, item, t):
        """Обновляет состояние после покупки; возвращает категорию расходника для переноса графика (или None)."""
        w = self.w
        c = int(w.item_cat[item])
        if w.is_cons_item[item]:
            self.last[item] = t
            return c if c in self.fav else None
        self.owned[item] = True
        self.owned_cat[c] = True
        return None


class _Proc:
    """
    Дискретно-событийный процесс одного пользователя: состояние + куча событий + его ГСЧ.
    Копируется (copy.deepcopy с общим World) в любой момент — так симулируется онлайн-окно,
    в котором π0 заменена другой политикой, без пересимуляции всего года.
    """

    def __init__(self, w, cfg, u, offset, snap_times):
        self.w, self.cfg, self.u, self.offset = w, cfg, u, offset
        self.rng = np.random.default_rng([cfg.seed, 1, u])
        self.us = _User(w, u, self.rng)
        self.heap, self.seq = [], 0
        rng, us = self.rng, self.us
        for c in us.fav:
            self.push(rng.uniform(0, us.interval[c]), CONS, c, 0)
        for t, item in us.durables:
            self.push(t, DUR, item)
        t = rng.exponential(1 / cfg.visit_rate)
        while t < cfg.days:
            self.push(t, VISIT, None)
            t += rng.exponential(1 / cfg.visit_rate)
        for ts in snap_times:
            self.push(ts, SNAP, ts)

    def push(self, t, kind, payload, ver=0):
        if t < self.cfg.days:
            heapq.heappush(self.heap, (t, kind, self.seq, payload, ver))
            self.seq += 1

    def reschedule(self, c, t_now):
        us = self.us
        us.version[c] += 1
        self.push(t_now + us.interval[c] * self.rng.lognormal(0, 0.25), CONS, c, us.version[c])

    def next_time(self):
        return self.heap[0][0] if self.heap else np.inf

    def run_until(self, t_end, ev_rows, imp=None, snaps=None, scores=None):
        """
        Обрабатывает события с t < t_end. scores — подмена π0: при визите показывается топ-K по scores
        (детерминированно, пропенсити = 1); None — обычная π0.
        """
        w, cfg, us, rng, u = self.w, self.cfg, self.us, self.rng, self.u
        K, N = cfg.slate, cfg.n_items
        while self.heap and self.heap[0][0] < t_end:
            t, kind, _, payload, ver = heapq.heappop(self.heap)
            if kind == SNAP:
                if snaps is not None:
                    util = us.utility(t)
                    snaps[payload]["util"][u] = util
                    snaps[payload]["pi0"][u] = us.pi0_score(util)
            elif kind == CONS:
                c = payload
                if ver != us.version[c]:
                    continue                                  # график сдвинут покупкой из рекомендации
                fav = us.fav[c]
                if rng.random() < cfg.brand_switch:           # сменил бренд насовсем
                    cand = w.items_by_cat[c]
                    fav[0] = int(rng.choice(cand, p=w.item_pop[cand] / w.item_pop[cand].sum()))
                    us._rebuild_fav()
                item = fav[0]
                if len(fav) > 1 and rng.random() < 0.2:
                    item = fav[1]
                if rng.random() < 0.1:                        # иногда пробует новое в категории
                    cand = w.items_by_cat[c]
                    item = int(rng.choice(cand, p=w.item_pop[cand] / w.item_pop[cand].sum()))
                ev_rows.append((u, int(item), t, True, np.nan, us.utility(t)[item]))
                us.buy(item, t)
                self.reschedule(c, t)
            elif kind == DUR:
                item = payload
                if us.owned[item]:
                    continue
                ev_rows.append((u, int(item), t, True, np.nan, us.utility(t)[item]))
                us.buy(item, t)
            else:                                             # VISIT
                util = us.utility(t)
                if scores is None:
                    p = _softmax(us.pi0_score(util) / cfg.tau)
                    shown = np.unique(rng.choice(N, K, p=p))
                    prop = 1.0 - (1.0 - p[shown]) ** K
                else:
                    shown = np.sort(np.argpartition(-scores, K)[:K])
                    prop = np.ones(K)
                pr = 1.0 / (1.0 + np.exp(-(cfg.beta * (1 - cfg.lam) * util[shown] + self.offset)))
                pr[us.owned[shown]] = 0.0
                bought = rng.random(len(shown)) < pr
                if imp is not None:
                    imp.append((np.full(len(shown), u), np.full(len(shown), t), shown, prop, bought))
                for item, pp in zip(shown[bought], prop[bought]):
                    ev_rows.append((u, int(item), t, False, pp, util[item]))
                    c = us.buy(item, t)
                    if c is not None:
                        self.reschedule(c, t)


EV_COLS = ["user", "item", "ts", "organic", "prop", "util"]


def _events_frame(ev_rows):
    ev = pd.DataFrame(ev_rows, columns=EV_COLS)
    return ev.sort_values(["user", "ts"], kind="stable").reset_index(drop=True)


def simulate(cfg, users=None, offset=None, keep_snapshots=True, capture_at=None):
    """
    Симулирует пользователей users (по умолчанию все). Возвращает SimResult.
    capture_at — момент, в который сохраняется копия процесса каждого пользователя (для онлайн-окна).
    """
    w = World(cfg)
    offset = cfg.offset if offset is None else offset
    users = range(cfg.n_users) if users is None else users
    N = cfg.n_items
    ev_rows, imp = [], []
    snaps = {t: {"util": np.zeros((cfg.n_users, N), np.float32), "pi0": np.zeros((cfg.n_users, N), np.float32)}
             for t in cfg.snapshot_times} if keep_snapshots else {}
    captures = {}
    for u in users:
        proc = _Proc(w, cfg, u, offset, list(snaps))
        if capture_at is not None:
            proc.run_until(capture_at, ev_rows, imp, snaps)
            captures[u] = copy.deepcopy(proc, {id(w): w, id(cfg): cfg})
        proc.run_until(np.inf, ev_rows, imp, snaps)

    if imp:
        cols = [np.concatenate([x[i] for x in imp]) for i in range(5)]
        impressions = pd.DataFrame(dict(zip(["user", "ts", "item", "prop", "bought"], cols)))
    else:
        impressions = pd.DataFrame(columns=["user", "ts", "item", "prop", "bought"])
    res = SimResult(replace(cfg, offset=offset), w, _events_frame(ev_rows), impressions, snaps)
    res.captures, res.capture_at = captures, capture_at
    return res


def simulate_window(res, window, scores=None):
    """
    Онлайн-окно [capture_at, capture_at + window) из сохранённых копий процессов.
    scores: (n_users, N) — политика вместо π0 (топ-K), None — та же π0 (воспроизводит исходный лог).
    Возвращает события окна (тот же формат, что SimResult.events).
    """
    assert res.captures, "нужна симуляция с capture_at"
    w, cfg = res.world, res.cfg
    ev_rows = []
    for u, proc in res.captures.items():
        p = copy.deepcopy(proc, {id(w): w, id(proc.cfg): proc.cfg})
        p.run_until(res.capture_at + window, ev_rows, scores=None if scores is None else scores[u])
    return _events_frame(ev_rows)


def calibrate_offset(cfg, n_users=200, iters=10, lo=-12.0, hi=6.0):
    """Подбирает offset бисекцией, чтобы доля рекомендованных покупок ≈ cfg.rec_share."""
    users = range(min(n_users, cfg.n_users))
    for _ in range(iters):
        mid = (lo + hi) / 2
        ev = simulate(cfg, users, offset=mid, keep_snapshots=False).events
        share = (~ev.organic).mean() if len(ev) else 0.0
        if share < cfg.rec_share:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def run(cfg, capture_at=None):
    """Калибровка (если offset не задан) + полная симуляция."""
    if cfg.offset is None:
        cfg = replace(cfg, offset=calibrate_offset(cfg))
    return simulate(cfg, capture_at=capture_at)
