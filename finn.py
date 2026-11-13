"""
Стенд C — FINN.no Slates (DESIGN.md §3): маркетплейс объявлений с реальными показами.

Данные: https://huggingface.co/datasets/simeneide/recsys_slates_dataset (Apache 2.0), data.npz:
  click (U, 20)             — что кликнул пользователь на каждом из 20 первых взаимодействий;
  slate (U, 20, ≤25)        — все показанные объявления (слейт);
  interaction_type (U, 20)  — 1 поиск, 2 рекомендация, 0 не определено;
  индекс 1 — «не кликнул» (особый элемент слейта), 0 — паддинг.
Меток времени нет — только порядок взаимодействий; «время» = номер шага.

Отличия от Yambda (оговорки для выводов):
  * «органика» здесь — поиск: тоже ранжирующая система, но выдача обусловлена запросом пользователя;
  * клик, а не покупка/прослушивание; объявления живут недолго, повторов мало;
  * зато есть показы → настоящая пропенсити (частота показа в рекомендательных слейтах) и имитатор,
    обученный на верхнем объявлении рекомендательного слейта, а не на кликах.
Сплит — по шагам: последний клик пользователя — тест, предпоследний — валидация (у пользователей с ≥ 3 кликами).
"""
import os

import numpy as np
import pandas as pd

from fl_data import FLSplit

PAD, NO_CLICK = 0, 1
SEARCH, REC = 1, 2


def download(data_dir="finn_data"):
    """Находит или скачивает data.npz: Hugging Face, затем pip-пакет авторов (recsys_slates_dataset)."""
    errors = []
    try:
        from huggingface_hub import hf_hub_download
        for fn in ("data/data.npz", "data.npz"):
            try:
                return hf_hub_download(repo_id="simeneide/recsys_slates_dataset", repo_type="dataset", filename=fn)
            except Exception as e:                       # noqa: BLE001 — пробуем следующий путь
                errors.append(f"{fn}: {e}")
    except ImportError as e:
        errors.append(str(e))
    try:
        from recsys_slates_dataset import data_helper
        data_helper.download_data_files(data_dir=data_dir)
        return os.path.join(data_dir, "data.npz")
    except Exception as e:                               # noqa: BLE001
        errors.append(f"recsys_slates_dataset: {e}")
    raise RuntimeError("не удалось получить data.npz:\n" + "\n".join(errors))


def load_raw(path, n_users=None, seed=0):
    d = np.load(path)
    print("массивы:", {k: d[k].shape for k in d.files if k != "slate"}, flush=True)
    U = d["click"].shape[0]
    idx = np.arange(U) if not n_users or n_users >= U else \
        np.sort(np.random.default_rng(seed).choice(U, n_users, replace=False))
    click = d["click"][idx].astype(np.int64)
    itype = d["interaction_type"][idx].astype(np.int64)
    slate = d["slate"][idx].astype(np.int64)
    print(f"пользователей {len(idx):,} из {U:,}; слейт {slate.shape}", flush=True)
    return click, itype, slate


def _first_real(slate):
    """Первое настоящее объявление в каждом слейте (пропуская паддинг и «не кликнул»); −1 если нет."""
    real = slate > NO_CLICK
    pos = real.argmax(axis=2)
    top = np.take_along_axis(slate, pos[..., None], axis=2)[..., 0]
    return np.where(real.any(axis=2), top, -1)


def build(click, itype, slate, top_items=20000, n_eval=(20000, 30000), seed=0):
    U, T = click.shape
    valid = (click > NO_CLICK) & np.isin(itype, (SEARCH, REC))
    uu, tt = np.nonzero(valid)                                     # по пользователю, затем по шагу
    ev = pd.DataFrame({"user": uu, "step": tt.astype(float), "item_raw": click[uu, tt],
                       "organic": itype[uu, tt] == SEARCH})
    ev["top_raw"] = np.where(itype[uu, tt] == REC, _first_real(slate)[uu, tt], -1)

    # сплит: последний клик — тест, предпоследний — валидация (если кликов ≥ 3)
    ev["rank"] = ev.groupby("user").cumcount()
    ev["n"] = ev.groupby("user")["user"].transform("size")
    elig = ev.n >= 3
    ev["part"] = "train"
    ev.loc[elig & (ev["rank"] == ev.n - 1), "part"] = "test"
    ev.loc[elig & (ev["rank"] == ev.n - 2), "part"] = "val"

    # каталог — самые кликабельные объявления обучения
    tr = ev[ev.part == "train"]
    catalog = tr["item_raw"].value_counts().index[:top_items].to_numpy()
    imap = pd.Series(np.arange(len(catalog)), index=catalog)
    ev["item"] = ev["item_raw"].map(imap).fillna(-1).astype(np.int64)
    ev["target_top"] = ev["top_raw"].map(imap).fillna(-1).astype(np.int64)

    # пропенсити по реальным показам: частота объявления в рекомендательных слейтах
    rec_steps = itype == REC
    shown = slate[rec_steps]
    shown = shown[shown > NO_CLICK]
    exp_cnt = pd.Series(shown).value_counts()
    ev["prop"] = np.where(ev.organic, np.nan,
                          ev["item_raw"].map(exp_cnt).fillna(1).to_numpy() / max(rec_steps.sum(), 1))

    stats = {"users": U, "clicks (search+rec)": len(ev), "rec share (clicks)": (~ev.organic).mean(),
             "rec share (test)": (~ev[ev.part == "test"].organic).mean(),
             "catalog": len(catalog), "catalog covers train clicks": tr.item_raw.isin(imap.index).mean(),
             "catalog covers test clicks": ev[ev.part == "test"].item_raw.isin(imap.index).mean(),
             "users evaluated": int(elig.groupby(ev.user).first().sum())}
    first = ~ev.duplicated(["user", "item_raw"])
    stats["repeat share search"] = 1 - first[ev.organic].mean()
    stats["repeat share rec"] = 1 - first[~ev.organic].mean()

    inc = ev[ev.item >= 0]                                        # в модели — только объявления каталога
    train = inc[inc.part == "train"].rename(columns={"step": "ts"})
    train = train[["user", "item", "ts", "organic", "prop", "target_top"]].reset_index(drop=True)

    slate_map = {}
    for part in ("val", "test"):
        p = ev[ev.part == part]
        s = slate[p.user.to_numpy(), p.step.to_numpy().astype(int)]
        slate_map[part] = {u: np.unique(imap.reindex(row[row > NO_CLICK]).dropna().astype(np.int64).to_numpy())
                           for u, row in zip(p.user.to_numpy(), s)}

    eval_rng = np.random.default_rng(seed)

    def split(part, n_max):
        tgt_ev = inc[inc.part == part]
        users = tgt_ev.user.unique()
        if n_max and len(users) > n_max:                       # оценка на выборке: матрица скоров U × N
            tgt_ev = tgt_ev[tgt_ev.user.isin(eval_rng.choice(users, n_max, replace=False))]
        before_parts = ("train",) if part == "val" else ("train", "val")
        hist_ev = inc[inc.part.isin(before_parts) & inc.user.isin(tgt_ev.user)]
        hist = {u: (g["item"].to_numpy(), g["step"].to_numpy()) for u, g in hist_ev.groupby("user")}
        tgt_ev = tgt_ev[tgt_ev.user.isin(list(hist))]
        targets = {"organic": {}, "rec": {}, "all": {}}
        t_query = {}
        for u, item, org, step in zip(tgt_ev.user, tgt_ev.item, tgt_ev.organic, tgt_ev.step):
            targets["organic" if org else "rec"][u] = np.array([item])
            targets["all"][u] = np.array([item])
            t_query[u] = step
        hist = {u: h for u, h in hist.items() if u in t_query}
        sp = FLSplit(hist, targets, t_query, np.nan, None, None, hist_ev)
        sp.shown = slate_map[part]                                 # показанный слейт на шаге цели
        sp.kind = {u: ("organic" if u in targets["organic"] else "rec") for u in t_query}
        return sp

    val, test = split("val", n_eval[0]), split("test", n_eval[1])
    log_pop = np.log1p(np.bincount(train["item"], minlength=len(catalog)))
    items = pd.DataFrame({"item": np.arange(len(catalog)), "is_consumable": np.nan, "log_pop": log_pop})
    return train, val, test, items, stats


def make_stub(path, n_users=3000, n_items=4000, seed=0):
    """Заглушка в формате data.npz для локальной проверки (структура как у настоящего датасета)."""
    rng = np.random.default_rng(seed)
    pop = 1.0 / np.arange(1, n_items + 1) ** 0.9
    pop /= pop.sum()
    T, S = 20, 25
    itype = rng.choice([0, 1, 2], size=(n_users, T), p=[0.05, 0.65, 0.30])
    slate = np.zeros((n_users, T, S), np.int64)
    click = np.zeros((n_users, T), np.int64)
    n_steps = rng.integers(3, T + 1, n_users)
    for u in range(n_users):
        pref = rng.choice(n_items, 50, p=pop) + 2
        for t in range(n_steps[u]):
            L = rng.integers(5, S)
            items = np.where(rng.random(L - 1) < 0.5, rng.choice(pref, L - 1), rng.choice(n_items, L - 1, p=pop) + 2)
            slate[u, t, 0] = NO_CLICK
            slate[u, t, 1:L] = items
            click[u, t] = NO_CLICK if rng.random() < 0.25 else rng.choice(items)
    np.savez(path, userId=np.arange(n_users), click=click, click_idx=np.zeros_like(click),
             slate_lengths=(slate > 0).sum(2), slate=slate, interaction_type=itype)
    return path
