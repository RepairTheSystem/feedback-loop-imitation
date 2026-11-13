"""
Стенд B — Yambda-50M (DESIGN.md §3, §7): загрузка, позитивы, каталог, сплит.

Данные: https://huggingface.co/datasets/yandex/yambda, файл flat/50m/listens.parquet
  колонки: uid, item_id, timestamp (5-секундные единицы), is_organic (0/1), played_ratio_pct, track_length_seconds.
Позитив (Listen+, как в статье авторов): прослушано больше 50% трека.
Сплит — глобальный временной, как у авторов: тест — последний день, перед ним зазор 30 минут;
валидация — последний день обучения с тем же зазором. Модель учится на событиях до val, на тесте не переобучается,
история пользователя — все позитивы до теста (минус зазор).

Каталог урезается до top_items самых частых треков обучения: полный softmax по ~0.9 млн треков не помещается
в память. События с треками вне каталога выпадают из истории и из целей; доля покрытых целей теста печатается.
"""
import numpy as np
import pandas as pd

from fl_data import FLSplit

# timestamp — секунды, округлённые до 5 с («binned into 5s units» в карточке). Первая версия считала его числом
# 5-секундных интервалов — длительность выходила 1504 дня вместо ~301 (300 дней обучения + 1 день теста у авторов),
# и «тестовый день» на деле был ~4.8 часа. Проверки в load_positives ловят такую ошибку.
UNIT_DAYS = 1.0 / 86400.0
GAP = 30.0 / (24 * 60)             # 30 минут в днях


def download_listens(cache_dir=None):
    """Скачивает flat/50m/listens.parquet с Hugging Face (нужен интернет). Возвращает путь."""
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id="yandex/yambda", repo_type="dataset",
                           filename="flat/50m/listens.parquet", cache_dir=cache_dir)


def load_positives(path, min_ratio=50, n_users=None, seed=0):
    """Позитивные прослушивания: user, item_raw, ts (дни), organic (bool)."""
    df = pd.read_parquet(path, columns=["uid", "item_id", "timestamp", "is_organic", "played_ratio_pct"])
    if n_users:
        uids = df["uid"].unique()
        keep = np.random.default_rng(seed).choice(uids, min(n_users, len(uids)), replace=False)
        df = df[df["uid"].isin(keep)]
    n_listens = len(df)
    ts = df["timestamp"].to_numpy(np.int64)
    span = (ts.max() - ts.min()) * UNIT_DAYS
    print(f"метки времени кратны 5 с: {(ts % 5 == 0).mean():.4f}; длительность {span:.1f} дн.", flush=True)
    if not 200 <= span <= 400 and not n_users:
        raise ValueError(f"длительность {span:.1f} дн. не похожа на ~301 день из статьи — проверь единицы времени")
    df = df[df["played_ratio_pct"].fillna(0) > min_ratio]
    out = pd.DataFrame({"user_raw": df["uid"].to_numpy(), "item_raw": df["item_id"].to_numpy(),
                        "ts": df["timestamp"].to_numpy(np.float64) * UNIT_DAYS,
                        "organic": df["is_organic"].to_numpy().astype(bool)})
    out = out.sort_values(["user_raw", "ts"], kind="stable").reset_index(drop=True)
    print(f"прослушиваний {n_listens:,}, позитивов (> {min_ratio}%) {len(out):,}, "
          f"доля рекомендованных среди позитивов {(~out.organic).mean():.3f}", flush=True)
    return out


def build(pos, top_items=20000, train_tail=500, test_days=1.0):
    """
    Возвращает (train, val, test, items, stats).
    train — позитивы до val (последние train_tail на пользователя), с колонками user, item, ts, organic, prop.
    """
    t_end = pos.ts.max()
    t_test = t_end - test_days
    t_val = t_test - GAP - 1.0
    train_all = pos[pos.ts < t_val - GAP]

    cnt = train_all["item_raw"].value_counts()
    catalog = cnt.index[:top_items].to_numpy()
    item_map = pd.Series(np.arange(len(catalog)), index=catalog)
    users = np.sort(pos["user_raw"].unique())
    user_map = pd.Series(np.arange(len(users)), index=users)

    ev = pos[pos["item_raw"].isin(item_map.index)].copy()
    ev["item"] = item_map[ev["item_raw"].to_numpy()].to_numpy()
    ev["user"] = user_map[ev["user_raw"].to_numpy()].to_numpy()
    ev["prop"] = np.nan                                   # показов в Yambda нет — истинных пропенсити тоже
    ev = ev[["user", "item", "ts", "organic", "prop"]]

    def split(t0, t1):
        before = ev[ev.ts < t0 - GAP]
        hist = {u: (g["item"].to_numpy(), g["ts"].to_numpy()) for u, g in before.groupby("user")}
        w = ev[(ev.ts >= t0) & (ev.ts < t1) & ev.user.isin(list(hist))]
        targets = {}
        for kind, part in (("organic", w[w.organic]), ("rec", w[~w.organic]), ("all", w)):
            targets[kind] = {u: np.unique(g["item"].to_numpy()) for u, g in part.groupby("user")}
        return FLSplit(hist, targets, {u: t0 for u in hist}, t0, None, None, before.reset_index(drop=True))

    val = split(t_val, t_test - GAP)
    test = split(t_test, t_end + 1e-9)
    train = ev[ev.ts < t_val - GAP]
    train = train.groupby("user", sort=False).tail(train_tail).sort_values(["user", "ts"], kind="stable")
    train = train.reset_index(drop=True)

    log_pop = np.log1p(np.bincount(train_all["item_raw"].map(item_map).dropna().astype(int), minlength=len(catalog)))
    items = pd.DataFrame({"item": np.arange(len(catalog)), "is_consumable": np.nan, "log_pop": log_pop})

    test_raw = pos[(pos.ts >= t_test)]
    stats = describe(pos, train_all, test_raw, item_map, t_val, t_test)
    return train, val, test, items, stats


def describe(pos, train_all, test_raw, item_map, t_val, t_test):
    """Описательная статистика: доли рекомендованных, повторов, покрытие каталогом."""
    s = {"days": pos.ts.max() - pos.ts.min(), "users": pos.user_raw.nunique(), "positives": len(pos),
         "rec share (positives)": (~pos.organic).mean(),
         "rec share (test day)": (~test_raw.organic).mean() if len(test_raw) else np.nan,
         "catalog": len(item_map),
         "catalog covers train positives": train_all.item_raw.isin(item_map.index).mean(),
         "catalog covers test positives": test_raw.item_raw.isin(item_map.index).mean() if len(test_raw) else np.nan}
    # повтор = пользователь уже слушал этот трек раньше (для H3: из чего состоят рекомендованные события)
    first = ~pos.duplicated(["user_raw", "item_raw"])
    s["repeat share organic"] = 1 - first[pos.organic].mean()
    s["repeat share rec"] = 1 - first[~pos.organic].mean()
    return s


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=None)
    ap.add_argument("--n-users", type=int, default=None)
    a = ap.parse_args()
    path = a.path or download_listens()
    pos = load_positives(path, n_users=a.n_users)
    *_, stats = build(pos)
    for k, v in stats.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
