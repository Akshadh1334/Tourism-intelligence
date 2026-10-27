"""Demand forecasting: a hybrid ensemble with honest backtesting.

Models
  * Seasonal naive          - baseline (same month last year)
  * Holt-Winters            - damped additive trend, multiplicative season,
                              fitted per destination on the post-COVID window
  * Trend + Gradient Boost  - ONE global model for all destinations. A
                              per-place log-linear trend is removed, and a
                              gradient-boosting model learns what is left from
                              calendar, weather comfort and place identity.
  * Ensemble                - inverse-MAPE weighted mean of the two main models

Protocol (no leakage)
  validation window = 12 months before the last 12  -> ensemble weights and
                      prediction-interval width are learned here
  test window       = final 12 months               -> reported metrics
  final forecast    = retrain on everything, predict the next 12 months

Training data is first cleaned with the anomaly detector (src/anomaly.py), so
one-off shocks do not distort the learned seasonality.
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from .anomaly import add_clean, in_covid
from .config import PLACES, SEED, START

warnings.filterwarnings("ignore")
HW_START = "2022-03-01"          # first month after the COVID regime
MODELS = ["hw", "gbm"]


def _tidx(series):
    d = pd.to_datetime(series)
    s = pd.Timestamp(START)
    y = d.dt.year if isinstance(d, pd.Series) else d.year
    m = d.dt.month if isinstance(d, pd.Series) else d.month
    return np.asarray((y - s.year) * 12 + m - s.month)


def make_future(train, horizon):
    """Future rows with climatological weather (mean by place and month)."""
    clim = (train[~in_covid(train["date"])]
            .assign(m=lambda d: d["date"].dt.month)
            .groupby(["place", "m"])[["avg_temp", "rainfall_mm", "comfort"]].mean())
    last = train["date"].max()
    dates = pd.date_range(last + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    rows = []
    for place in train["place"].unique():
        for d in dates:
            c = clim.loc[(place, d.month)]
            rows.append(dict(date=d, place=place, avg_temp=c.avg_temp,
                             rainfall_mm=c.rainfall_mm, comfort=c.comfort,
                             is_festive=int(d.month in (10, 11, 12))))
    return pd.DataFrame(rows)


def _features(df):
    m = df["date"].dt.month
    X = pd.DataFrame({"m_sin": np.sin(2 * np.pi * m / 12), "m_cos": np.cos(2 * np.pi * m / 12),
                      "temp": df["avg_temp"].values, "rain": df["rainfall_mm"].values,
                      "comfort": df["comfort"].values, "festive": df["is_festive"].values})
    for p in PLACES:
        X["pl_" + p] = (df["place"].values == p).astype(int)
    return X


def fit_gbm(train):
    ok = train[~in_covid(train["date"])]
    coefs = {}
    for p, g in ok.groupby("place"):
        b, a = np.polyfit(_tidx(g["date"]), np.log(g["arrivals_clean"]), 1)
        coefs[p] = (a, b)
    trend = np.array([coefs[p][0] + coefs[p][1] * t
                      for p, t in zip(ok["place"], _tidx(ok["date"]))])
    resid = np.log(ok["arrivals_clean"].values) - trend
    model = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                      subsample=0.8, random_state=SEED)
    model.fit(_features(ok), resid)
    return coefs, model


def predict_gbm(coefs, model, future):
    trend = np.array([coefs[p][0] + coefs[p][1] * t
                      for p, t in zip(future["place"], _tidx(future["date"]))])
    return np.exp(trend + model.predict(_features(future)))


def predict_hw(train, future, horizon):
    out = []
    for p in train["place"].unique():
        g = train[(train["place"] == p) & (train["date"] >= HW_START)]
        s = g.set_index("date")["arrivals_clean"].asfreq("MS")
        try:
            fit = ExponentialSmoothing(s, trend="add", damped_trend=True, seasonal="mul",
                                       seasonal_periods=12).fit()
            f = fit.forecast(horizon).values
            if not np.all(np.isfinite(f)):
                raise ValueError
        except Exception:
            f = s.values[-12:][:horizon]
        out.append(pd.DataFrame(dict(place=p, date=future.loc[future.place == p, "date"].values, hw=f)))
    return pd.concat(out, ignore_index=True)


def run_fold(df, train_end, horizon=12):
    """Train on data up to `train_end`, forecast the following `horizon` months."""
    train = add_clean(df[df["date"] <= train_end].copy())
    future = make_future(train, horizon)
    coefs, gbm = fit_gbm(train)
    future["gbm"] = predict_gbm(coefs, gbm, future)
    future = future.merge(predict_hw(train, future, horizon), on=["place", "date"])
    naive = []
    for p in train["place"].unique():
        s = train[train["place"] == p].sort_values("date")["arrivals_clean"].values[-12:]
        naive.append(pd.DataFrame(dict(place=p, date=future.loc[future.place == p, "date"].values,
                                       naive=s[:horizon])))
    return future.merge(pd.concat(naive), on=["place", "date"])


def mape(a, f):
    return float(np.mean(np.abs(a - f) / a))


def rmse(a, f):
    return float(np.sqrt(np.mean((a - f) ** 2)))


def backtest_and_forecast(df, horizon=12):
    end = df["date"].max()
    test_start = end - pd.DateOffset(months=horizon - 1)
    val_start = test_start - pd.DateOffset(months=horizon)

    # ---- validation fold: learn ensemble weights + interval width
    val = run_fold(df, val_start - pd.DateOffset(months=1), horizon)
    val = val.merge(df[["place", "date", "arrivals"]], on=["place", "date"])
    inv = {m: 1 / mape(val["arrivals"], val[m]) for m in MODELS}
    weights = {m: inv[m] / sum(inv.values()) for m in MODELS}
    val["ens"] = sum(weights[m] * val[m] for m in MODELS)
    rel_err = (val["arrivals"] - val["ens"]) / val["ens"]
    q_lo, q_hi = np.quantile(rel_err, [0.10, 0.90])

    # ---- test fold: honest, untouched metrics
    test = run_fold(df, test_start - pd.DateOffset(months=1), horizon)
    test = test.merge(df[["place", "date", "arrivals"]], on=["place", "date"])
    test["ens"] = sum(weights[m] * test[m] for m in MODELS)
    test["lower"], test["upper"] = test["ens"] * (1 + q_lo), test["ens"] * (1 + q_hi)

    rows = []
    for name, col in [("Seasonal naive", "naive"), ("Holt-Winters", "hw"),
                      ("Trend + Gradient Boosting", "gbm"), ("Ensemble", "ens")]:
        rows.append(dict(model=name, MAPE_validation=mape(val["arrivals"], val[col]) if col in val else np.nan,
                         MAPE_test=mape(test["arrivals"], test[col]),
                         RMSE_test=rmse(test["arrivals"], test[col])))
    metrics = pd.DataFrame(rows)
    coverage = float(((test["arrivals"] >= test["lower"]) & (test["arrivals"] <= test["upper"])).mean())
    per_place = (test.assign(ape=lambda d: (d["arrivals"] - d["ens"]).abs() / d["arrivals"])
                 .groupby("place")["ape"].mean().rename("ensemble_MAPE_test").reset_index())

    # ---- final forecast beyond the last observed month
    final = run_fold(df, end, horizon)
    final["forecast"] = sum(weights[m] * final[m] for m in MODELS)
    final["lower"], final["upper"] = final["forecast"] * (1 + q_lo), final["forecast"] * (1 + q_hi)

    info = dict(weights=weights, interval=(float(q_lo), float(q_hi)), coverage_test=coverage,
                test_window=(str(test_start.date()), str(end.date())))
    return final, metrics, per_place, test, info
