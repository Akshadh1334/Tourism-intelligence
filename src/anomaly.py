"""Anomaly detection on tourist arrivals.

Two complementary detectors are combined by a voting rule:
  1. STL decomposition (robust) -> residual -> robust z-score (median/MAD)
  2. Isolation Forest on [STL residual, month-on-month log change]
A month is flagged when STL is very sure (|z| > 6) OR when Isolation Forest
agrees and STL is moderately sure (|z| > 3.5). Residuals are heavy-tailed,
so thresholds sit above the textbook 3.5; they were calibrated on the
synthetic data and should be re-tuned on real series.

The same machinery is reused to *clean* training data before forecasting,
so one-off shocks do not leak into the learned seasonality.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.seasonal import STL
from .config import SEED

COVID_START, COVID_END = "2020-03-01", "2022-03-01"


def in_covid(dates):
    d = pd.to_datetime(dates)
    return (d >= COVID_START) & (d <= COVID_END)


def _robust_z(x):
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826
    return (x - med) / (mad if mad > 0 else 1.0)


def _weather_adjust(dates, y_log, comfort, mask):
    """Remove the COVID regime and the weather effect from log arrivals.

    OLS on non-COVID months: log y ~ trend + month dummies + weather comfort
    (one round of outlier trimming). COVID months are replaced by the model's
    counterfactual and the fitted weather effect is subtracted, so what is
    left for STL is demand that weather cannot explain."""
    t = np.arange(len(dates))
    X = np.column_stack([np.ones(len(t)), t, comfort] +
                        [(dates.month == m).astype(float) for m in range(2, 13)])
    keep = ~mask
    for _ in range(2):
        beta, *_ = np.linalg.lstsq(X[keep], y_log[keep], rcond=None)
        r = y_log - X @ beta
        mad = np.median(np.abs(r[~mask] - np.median(r[~mask]))) * 1.4826
        keep = ~mask & (np.abs(r) < 3 * mad)
    filled = y_log.copy()
    filled[mask] = (X @ beta)[mask]
    return filled - beta[2] * comfort, beta[2] * comfort


def annotate(df, col="arrivals"):
    """Return df with stl_z, iso_flag, fitted, anomaly (bool) per place.

    Weather-aware: arrivals are adjusted for weather comfort and the COVID
    regime is replaced by a counterfactual before STL, so neither a rainy
    month nor the pandemic is mistaken for a point anomaly. COVID months are
    reported through `covid_regime` instead."""
    out = []
    for place, g in df.sort_values("date").groupby("place"):
        g = g.copy()
        dates = pd.DatetimeIndex(g["date"])
        mask = np.asarray(in_covid(dates))
        y, w_eff = _weather_adjust(dates, np.log(g[col].values), g['comfort'].values, mask)
        s = pd.Series(y, index=dates).asfreq("MS")
        res = STL(s, period=12, seasonal=25, robust=True).fit()
        resid = res.resid.values
        z = np.zeros(len(resid))
        z[~mask] = _robust_z(resid[~mask])        # scale from real months only
        mom = np.r_[0, np.diff(y)]
        mom[mask] = 0.0
        iso = IsolationForest(contamination=0.04, random_state=SEED)
        iso_flag = iso.fit_predict(np.column_stack([np.where(mask, 0, resid), mom])) == -1
        iso_flag &= ~mask
        g["stl_z"] = z
        g["iso_flag"] = iso_flag
        g["covid_regime"] = mask
        g["fitted"] = np.exp(res.trend.values + res.seasonal.values + w_eff)
        g["anomaly_stl"] = np.abs(z) > 5.0
        g["anomaly"] = (np.abs(z) > 6.0) | (iso_flag & (np.abs(z) > 3.5))
        out.append(g)
    return pd.concat(out, ignore_index=True)


def evaluate(df_annotated, events):
    """Precision / recall / F1 of each detector against injected ground truth.
    The COVID window is excluded (it is a regime change, not a point anomaly)."""
    d = df_annotated[~in_covid(df_annotated["date"])].copy()
    truth = set(zip(events["place"], pd.to_datetime(events["date"])))
    d["truth"] = [(p, t) in truth for p, t in zip(d["place"], d["date"])]
    rows = []
    for name, col in [("STL robust z-score", "anomaly_stl"),
                      ("Isolation Forest", "iso_flag"),
                      ("Combined (voting)", "anomaly")]:
        tp = int((d[col] & d.truth).sum())
        fp = int((d[col] & ~d.truth).sum())
        fn = int((~d[col] & d.truth).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append(dict(detector=name, precision=prec, recall=rec, f1=f1, tp=tp, fp=fp, fn=fn))
    return pd.DataFrame(rows)


def add_clean(df):
    """Add `arrivals_clean`: anomalies (outside COVID window) replaced by the
    STL fitted value (trend + seasonality). Fit only on the rows passed in."""
    a = annotate(df)
    replace = a["anomaly"] & ~in_covid(a["date"])
    a["arrivals_clean"] = np.where(replace, a["fitted"], a["arrivals"])
    return a
