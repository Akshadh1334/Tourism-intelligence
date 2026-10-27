"""Synthetic monthly tourism dataset (2015-2026) for 12 Indian destinations.

Signals deliberately built in so the models have something real to find:
  * yearly growth trend (4.5 %/yr) and destination-specific seasonality
  * weather comfort (temperature + rainfall) that pulls demand up or down
  * festive-season uplift (Oct-Dec)
  * a COVID-19 style collapse and staged recovery (2020-2022)
  * randomly injected one-off disruptions / mega-events (ground truth kept
    separately so the anomaly detector can be scored honestly)
  * hotel ADR (price) and occupancy with a known price elasticity
"""
import numpy as np
import pandas as pd
from .config import PLACES, START, END, SEED

COVID_ANCHORS = [("2020-02-01", 1.0), ("2020-03-01", .55), ("2020-04-01", .03),
                 ("2020-06-01", .03), ("2020-12-01", .40), ("2021-03-01", .50),
                 ("2021-04-01", .20), ("2021-06-01", .15), ("2021-12-01", .85),
                 ("2022-03-01", 1.0)]


def circ_dist(m, peak):
    d = np.abs(m - peak) % 12
    return np.minimum(d, 12 - d)


def comfort_index(temp, rain):
    """0-1 weather comfort: best near 22 C, penalised by heavy rain."""
    return np.exp(-((temp - 22) / 8) ** 2) / (1 + rain / 150)


def covid_factor(dates):
    xs = pd.to_datetime([a for a, _ in COVID_ANCHORS]).asi8
    ys = [b for _, b in COVID_ANCHORS]
    return np.interp(dates.asi8, xs, ys)


def generate(seed=SEED):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START, END, freq="MS")
    n = len(dates)
    t = np.arange(n)
    m = dates.month.values
    covid = covid_factor(dates)
    safe_idx = np.where(((dates < "2020-03-01") | (dates > "2022-04-01")))[0]
    safe_idx = safe_idx[safe_idx >= 18]
    rows, events = [], []

    for name, p in PLACES.items():
        tmean, tamp, tpeak = p["temp"]
        rmax, rpeak = p["rain"]
        temp = tmean + tamp * np.cos(2 * np.pi * (m - tpeak) / 12) + rng.normal(0, .8, n)
        rain = rmax * np.exp(-(circ_dist(m, rpeak) / 1.5) ** 2) * rng.lognormal(0, .15, n) + 10
        comfort = comfort_index(temp, rain)

        seasonal = 1 + .25 * np.cos(2 * np.pi * (m - p["peak"]) / 12)
        festive = np.isin(m, [10, 11, 12]).astype(int)
        trend = 1.045 ** (t / 12)
        arrivals = (p["base"] * trend * seasonal * (0.7 + 0.6 * comfort)
                    * (1 + 0.08 * festive) * covid * rng.lognormal(0, .05, n))

        for i in rng.choice(safe_idx, 4, replace=False):      # injected events
            kind = rng.choice(["disruption", "mega_event"])
            arrivals[i] *= 0.55 if kind == "disruption" else 1.6
            events.append(dict(place=name, date=dates[i], type=kind))

        # hotel market: occupancy responds to price with elasticity e
        ref_arr = p["base"] * 1.35 * trend
        adr_ref = p["adr"] * 1.04 ** (t / 12)
        adr = adr_ref * (1 + .15 * (seasonal - 1)) * rng.lognormal(0, .12, n)
        occ = arrivals / ref_arr * (adr / adr_ref) ** (-p["elasticity"]) * rng.lognormal(0, .03, n)
        occ = np.clip(occ, .05, .98)

        foreign = arrivals * np.clip(p["foreign"] * rng.lognormal(0, .06, n), 0, 1)
        rows.append(pd.DataFrame(dict(
            date=dates, place=name, arrivals=arrivals.round().astype(int),
            foreign_arrivals=foreign.round().astype(int),
            avg_temp=temp.round(1), rainfall_mm=rain.round(0), comfort=comfort.round(3),
            is_festive=festive, covid_factor=covid.round(3),
            hotel_adr=adr.round(0), hotel_occupancy=occ.round(3), rooms=p["rooms"])))

    return pd.concat(rows, ignore_index=True), pd.DataFrame(events)


if __name__ == "__main__":
    from .config import DATA_DIR
    DATA_DIR.mkdir(exist_ok=True)
    d, e = generate()
    d.to_csv(DATA_DIR / "tourism_monthly.csv", index=False)
    e.to_csv(DATA_DIR / "injected_events.csv", index=False)
    print(d.shape, e.shape)
