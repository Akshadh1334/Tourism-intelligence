"""Dynamic hotel pricing from forecast demand and estimated price elasticity.

1. Elasticity: per destination, a log-log regression on non-COVID months
       log(occupancy) = a + b*log(arrivals) + c*log(ADR) + d*t
   gives the price elasticity  e = -c.  Months where occupancy hit the
   capacity ceiling are dropped (censored observations).
2. Optimisation: for every forecast month, search a grid of price
   multipliers and choose the one that maximises
       revenue = price * min(occupancy_cap, occupancy(price)) * rooms * 30
   Under capacity constraints this raises prices in peak months and
   discounts in lean months for price-sensitive destinations.
"""
import numpy as np
import pandas as pd
from .anomaly import in_covid
from .config import PLACES
from .forecasting import _tidx

OCC_CAP = 0.98
GRID = np.round(np.arange(0.75, 1.351, 0.025), 3)


def estimate_elasticity(df):
    d = df[~in_covid(df["date"]) & (df["hotel_occupancy"] < 0.95)]
    rows = []
    for place, g in d.groupby("place"):
        X = np.column_stack([np.ones(len(g)), np.log(g["arrivals"]),
                             np.log(g["hotel_adr"]), _tidx(g["date"])])
        y = np.log(g["hotel_occupancy"])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        rows.append(dict(place=place, intercept=beta[0], arrivals_coef=beta[1],
                         price_coef=beta[2], trend_coef=beta[3],
                         elasticity=-beta[2], true_elasticity=PLACES[place]["elasticity"]))
    return pd.DataFrame(rows)


def optimise_prices(df, forecast, est):
    """Return a monthly price plan with expected revenue vs a flat price."""
    last = df["date"].max()
    recent = df[df["date"] > last - pd.DateOffset(months=12)]
    adr_ref = {(r.place, r.date.month): r.hotel_adr * 1.04 for r in recent.itertuples()}
    coef = est.set_index("place")
    rows = []
    for r in forecast.itertuples():
        c = coef.loc[r.place]
        base = adr_ref[(r.place, r.date.month)]
        t = _tidx(pd.Series([r.date]))[0]
        rooms = PLACES[r.place]["rooms"]

        def occ(price):
            val = np.exp(c.intercept + c.arrivals_coef * np.log(r.forecast)
                         + c.price_coef * np.log(price) + c.trend_coef * t)
            return min(OCC_CAP, val)

        revenue = np.array([m * base * occ(m * base) * rooms * 30 for m in GRID])
        best = int(np.argmax(revenue))
        flat = float(base * occ(base) * rooms * 30)
        rows.append(dict(place=r.place, date=r.date, forecast=r.forecast,
                         base_price=base, multiplier=GRID[best], optimal_price=GRID[best] * base,
                         occupancy_flat=occ(base), occupancy_optimal=occ(GRID[best] * base),
                         revenue_flat=flat, revenue_optimal=float(revenue[best]),
                         uplift_pct=100 * (revenue[best] / flat - 1)))
    plan = pd.DataFrame(rows)
    plan["action"] = np.select([plan.multiplier >= 1.05, plan.multiplier <= 0.95],
                               ["RAISE price", "OFFER discount"], "HOLD price")
    return plan
