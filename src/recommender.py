"""Explainable hybrid recommender and best-time-to-visit scorer.

Recommendation score for a traveller (preferences + travel month):
    0.40 * content match   cosine similarity between preferences and traits
  + 0.25 * weather         climatological comfort in that month
  + 0.15 * low crowds      1 - forecast arrivals / the place's yearly peak
  + 0.20 * reviews         mean sentiment of traveller reviews
Every component is returned so each recommendation can be explained.

Best-time score for a place (per month, next 12 months):
    100 * (0.70 * weather comfort + 0.30 * (1 - normalised forecast crowd))
"""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from .anomaly import in_covid
from .config import FEATURES, PLACES

W = dict(content=0.40, weather=0.25, crowd=0.15, reviews=0.20)
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


class Recommender:
    def __init__(self, df, forecast, sentiment):
        self.traits = pd.DataFrame({k: v["traits"] for k, v in PLACES.items()}, index=FEATURES).T
        clean = df[~in_covid(df["date"])]
        self.comfort = clean.groupby(["place", clean["date"].dt.month])["comfort"].mean().unstack()
        self.forecast = forecast.pivot_table(index="place", columns=forecast["date"].dt.month,
                                             values="forecast")
        self.sentiment = sentiment.set_index("place")["sentiment"]

    # ------------------------------------------------------------ best time
    def best_time(self, place):
        f = self.forecast.loc[place]
        crowd = (f - f.min()) / (f.max() - f.min())
        comfort = self.comfort.loc[place].reindex(f.index)
        score = 100 * (0.7 * comfort + 0.3 * (1 - crowd))
        out = pd.DataFrame({"month": [MONTHS[m - 1] for m in f.index], "month_no": f.index,
                            "score": score.round(0).values, "comfort": comfort.round(2).values,
                            "expected_tourists": f.round(0).astype(int).values})
        out["weather"] = np.select([out.comfort >= .6, out.comfort >= .3], ["Ideal", "Okay"], "Avoid")
        out["crowd"] = np.select([crowd.values > .66, crowd.values > .33], ["High", "Medium"], "Low")
        return out.sort_values("score", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------- recommendation
    def recommend(self, prefs, month, top_n=3):
        user = np.array([[prefs.get(k, 0) for k in FEATURES]], dtype=float)
        content = cosine_similarity(user, self.traits.values)[0]
        rows = []
        for i, place in enumerate(self.traits.index):
            f = self.forecast.loc[place]
            crowd = 1 - f[month] / f.max()
            rows.append(dict(place=place, content=content[i], weather=float(self.comfort.loc[place, month]),
                             crowd=float(crowd), reviews=float(self.sentiment[place])))
        r = pd.DataFrame(rows)
        r["score"] = sum(W[k] * r[k] for k in W)
        r = r.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)
        r["why"] = r.apply(self._why, axis=1)
        return r

    @staticmethod
    def _why(row):
        parts = {k: W[k] * row[k] for k in W}
        top = sorted(parts, key=parts.get, reverse=True)[:2]
        text = {"content": "matches your interests", "weather": "pleasant weather that month",
                "crowd": "fewer crowds expected", "reviews": "strong traveller reviews"}
        return " and ".join(text[t] for t in top)

    def similar_places(self, place, top_n=3):
        sims = cosine_similarity(self.traits.loc[[place]].values, self.traits.values)[0]
        s = pd.Series(sims, index=self.traits.index).drop(place)
        return s.sort_values(ascending=False).head(top_n)
