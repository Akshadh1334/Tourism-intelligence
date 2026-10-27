"""Destination segmentation: unsupervised clustering of tourism 'personalities'.

Each destination is described by behavioural features computed from the data
(seasonality strength, peak month, international share, growth, COVID
resilience, price sensitivity, hotel price level, weather comfort, review
sentiment). K-Means groups them; k is chosen by silhouette score, and each
cluster is named from the features where its centroid deviates most.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from .anomaly import in_covid
from .config import SEED

NICE = {"log_arrivals": "high volume", "seasonality": "strong seasonality",
        "foreign_share": "international", "growth": "fast growth",
        "covid_resilience": "covid resilient", "elasticity": "price sensitive",
        "log_adr": "premium priced", "comfort": "mild climate", "sentiment": "well reviewed"}


def build_features(df, elasticity, sentiment):
    rows = []
    for place, g in df.groupby("place"):
        g = g.sort_values("date")
        clean = g[~in_covid(g["date"])]
        monthly = clean.groupby(clean["date"].dt.month)["arrivals"].mean()
        peak_m = int(monthly.idxmax())
        pre = g[(g["date"].dt.year == 2019)]["arrivals"].mean()
        post = g[(g["date"] >= "2022-04-01") & (g["date"] < "2023-04-01")]["arrivals"].mean()
        y15 = g[g["date"].dt.year == 2016]["arrivals"].mean()
        y19 = g[g["date"].dt.year == 2019]["arrivals"].mean()
        rows.append(dict(
            place=place, peak_month=peak_m,
            log_arrivals=np.log(clean["arrivals"].mean()),
            seasonality=monthly.std() / monthly.mean(),
            peak_sin=np.sin(2 * np.pi * peak_m / 12), peak_cos=np.cos(2 * np.pi * peak_m / 12),
            foreign_share=clean["foreign_arrivals"].sum() / clean["arrivals"].sum(),
            growth=(y19 / y15) ** (1 / 3) - 1,
            covid_resilience=g[(g["date"] >= "2021-01-01") & (g["date"] <= "2021-12-01")]["arrivals"].mean() / pre,
            log_adr=np.log(clean["hotel_adr"].mean()),
            comfort=clean["comfort"].mean(),
        ))
    f = pd.DataFrame(rows)
    f = f.merge(elasticity[["place", "elasticity"]], on="place").merge(sentiment, on="place")
    return f


def cluster(features, k_range=range(2, 6)):
    cols = [c for c in features.columns if c not in ("place", "peak_month")]
    Z = StandardScaler().fit_transform(features[cols])
    scores = {}
    for k in k_range:
        labels = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit_predict(Z)
        scores[k] = silhouette_score(Z, labels)
    best_k = max(scores, key=scores.get)
    km = KMeans(n_clusters=best_k, n_init=20, random_state=SEED).fit(Z)
    out = features.copy()
    out["cluster"] = km.labels_
    pcs = PCA(n_components=2, random_state=SEED).fit(Z)
    xy = pcs.transform(Z)
    out["pc1"], out["pc2"] = xy[:, 0], xy[:, 1]

    names = {}
    cent = pd.DataFrame(km.cluster_centers_, columns=cols)
    for c in range(best_k):
        cand = cent.loc[c, [k for k in NICE if k in cols]].sort_values(ascending=False)
        tags = [NICE[t] for t in cand.index[:2]]
        months = out.loc[out["cluster"] == c, "peak_month"]
        mode = int(months.mode().iloc[0])
        season = ("winter-peak" if mode in (11, 12, 1, 2) else
                  "summer-peak" if mode in (3, 4, 5, 6) else "monsoon/autumn-peak")
        names[c] = f"{tags[0]} + {tags[1]} ({season})"
    out["segment"] = out["cluster"].map(names)
    info = dict(best_k=best_k, silhouette={int(k): float(v) for k, v in scores.items()},
                explained_variance=[float(v) for v in pcs.explained_variance_ratio_])
    return out, info
