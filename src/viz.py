"""Matplotlib figures saved to /results (no UI required)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def forecast_grid(df, final, path):
    places = sorted(df["place"].unique())
    fig, axes = plt.subplots(4, 3, figsize=(14, 11), sharex=False)
    for ax, p in zip(axes.ravel(), places):
        h = df[df.place == p].tail(36)
        f = final[final.place == p]
        ax.plot(h["date"], h["arrivals"] / 1000, color="#1f77b4", label="Actual")
        ax.plot(f["date"], f["forecast"] / 1000, "--o", ms=3, color="#d62728", label="Forecast")
        ax.fill_between(f["date"], f["lower"] / 1000, f["upper"] / 1000, color="#d62728",
                        alpha=.15, label="80% interval")
        ax.set_title(p, fontweight="bold")
        ax.tick_params(axis="x", rotation=30)
        ax.set_ylabel("tourists (000)")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("12-month tourist arrival forecast (ensemble, 80% prediction interval)", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def anomalies(a, path, places=("Goa", "Manali", "Kerala")):
    fig, axes = plt.subplots(len(places), 1, figsize=(12, 3.2 * len(places)), sharex=True)
    for ax, p in zip(np.atleast_1d(axes), places):
        g = a[a.place == p]
        ax.plot(g["date"], g["arrivals"] / 1000, color="#555", lw=1)
        ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2022-03-01"), color="grey", alpha=.18,
                   label="COVID regime (handled separately)")
        flag = g[g["anomaly"]]
        ax.scatter(flag["date"], flag["arrivals"] / 1000, color="#d62728", zorder=3, label="Detected anomaly")
        ax.set_title(p, loc="left", fontweight="bold")
        ax.set_ylabel("tourists (000)")
    np.atleast_1d(axes)[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Weather-aware anomaly detection (STL + Isolation Forest voting)", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def model_comparison(metrics, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    m = metrics.sort_values("MAPE_test", ascending=False)
    bars = ax.barh(m["model"], m["MAPE_test"] * 100, color=["#bbb", "#9ecae1", "#6baed6", "#2171b5"][:len(m)][::-1])
    for b, v in zip(bars, m["MAPE_test"] * 100):
        ax.text(v + .1, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center")
    ax.set_xlabel("MAPE on untouched test year (lower is better)")
    ax.set_title("Forecast accuracy by model")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def segments(seg, path):
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("tab10")
    for c, g in seg.groupby("cluster"):
        ax.scatter(g["pc1"], g["pc2"], s=90, color=cmap(c), label=g["segment"].iloc[0])
        for _, r in g.iterrows():
            ax.annotate(r["place"], (r["pc1"], r["pc2"]), xytext=(5, 4), textcoords="offset points")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Destination segments (K-Means on PCA-projected behaviour features)")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def pricing(plan, path):
    piv = plan.assign(m=plan["date"].dt.month).pivot_table(index="place", columns="date", values="multiplier")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [2, 1]})
    im = axes[0].imshow(piv.values, aspect="auto", cmap="RdYlGn", vmin=.75, vmax=1.35)
    axes[0].set_yticks(range(len(piv.index)))
    axes[0].set_yticklabels(piv.index)
    axes[0].set_xticks(range(piv.shape[1]))
    axes[0].set_xticklabels([d.strftime("%b %y") for d in piv.columns], rotation=45)
    axes[0].set_title("Recommended price multiplier (green = raise, red = discount)")
    fig.colorbar(im, ax=axes[0], fraction=.03)
    up = plan.groupby("place").apply(lambda g: 100 * (g.revenue_optimal.sum() / g.revenue_flat.sum() - 1),
                                     include_groups=False).sort_values()
    axes[1].barh(up.index, up.values, color="#2171b5")
    axes[1].set_xlabel("Modelled revenue uplift vs flat price (%)")
    axes[1].set_title("Expected uplift (model-based)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def aspects(am, path):
    am = am.drop(columns=[c for c in ["other"] if c in am.columns])
    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.imshow(am.values, cmap="RdYlGn", vmin=.4, vmax=.95, aspect="auto")
    ax.set_xticks(range(am.shape[1]))
    ax.set_xticklabels(am.columns, rotation=30)
    ax.set_yticks(range(am.shape[0]))
    ax.set_yticklabels(am.index)
    for i in range(am.shape[0]):
        for j in range(am.shape[1]):
            ax.text(j, i, f"{am.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("Aspect-based sentiment (probability of positive)")
    fig.colorbar(im, ax=ax, fraction=.04)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
