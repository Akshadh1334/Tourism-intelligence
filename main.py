"""Run the whole Smart Tourism Intelligence pipeline end to end.

    python main.py            # use data/ if present, otherwise generate it
    python main.py --regen    # force regeneration of the synthetic data

Outputs (CSV, JSON, PNG) are written to ./results
"""
import json
import sys
import warnings
import pandas as pd

from src import anomaly, data_gen, forecasting, pricing, segmentation, sentiment, viz
from src.config import DATA_DIR, RESULTS_DIR
from src.recommender import Recommender

warnings.filterwarnings("ignore")


def banner(text):
    print("\n" + "=" * 72 + f"\n{text}\n" + "=" * 72)


def main(regen=False):
    DATA_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    # 1 ------------------------------------------------------------- data
    banner("1. DATA")
    f_data, f_events, f_rev = (DATA_DIR / n for n in
                               ("tourism_monthly.csv", "injected_events.csv", "reviews.csv"))
    if regen or not (f_data.exists() and f_events.exists() and f_rev.exists()):
        df, events = data_gen.generate()
        df.to_csv(f_data, index=False)
        events.to_csv(f_events, index=False)
        sentiment.generate_reviews().to_csv(f_rev, index=False)
        print("Generated synthetic dataset.")
    df = pd.read_csv(f_data, parse_dates=["date"])
    events = pd.read_csv(f_events, parse_dates=["date"])
    reviews = pd.read_csv(f_rev)
    print(f"{len(df):,} monthly rows | {df.place.nunique()} destinations | "
          f"{df.date.min():%b %Y} - {df.date.max():%b %Y} | {len(reviews):,} review sentences")

    summary = {}

    # 2 ------------------------------------------------------- anomalies
    banner("2. ANOMALY DETECTION (weather-aware STL + Isolation Forest)")
    ann = anomaly.annotate(df)
    ev = anomaly.evaluate(ann, events)
    print(ev.round(3).to_string(index=False))
    ann[ann.anomaly & ~ann.covid_regime][["place", "date", "arrivals", "fitted", "stl_z"]] \
        .round(2).to_csv(RESULTS_DIR / "anomalies_detected.csv", index=False)
    ev.to_csv(RESULTS_DIR / "anomaly_metrics.csv", index=False)
    summary["anomaly"] = ev.round(3).to_dict(orient="records")

    # 3 ---------------------------------------------------- forecasting
    banner("3. DEMAND FORECASTING (hybrid ensemble, rolling-origin backtest)")
    final, metrics, per_place, test, info = forecasting.backtest_and_forecast(df)
    print(metrics.round(4).to_string(index=False))
    print(f"\nEnsemble weights   : {({k: round(v, 3) for k, v in info['weights'].items()})}")
    print(f"80% interval cover : {info['coverage_test']:.1%} of test months (target 80%)")
    print(f"Test window        : {info['test_window'][0]} to {info['test_window'][1]}")
    metrics.to_csv(RESULTS_DIR / "forecast_metrics.csv", index=False)
    per_place.to_csv(RESULTS_DIR / "forecast_metrics_by_place.csv", index=False)
    keep = ["place", "date", "forecast", "lower", "upper", "hw", "gbm"]
    final[keep].round(0).to_csv(RESULTS_DIR / "forecast_next_12_months.csv", index=False)
    summary["forecast"] = dict(metrics=metrics.round(4).to_dict(orient="records"),
                               weights=info["weights"], coverage=info["coverage_test"])

    # 4 ------------------------------------------------------- pricing
    banner("4. DYNAMIC PRICING (elasticity estimation + revenue optimisation)")
    est = pricing.estimate_elasticity(df)
    plan = pricing.optimise_prices(df, final, est)
    mae = (est.elasticity - est.true_elasticity).abs().mean()
    total = 100 * (plan.revenue_optimal.sum() / plan.revenue_flat.sum() - 1)
    print(est[["place", "elasticity", "true_elasticity"]].round(2).to_string(index=False))
    print(f"\nMean abs. error of estimated elasticity: {mae:.3f}")
    print(f"Modelled revenue uplift vs flat pricing : {total:+.1f}%")
    est.round(3).to_csv(RESULTS_DIR / "price_elasticity.csv", index=False)
    plan.round(2).to_csv(RESULTS_DIR / "pricing_plan.csv", index=False)
    summary["pricing"] = dict(elasticity_mae=float(mae), revenue_uplift_pct=float(total))

    # 5 ------------------------------------------------------ sentiment
    banner("5. REVIEW NLP (TF-IDF + logistic regression, aspect scores)")
    place_sent, aspect_matrix, sm = sentiment.analyse(reviews)
    print({k: round(v, 3) if isinstance(v, float) else v for k, v in sm.items()})
    place_sent.round(3).to_csv(RESULTS_DIR / "place_sentiment.csv", index=False)
    aspect_matrix.round(3).to_csv(RESULTS_DIR / "aspect_sentiment.csv")
    summary["sentiment"] = {k: float(v) for k, v in sm.items()}

    # 6 ------------------------------------------------- segmentation
    banner("6. DESTINATION SEGMENTATION (K-Means + PCA)")
    feats = segmentation.build_features(df, est, place_sent)
    seg, sinfo = segmentation.cluster(feats)
    print(f"k = {sinfo['best_k']} (silhouette scores: "
          f"{ {k: round(v, 2) for k, v in sinfo['silhouette'].items()} })")
    print(seg[["place", "segment"]].sort_values("segment").to_string(index=False))
    seg.round(3).to_csv(RESULTS_DIR / "destination_segments.csv", index=False)
    summary["segmentation"] = dict(k=sinfo["best_k"], silhouette=sinfo["silhouette"])

    # 7 ------------------------------------------------ recommender
    banner("7. RECOMMENDER DEMO (beach + nightlife lover travelling in December)")
    rec = Recommender(df, final, place_sent)
    demo = rec.recommend({"beach": 9, "nightlife": 8, "budget": 5}, month=12)
    print(demo[["place", "score", "content", "weather", "crowd", "reviews", "why"]].round(2).to_string(index=False))

    # 8 --------------------------------------------------------- plots
    banner("8. FIGURES")
    viz.forecast_grid(df, final, RESULTS_DIR / "forecast_grid.png")
    viz.anomalies(ann, RESULTS_DIR / "anomalies.png")
    viz.model_comparison(metrics, RESULTS_DIR / "model_comparison.png")
    viz.segments(seg, RESULTS_DIR / "segments.png")
    viz.pricing(plan, RESULTS_DIR / "pricing.png")
    viz.aspects(aspect_matrix, RESULTS_DIR / "aspect_sentiment.png")
    print("Saved 6 figures to results/")

    with open(RESULTS_DIR / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print("\nDone. Try:  python explore.py Goa")


if __name__ == "__main__":
    main(regen="--regen" in sys.argv)
