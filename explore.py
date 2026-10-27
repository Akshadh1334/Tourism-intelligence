"""Command-line explorer (no UI framework needed).

    python explore.py Goa
    python explore.py --recommend beach=9 nightlife=7 budget=5 --month 12

Run `python main.py` first - this script reads the files it writes to results/.
"""
import difflib
import sys
import pandas as pd
from src.config import DATA_DIR, FEATURES, PLACES, RESULTS_DIR
from src.recommender import MONTHS, Recommender


def load():
    need = [RESULTS_DIR / n for n in ("forecast_next_12_months.csv", "place_sentiment.csv",
                                      "aspect_sentiment.csv", "destination_segments.csv",
                                      "pricing_plan.csv")]
    if not all(p.exists() for p in need):
        sys.exit("Results not found - run `python main.py` first.")
    df = pd.read_csv(DATA_DIR / "tourism_monthly.csv", parse_dates=["date"])
    fc = pd.read_csv(RESULTS_DIR / "forecast_next_12_months.csv", parse_dates=["date"])
    sent = pd.read_csv(RESULTS_DIR / "place_sentiment.csv")
    asp = pd.read_csv(RESULTS_DIR / "aspect_sentiment.csv", index_col=0)
    seg = pd.read_csv(RESULTS_DIR / "destination_segments.csv")
    plan = pd.read_csv(RESULTS_DIR / "pricing_plan.csv", parse_dates=["date"])
    return df, fc, sent, asp, seg, plan


def explore(query):
    df, fc, sent, asp, seg, plan = load()
    hit = difflib.get_close_matches(query.strip().title(), PLACES, n=1, cutoff=0.5)
    if not hit:
        print(f"'{query}' not found. Available: {', '.join(PLACES)}")
        return
    place = hit[0]
    rec = Recommender(df, fc, sent)
    f = fc[fc.place == place].sort_values("date")
    bt = rec.best_time(place)
    pl = plan[plan.place == place].sort_values("date")

    print(f"\n{'=' * 64}\n  {place.upper()}  |  segment: {seg.loc[seg.place == place, 'segment'].iloc[0]}\n{'=' * 64}")

    print("\n1) FORECAST - next 12 months (tourists, 80% interval)")
    print(f"   {'Month':<9}{'Forecast':>10}{'Low':>10}{'High':>10}   Hotel price action")
    for (_, r), (_, p) in zip(f.iterrows(), pl.iterrows()):
        print(f"   {r.date:%b %Y}  {r.forecast:>10,.0f}{r.lower:>10,.0f}{r.upper:>10,.0f}   "
              f"{p.action} (x{p.multiplier:.2f})")

    print("\n2) BEST TIME TO VISIT")
    for _, r in bt.head(3).iterrows():
        print(f"   * {r.month:<4} score {int(r.score)}/100 | weather {r.weather} | crowd {r.crowd} "
              f"| ~{r.expected_tourists:,} tourists")
    avoid = bt[bt.weather == "Avoid"].sort_values("month_no")["month"].tolist()
    quiet = bt[bt.weather != "Avoid"].sort_values("expected_tourists").iloc[0]
    print(f"   Budget pick (quietest with acceptable weather): {quiet.month}")
    if avoid:
        print(f"   Avoid: {', '.join(avoid)} (poor weather)")

    print("\n3) TRAVELLER REVIEWS")
    a = asp.loc[place].drop(labels=["other"], errors="ignore").sort_values()
    s = sent.set_index("place").loc[place, "sentiment"]
    print(f"   Overall positivity {s:.0%} | strongest: {a.index[-1]} ({a.iloc[-1]:.0%}) "
          f"| weakest: {a.index[0]} ({a.iloc[0]:.0%})")

    print("\n4) SIMILAR PLACES")
    for other, score in rec.similar_places(place).items():
        b = rec.best_time(other).iloc[0]
        print(f"   - {other:<12} match {score:.0%} | best time {b.month}")


def recommend(args):
    prefs, month = {}, 12
    it = iter(args)
    for a in it:
        if a == "--month":
            month = int(next(it))
        elif "=" in a:
            k, v = a.split("=")
            if k not in FEATURES:
                sys.exit(f"Unknown interest '{k}'. Use: {', '.join(FEATURES)}")
            prefs[k] = float(v)
    df, fc, sent, *_ = load()
    r = Recommender(df, fc, sent).recommend(prefs, month, top_n=5)
    print(f"\nTop picks for {MONTHS[month - 1]} with interests {prefs}:")
    for i, row in r.iterrows():
        print(f"  {i + 1}. {row.place:<12} score {row.score:.2f}  - {row.why}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    if not argv:
        print("Available places:", ", ".join(PLACES))
        while True:
            q = input("\nEnter a tourist place (or 'quit'): ").strip()
            if q.lower() in ("quit", "exit", "q"):
                break
            explore(q)
    elif argv[0] == "--recommend":
        recommend(argv[1:])
    else:
        explore(" ".join(argv))
