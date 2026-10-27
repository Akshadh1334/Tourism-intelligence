"""Central configuration: paths, time window and destination profiles.

All destination parameters below are *assumed* profiles used to build the
synthetic dataset (see data_gen.py). Replace the CSVs in /data with real
statistics (e.g. India Tourism Statistics) and the rest of the pipeline
runs unchanged, as long as the column names are kept.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

START, END = "2015-01-01", "2026-09-01"   # monthly data window
SEED = 42
HORIZON = 12                               # months to forecast

FEATURES = ["beach", "hills", "heritage", "nightlife", "nature", "budget"]

# traits: 0-10 score for FEATURES | temp=(mean C, amplitude, hottest month)
# rain=(peak mm, wettest month)  | foreign = share of foreign tourists
# rooms = hotel rooms | adr = base average daily rate (INR)
# elasticity = TRUE price elasticity used only to simulate data
# pos = baseline share of positive reviews
PLACES = {
    "Goa":         dict(base=90_000, peak=12, traits=[10, 0, 4, 10, 4, 5], temp=(27, 3, 5),   rain=(600, 7),  foreign=.25, rooms=9000, adr=5500, elasticity=1.1, pos=.82),
    "Kerala":      dict(base=70_000, peak=1,  traits=[7, 3, 5, 3, 10, 6],  temp=(28, 2, 4),   rain=(500, 6),  foreign=.12, rooms=7000, adr=4800, elasticity=0.9, pos=.85),
    "Jaipur":      dict(base=60_000, peak=11, traits=[0, 0, 10, 5, 2, 7],  temp=(26, 9, 5),   rain=(250, 8),  foreign=.20, rooms=6500, adr=4200, elasticity=1.2, pos=.78),
    "Manali":      dict(base=50_000, peak=5,  traits=[0, 10, 2, 4, 9, 6],  temp=(13, 9, 6),   rain=(300, 7),  foreign=.06, rooms=4000, adr=3800, elasticity=1.3, pos=.80),
    "Chennai":     dict(base=55_000, peak=2,  traits=[7, 0, 7, 5, 2, 8],   temp=(29, 3, 5),   rain=(350, 11), foreign=.10, rooms=8000, adr=4500, elasticity=1.0, pos=.72),
    "Ooty":        dict(base=45_000, peak=5,  traits=[0, 9, 3, 2, 9, 7],   temp=(16, 3, 4),   rain=(200, 10), foreign=.03, rooms=3500, adr=3200, elasticity=1.4, pos=.83),
    "Varanasi":    dict(base=65_000, peak=11, traits=[0, 0, 10, 2, 2, 8],  temp=(26, 10, 5),  rain=(300, 8),  foreign=.15, rooms=6000, adr=3000, elasticity=1.5, pos=.74),
    "Pondicherry": dict(base=40_000, peak=1,  traits=[8, 0, 7, 6, 3, 6],   temp=(28, 3, 5),   rain=(300, 11), foreign=.22, rooms=3000, adr=4600, elasticity=1.0, pos=.84),
    "Darjeeling":  dict(base=38_000, peak=4,  traits=[0, 9, 5, 2, 9, 6],   temp=(14, 6, 6),   rain=(600, 7),  foreign=.10, rooms=2800, adr=3500, elasticity=1.3, pos=.81),
    "Andaman":     dict(base=35_000, peak=1,  traits=[10, 0, 3, 3, 9, 3],  temp=(28, 1.5, 4), rain=(400, 7),  foreign=.18, rooms=2500, adr=7000, elasticity=0.8, pos=.88),
    "Udaipur":     dict(base=48_000, peak=12, traits=[0, 2, 10, 5, 5, 5],  temp=(25, 8, 5),   rain=(300, 8),  foreign=.30, rooms=4500, adr=6000, elasticity=1.0, pos=.86),
    "Munnar":      dict(base=42_000, peak=12, traits=[0, 9, 2, 1, 10, 7],  temp=(17, 3, 4),   rain=(600, 7),  foreign=.08, rooms=3000, adr=4000, elasticity=1.2, pos=.85),
}
