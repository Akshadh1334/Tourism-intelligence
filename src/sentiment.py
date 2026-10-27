"""Traveller-review NLP: sentence-level sentiment + aspect-based scores.

Reviews are synthetic (template sentences with per-destination positivity and
a few percent label noise), so absolute accuracy is optimistic. The pipeline
itself - TF-IDF + logistic regression, keyword aspect tagging, aggregation to
destination x aspect scores - carries over unchanged to real review text
(e.g. scraped Google/TripAdvisor reviews with a `place` and `text` column).
"""
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from .config import PLACES, SEED

TEMPLATES = {
    "hotel": (
        ["Our hotel room was spacious, clean and comfortable.", "Staff were warm, helpful and the stay was wonderful.",
         "Great hotel with excellent amenities and a lovely view.", "Check-in was smooth and the room exceeded expectations.",
         "Very comfortable beds and friendly hosts at the hotel."],
        ["The hotel room was small, dirty and noisy.", "Staff were rude and the check-in took forever.",
         "Hotel was overpriced for the poor facilities.", "The room smelled damp and the AC did not work.",
         "Booking details did not match and the stay was disappointing."]),
    "food": (
        ["The local food was absolutely delicious and fresh.", "Loved the street food, amazing flavours everywhere.",
         "Restaurants served tasty meals at fair prices.", "Great variety of food, we enjoyed every meal.",
         "The seafood and regional dishes were superb."],
        ["The food was bland and overpriced.", "Got a stomach upset after a meal at a roadside stall.",
         "Restaurants were mediocre and the service was slow.", "Very limited food options and poor quality.",
         "Meals tasted stale and not worth the money."]),
    "cleanliness": (
        ["The streets and beaches were clean and well maintained.", "Public areas were tidy and hygienic.",
         "Impressed by the cleanliness of the main attractions.", "Washrooms were clean and well kept.",
         "Well maintained surroundings with no litter anywhere."],
        ["There was garbage piled up near the main attractions.", "Public toilets were filthy and unhygienic.",
         "Litter everywhere, the place felt neglected.", "Very dirty surroundings and a terrible smell.",
         "Poor waste management spoiled the visit."]),
    "safety": (
        ["I felt very safe walking around even at night.", "Police and guides were helpful and the area felt secure.",
         "Safe and welcoming for families and solo travellers.", "Well lit streets and trustworthy locals.",
         "Never felt unsafe during the entire trip."],
        ["Touts and scammers constantly harassed tourists.", "Did not feel safe after dark, poorly lit lanes.",
         "Pickpocketing was common in the market areas.", "Overcharging and cheating by local drivers.",
         "Felt uneasy because of aggressive vendors."]),
    "transport": (
        ["Getting around was easy with reliable local transport.", "Roads were good and the cabs were affordable.",
         "Well connected by train and flights, smooth journey.", "Public transport was punctual and convenient.",
         "Easy to reach with comfortable travel options."],
        ["Roads were terrible and the journey was exhausting.", "Taxis refused meters and charged way too much.",
         "Poor connectivity, buses were late and overfull.", "Hard to reach with very limited transport options.",
         "Flights were delayed and local travel was chaotic."]),
    "crowd": (
        ["Peaceful and not too crowded, we could explore freely.", "Short queues and plenty of space at every spot.",
         "Calm atmosphere and easy access to the sights.", "Manageable crowds even on the weekend.",
         "Quiet mornings made the sightseeing relaxing."],
        ["Extremely crowded with endless queues everywhere.", "Overcrowded sights, we could barely move.",
         "Long waiting times and noisy mobs at every spot.", "Too many tourists ruined the experience.",
         "Packed streets and traffic jams all day."]),
}

# order matters: first matching aspect wins
ASPECT_RULES = [
    ("hotel", r"\b(hotel|room|check-in|staff|booking|beds)"),
    ("food", r"\b(food|meal|restaurant|dish|seafood|flavour)"),
    ("cleanliness", r"\b(clean|garbage|toilet|litter|dirty|hygien|waste|tidy|washroom|maintained)"),
    ("safety", r"\b(safe|unsafe|secure|police|scam|pickpocket|touts|harass|cheating|overcharging|uneasy|vendors)|\blit\b"),
    ("transport", r"\b(transport|roads|cab|taxi|taxis|train|flights|buses|connect|reach|travel|journey)"),
    ("crowd", r"\b(crowd|overcrowd|queue|tourists|mobs|packed|waiting|peaceful|quiet|calm|space)"),
]


def tag_aspect(text):
    low = text.lower()
    for name, pat in ASPECT_RULES:
        if re.search(pat, low):
            return name
    return "other"


def generate_reviews(n_reviews=200, sentences=3, seed=SEED, label_noise=0.06):
    rng = np.random.default_rng(seed)
    aspects = list(TEMPLATES)
    rows = []
    for place, p in PLACES.items():
        # each destination is better/worse at different aspects
        aspect_pos = {a: float(np.clip(p["pos"] + rng.normal(0, .12), .25, .95)) for a in aspects}
        for rid in range(n_reviews):
            for a in rng.choice(aspects, sentences, replace=False):
                positive = rng.random() < aspect_pos[a]
                text = rng.choice(TEMPLATES[a][0 if positive else 1])
                label = int(positive)
                if rng.random() < label_noise:      # annotation noise
                    label = 1 - label
                rows.append(dict(place=place, review_id=f"{place[:3].upper()}{rid:04d}",
                                 text=text, label=label))
    return pd.DataFrame(rows)


def analyse(reviews):
    """Train sentiment model, score every sentence, aggregate per place/aspect."""
    X_tr, X_te, y_tr, y_te = train_test_split(reviews["text"], reviews["label"], test_size=.2,
                                              stratify=reviews["label"], random_state=SEED)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(vec.fit_transform(X_tr), y_tr)
    pred = clf.predict(vec.transform(X_te))
    metrics = dict(accuracy=accuracy_score(y_te, pred), f1=f1_score(y_te, pred),
                   n_train=len(X_tr), n_test=len(X_te))

    scored = reviews.copy()
    scored["p_pos"] = clf.predict_proba(vec.transform(scored["text"]))[:, 1]
    scored["aspect"] = scored["text"].map(tag_aspect)
    place_score = scored.groupby("place")["p_pos"].mean().rename("sentiment").reset_index()
    aspect_matrix = scored.pivot_table(index="place", columns="aspect", values="p_pos", aggfunc="mean")
    metrics["aspect_tag_coverage"] = float((scored["aspect"] != "other").mean())
    return place_score, aspect_matrix, metrics
