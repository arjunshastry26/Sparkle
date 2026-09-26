"""
generate_data.py

Generates a realistic synthetic retail sales transaction dataset.

Why synthetic instead of a real download:
- Fully reproducible (fixed random seed)
- We can deliberately bake in patterns the rest of the project needs to
  demonstrate: seasonality, regional differences, category differences,
  growing/declining products, and one clear "anomaly" event that the
  AI analyst can later explain with real numbers.

Output:
    data/raw/retail_sales_raw.csv       (full dataset, ~100k-150k rows)
    data/sample/retail_sales_sample.csv (5,000-row sample for quick preview)
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 42
START_MONTH = pd.Timestamp("2022-01-01")
END_MONTH = pd.Timestamp("2025-12-01")          # inclusive, 48 months total
BASE_ORDERS_PER_MONTH = 2500                     # scaled by seasonality/growth
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"

rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# Reference data: regions, cities, product catalog
# ---------------------------------------------------------------------------

REGIONS = {
    # region: (relative order weight, monthly growth rate, cities)
    "West":    {"weight": 1.30, "growth": 0.010, "cities": ["Los Angeles", "San Francisco", "Seattle", "Denver"]},
    "East":    {"weight": 1.20, "growth": 0.008, "cities": ["New York", "Boston", "Philadelphia", "Washington DC"]},
    "Central": {"weight": 0.90, "growth": 0.006, "cities": ["Chicago", "Dallas", "Minneapolis", "St. Louis"]},
    "South":   {"weight": 0.80, "growth": 0.002, "cities": ["Atlanta", "Miami", "Houston", "Charlotte"]},
}

# category -> (sub_categories, price_range, base monthly growth rate)
CATEGORY_TREE = {
    "Technology": {
        "growth": 0.014,
        "sub_categories": {
            "Phones":      (150, 700),
            "Machines":    (300, 1100),
            "Copiers":     (250, 900),
            "Accessories": (15, 160),
        },
    },
    "Furniture": {
        "growth": 0.004,
        "sub_categories": {
            "Chairs":      (80, 400),
            "Tables":      (150, 700),
            "Bookcases":   (100, 500),
            "Furnishings": (20, 200),
        },
    },
    "Office Supplies": {
        "growth": 0.006,
        "sub_categories": {
            "Paper":     (5, 40),
            "Binders":   (3, 30),
            "Storage":   (10, 120),
            "Art":       (5, 50),
            "Labels":    (3, 20),
            "Envelopes": (3, 25),
            "Fasteners": (2, 15),
            "Appliances": (30, 300),
            "Supplies":  (5, 60),
        },
    },
}

ADJECTIVES = ["Standard", "Deluxe", "Premium", "Compact", "Executive", "Classic",
              "Pro", "Basic", "Heavy-Duty", "Portable", "Wireless", "Ergonomic"]


def _singularize(word: str) -> str:
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith("s"):
        return word[:-1]
    return word


def build_product_catalog(n_products: int = 70) -> pd.DataFrame:
    """Builds a fixed product catalog with a price and a long-run trend per product.

    ~15% of products are deliberately 'declining' and ~10% are deliberately
    'growing' faster than their category, on top of the category-level trend.
    This is what gold_product_performance.sales_growth is meant to surface.
    """
    rows = []
    cats = list(CATEGORY_TREE.keys())
    weights = [0.28, 0.30, 0.42]  # Technology, Furniture, Office Supplies mix

    for i in range(n_products):
        category = rng.choice(cats, p=weights)
        sub_cats = list(CATEGORY_TREE[category]["sub_categories"].keys())
        sub_category = rng.choice(sub_cats)
        lo, hi = CATEGORY_TREE[category]["sub_categories"][sub_category]
        base_price = round(rng.uniform(lo, hi), 2)

        # per-product trend on top of category trend
        r = rng.random()
        if r < 0.15:
            trend_adjust = rng.uniform(-0.025, -0.012)   # declining product
        elif r < 0.25:
            trend_adjust = rng.uniform(0.012, 0.028)     # rising star
        else:
            trend_adjust = rng.uniform(-0.003, 0.003)    # roughly flat

        popularity = rng.lognormal(mean=0.0, sigma=0.6)  # skewed: few very popular products

        product_name = f"{rng.choice(ADJECTIVES)} {_singularize(sub_category)} {i+1:03d}"

        rows.append({
            "product_id": f"P{i+1:04d}",
            "product_name": product_name,
            "category": category,
            "sub_category": sub_category,
            "base_price": base_price,
            "trend_adjust": trend_adjust,
            "popularity": popularity,
        })

    return pd.DataFrame(rows)


def seasonal_multiplier(month: int) -> float:
    """1-indexed month -> demand multiplier. Holiday peak, January dip,
    mild back-to-school bump in Aug/Sep."""
    base = {
        1: 0.78, 2: 0.85, 3: 0.92, 4: 0.95, 5: 0.98, 6: 1.00,
        7: 0.97, 8: 1.05, 9: 1.08, 10: 1.10, 11: 1.35, 12: 1.55,
    }
    return base[month]


def months_between(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    return list(pd.date_range(start, end, freq="MS"))


# ---------------------------------------------------------------------------
# Anomaly definition (deliberately grounded so the AI analyst has a real,
# data-backed answer to "why did sales decrease?")
# ---------------------------------------------------------------------------
ANOMALY_MONTH = pd.Timestamp("2024-07-01")
ANOMALY_REGION = "South"
ANOMALY_CATEGORY = "Furniture"
ANOMALY_ORDER_MULTIPLIER = 0.40   # 60% fewer Furniture orders in South that month

# a short-lived viral spike, unrelated to the anomaly above
SPIKE_MONTHS = [pd.Timestamp("2023-11-01"), pd.Timestamp("2023-12-01")]
SPIKE_MULTIPLIER = 4.0


def generate_orders() -> pd.DataFrame:
    products = build_product_catalog()
    region_names = list(REGIONS.keys())
    region_weights = np.array([REGIONS[r]["weight"] for r in region_names])
    region_weights = region_weights / region_weights.sum()

    # pick 1 product per spike event (a viral gift-season item)
    spike_product_id = products.sample(1, random_state=SEED)["product_id"].iloc[0]

    all_rows = []
    order_counter = 1
    months = months_between(START_MONTH, END_MONTH)
    total_months = len(months)

    for m_idx, month_start in enumerate(months):
        month_num = month_start.month
        year = month_start.year

        # overall slow business growth over the 4 years
        overall_growth = (1.0 + 0.006) ** m_idx
        seasonal = seasonal_multiplier(month_num)
        n_orders_month = int(BASE_ORDERS_PER_MONTH * seasonal * overall_growth * rng.uniform(0.95, 1.05))

        # --- region weighting for this month (with region growth + anomaly) ---
        month_region_weights = region_weights.copy()
        for i, r in enumerate(region_names):
            month_region_weights[i] *= (1.0 + REGIONS[r]["growth"]) ** m_idx
        month_region_weights = month_region_weights / month_region_weights.sum()

        # sample region per order
        regions_sampled = rng.choice(region_names, size=n_orders_month, p=month_region_weights)

        # --- product weighting for this month (category + product trend) ---
        prod_weights = products["popularity"].to_numpy().copy()
        cat_growth = products["category"].map(lambda c: CATEGORY_TREE[c]["growth"]).to_numpy()
        trend_adjust = products["trend_adjust"].to_numpy()
        monthly_growth_factor = (1.0 + cat_growth + trend_adjust) ** m_idx
        # Bound the compounding growth/decline so a single lucky product
        # (high popularity + rising trend + high-growth category) can't
        # snowball into an unrealistic winner-take-all share of demand.
        # Even the most extreme "rising star" tops out at 3x its starting
        # relative demand; the most extreme decliner bottoms out at 30%.
        monthly_growth_factor = np.clip(monthly_growth_factor, 0.3, 3.0)
        prod_weights = prod_weights * monthly_growth_factor
        prod_weights = np.clip(prod_weights, 1e-6, None)

        if month_start in SPIKE_MONTHS:
            spike_idx = products.index[products["product_id"] == spike_product_id][0]
            prod_weights[spike_idx] *= SPIKE_MULTIPLIER

        prod_probs = prod_weights / prod_weights.sum()
        product_idx_sampled = rng.choice(products.index.to_numpy(), size=n_orders_month, p=prod_probs)

        # apply the anomaly: for South region + Furniture category in the anomaly
        # month, drop most of those orders (simulates a supply-chain disruption)
        is_anomaly_month = (month_start == ANOMALY_MONTH)

        for i in range(n_orders_month):
            region = regions_sampled[i]
            prod_row = products.loc[product_idx_sampled[i]]

            if is_anomaly_month and region == ANOMALY_REGION and prod_row["category"] == ANOMALY_CATEGORY:
                if rng.random() > ANOMALY_ORDER_MULTIPLIER:
                    continue  # this order simply didn't happen (lost sales)

            # a few random days into the month
            day = rng.integers(1, 29)  # keep within all months safely
            order_date = month_start + pd.Timedelta(days=int(day) - 1)

            price_growth = (1.0 + trend_adjust[product_idx_sampled[i]] * 0.5) ** m_idx
            unit_price = round(prod_row["base_price"] * price_growth * rng.uniform(0.95, 1.05), 2)
            unit_price = max(unit_price, 1.0)

            quantity = int(rng.poisson(lam=2.2) + 1)
            quantity = min(quantity, 14)

            discount_roll = rng.random()
            if discount_roll < 0.55:
                discount = 0.0
            elif discount_roll < 0.75:
                discount = 0.10
            elif discount_roll < 0.88:
                discount = 0.15
            elif discount_roll < 0.96:
                discount = 0.20
            else:
                discount = 0.30

            revenue = round(quantity * unit_price, 2)
            city = rng.choice(REGIONS[region]["cities"])

            all_rows.append({
                "order_id": f"ORD-{order_counter:07d}",
                "order_date": order_date.strftime("%Y-%m-%d"),
                "product_id": prod_row["product_id"],
                "product_name": prod_row["product_name"],
                "category": prod_row["category"],
                "sub_category": prod_row["sub_category"],
                "region": region,
                "city": city,
                "quantity": quantity,
                "unit_price": unit_price,
                "discount": discount,
                "revenue": revenue,
            })
            order_counter += 1

        if (m_idx + 1) % 12 == 0 or (m_idx + 1) == total_months:
            print(f"  generated through {month_start.strftime('%Y-%m')} "
                  f"({m_idx + 1}/{total_months} months, {len(all_rows):,} rows so far)")

    df = pd.DataFrame(all_rows)

    # inject a small amount of realistic messiness for the cleaning phase to handle
    df = inject_messiness(df)

    return df, products


def inject_messiness(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a controlled amount of dirty data: a few duplicate orders,
    a few nulls, a couple of inconsistent-case region strings, and a
    couple of invalid numeric rows. This is what the Silver layer's
    cleaning logic is written to catch."""
    df = df.copy()
    n = len(df)

    dup_idx = rng.choice(df.index, size=max(1, int(n * 0.002)), replace=False)
    dup_rows = df.loc[dup_idx]
    df = pd.concat([df, dup_rows], ignore_index=True)

    null_idx = rng.choice(df.index, size=max(1, int(n * 0.001)), replace=False)
    df.loc[null_idx, "discount"] = np.nan

    case_idx = rng.choice(df.index, size=max(1, int(n * 0.01)), replace=False)
    df.loc[case_idx, "region"] = df.loc[case_idx, "region"].str.lower()

    bad_idx = rng.choice(df.index, size=max(1, int(n * 0.0005)), replace=False)
    df.loc[bad_idx, "quantity"] = -1

    return df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)


def main():
    print(f"Generating retail sales data from {START_MONTH.date()} to {END_MONTH.date()}...")
    df, products = generate_orders()

    OUTPUT_DIR.joinpath("raw").mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.joinpath("sample").mkdir(parents=True, exist_ok=True)

    raw_path = OUTPUT_DIR / "raw" / "retail_sales_raw.csv"
    df.to_csv(raw_path, index=False)

    sample_path = OUTPUT_DIR / "sample" / "retail_sales_sample.csv"
    df.sample(n=min(5000, len(df)), random_state=SEED).sort_values("order_date").to_csv(sample_path, index=False)

    catalog_path = OUTPUT_DIR / "sample" / "product_catalog_reference.csv"
    products.to_csv(catalog_path, index=False)

    print(f"\nDone.")
    print(f"  Total rows: {len(df):,}")
    print(f"  Date range: {df['order_date'].min()} to {df['order_date'].max()}")
    print(f"  Raw file:    {raw_path}")
    print(f"  Sample file: {sample_path}")
    print(f"  (product_catalog_reference.csv is a generator-internal reference, "
          f"not part of the raw feed)")


if __name__ == "__main__":
    main()
