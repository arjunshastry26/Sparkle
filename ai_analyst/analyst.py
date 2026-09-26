"""
analyst.py

The AI analyst, end to end:

    User question
         |
         v
    Intent identification      <- rule-based keyword matching (deterministic,
         |                         no LLM involved — this is the part that
         v                         MUST NOT hallucinate what the user asked)
    Relevant analytical function
         |
         v
    Structured result (plain dict of real numbers pulled from the gold tables)
         |
         v
    LLM explanation (if an API key is configured) OR a deterministic
    template (if not) — either way, the wording layer is never allowed to
    introduce a number that isn't already in the structured result.
         |
         v
    Answer

This module has no Spark dependency and no dependency on Streamlit — it's
plain pandas + (optionally) LangChain, so it can be unit tested directly
(see tests/) and reused from app.py or a notebook.
"""

from __future__ import annotations

import glob
import json
import os
import re
from datetime import datetime, timezone

import pandas as pd

from prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD_CSV_DIR = os.path.join(PROJECT_ROOT, "dbfs_local", "gold", "csv_export")
FORECAST_PATH = os.path.join(PROJECT_ROOT, "dbfs_local", "gold", "sales_forecast.csv")

MONTH_NAMES = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9, "october": 10,
    "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _resolve_csv(path: str) -> str | None:
    if os.path.isfile(path):
        return path
    matches = glob.glob(os.path.join(path, "part-*.csv"))
    return matches[0] if matches else None


def load_data() -> dict:
    """Loads all Gold tables the analyst can draw on. Any table that isn't
    available yet (e.g. forecast not generated) is simply absent from the
    dict — callers check for its presence rather than assuming it exists."""
    data = {}

    monthly_csv = _resolve_csv(os.path.join(GOLD_CSV_DIR, "gold_monthly_sales"))
    if monthly_csv:
        df = pd.read_csv(monthly_csv)
        df["month"] = pd.to_datetime(df["month"])
        data["monthly"] = df

    product_csv = _resolve_csv(os.path.join(GOLD_CSV_DIR, "gold_product_performance"))
    if product_csv:
        data["product"] = pd.read_csv(product_csv)

    regional_csv = _resolve_csv(os.path.join(GOLD_CSV_DIR, "gold_regional_performance"))
    if regional_csv:
        data["regional"] = pd.read_csv(regional_csv)

    if os.path.isfile(FORECAST_PATH):
        fdf = pd.read_csv(FORECAST_PATH)
        fdf["forecast_month"] = pd.to_datetime(fdf["forecast_month"])
        data["forecast"] = fdf

    return data


# ---------------------------------------------------------------------------
# Intent identification (deterministic — no LLM)
# ---------------------------------------------------------------------------

INTENT_RULES = [
    # (intent_name, [keyword patterns that must ALL appear (any-order) — first full match wins])
    ("forecast_growth_region", [r"region", r"(grow|grew|grown|growth|increase)", r"(forecast|predict|next month|expected)"]),
    ("forecast", [r"(forecast|predict|next month)"]),
    ("why_decrease", [r"why", r"(decrease|drop|decline|fell|fall)"]),
    # Region questions must be checked before "worst"/"best" product rules.
    ("region_performance", [r"region", r"(best|worst|highest|lowest|performance|perform)"]),
    ("declining_products", [r"(declin|worst|underperform|lowest.?perform)", r"product"]),
    ("top_products", [r"(top|best|highest|most|leading|perform)", r"product"]),
    ("category_growth", [r"category", r"(grow|grew|grown|growth|increase)"]),
    # a specific month/period reference should route to monthly_trend even if
    # the question also says "total revenue" — the period reference is more
    # specific than the metric name, so it's checked before total_revenue
    ("monthly_trend", [
        r"(trend|over time|by month|monthly|last month|this month|current month|"
        r"january|february|march|april|may|june|july|august|september|october|"
        r"november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)"
    ]),
    ("region_performance", [r"region"]),
    ("category_performance", [r"category"]),
    ("total_revenue", [r"(total|overall).*(revenue|sales)"]),
]


def identify_intent(question: str) -> str:
    q = question.lower()
    for intent, patterns in INTENT_RULES:
        if all(re.search(p, q) for p in patterns):
            return intent
    return "total_revenue"  # safe, always-answerable default


def extract_month_year(question: str, available_months: pd.Series) -> tuple[int, int] | None:
    """Looks for an explicit month name (optionally with a year) in the
    question. If a month is named without a year, resolves to the most
    recent year that month appears in the dataset. Returns None if no month
    name is found (caller should then default to "the most recent month")."""
    q = question.lower()
    found_month = None
    for name, num in MONTH_NAMES.items():
        if re.search(rf"\b{name}\b", q):
            found_month = num
            break
    if found_month is None:
        return None

    year_match = re.search(r"\b(20\d{2})\b", q)
    if year_match:
        return found_month, int(year_match.group(1))

    candidates = sorted({d.year for d in available_months if d.month == found_month}, reverse=True)
    if not candidates:
        return None
    return found_month, candidates[0]


# ---------------------------------------------------------------------------
# Analytical functions — each returns a small, JSON-serializable dict of
# REAL numbers pulled straight from the gold tables. Nothing here is
# generated by an LLM.
# ---------------------------------------------------------------------------

def get_total_revenue(data: dict) -> dict:
    monthly = data.get("monthly")
    if monthly is None:
        return {"available": False, "reason": "monthly sales data not loaded"}
    return {
        "available": True,
        "metric": "total_revenue_all_time",
        "total_revenue": round(float(monthly["revenue"].sum()), 2),
        "total_orders": int(monthly["orders"].sum()),
        "date_range": [str(monthly["month"].min().date()), str(monthly["month"].max().date())],
    }


def get_monthly_sales(data: dict, month_year: tuple | None) -> dict:
    monthly = data.get("monthly")
    if monthly is None:
        return {"available": False, "reason": "monthly sales data not loaded"}

    if month_year is None:
        target = monthly["month"].max()
    else:
        m, y = month_year
        candidates = monthly[(monthly["month"].dt.month == m) & (monthly["month"].dt.year == y)]
        if candidates.empty:
            return {"available": False, "reason": f"no data for {y}-{m:02d}"}
        target = candidates["month"].iloc[0]

    cur = monthly[monthly["month"] == target]
    prev_target = target - pd.offsets.MonthBegin(1)
    prev = monthly[monthly["month"] == prev_target]

    result = {
        "available": True,
        "metric": "monthly_sales",
        "month": str(target.date()),
        "revenue": round(float(cur["revenue"].sum()), 2),
        "orders": int(cur["orders"].sum()),
        "units_sold": int(cur["units_sold"].sum()),
    }
    if not prev.empty:
        prev_revenue = float(prev["revenue"].sum())
        result["previous_month"] = str(prev_target.date())
        result["previous_month_revenue"] = round(prev_revenue, 2)
        result["mom_change_pct"] = round((result["revenue"] - prev_revenue) / prev_revenue * 100, 2) if prev_revenue else None
    return result


def get_region_performance(data: dict, month_year: tuple | None = None) -> dict:
    monthly = data.get("monthly")
    if monthly is None:
        return {"available": False, "reason": "monthly sales data not loaded"}

    df = monthly
    scope = "all_time"
    if month_year is not None:
        m, y = month_year
        df = monthly[(monthly["month"].dt.month == m) & (monthly["month"].dt.year == y)]
        scope = f"{y}-{m:02d}"
        if df.empty:
            return {"available": False, "reason": f"no data for {scope}"}

    by_region = df.groupby("region")["revenue"].sum().sort_values(ascending=False)
    return {
        "available": True,
        "metric": "region_performance",
        "scope": scope,
        "best_region": by_region.index[0],
        "best_region_revenue": round(float(by_region.iloc[0]), 2),
        "worst_region": by_region.index[-1],
        "worst_region_revenue": round(float(by_region.iloc[-1]), 2),
        "all_regions": {k: round(float(v), 2) for k, v in by_region.items()},
    }


def get_category_performance(data: dict, month_year: tuple | None = None) -> dict:
    monthly = data.get("monthly")
    if monthly is None:
        return {"available": False, "reason": "monthly sales data not loaded"}

    df = monthly
    scope = "all_time"
    if month_year is not None:
        m, y = month_year
        df = monthly[(monthly["month"].dt.month == m) & (monthly["month"].dt.year == y)]
        scope = f"{y}-{m:02d}"
        if df.empty:
            return {"available": False, "reason": f"no data for {scope}"}

    by_cat = df.groupby("category")["revenue"].sum().sort_values(ascending=False)
    return {
        "available": True,
        "metric": "category_performance",
        "scope": scope,
        "top_category": by_cat.index[0],
        "top_category_revenue": round(float(by_cat.iloc[0]), 2),
        "all_categories": {k: round(float(v), 2) for k, v in by_cat.items()},
    }


def get_category_growth(data: dict) -> dict:
    """First full calendar year vs. most recent full calendar year, per
    category — avoids seasonal noise from comparing single months."""
    monthly = data.get("monthly")
    if monthly is None:
        return {"available": False, "reason": "monthly sales data not loaded"}

    yearly = monthly.groupby([monthly["month"].dt.year.rename("year"), "category"])["revenue"].sum().reset_index()
    first_year, last_year = yearly["year"].min(), yearly["year"].max()
    if first_year == last_year:
        return {"available": False, "reason": "not enough years of data to compute growth"}

    first = yearly[yearly["year"] == first_year].set_index("category")["revenue"]
    last = yearly[yearly["year"] == last_year].set_index("category")["revenue"]
    growth = ((last - first) / first * 100).round(2).sort_values(ascending=False)

    return {
        "available": True,
        "metric": "category_growth",
        "first_year": int(first_year),
        "last_year": int(last_year),
        "fastest_growing_category": growth.index[0],
        "fastest_growing_pct": float(growth.iloc[0]),
        "all_categories_growth_pct": {k: float(v) for k, v in growth.items()},
    }


def get_top_products(data: dict, n: int = 10) -> dict:
    product = data.get("product")
    if product is None:
        return {"available": False, "reason": "product performance data not loaded"}
    top = product.sort_values("total_revenue", ascending=False).head(n)
    return {
        "available": True,
        "metric": "top_products",
        "n": n,
        "products": top[["product_id", "product_name", "category", "total_revenue", "units_sold"]].round(2).to_dict("records"),
    }


def get_declining_products(data: dict, n: int = 10) -> dict:
    product = data.get("product")
    if product is None:
        return {"available": False, "reason": "product performance data not loaded"}
    declining = product[product["sales_growth"].notna() & (product["sales_growth"] < 0)]
    declining = declining.sort_values("sales_growth", ascending=True).head(n)
    if declining.empty:
        return {"available": True, "metric": "declining_products", "n": 0, "products": [],
                "note": "No products currently show negative month-over-month growth."}
    return {
        "available": True,
        "metric": "declining_products",
        "n": len(declining),
        "products": declining[["product_id", "product_name", "category", "total_revenue", "sales_growth"]].round(4).to_dict("records"),
    }


def get_why_decrease(data: dict, month_year: tuple | None) -> dict:
    """Root-cause style breakdown: current vs. previous month, at the
    region x category grain, so the LLM has the SAME data a human analyst
    would pull to answer "why did sales decrease"."""
    monthly = data.get("monthly")
    if monthly is None:
        return {"available": False, "reason": "monthly sales data not loaded"}

    if month_year is None:
        target = monthly["month"].max()
    else:
        m, y = month_year
        candidates = monthly[(monthly["month"].dt.month == m) & (monthly["month"].dt.year == y)]
        if candidates.empty:
            return {"available": False, "reason": f"no data for {y}-{m:02d}"}
        target = candidates["month"].iloc[0]

    prev_target = target - pd.offsets.MonthBegin(1)
    cur = monthly[monthly["month"] == target]
    prev = monthly[monthly["month"] == prev_target]
    if cur.empty or prev.empty:
        return {"available": False, "reason": f"missing current or prior month data around {target.date()}"}

    total_cur, total_prev = float(cur["revenue"].sum()), float(prev["revenue"].sum())
    total_change_pct = round((total_cur - total_prev) / total_prev * 100, 2) if total_prev else None

    seg_cur = cur.groupby(["region", "category"])["revenue"].sum()
    seg_prev = prev.groupby(["region", "category"])["revenue"].sum()
    seg = pd.concat([seg_cur.rename("current"), seg_prev.rename("previous")], axis=1).fillna(0)
    seg["change_abs"] = seg["current"] - seg["previous"]
    seg["change_pct"] = ((seg["current"] - seg["previous"]) / seg["previous"].replace(0, pd.NA) * 100).round(2)
    seg = seg.sort_values("change_abs")

    top_decliners = seg.head(3).reset_index()
    top_decliners_list = [
        {
            "region": r["region"], "category": r["category"],
            "previous_revenue": round(float(r["previous"]), 2),
            "current_revenue": round(float(r["current"]), 2),
            "change_abs": round(float(r["change_abs"]), 2),
            "change_pct": float(r["change_pct"]) if pd.notna(r["change_pct"]) else None,
        }
        for _, r in top_decliners.iterrows()
    ]

    return {
        "available": True,
        "metric": "why_decrease",
        "month": str(target.date()),
        "previous_month": str(prev_target.date()),
        "total_revenue_current": round(total_cur, 2),
        "total_revenue_previous": round(total_prev, 2),
        "total_change_pct": total_change_pct,
        "direction": "decrease" if total_change_pct is not None and total_change_pct < 0 else "increase_or_flat",
        "largest_declining_segments": top_decliners_list,
    }


def get_forecast(data: dict, region: str | None = None, category: str | None = None) -> dict:
    forecast = data.get("forecast")
    if forecast is None:
        return {"available": False, "reason": "forecast not yet generated — run ml/predict.py first"}

    df = forecast
    if region:
        df = df[df["region"].str.lower() == region.lower()]
    if category:
        df = df[df["category"].str.lower() == category.lower()]
    if df.empty:
        return {"available": False, "reason": "no forecast rows matched the requested region/category"}

    total_predicted = float(df["predicted_revenue"].sum())
    return {
        "available": True,
        "metric": "forecast",
        "forecast_month": str(df["forecast_month"].iloc[0].date()),
        "model_version": df["model_version"].iloc[0],
        "total_predicted_revenue": round(total_predicted, 2),
        "breakdown": df[["region", "category", "predicted_revenue", "lower_bound", "upper_bound"]].round(2).to_dict("records"),
    }


def get_forecast_growth_region(data: dict) -> dict:
    """Compares each region's forecasted next-month revenue to its most
    recent actual month, to answer "which region is expected to grow?"."""
    forecast, monthly = data.get("forecast"), data.get("monthly")
    if forecast is None:
        return {"available": False, "reason": "forecast not yet generated — run ml/predict.py first"}
    if monthly is None:
        return {"available": False, "reason": "monthly sales data not loaded"}

    latest_month = monthly["month"].max()
    latest_actual = monthly[monthly["month"] == latest_month].groupby("region")["revenue"].sum()
    forecast_total = forecast.groupby("region")["predicted_revenue"].sum()

    comparison = pd.concat([latest_actual.rename("latest_actual"), forecast_total.rename("forecast")], axis=1).dropna()
    comparison["growth_pct"] = ((comparison["forecast"] - comparison["latest_actual"]) / comparison["latest_actual"] * 100).round(2)
    comparison = comparison.sort_values("growth_pct", ascending=False)

    return {
        "available": True,
        "metric": "forecast_growth_by_region",
        "latest_actual_month": str(latest_month.date()),
        "forecast_month": str(forecast["forecast_month"].iloc[0].date()),
        "fastest_growing_region": comparison.index[0],
        "fastest_growing_region_growth_pct": float(comparison["growth_pct"].iloc[0]),
        "all_regions": comparison.round(2).reset_index().rename(columns={"index": "region"}).to_dict("records"),
    }


# ---------------------------------------------------------------------------
# LLM explanation (optional) + deterministic fallback
# ---------------------------------------------------------------------------

LAST_LLM_ERROR = None


def explain_with_llm(question: str, structured_result: dict) -> str | None:
    global LAST_LLM_ERROR
    LAST_LLM_ERROR = None
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None

    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate

        llm = ChatOpenAI(
            model=os.environ.get("LLM_MODEL", "openai/gpt-oss-20b"),
            base_url=os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
            api_key=api_key,
            temperature=0.2,
            max_tokens=300,
            timeout=15,
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("user", USER_PROMPT_TEMPLATE),
        ])
        chain = prompt | llm
        response = chain.invoke({
            "question": question,
            "structured_result_json": json.dumps(structured_result, indent=2, default=str),
        })
        return response.content.strip()
    except Exception as e:
        LAST_LLM_ERROR = str(e)
        print(f"[ai_analyst] LLM explanation failed, using deterministic fallback: {e}")
        return None


def format_deterministic_answer(intent: str, result: dict) -> str:
    """A plain-language template for every intent. Used whenever no LLM API
    key is configured, or the LLM call fails — the analyst always produces
    a correct answer, just a more mechanically-worded one without an LLM."""
    if not result.get("available"):
        return f"I can't answer that from the current dataset: {result.get('reason', 'data not available')}."

    m = result["metric"]

    if m == "total_revenue_all_time":
        return (f"Total revenue from {result['date_range'][0]} to {result['date_range'][1]} was "
                f"${result['total_revenue']:,.2f} across {result['total_orders']:,} orders.")

    if m == "monthly_sales":
        base = f"Revenue in {result['month']} was ${result['revenue']:,.2f} across {result['orders']:,} orders."
        if "mom_change_pct" in result and result["mom_change_pct"] is not None:
            direction = "up" if result["mom_change_pct"] >= 0 else "down"
            base += f" That's {direction} {abs(result['mom_change_pct']):.1f}% from {result['previous_month']} (${result['previous_month_revenue']:,.2f})."
        return base

    if m == "region_performance":
        return (f"For {result['scope']}, {result['best_region']} had the highest revenue "
                f"(${result['best_region_revenue']:,.2f}), and {result['worst_region']} had the lowest "
                f"(${result['worst_region_revenue']:,.2f}).")

    if m == "category_performance":
        return (f"For {result['scope']}, {result['top_category']} led with "
                f"${result['top_category_revenue']:,.2f} in revenue.")

    if m == "category_growth":
        return (f"Between {result['first_year']} and {result['last_year']}, {result['fastest_growing_category']} "
                f"grew the most, at {result['fastest_growing_pct']:+.1f}%.")

    if m == "top_products":
        lines = [f"{i+1}. {p['product_name']} ({p['category']}) — ${p['total_revenue']:,.2f}"
                 for i, p in enumerate(result["products"])]
        return f"Top {result['n']} products by revenue:\n" + "\n".join(lines)

    if m == "declining_products":
        if result["n"] == 0:
            return result.get("note", "No declining products found.")
        lines = [f"{p['product_name']} ({p['category']}) — {p['sales_growth']*100:+.1f}% month-over-month"
                 for p in result["products"]]
        return f"{result['n']} products with declining sales:\n" + "\n".join(lines)

    if m == "why_decrease":
        direction = "decreased" if result["direction"] == "decrease" else "changed"
        base = (f"Total revenue {direction} {result['total_change_pct']:+.1f}% in {result['month']} "
                f"(${result['total_revenue_current']:,.2f}) versus {result['previous_month']} "
                f"(${result['total_revenue_previous']:,.2f}).")
        if result["largest_declining_segments"]:
            seg = result["largest_declining_segments"][0]
            base += (f" The largest contributor was {seg['category']} in the {seg['region']} region, "
                     f"which fell {seg['change_pct']:+.1f}% (${seg['change_abs']:,.2f}).")
        return base

    if m == "forecast":
        return (f"Predicted revenue for {result['forecast_month']} is ${result['total_predicted_revenue']:,.2f} "
                f"(model: {result['model_version']}).")

    if m == "forecast_growth_by_region":
        pct = result["fastest_growing_region_growth_pct"]
        region = result["fastest_growing_region"]
        if pct >= 0:
            return (f"{region} is forecast to grow the most next month ({result['forecast_month']}), "
                    f"at {pct:+.1f}% versus {result['latest_actual_month']}.")
        all_negative = all(r["growth_pct"] < 0 for r in result["all_regions"])
        if all_negative:
            return (f"Every region is forecast to decline next month ({result['forecast_month']}) versus "
                    f"{result['latest_actual_month']} — likely the normal post-holiday seasonal dip, not a "
                    f"business problem. {region} is forecast to decline the least, at {pct:+.1f}%.")
        return (f"{region} has the highest forecast growth among regions ({pct:+.1f}%) for "
                f"{result['forecast_month']} versus {result['latest_actual_month']}, though that's still a decline.")

    return "I calculated a result but don't have a template to phrase it — check the raw metrics."


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def answer_question(question: str, data: dict | None = None) -> dict:
    if data is None:
        data = load_data()

    intent = identify_intent(question)
    month_year = None
    if "monthly" in data:
        month_year = extract_month_year(question, data["monthly"]["month"])

    if intent == "total_revenue":
        result = get_total_revenue(data)
    elif intent == "monthly_trend":
        result = get_monthly_sales(data, month_year)
    elif intent == "region_performance":
        result = get_region_performance(data, month_year)
    elif intent == "category_performance":
        result = get_category_performance(data, month_year)
    elif intent == "category_growth":
        result = get_category_growth(data)
    elif intent == "top_products":
        result = get_top_products(data)
    elif intent == "declining_products":
        result = get_declining_products(data)
    elif intent == "why_decrease":
        result = get_why_decrease(data, month_year)
    elif intent == "forecast":
        result = get_forecast(data)
    elif intent == "forecast_growth_region":
        result = get_forecast_growth_region(data)
    else:
        result = get_total_revenue(data)

    llm_answer = explain_with_llm(question, result)
    used_llm = llm_answer is not None
    answer = llm_answer if used_llm else format_deterministic_answer(intent, result)

    return {
        "question": question,
        "intent": intent,
        "answer": answer,
        "used_llm": used_llm,
        "llm_error": LAST_LLM_ERROR,
        "structured_result": result,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


if __name__ == "__main__":
    data = load_data()
    demo_questions = [
        "What was our total revenue last month?",
        "Which region had the highest sales?",
        "Which category grew the most?",
        "Why did sales decrease in July 2024?",
        "What are the top 10 products?",
        "Which products are declining?",
        "What is the predicted sales for next month?",
        "Which region is expected to grow next month?",
    ]
    for q in demo_questions:
        r = answer_question(q, data)
        print(f"\nQ: {q}")
        print(f"[intent: {r['intent']}, used_llm: {r['used_llm']}]")
        print(r["answer"])
