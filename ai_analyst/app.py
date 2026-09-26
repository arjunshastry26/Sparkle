"""
app.py

Minimal Streamlit interface for the AI analyst. No session/chat history
complexity, no multi-page app — one question in, one answer out, with the
underlying numbers shown so the user can verify the answer themselves.

Run with:
    streamlit run ai_analyst/app.py
"""

import os
import sys

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"), override=True)

from analyst import load_data, answer_question  # noqa: E402

st.set_page_config(page_title="Retail Sales AI Analyst", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
    .hero {
        padding: 1.4rem 1.6rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #172554 0%, #1d4ed8 55%, #06b6d4 100%);
        color: white;
        margin-bottom: 1.2rem;
    }
    .hero h1 { margin: 0; color: white; }
    .hero p { margin: .35rem 0 0; color: #dbeafe; }
    [data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: .8rem;
        border-radius: 14px;
    }
    [data-testid="stMetric"] label,
    [data-testid="stMetric"] [data-testid="stMetricValue"],
    [data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #0f172a !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero"><h1>Retail Sales AI Analyst</h1>'
    '<p>Explore performance, trends, and forecasts from your Gold tables.</p></div>',
    unsafe_allow_html=True,
)
st.caption("Ask a question about the business data — answers are calculated "
           "from the Gold tables, not guessed by the AI.")

EXAMPLE_QUESTIONS = [
    "What were total sales last month?",
    "Which region performed best?",
    "What are the top 5 products?",
    "Why did sales decrease in July 2024?",
    "What is next month's predicted sales?",
]


@st.cache_data(show_spinner=False)
def _load_data_cached():
    return load_data()


data = _load_data_cached()

missing = [k for k in ["monthly", "product", "regional"] if k not in data]
if missing:
    st.warning(
        f"Some Gold tables aren't available yet ({', '.join(missing)}). "
        f"Run the Databricks pipeline (01-03) and export to CSV first — "
        f"see the README for the exact commands."
    )
if "forecast" not in data:
    st.info("Forecast data not found — run `python ml/predict.py` first if you want to ask forecast questions.")

llm_api_key_configured = bool(
    os.environ.get("LLM_API_KEY") or os.environ.get("GROQ_API_KEY")
)
if not llm_api_key_configured:
    st.caption("ℹ️ No LLM_API_KEY set — answers are generated with the built-in deterministic "
               "template rather than an LLM. Both are equally grounded in real numbers; "
               "the LLM just phrases it more naturally. Set LLM_API_KEY in .env to enable it.")
else:
    st.caption("✅ Groq API key detected — answer wording will use Groq when the request succeeds.")


def _money(value):
    return f"${value:,.0f}"


def _render_dashboard():
    """Render the high-level dashboard from the loaded Gold tables."""
    monthly = data.get("monthly")
    if monthly is None or monthly.empty:
        return

    total_revenue = monthly["revenue"].sum()
    total_orders = monthly["orders"].sum()
    total_units = monthly["units_sold"].sum()
    latest_month = monthly["month"].max()
    latest_revenue = monthly.loc[monthly["month"] == latest_month, "revenue"].sum()

    st.subheader("Performance overview")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Total revenue", _money(total_revenue))
    metric_cols[1].metric("Total orders", f"{total_orders:,.0f}")
    metric_cols[2].metric("Units sold", f"{total_units:,.0f}")
    metric_cols[3].metric(
        f"{latest_month.strftime('%b %Y')} revenue",
        _money(latest_revenue),
    )

    monthly_chart = (
        monthly.groupby("month", as_index=True)["revenue"]
        .sum()
        .rename("Revenue")
        .sort_index()
    )
    st.markdown("#### Revenue trend")
    st.area_chart(monthly_chart, color="#2563eb", height=300)

    chart_cols = st.columns(2)
    regional = data.get("regional")
    if regional is not None and not regional.empty:
        regional_chart = (
            regional.set_index("region")["total_revenue"]
            .sort_values(ascending=True)
            .rename("Revenue")
        )
        with chart_cols[0]:
            st.markdown("#### Revenue by region")
            st.bar_chart(regional_chart, color="#06b6d4", height=280)

    product = data.get("product")
    if product is not None and not product.empty:
        category_chart = (
            product.groupby("category")["total_revenue"]
            .sum()
            .sort_values(ascending=True)
            .rename("Revenue")
        )
        with chart_cols[1]:
            st.markdown("#### Revenue by category")
            st.bar_chart(category_chart, color="#8b5cf6", height=280)

    forecast = data.get("forecast")
    if forecast is not None and not forecast.empty:
        forecast_chart = (
            forecast.groupby("forecast_month")[
                ["predicted_revenue", "lower_bound", "upper_bound"]
            ]
            .sum()
            .rename(
                columns={
                    "predicted_revenue": "Predicted",
                    "lower_bound": "Lower bound",
                    "upper_bound": "Upper bound",
                }
            )
            .sort_index()
        )
        st.markdown("#### Forecast with confidence range")
        if len(forecast_chart) == 1:
            st.caption(
                "The current forecast contains one month, so it is shown as bars "
                "instead of a line. Lower and upper bars show the confidence range."
            )
            st.bar_chart(
                forecast_chart,
                color=["#f97316", "#94a3b8", "#cbd5e1"],
                height=280,
            )
        else:
            st.line_chart(
                forecast_chart,
                color=["#f97316", "#cbd5e1", "#cbd5e1"],
                height=280,
            )


_render_dashboard()

st.subheader("Try an example question")
cols = st.columns(len(EXAMPLE_QUESTIONS))
clicked_question = None
for col, q in zip(cols, EXAMPLE_QUESTIONS):
    # short label on the button, full question underneath as a tooltip
    if col.button(q.split("?")[0][:18] + "…", help=q, width="stretch"):
        clicked_question = q

with st.form("question_form", clear_on_submit=False):
    question = st.text_input(
        "Ask a question about the business data...",
        value=clicked_question or "",
        placeholder="e.g. Which region had the highest sales last quarter?",
    )
    st.caption(
        "Ask about revenue, monthly trends, regions, categories, products, "
        "declines, or forecasts."
    )
    ask_clicked = st.form_submit_button("Ask", type="primary")

if question and (ask_clicked or clicked_question):
    with st.spinner("Calculating from the sales data..."):
        result = answer_question(question, data)

    st.markdown("### Answer")
    st.markdown(result["answer"].replace("$", r"\$"))

    badge = "🤖 LLM-generated wording" if result["used_llm"] else "📐 Deterministic template"
    st.caption(f"{badge} · intent detected: `{result['intent']}`")
    if llm_api_key_configured and not result["used_llm"]:
        st.warning(
            "Groq was reached but did not return an answer, so the app used the "
            "deterministic fallback. Check the error below."
        )
        if result.get("llm_error"):
            st.code(result["llm_error"], language="text")

    with st.expander("Relevant metrics (the exact numbers behind this answer)"):
        st.json(result["structured_result"])

    # show an underlying table where one naturally exists in the result
    sr = result["structured_result"]
    if sr.get("available"):
        if "products" in sr and sr["products"]:
            st.markdown("#### Underlying data")
            st.dataframe(pd.DataFrame(sr["products"]), width="stretch")
        elif "all_regions" in sr and isinstance(sr["all_regions"], dict):
            st.markdown("#### Underlying data")
            st.dataframe(
                pd.DataFrame(list(sr["all_regions"].items()), columns=["region", "revenue"]),
                width="stretch",
            )
        elif "all_categories" in sr and isinstance(sr["all_categories"], dict):
            st.markdown("#### Underlying data")
            st.dataframe(
                pd.DataFrame(list(sr["all_categories"].items()), columns=["category", "revenue"]),
                width="stretch",
            )
        elif "breakdown" in sr:
            st.markdown("#### Underlying data")
            st.dataframe(pd.DataFrame(sr["breakdown"]), width="stretch")
        elif "largest_declining_segments" in sr and sr["largest_declining_segments"]:
            st.markdown("#### Largest declining segments")
            st.dataframe(pd.DataFrame(sr["largest_declining_segments"]), width="stretch")

st.divider()
st.caption(
    "This analyst never invents numbers: every answer is generated from a fixed set of "
    "analytical functions run against the Gold tables. If the data doesn't support an "
    "answer, it says so instead of guessing."
)
