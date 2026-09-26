# Power BI Dashboard — Build Instructions

This is not a `.pbix` file (Power BI Desktop isn't available in this build
environment), but everything needed to build one is here: exact data
sources, exact visuals, and the DAX measures behind anything that isn't a
plain column. Following these steps end to end takes about 30-45 minutes in
Power BI Desktop.

## 1. Data sources

After running the Databricks pipeline (or the local equivalent — see the
main README), you'll have these CSV exports under `dbfs_local/gold/csv_export/`
(on real Databricks, these are Delta/Parquet tables in Unity Catalog or DBFS
— connect Power BI to Databricks directly via **Get Data → Databricks**
instead, using the same table names):

| Table | Path (local demo) |
|---|---|
| `gold_monthly_sales` | `dbfs_local/gold/csv_export/gold_monthly_sales/part-*.csv` |
| `gold_product_performance` | `dbfs_local/gold/csv_export/gold_product_performance/part-*.csv` |
| `gold_regional_performance` | `dbfs_local/gold/csv_export/gold_regional_performance/part-*.csv` |
| `gold_sales_forecast` | `dbfs_local/gold/sales_forecast.csv` |

**Steps:**
1. Power BI Desktop → **Home → Get Data → Text/CSV**.
2. Load all four files above (rename each query to match the table name,
   e.g. `gold_monthly_sales`, via the Query Editor's "Rename" so DAX measures
   below can reference them directly).
3. In Power Query, set `month` / `forecast_month` columns to **Date** type
   explicitly (Power BI sometimes infers CSV dates as text).
4. **Home → Close & Apply**.

## 2. Relationships

Create a **Date table** (Modeling → New Table):
```
DateTable = CALENDAR(MIN(gold_monthly_sales[month]), MAX(gold_sales_forecast[forecast_month]))
```
Mark it as a Date table (Modeling → Mark as Date Table), then relate:
- `DateTable[Date]` → `gold_monthly_sales[month]` (many-to-one)
- `DateTable[Date]` → `gold_sales_forecast[forecast_month]` (many-to-one)

Relate `gold_product_performance[category]` → `gold_monthly_sales[category]`
(many-to-one) so category slicers filter both tables together.

## 3. Core DAX measures

Create these once (Modeling → New Measure) — every page below reuses them:

```dax
Total Revenue = SUM(gold_monthly_sales[revenue])

Total Orders = SUM(gold_monthly_sales[orders])

Total Units Sold = SUM(gold_monthly_sales[units_sold])

Average Order Value = DIVIDE([Total Revenue], [Total Orders])

Prior Month Revenue =
CALCULATE([Total Revenue], DATEADD(DateTable[Date], -1, MONTH))

MoM Growth % =
DIVIDE([Total Revenue] - [Prior Month Revenue], [Prior Month Revenue])

Forecasted Revenue = SUM(gold_sales_forecast[predicted_revenue])
```

## 4. Page 1 — Executive Overview

**Cards** (Insert → Card), one each:
- `[Total Revenue]` — format as currency, no decimals
- `[Total Orders]`
- `[Total Units Sold]`
- `[Average Order Value]` — currency, 2 decimals

**Charts:**
- **Revenue over time** — Line chart. X-axis: `DateTable[Date]` (Month
  hierarchy, drilled to Month). Y-axis: `[Total Revenue]`.
- **Revenue by region** — Clustered bar chart. Axis: `gold_monthly_sales[region]`.
  Values: `[Total Revenue]`. Sort descending.
- **Revenue by category** — Donut or clustered bar chart. Axis:
  `gold_monthly_sales[category]`. Values: `[Total Revenue]`.

## 5. Page 2 — Product & Regional Analysis

- **Top 10 products** — Bar chart. Axis: `gold_product_performance[product_name]`.
  Values: `SUM(gold_product_performance[total_revenue])`. Use the visual-level
  **Top N** filter (Top 10 by total_revenue).
- **Bottom 10 products** — Same, but Top N → **Bottom 10**.
- **Region comparison** — Clustered column chart, `region` on axis, both
  `[Total Revenue]` and `[Total Orders]` as values (dual value bars, or use
  a small-multiples layout by region).
- **Category performance** — Treemap. Group: `category`. Values:
  `[Total Revenue]`.
- **Month-over-month growth** — Line chart. Axis: `DateTable[Date]` (Month).
  Values: `[MoM Growth %]`. Format as percentage; add a 0% reference line
  (Format → Analytics → Constant line) so growth vs. decline is visible at a
  glance.

## 6. Page 3 — Forecast

- **Actual vs. predicted sales** — Line chart. Axis: `DateTable[Date]`.
  Values: `[Total Revenue]` (actual, solid line) AND `[Forecasted Revenue]`
  (predicted, will only plot on the single forecast month — format that
  series as a dashed line or distinct marker so it reads as "the forecast"
  rather than a continuation of history).
- **Next month's forecast** — Card showing `[Forecasted Revenue]`, plus a
  subtitle text box noting the `model_version` value from
  `gold_sales_forecast` (so viewers know whether they're looking at the
  baseline or a trained model — see the main README's Results section for
  why that matters here).
- **Forecast by region** — Clustered bar chart. Axis: `region`. Values:
  `SUM(gold_sales_forecast[predicted_revenue])`. Add error bars using
  `lower_bound` / `upper_bound` (Format → Error bars → Custom, bind to those
  columns) if you want the uncertainty range visible.
- **Forecast by category** — Same pattern, axis = `category`.

## 7. Slicers (add to every page, or use a synced slicer panel)

- **Date** — slicer bound to `DateTable[Date]`, "Between" style.
- **Region** — slicer bound to `gold_monthly_sales[region]`.
- **Category** — slicer bound to `gold_monthly_sales[category]`.

To sync slicers across all three pages: select each slicer → **View → Sync
Slicers** → check all three pages.

## 8. A note on the forecast page

`ml/train.py` compares a naive "last month repeats" baseline against
XGBoost and always deploys whichever wins on measured test error — see the
main README's Results section for the actual numbers. The `model_version`
field on `gold_sales_forecast` records which one produced a given forecast,
so put it on the Page 3 card (see step 6 above) rather than assuming it's
always the same model — re-running `04_forecasting.py`/`05_prediction.py`
on refreshed data can change the winner.
