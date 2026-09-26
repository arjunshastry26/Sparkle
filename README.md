# Retail Sales Analytics & Forecasting

<p align="center">
  <strong>A Databricks + PySpark + Streamlit retail intelligence platform</strong><br>
  <sub>Real-world grocery sales data • Gold-table analytics • Natural-language questions • Forecasting-ready</sub>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PySpark-3.5-orange?logo=apachespark&logoColor=white" alt="PySpark">
  <img src="https://img.shields.io/badge/Streamlit-dashboard-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Databricks-Unity%20Catalog-EF3E42?logo=databricks&logoColor=white" alt="Databricks">
  <img src="https://img.shields.io/badge/tests-20%20passed-2EA44F" alt="Tests">
</p>

## What this project does

This project turns retail transactions into decision-ready analytics:

```text
Favorita CSV files
      ↓
Databricks / PySpark aggregation
      ↓
Gold tables: monthly • products • regions • forecast input
      ↓
Streamlit dashboard + deterministic analyst + optional Groq wording
```

The dashboard answers questions such as:

- How are sales changing over time?
- Which states and product families perform best?
- Which products are declining?
- What happened in a specific month?
- What is the next-month forecast? (after a forecast artifact is generated)

## Highlights

- **Real dataset:** Corporación Favorita Grocery Sales Forecasting data.
- **Scalable processing:** 4.65 GB `train.csv` is aggregated with Spark in Databricks.
- **Stable analytics contract:** the dashboard consumes consistent monthly, product,
  regional, and forecast-input Gold outputs.
- **Polished dashboard:** KPI cards, trend charts, regional/category comparisons,
  forecast visualizations, and dark-theme styling.
- **Grounded analyst:** rule-based intent routing and data-backed answers; an
  OpenAI-compatible Groq model only improves phrasing.
- **Safe fallback:** the dashboard remains usable without an LLM API key.
- **Tested behavior:** intent-routing and feature-regression tests are included.

## Dashboard preview

Run the app locally:

```powershell
streamlit run ai_analyst\app.py
```

Then open <http://localhost:8501>.

For the Favorita migration, the dashboard uses these meanings:

| Dashboard term | Favorita meaning |
|---|---|
| Revenue | `unit_sales` (Favorita does not provide prices) |
| Orders | Count of source sales records |
| Region | Store `state` |
| Category | Item `family` |

## Quick start

### 1. Create an environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure optional LLM wording

Copy `.env.example` to `.env` and add a replacement Groq key:

```env
LLM_API_KEY=gsk_your_key_here
LLM_MODEL=openai/gpt-oss-20b
LLM_BASE_URL=https://api.groq.com/openai/v1
```

Never commit `.env`, API keys, Kaggle credentials, or downloaded datasets.

### 3. Run the dashboard

```powershell
streamlit run ai_analyst\app.py
```

Without `LLM_API_KEY`, the deterministic answer path still uses the real Gold
numbers and does not require an external service.

## Databricks Favorita pipeline

The large Favorita files should be processed in Databricks rather than locally.
The pipeline is [`databricks/favorita_pipeline.py`](databricks/favorita_pipeline.py).

### Input files

Place these files in the Unity Catalog volume:

```text
/Volumes/workspace/default/favorita/
├── train.csv
├── test.csv
├── stores.csv
├── items.csv
├── transactions.csv
├── oil.csv
├── holidays_events.csv
└── sample_submission.csv
```

### Run configuration

In a Databricks Python notebook, set:

```python
import os

os.environ["FAVORITA_DATA_DIR"] = "/Volumes/workspace/default/favorita"
os.environ["RETAIL_BASE_PATH"] = "/Volumes/workspace/default/favorita/gold"
os.environ["SPARK_SHUFFLE_PARTITIONS"] = "200"
os.environ["SPARK_DEFAULT_PARALLELISM"] = "200"
```

Then run the pipeline on Serverless or a suitable Spark cluster. It writes:

```text
/Volumes/workspace/default/favorita/gold/gold/
├── monthly_sales
├── product_performance
├── regional_performance
└── forecast_input
```

The CSV exports are under `gold/csv_export/`. Download those four folders into:

```text
dbfs_local\gold\csv_export\
```

The application automatically reads Spark's `part-*.csv` output files.

### Verified Favorita output

The completed Databricks run produced:

| Gold output | Rows |
|---|---:|
| Monthly sales | 23,517 |
| Product performance | 4,036 |
| Regional performance | 16 |
| Forecast input | 23,517 |

## Repository layout

```text
ai_analyst/                 Streamlit UI and grounded analyst logic
databricks/                 Spark pipelines and Databricks notebooks
data_generator/             Reproducible synthetic-data generator
ml/                         Feature engineering, training, and prediction
powerbi/                    Power BI build notes and dashboard guidance
sql/                        Business SQL queries
tests/                      Regression and feature tests
```

## Forecasting

The original forecasting workflow operates at:

```text
month × region × category
```

It includes lag and rolling features, chronological validation, a naive
last-month baseline, and XGBoost models. The Favorita Gold pipeline creates the
compatible `forecast_input` table; run the forecasting workflow after the Gold
outputs have been downloaded locally or consumed directly in Databricks.

## Validation

Run the targeted test suite:

```powershell
python -m pytest tests\test_features.py -q
```

Expected result:

```text
20 passed
```

Validate syntax:

```powershell
python -m py_compile ai_analyst\app.py ai_analyst\analyst.py databricks\favorita_pipeline.py
```

