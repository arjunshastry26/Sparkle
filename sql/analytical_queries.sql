-- =============================================================================
-- analytical_queries.sql
--
-- Business-facing SQL queries against the Silver/Gold tables produced by the
-- Databricks pipeline. Written in ANSI SQL (Databricks SQL / Spark SQL
-- compatible). Each query is preceded by a comment explaining what business
-- question it answers and why it's written the way it is.
--
-- Assumes the following tables are registered (in Databricks, register the
-- Parquet outputs from the pipeline as tables, e.g.:
--   CREATE TABLE IF NOT EXISTS silver_sales
--   USING PARQUET LOCATION '/FileStore/retail_sales_analytics/silver/sales';
-- ... repeat for gold_monthly_sales, gold_product_performance,
--     gold_regional_performance, using the paths under gold/ ).
-- =============================================================================


-- -----------------------------------------------------------------------------
-- 1. Total revenue (all-time)
-- Straightforward headline metric. We use net_revenue because that's what the
-- business actually earns after discounts.
-- -----------------------------------------------------------------------------
SELECT ROUND(SUM(revenue), 2) AS total_revenue
FROM gold_monthly_sales;


-- -----------------------------------------------------------------------------
-- 2. Monthly revenue trend
-- Powers the "Revenue over time" line chart on the Executive Overview page.
-- Aggregating gold_monthly_sales (already at month grain) rather than
-- silver_sales keeps this query fast even as raw transaction volume grows.
-- -----------------------------------------------------------------------------
SELECT
    month,
    ROUND(SUM(revenue), 2) AS revenue,
    SUM(orders)            AS orders,
    SUM(units_sold)        AS units_sold
FROM gold_monthly_sales
GROUP BY month
ORDER BY month;


-- -----------------------------------------------------------------------------
-- 3. Revenue by region
-- -----------------------------------------------------------------------------
SELECT
    region,
    ROUND(SUM(revenue), 2) AS total_revenue,
    SUM(orders)            AS total_orders,
    ROUND(SUM(revenue) / NULLIF(SUM(orders), 0), 2) AS average_order_value
FROM gold_monthly_sales
GROUP BY region
ORDER BY total_revenue DESC;


-- -----------------------------------------------------------------------------
-- 4. Revenue by category
-- -----------------------------------------------------------------------------
SELECT
    category,
    ROUND(SUM(revenue), 2) AS total_revenue,
    SUM(units_sold)        AS total_units
FROM gold_monthly_sales
GROUP BY category
ORDER BY total_revenue DESC;


-- -----------------------------------------------------------------------------
-- 5. Top 10 products by total revenue
-- -----------------------------------------------------------------------------
SELECT
    product_id,
    product_name,
    category,
    total_revenue,
    units_sold
FROM gold_product_performance
ORDER BY total_revenue DESC
LIMIT 10;


-- -----------------------------------------------------------------------------
-- 6. Lowest-performing products
-- Bottom 10 by revenue AND explicitly negative-growth products, since "worst
-- performer" can mean either "smallest" or "declining" — the dashboard shows
-- both views on the Product & Regional Analysis page.
-- -----------------------------------------------------------------------------
SELECT product_id, product_name, category, total_revenue, units_sold
FROM gold_product_performance
ORDER BY total_revenue ASC
LIMIT 10;

SELECT product_id, product_name, category, total_revenue, sales_growth
FROM gold_product_performance
WHERE sales_growth IS NOT NULL AND sales_growth < 0
ORDER BY sales_growth ASC
LIMIT 10;


-- -----------------------------------------------------------------------------
-- 7. Month-over-month revenue growth (overall business)
-- LAG() over the ordered month sequence is the standard, readable way to do
-- period-over-period comparisons in SQL — avoids a self-join.
-- -----------------------------------------------------------------------------
WITH monthly AS (
    SELECT month, SUM(revenue) AS revenue
    FROM gold_monthly_sales
    GROUP BY month
)
SELECT
    month,
    revenue,
    LAG(revenue) OVER (ORDER BY month) AS prev_month_revenue,
    ROUND(
        (revenue - LAG(revenue) OVER (ORDER BY month))
        / NULLIF(LAG(revenue) OVER (ORDER BY month), 0) * 100, 2
    ) AS mom_growth_pct
FROM monthly
ORDER BY month;


-- -----------------------------------------------------------------------------
-- 8. Average order value, overall and by region
-- -----------------------------------------------------------------------------
SELECT ROUND(SUM(revenue) / NULLIF(SUM(orders), 0), 2) AS overall_avg_order_value
FROM gold_monthly_sales;

SELECT region, ROUND(SUM(revenue) / NULLIF(SUM(orders), 0), 2) AS avg_order_value
FROM gold_monthly_sales
GROUP BY region
ORDER BY avg_order_value DESC;


-- -----------------------------------------------------------------------------
-- 9. Regional contribution to total revenue (%)
-- Uses a window function to compute each region's share without a separate
-- subquery for the grand total.
-- -----------------------------------------------------------------------------
SELECT
    region,
    ROUND(SUM(revenue), 2) AS region_revenue,
    ROUND(100.0 * SUM(revenue) / SUM(SUM(revenue)) OVER (), 2) AS pct_of_total_revenue
FROM gold_monthly_sales
GROUP BY region
ORDER BY region_revenue DESC;


-- -----------------------------------------------------------------------------
-- 10. Category growth (first year vs. most recent full year)
-- Compares full calendar years instead of single months to avoid seasonal
-- noise (e.g. comparing December to January would always look like a decline).
-- -----------------------------------------------------------------------------
WITH yearly AS (
    SELECT category, YEAR(month) AS yr, SUM(revenue) AS revenue
    FROM gold_monthly_sales
    GROUP BY category, YEAR(month)
),
bounds AS (
    SELECT MIN(yr) AS first_year, MAX(yr) AS last_year FROM yearly
)
SELECT
    y1.category,
    y1.revenue AS first_year_revenue,
    y2.revenue AS last_year_revenue,
    ROUND((y2.revenue - y1.revenue) / NULLIF(y1.revenue, 0) * 100, 2) AS growth_pct
FROM yearly y1
JOIN bounds b ON y1.yr = b.first_year
JOIN yearly y2 ON y2.category = y1.category AND y2.yr = b.last_year
ORDER BY growth_pct DESC;


-- -----------------------------------------------------------------------------
-- Bonus: root-cause style query for "why did sales decrease in <month>?"
-- Finds the region × category combination(s) with the largest absolute
-- revenue decline vs. the prior month — this is the query the AI analyst's
-- get_region_performance()/get_category_performance() functions are built on.
-- -----------------------------------------------------------------------------
WITH by_segment AS (
    SELECT month, region, category, SUM(revenue) AS revenue
    FROM gold_monthly_sales
    GROUP BY month, region, category
),
with_prev AS (
    SELECT
        month, region, category, revenue,
        LAG(revenue) OVER (PARTITION BY region, category ORDER BY month) AS prev_revenue
    FROM by_segment
)
SELECT
    month, region, category, revenue, prev_revenue,
    ROUND((revenue - prev_revenue) / NULLIF(prev_revenue, 0) * 100, 2) AS mom_change_pct,
    ROUND(revenue - prev_revenue, 2) AS mom_change_abs
FROM with_prev
WHERE prev_revenue IS NOT NULL
ORDER BY mom_change_abs ASC
LIMIT 10;
