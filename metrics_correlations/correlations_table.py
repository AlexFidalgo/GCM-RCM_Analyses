import os
import pandas as pd
from itertools import combinations
import sys
from sqlalchemy import text
from tqdm.auto import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_db_connection

# Regions to include. Set to None to include all available regions found in the data.
REGIONS = None  # e.g., ['RegionA', 'RegionB']
# Physical variable to include ('ppt', 'tas', etc.). Set to None to include all.
PHYSICAL_VARIABLE = None  # e.g., 'ppt'
# Metric abbreviations to include. Set to None to include all metrics present in the data.
METRIC_ABBREVIATIONS = None  # e.g., ['ACC', 'KGE (2009)']
# Correlation method: 'pearson', 'spearman', or 'kendall'
CORRELATION_METHOD = 'pearson'

OUTPUT_DIR = "metrics_correlations/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def _normalize_filter(value):
    """Turn user input into a list or None."""
    if value in (None, [], ()):
        return None
    if isinstance(value, str):
        return [value]
    if len(value) == 0:
        return None
    return list(value)


def fetch_available_values(conn):
    """Fetch distinct regions, physical variables and metrics from the database."""
    regions = sorted(row[0] for row in conn.execute(text("SELECT DISTINCT region FROM error")))
    physicals = sorted(row[0] for row in conn.execute(text("SELECT DISTINCT physical_variable FROM error")))
    metrics = sorted(row[0] for row in conn.execute(text("SELECT DISTINCT metric_name FROM metrics")))
    return regions, physicals, metrics


def build_metric_placeholders(metric_list):
    """Build IN clause placeholders and params for a metric list."""
    placeholders = []
    params = {}
    for idx, metric in enumerate(metric_list):
        key = f"metric_{idx}"
        placeholders.append(f":{key}")
        params[key] = metric
    return ", ".join(placeholders), params


def load_subset(conn, region, physical, metric_subset):
    """Load only one region + one physical_variable subset to keep memory small."""
    metric_clause, metric_params = build_metric_placeholders(metric_subset)
    query = text(
        f"""
        SELECT
            error.region,
            error.gridpoint,
            error.physical_variable,
            error.model,
            error.rcm_id,
            error.gcm_id,
            metrics.metric_name,
            error.mat_vector
        FROM error
        LEFT JOIN metrics ON metrics.id = error.metric_id
        WHERE error.region = :region
          AND error.physical_variable = :physical
          AND metrics.metric_name IN ({metric_clause})
        """
    )
    params = {"region": region, "physical": physical}
    params.update(metric_params)
    return pd.read_sql_query(query, conn, params=params)


def load_physical_all(conn, physical, metric_subset):
    """Load all regions for a given physical_variable."""
    metric_clause, metric_params = build_metric_placeholders(metric_subset)
    query = text(
        f"""
        SELECT
            error.region,
            error.gridpoint,
            error.physical_variable,
            error.model,
            error.rcm_id,
            error.gcm_id,
            metrics.metric_name,
            error.mat_vector
        FROM error
        LEFT JOIN metrics ON metrics.id = error.metric_id
        WHERE error.physical_variable = :physical
          AND metrics.metric_name IN ({metric_clause})
        """
    )
    params = {"physical": physical}
    params.update(metric_params)
    return pd.read_sql_query(query, conn, params=params)


def compute_corr_matrix(sub_df: pd.DataFrame) -> pd.DataFrame:
    """Compute correlation matrix for the subset; return empty DF if insufficient data."""
    if sub_df.empty:
        return pd.DataFrame()
    pivot = sub_df.pivot_table(
        index=['region', 'gridpoint', 'physical_variable', 'model', 'rcm_id', 'gcm_id'],
        columns='metric_name',
        values='mat_vector',
        aggfunc='first'
    )
    # If after pivot there are fewer than 2 rows or columns, corr() will be empty.
    if pivot.shape[0] < 2 or pivot.shape[1] < 2:
        return pd.DataFrame()
    return pivot.corr(method=CORRELATION_METHOD)


def main():
    regions_filter = _normalize_filter(REGIONS)
    physical_filter = _normalize_filter(PHYSICAL_VARIABLE)
    metrics_filter = _normalize_filter(METRIC_ABBREVIATIONS)

    engine = get_db_connection()
    with engine.connect() as conn:
        available_regions, available_physicals, available_metrics = fetch_available_values(conn)

        region_columns = regions_filter if regions_filter else available_regions
        physical_values = physical_filter if physical_filter else available_physicals
        metrics_to_use = metrics_filter if metrics_filter else available_metrics

        missing_regions = set(region_columns) - set(available_regions)
        missing_physicals = set(physical_values) - set(available_physicals)
        missing_metrics = set(metrics_to_use) - set(available_metrics)
        if missing_regions:
            raise ValueError(f"Regions not found in data: {sorted(missing_regions)}")
        if missing_physicals:
            raise ValueError(f"Physical variables not found in data: {sorted(missing_physicals)}")
        if missing_metrics:
            raise ValueError(f"Metrics not found in data: {sorted(missing_metrics)}")
        if not metrics_to_use:
            raise ValueError("No metrics available to compute correlations.")
        if len(metrics_to_use) < 2:
            raise ValueError("Need at least two metrics to compute correlations.")

        print(f"Using regions: {region_columns}")
        print(f"Using physical variables: {physical_values}")
        print(f"Using metrics: {metrics_to_use}")

        metric_pairs = list(combinations(metrics_to_use, 2))
        rows = []

        for phys in tqdm(physical_values, desc="Physical variables"):
            phys_all_df = load_physical_all(conn, phys, metrics_to_use)
            total_corr_matrix = compute_corr_matrix(phys_all_df)

            # Pre-build row shells so we only set correlations as we compute them.
            pair_rows = {
                (m1, m2): {
                    'physical_variable': phys,
                    'metric1': m1,
                    'metric2': m2,
                    'total_corr': round(total_corr_matrix.loc[m1, m2], 3)
                    if (m1 in total_corr_matrix.index and m2 in total_corr_matrix.columns and pd.notnull(total_corr_matrix.loc[m1, m2]))
                    else None,
                    **{region: None for region in region_columns}
                }
                for m1, m2 in metric_pairs
            }

            for region in tqdm(region_columns, desc=f"{phys}: regions", leave=False):
                subset = load_subset(conn, region, phys, metrics_to_use)
                corr_matrix = compute_corr_matrix(subset)
                if corr_matrix.empty:
                    continue

                for m1, m2 in metric_pairs:
                    if m1 in corr_matrix.index and m2 in corr_matrix.columns:
                        corr_val = corr_matrix.loc[m1, m2]
                        if pd.notnull(corr_val):
                            pair_rows[(m1, m2)][region] = round(corr_val, 3)

            rows.extend(pair_rows.values())

    correlation_table = pd.DataFrame(rows)

    if correlation_table.empty:
        raise ValueError("No correlation rows were produced for the given filters.")

    # Drop rows where all region values and total_corr are missing
    region_fields = region_columns
    drop_fields = region_fields + ['total_corr']
    has_data = correlation_table[drop_fields].notnull().any(axis=1)
    correlation_table = correlation_table[has_data]

    output_path = os.path.join(OUTPUT_DIR, 'correlations_table.xlsx')
    correlation_table.to_excel(output_path, index=False)
    print(f"Saved correlations table to: {output_path}")

    # Persist results back to the database
    correlation_table.to_sql(
        "correlations_between_metrics",
        engine,
        if_exists="replace",
        index=False,
    )
    print('Saved correlations table to database table "correlations_between_metrics".')


if __name__ == "__main__":
    main()
