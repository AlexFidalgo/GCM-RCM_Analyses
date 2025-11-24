import os
import pandas as pd
from itertools import combinations
import sys
from sqlalchemy import text
from tqdm.auto import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_db_connection

# ------------------ USER CONFIGURATIONS ------------------ #

# Regions to include. Set to None to include all available regions found in the data.
REGIONS = None  # e.g., ['RegionA', 'RegionB']

# Physical variable to include ('ppt', 'tas', etc.). Set to None to include all.
PHYSICAL_VARIABLE = None  # e.g., 'ppt'

# Metric abbreviations to include. Set to None to include all metrics present in the data.
METRIC_ABBREVIATIONS = None  # e.g., ['ACC', 'KGE (2009)']

# RCM and GCM model IDs to include as filters. Set to None to include all available IDs.
RCM_MODEL_IDS = None  # e.g., [1, 2, 3]
GCM_MODEL_IDS = None  # e.g., [10, 11, 12]
RCM_MODEL_IDS = [1,2,5,6,8,11,12]
GCM_MODEL_IDS = [3,7,8]

# Correlation method: 'pearson', 'spearman', or 'kendall'
CORRELATION_METHOD = 'pearson'

# Name of the output SQL table where the correlations will be stored
OUTPUT_SQL_TABLE = "correlations_between_metrics_reduced_matrix"

# --------------------------------------------------------- #

OUTPUT_DIR = "metrics_correlations/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _normalize_filter(value):
    """Turn user input into a list or None."""
    if value in (None, [], ()):
        return None
    if isinstance(value, str):
        return [value]
    # If it's not a string but is iterable, convert to list
    try:
        return list(value)
    except TypeError:
        # Scalar value (e.g., int, float); wrap in a list
        return [value]


def fetch_available_values(conn):
    """
    Fetch distinct regions, physical variables, metrics, rcm_ids and gcm_ids
    from the database.
    """
    regions = sorted(row[0] for row in conn.execute(text("SELECT DISTINCT region FROM error")))
    physicals = sorted(row[0] for row in conn.execute(text("SELECT DISTINCT physical_variable FROM error")))
    metrics = sorted(row[0] for row in conn.execute(text("SELECT DISTINCT metric_name FROM metrics")))
    rcm_ids = sorted(row[0] for row in conn.execute(text("SELECT DISTINCT rcm_id FROM error")))
    gcm_ids = sorted(row[0] for row in conn.execute(text("SELECT DISTINCT gcm_id FROM error")))
    return regions, physicals, metrics, rcm_ids, gcm_ids


def build_metric_placeholders(metric_list):
    """Build IN clause placeholders and params for a metric list."""
    placeholders = []
    params = {}
    for idx, metric in enumerate(metric_list):
        key = f"metric_{idx}"
        placeholders.append(f":{key}")
        params[key] = metric
    return ", ".join(placeholders), params


def build_id_placeholders(id_list, prefix):
    """Build IN clause placeholders and params for a list of IDs (RCM/GCM)."""
    placeholders = []
    params = {}
    for idx, value in enumerate(id_list):
        key = f"{prefix}_{idx}"
        placeholders.append(f":{key}")
        params[key] = value
    return ", ".join(placeholders), params


def load_subset(conn, region, physical, metric_subset, rcm_ids=None, gcm_ids=None):
    """
    Load only one region + one physical_variable subset to keep memory small.
    Optionally filter by RCM and GCM IDs.
    """
    metric_clause, metric_params = build_metric_placeholders(metric_subset)

    where_clauses = [
        "error.region = :region",
        "error.physical_variable = :physical",
        f"metrics.metric_name IN ({metric_clause})"
    ]
    params = {"region": region, "physical": physical}
    params.update(metric_params)

    if rcm_ids is not None:
        rcm_clause, rcm_params = build_id_placeholders(rcm_ids, "rcm")
        where_clauses.append(f"error.rcm_id IN ({rcm_clause})")
        params.update(rcm_params)

    if gcm_ids is not None:
        gcm_clause, gcm_params = build_id_placeholders(gcm_ids, "gcm")
        where_clauses.append(f"error.gcm_id IN ({gcm_clause})")
        params.update(gcm_params)

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
        WHERE {' AND '.join(where_clauses)}
        """
    )

    return pd.read_sql_query(query, conn, params=params)


def load_physical_all(conn, physical, metric_subset, rcm_ids=None, gcm_ids=None):
    """
    Load all regions for a given physical_variable.
    Optionally filter by RCM and GCM IDs.
    """
    metric_clause, metric_params = build_metric_placeholders(metric_subset)

    where_clauses = [
        "error.physical_variable = :physical",
        f"metrics.metric_name IN ({metric_clause})"
    ]
    params = {"physical": physical}
    params.update(metric_params)

    if rcm_ids is not None:
        rcm_clause, rcm_params = build_id_placeholders(rcm_ids, "rcm")
        where_clauses.append(f"error.rcm_id IN ({rcm_clause})")
        params.update(rcm_params)

    if gcm_ids is not None:
        gcm_clause, gcm_params = build_id_placeholders(gcm_ids, "gcm")
        where_clauses.append(f"error.gcm_id IN ({gcm_clause})")
        params.update(gcm_params)

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
        WHERE {' AND '.join(where_clauses)}
        """
    )

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
    rcm_filter = _normalize_filter(RCM_MODEL_IDS)
    gcm_filter = _normalize_filter(GCM_MODEL_IDS)

    engine = get_db_connection()
    with engine.connect() as conn:
        (
            available_regions,
            available_physicals,
            available_metrics,
            available_rcm_ids,
            available_gcm_ids
        ) = fetch_available_values(conn)

        region_columns = regions_filter if regions_filter else available_regions
        physical_values = physical_filter if physical_filter else available_physicals
        metrics_to_use = metrics_filter if metrics_filter else available_metrics
        rcm_values = rcm_filter if rcm_filter else available_rcm_ids
        gcm_values = gcm_filter if gcm_filter else available_gcm_ids

        missing_regions = set(region_columns) - set(available_regions)
        missing_physicals = set(physical_values) - set(available_physicals)
        missing_metrics = set(metrics_to_use) - set(available_metrics)
        missing_rcm_ids = set(rcm_values) - set(available_rcm_ids)
        missing_gcm_ids = set(gcm_values) - set(available_gcm_ids)

        if missing_regions:
            raise ValueError(f"Regions not found in data: {sorted(missing_regions)}")
        if missing_physicals:
            raise ValueError(f"Physical variables not found in data: {sorted(missing_physicals)}")
        if missing_metrics:
            raise ValueError(f"Metrics not found in data: {sorted(missing_metrics)}")
        if missing_rcm_ids:
            raise ValueError(f"RCM IDs not found in data: {sorted(missing_rcm_ids)}")
        if missing_gcm_ids:
            raise ValueError(f"GCM IDs not found in data: {sorted(missing_gcm_ids)}")
        if not metrics_to_use:
            raise ValueError("No metrics available to compute correlations.")
        if len(metrics_to_use) < 2:
            raise ValueError("Need at least two metrics to compute correlations.")

        print(f"Using regions: {region_columns}")
        print(f"Using physical variables: {physical_values}")
        print(f"Using metrics: {metrics_to_use}")
        print(f"Using RCM model IDs: {rcm_values}")
        print(f"Using GCM model IDs: {gcm_values}")
        print(f"Output SQL table: {OUTPUT_SQL_TABLE}")

        metric_pairs = list(combinations(metrics_to_use, 2))
        rows = []

        for phys in tqdm(physical_values, desc="Physical variables"):
            phys_all_df = load_physical_all(conn, phys, metrics_to_use, rcm_values, gcm_values)
            total_corr_matrix = compute_corr_matrix(phys_all_df)

            # Pre-build row shells so we only set correlations as we compute them.
            pair_rows = {
                (m1, m2): {
                    'physical_variable': phys,
                    'metric1': m1,
                    'metric2': m2,
                    'total_corr': round(total_corr_matrix.loc[m1, m2], 3)
                    if (
                        (m1 in total_corr_matrix.index)
                        and (m2 in total_corr_matrix.columns)
                        and pd.notnull(total_corr_matrix.loc[m1, m2])
                    )
                    else None,
                    **{region: None for region in region_columns}
                }
                for m1, m2 in metric_pairs
            }

            for region in tqdm(region_columns, desc=f"{phys}: regions", leave=False):
                subset = load_subset(conn, region, phys, metrics_to_use, rcm_values, gcm_values)
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
        OUTPUT_SQL_TABLE,
        engine,
        if_exists="replace",
        index=False,
    )
    print(f'Saved correlations table to database table "{OUTPUT_SQL_TABLE}".')


if __name__ == "__main__":
    main()
