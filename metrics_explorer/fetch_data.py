"""
Fetch error metrics data from the climate_data database.

This script queries the database for error metrics across regions, gridpoints,
physical variables, and models, then saves the data for visualization.

The database contains 28M+ rows, so this script filters server-side to only
retrieve the target metrics, reducing data transfer and processing time.
"""

import os
import pandas as pd
from sqlalchemy import create_engine, text

from utils import get_db_connection

# Metrics of interest for exploration
TARGET_METRICS = [
    'H10 (MAHE)',
    'd',
    'dr',
    'NED',
    'MV',
    'KGE (2009)'
]

def fetch_error_metrics():
    """
    Fetch error metrics data from the database.
    
    Filters server-side for target metrics only to avoid loading all 28M+ rows.
    
    Returns:
        pd.DataFrame: DataFrame with columns: region, gridpoint, physical_variable,
                     model, rcm_id, gcm_id, metric_id, metric_name, mat_vector
    """
    query = """
    SELECT 
        error.region,
        error.gridpoint,
        error.physical_variable,
        error.model,
        error.rcm_id,
        error.gcm_id,
        error.metric_id,
        metrics.metric_name,
        error.mat_vector
    FROM 
        error 
    LEFT JOIN 
        metrics
    ON 
        metrics.id = error.metric_id
    WHERE
        metrics.metric_name IN :metric_names
    ORDER BY
        metrics.metric_name, error.region, error.physical_variable, error.gridpoint;
    """
    
    engine = get_db_connection()
    
    print("Fetching error metrics from database...")
    print(f"Target metrics: {TARGET_METRICS}")
    print("(Filtering server-side to avoid loading all 28M+ rows)")
    
    with engine.connect() as conn:
        df = pd.read_sql(
            text(query),
            conn,
            params={"metric_names": tuple(TARGET_METRICS)}
        )
    
    print(f"\nRetrieved {len(df):,} rows")
    print(f"Metrics found: {df['metric_name'].unique().tolist()}")
    print(f"Regions: {sorted(df['region'].unique().tolist())}")
    print(f"Physical variables: {sorted(df['physical_variable'].unique().tolist())}")
    
    return df


def get_summary_statistics(df):
    """
    Calculate summary statistics for each metric, region, and physical variable.
    
    Args:
        df: DataFrame with error metrics data
        
    Returns:
        pd.DataFrame: Summary statistics grouped by metric_name, region, and physical_variable
    """
    summary = df.groupby(['metric_name', 'region', 'physical_variable'])['mat_vector'].agg([
        ('count', 'count'),
        ('mean', 'mean'),
        ('std', 'std'),
        ('min', 'min'),
        ('q25', lambda x: x.quantile(0.25)),
        ('median', 'median'),
        ('q75', lambda x: x.quantile(0.75)),
        ('max', 'max')
    ]).reset_index()
    
    return summary


def save_data(df, output_dir=None):
    """
    Save the fetched data to CSV files.
    
    Args:
        df: DataFrame with error metrics data
        output_dir: Directory to save output files (defaults to metrics_explorer/output)
    """
    if output_dir is None:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, 'output')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save full dataset
    full_path = os.path.join(output_dir, 'error_metrics_full.csv')
    df.to_csv(full_path, index=False)
    print(f"\nSaved full dataset to: {full_path}")
    
    # Save summary statistics
    summary = get_summary_statistics(df)
    summary_path = os.path.join(output_dir, 'error_metrics_summary.csv')
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary statistics to: {summary_path}")
    
    # Save separate files for each metric
    for metric in df['metric_name'].unique():
        metric_df = df[df['metric_name'] == metric]
        # Clean metric name for filename (remove parentheses and spaces)
        safe_metric_name = metric.replace(' ', '_').replace('(', '').replace(')', '')
        metric_path = os.path.join(output_dir, f'{safe_metric_name}_data.csv')
        metric_df.to_csv(metric_path, index=False)
        print(f"Saved {metric} data to: {metric_path}")


def main():
    """Main execution function."""
    print("=" * 70)
    print("FETCHING ERROR METRICS DATA")
    print("=" * 70)
    
    try:
        # Fetch data from database
        df = fetch_error_metrics()
        
        if df.empty:
            print("\nWarning: No data retrieved from database!")
            print("Check that the target metrics exist in the database.")
            return
        
        # Save data to CSV files
        save_data(df)
        
        print("\n" + "=" * 70)
        print("DATA FETCH COMPLETE")
        print("=" * 70)
        print("\nNext step: Run visualize_metrics.py to generate visualizations")
        
    except Exception as e:
        print(f"\nError occurred: {e}")
        raise


if __name__ == "__main__":
    main()
