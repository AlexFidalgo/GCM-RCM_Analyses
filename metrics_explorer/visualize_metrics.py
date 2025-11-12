"""
Generate visualizations for error metrics distributions.

This script creates multiple types of visualizations (histograms, box plots, 
violin plots) to explore the distribution of error metrics across regions,
physical variables, and models.

The data should first be fetched using fetch_data.py, which filters the 
28M+ row database table down to only the target metrics before loading.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Define consistent colors for regions
REGION_COLORS = {
    'AL': '#1f77b4',  # blue
    'BI': '#ff7f0e',  # orange
    'EA': '#2ca02c',  # green
    'FR': '#d62728',  # red
    'IP': '#9467bd',  # purple
    'MD': '#8c564b',  # brown
    'ME': '#e377c2',  # pink
    'SC': '#7f7f7f',  # gray
}


def _compute_shared_axis(series: pd.Series, n_bins: int = 30):
    """
    Compute shared histogram bins and axis limits for a set of values.

    Ensures all histograms use the same x-axis range so that regions are
    comparable at a glance.
    """
    clean_values = series.replace([np.inf, -np.inf], np.nan).dropna()
    if clean_values.empty:
        return None, None

    min_val = clean_values.min()
    max_val = clean_values.max()

    if min_val < 0 < max_val:
        max_abs = max(abs(min_val), abs(max_val))
        x_min, x_max = -max_abs, max_abs
    else:
        spread = max_val - min_val
        padding = 0.05 * spread if spread > 0 else 0.05 * max(abs(min_val), abs(max_val), 1)
        x_min = min_val - padding
        x_max = max_val + padding

    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5

    bins = np.linspace(x_min, x_max, n_bins + 1)
    return bins, (x_min, x_max)


def load_data(input_file=None):
    """
    Load the error metrics data from CSV.
    
    The CSV file should be generated first using fetch_data.py, which 
    filters the 28M+ row database down to only the target metrics.
    
    Args:
        input_file: Path to the CSV file with error metrics (defaults to metrics_explorer/output/error_metrics_full.csv)
        
    Returns:
        pd.DataFrame: Loaded data
    """
    if input_file is None:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        input_file = os.path.join(script_dir, 'output', 'error_metrics_full.csv')
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(
            f"Data file not found: {input_file}\n"
            "Please run fetch_data.py first to retrieve and filter data from the database.\n"
            "The database has 28M+ rows, so we filter it first to avoid loading everything."
        )
    
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df):,} rows")
    return df


def create_histogram_grid(df, metric_name, physical_var, output_dir):
    """
    Create a grid of histograms, one for each region.
    
    Args:
        df: DataFrame filtered for specific metric and physical variable
        metric_name: Name of the metric
        physical_var: Physical variable (ppt or tas)
        output_dir: Directory to save the figure
    """
    regions = sorted(df['region'].unique())
    n_regions = len(regions)
    
    # Create subplot grid (2 rows x 4 columns for 8 regions)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    shared_bins, x_limits = _compute_shared_axis(df['mat_vector'], n_bins=30)

    for idx, region in enumerate(regions):
        ax = axes[idx]
        region_data = df[df['region'] == region]['mat_vector']
        
        # Plot histogram
        hist_kwargs = dict(
            color=REGION_COLORS.get(region, 'gray'),
                alpha=0.7, edgecolor='black', linewidth=0.5)
        if shared_bins is not None:
            ax.hist(region_data, bins=shared_bins, **hist_kwargs)
            ax.set_xlim(x_limits)
        else:
            ax.hist(region_data, bins=30, **hist_kwargs)
        
        # Add statistics text
        mean_val = region_data.mean()
        std_val = region_data.std()
        median_val = region_data.median()
        
        stats_text = f'n={len(region_data)}\nμ={mean_val:.3f}\nσ={std_val:.3f}\nmed={median_val:.3f}'
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=8)
        
        ax.set_title(f'{region}', fontweight='bold', fontsize=12)
        ax.set_xlabel('mat_vector', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # Hide any unused subplots
    for idx in range(n_regions, len(axes)):
        axes[idx].axis('off')
    
    # Overall title
    fig.suptitle(f'{metric_name} - {physical_var.upper()}\nDistribution by Region', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    # Save figure
    safe_metric_name = metric_name.replace(' ', '_').replace('(', '').replace(')', '')
    output_path = os.path.join(output_dir, f'{physical_var}_histograms.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved histogram grid: {output_path}")


def create_boxplot_comparison(df, metric_name, physical_var, output_dir):
    """
    Create box plots comparing all regions side by side.
    
    Args:
        df: DataFrame filtered for specific metric and physical variable
        metric_name: Name of the metric
        physical_var: Physical variable (ppt or tas)
        output_dir: Directory to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    regions = sorted(df['region'].unique())
    colors = [REGION_COLORS.get(r, 'gray') for r in regions]
    
    # Create box plot
    box_parts = ax.boxplot(
        [df[df['region'] == r]['mat_vector'].values for r in regions],
        labels=regions,
        patch_artist=True,
        widths=0.6,
        showmeans=True,
        meanprops=dict(marker='D', markerfacecolor='red', markersize=6)
    )
    
    # Color the boxes
    for patch, color in zip(box_parts['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Region', fontsize=12, fontweight='bold')
    ax.set_ylabel('mat_vector', fontsize=12, fontweight='bold')
    ax.set_title(f'{metric_name} - {physical_var.upper()}\nBox Plot Comparison Across Regions', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    safe_metric_name = metric_name.replace(' ', '_').replace('(', '').replace(')', '')
    output_path = os.path.join(output_dir, f'{physical_var}_boxplot.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved box plot: {output_path}")


def create_violin_plot(df, metric_name, physical_var, output_dir):
    """
    Create violin plots showing distribution shapes across regions.
    
    Args:
        df: DataFrame filtered for specific metric and physical variable
        metric_name: Name of the metric
        physical_var: Physical variable (ppt or tas)
        output_dir: Directory to save the figure
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    regions = sorted(df['region'].unique())
    
    # Create violin plot
    parts = ax.violinplot(
        [df[df['region'] == r]['mat_vector'].values for r in regions],
        positions=range(len(regions)),
        showmeans=True,
        showmedians=True,
        widths=0.7
    )
    
    # Color the violins
    for idx, pc in enumerate(parts['bodies']):
        pc.set_facecolor(REGION_COLORS.get(regions[idx], 'gray'))
        pc.set_alpha(0.7)
    
    ax.set_xticks(range(len(regions)))
    ax.set_xticklabels(regions)
    ax.set_xlabel('Region', fontsize=12, fontweight='bold')
    ax.set_ylabel('mat_vector', fontsize=12, fontweight='bold')
    ax.set_title(f'{metric_name} - {physical_var.upper()}\nViolin Plot Comparison Across Regions', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    safe_metric_name = metric_name.replace(' ', '_').replace('(', '').replace(')', '')
    output_path = os.path.join(output_dir, f'{physical_var}_violin.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved violin plot: {output_path}")


def create_combined_comparison(df, metric_name, physical_var, output_dir):
    """
    Create a combined figure with histogram, box plot, and violin plot.
    
    Args:
        df: DataFrame filtered for specific metric and physical variable
        metric_name: Name of the metric
        physical_var: Physical variable (ppt or tas)
        output_dir: Directory to save the figure
    """
    fig = plt.figure(figsize=(18, 12))
    
    # Create grid for subplots
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    regions = sorted(df['region'].unique())
    
    # Top 2 rows: Histograms (8 regions)
    shared_bins, x_limits = _compute_shared_axis(df['mat_vector'], n_bins=25)

    for idx, region in enumerate(regions):
        row = idx // 4
        col = idx % 4
        ax = fig.add_subplot(gs[row, col])
        
        region_data = df[df['region'] == region]['mat_vector']
        
        hist_kwargs = dict(
            color=REGION_COLORS.get(region, 'gray'),
            alpha=0.7, edgecolor='black', linewidth=0.5)
        if shared_bins is not None:
            ax.hist(region_data, bins=shared_bins, **hist_kwargs)
            ax.set_xlim(x_limits)
        else:
            ax.hist(region_data, bins=25, **hist_kwargs)
        
        mean_val = region_data.mean()
        median_val = region_data.median()
        
        stats_text = f'n={len(region_data)}\nμ={mean_val:.3f}\nmed={median_val:.3f}'
        ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=7)
        
        ax.set_title(f'{region}', fontweight='bold', fontsize=11)
        ax.set_xlabel('mat_vector', fontsize=9)
        ax.set_ylabel('Frequency', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Bottom row: Box plot (left) and Violin plot (right)
    ax_box = fig.add_subplot(gs[2, :2])
    ax_violin = fig.add_subplot(gs[2, 2:])
    
    # Box plot
    colors = [REGION_COLORS.get(r, 'gray') for r in regions]
    box_parts = ax_box.boxplot(
        [df[df['region'] == r]['mat_vector'].values for r in regions],
        labels=regions,
        patch_artist=True,
        widths=0.6,
        showmeans=True,
        meanprops=dict(marker='D', markerfacecolor='red', markersize=5)
    )
    
    for patch, color in zip(box_parts['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax_box.set_xlabel('Region', fontsize=10, fontweight='bold')
    ax_box.set_ylabel('mat_vector', fontsize=10, fontweight='bold')
    ax_box.set_title('Box Plot Comparison', fontsize=11, fontweight='bold')
    ax_box.grid(True, alpha=0.3, axis='y')
    
    # Violin plot
    parts = ax_violin.violinplot(
        [df[df['region'] == r]['mat_vector'].values for r in regions],
        positions=range(len(regions)),
        showmeans=True,
        showmedians=True,
        widths=0.7
    )
    
    for idx, pc in enumerate(parts['bodies']):
        pc.set_facecolor(REGION_COLORS.get(regions[idx], 'gray'))
        pc.set_alpha(0.7)
    
    ax_violin.set_xticks(range(len(regions)))
    ax_violin.set_xticklabels(regions)
    ax_violin.set_xlabel('Region', fontsize=10, fontweight='bold')
    ax_violin.set_ylabel('mat_vector', fontsize=10, fontweight='bold')
    ax_violin.set_title('Violin Plot Comparison', fontsize=11, fontweight='bold')
    ax_violin.grid(True, alpha=0.3, axis='y')
    
    # Overall title
    fig.suptitle(f'{metric_name} - {physical_var.upper()}\nComprehensive Distribution Analysis', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Save figure
    safe_metric_name = metric_name.replace(' ', '_').replace('(', '').replace(')', '')
    output_path = os.path.join(output_dir, f'{physical_var}_combined.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved combined plot: {output_path}")


def save_metric_summary_stats(df, metric_name, output_dir):
    """
    Save summary statistics for a specific metric to CSV.
    
    Args:
        df: DataFrame filtered for specific metric
        metric_name: Name of the metric
        output_dir: Directory to save the summary
    """
    summary = df.groupby(['region', 'physical_variable'])['mat_vector'].agg([
        ('count', 'count'),
        ('mean', 'mean'),
        ('std', 'std'),
        ('min', 'min'),
        ('q25', lambda x: x.quantile(0.25)),
        ('median', 'median'),
        ('q75', lambda x: x.quantile(0.75)),
        ('max', 'max'),
        ('skew', lambda x: pd.Series(x).skew()),
        ('kurtosis', lambda x: pd.Series(x).kurtosis())
    ]).reset_index()
    
    safe_metric_name = metric_name.replace(' ', '_').replace('(', '').replace(')', '')
    output_path = os.path.join(output_dir, f'summary_statistics.csv')
    summary.to_csv(output_path, index=False)
    
    print(f"  Saved summary statistics: {output_path}")


def process_metric(df, metric_name, base_output_dir=None):
    """
    Process a single metric: create all visualizations and statistics.
    
    Args:
        df: Full DataFrame
        metric_name: Name of the metric to process
        base_output_dir: Base output directory (defaults to metrics_explorer/output)
    """
    if base_output_dir is None:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_output_dir = os.path.join(script_dir, 'output')
    
    print(f"\nProcessing: {metric_name}")
    print("-" * 70)
    
    # Create metric-specific output directory
    safe_metric_name = metric_name.replace(' ', '_').replace('(', '').replace(')', '')
    metric_dir = os.path.join(base_output_dir, safe_metric_name)
    os.makedirs(metric_dir, exist_ok=True)
    
    # Filter data for this metric
    metric_df = df[df['metric_name'] == metric_name].copy()
    
    if metric_df.empty:
        print(f"  Warning: No data found for {metric_name}")
        return
    
    print(f"  Found {len(metric_df):,} records")
    
    # Save summary statistics
    save_metric_summary_stats(metric_df, metric_name, metric_dir)
    
    # Process each physical variable
    for physical_var in sorted(metric_df['physical_variable'].unique()):
        print(f"\n  Processing {physical_var}:")
        var_df = metric_df[metric_df['physical_variable'] == physical_var]
        
        # Create all visualization types
        create_histogram_grid(var_df, metric_name, physical_var, metric_dir)
        create_boxplot_comparison(var_df, metric_name, physical_var, metric_dir)
        create_violin_plot(var_df, metric_name, physical_var, metric_dir)
        create_combined_comparison(var_df, metric_name, physical_var, metric_dir)


def main():
    """Main execution function."""
    print("=" * 70)
    print("GENERATING METRIC VISUALIZATIONS")
    print("=" * 70)
    
    try:
        # Load data from CSV (filtered by fetch_data.py from 28M+ rows)
        df = load_data()
        
        if df.empty:
            print("\nWarning: No data in CSV file!")
            return
        
        # Get list of metrics to process
        metrics = sorted(df['metric_name'].unique())
        print(f"\nMetrics to process: {metrics}")
        
        # Process each metric
        for metric in metrics:
            process_metric(df, metric)
        
        # Get output directory path for display
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, 'output')
        
        print("\n" + "=" * 70)
        print("VISUALIZATION GENERATION COMPLETE")
        print("=" * 70)
        print(f"\nAll outputs saved to: {output_path}")
        print("\nGenerated for each metric and physical variable:")
        print("  - Histogram grid (8 regions)")
        print("  - Box plot comparison")
        print("  - Violin plot comparison")
        print("  - Combined visualization")
        print("  - Summary statistics (CSV)")
        
    except Exception as e:
        print(f"\nError occurred: {e}")
        raise


if __name__ == "__main__":
    main()
