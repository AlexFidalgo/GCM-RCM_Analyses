# Metrics Correlation Analysis

## Overview

This project analyzes **correlations between error metrics** computed for GCM-RCM (Global Climate Model - Regional Climate Model) climate simulations. 

The goal is to:
1. Identify redundancy among error metrics (high correlation → similar information)
2. Discover orthogonal error facets (low/negative correlation → complementary information)
3. Generate visualizations (scatter plot matrix, heatmaps) to understand metric relationships

## Data Structure

For each combination of `(region, gridpoint, physical_variable, model)`, we have multiple error metrics computed. The correlation analysis compares these metrics **only within the same group**, ensuring we're correlating values from the same data.

Example:
- Region: France (FR)
- Gridpoint: 652
- Physical Variable: ppt (precipitation)
- Model: 1 (GCM-RCM pair)

This group has multiple metrics: ACC, d, KGE (2009), etc. We correlate these metric values across all such groups.

## Files

- **metrics_correlation_analysis.ipynb**: Main Jupyter notebook with all analysis code
- **README.md**: This file
- **output/**: Directory for saving results (automatically created)

## Requirements

```
pandas
numpy
scipy
matplotlib
seaborn
sqlite3 (built-in)
```

Install with:
```bash
pip install pandas numpy scipy matplotlib seaborn
```

## Configuration

Edit these parameters in the notebook (Section 1) to customize your analysis:

```python
REGIONS = ['FR']  # or None for all regions
PHYSICAL_VARIABLES = ['ppt']  # or None for all variables
METRIC_ABBREVIATIONS = None  # None for all metrics, or list specific ones
CORRELATION_METHOD = 'pearson'  # 'pearson', 'spearman', or 'kendall'
```

### Available Values

**Regions**: `AL`, `BI`, `EA`, `FR`, `IP`, `MD`, `ME`, `SC`

**Physical Variables**: `ppt` (precipitation), `tas` (temperature)

**Metric Abbreviations** (examples): `ACC`, `D1`, `d`, `dr`, `KGE (2009)`, `KGE (2012)`, `BM`, `H10 (MAHE)`, `MdAE`, `NED`

## Workflow

1. **Load Configuration**: Set filters for regions, variables, and metrics
2. **Load Data**: Query from database
3. **Filter & Preprocess**: Apply user-defined filters
4. **Create Pivot Table**: Transform data so each row is a `(region, gridpoint, physical_variable, model)` group and each column is a metric
5. **Compute Correlations**: Calculate pairwise correlations with p-values
6. **Visualize**: Generate scatter plot matrix and correlation heatmap
7. **Export**: Save results to CSV files

## Output Files

The notebook generates the following files in `output/`:

- **pairplot_{method}.png**: Scatter plot matrix showing all pairwise relationships
- **heatmap_clustered_{method}.png**: Hierarchical correlation heatmap
- **correlation_matrix_{method}.csv**: Full correlation matrix
- **pairwise_correlations_{method}.csv**: Table of all pairwise correlations with p-values
- **metrics_summary_statistics.csv**: Mean, std, min, max for each metric
- **analysis_metadata.csv**: Analysis parameters and metadata

## Database Connection

The notebook expects a SQLite database with the following tables:
- **error**: Error values with columns `(region, gridpoint, physical_variable, model, rcm_id, gcm_id, metric_id, mat_vector)`
- **metrics**: Metric metadata with columns `(id, metric_name)` and optionally `(region, physical_variable)`

Adjust `DB_PATH` in the notebook if your database is in a different location.

## Interpretation Guide

### Correlation Values
- **Correlation ≈ 1**: Metrics move together (likely redundant)
- **Correlation ≈ 0**: Metrics are independent (complementary information)
- **Correlation ≈ -1**: Metrics move in opposite directions

### Statistical Significance
- **p < 0.05** (*): Statistically significant correlation
- **p < 0.01** (**): Highly significant
- **p < 0.001** (***): Very highly significant
- **p ≥ 0.05** (ns): Not significant

## Example Usage

### Run analysis for all precipitation metrics in France:
```python
REGIONS = ['FR']
PHYSICAL_VARIABLES = ['ppt']
METRIC_ABBREVIATIONS = None  # All metrics for ppt in FR
CORRELATION_METHOD = 'pearson'
```

### Run analysis for specific metrics across all regions:
```python
REGIONS = None  # All regions
PHYSICAL_VARIABLES = None  # All variables
METRIC_ABBREVIATIONS = ['ACC', 'd', 'KGE (2009)', 'BM']
CORRELATION_METHOD = 'spearman'
```

## Notes

- The analysis requires at least a few observations (rows) with complete metric data
- If many metrics are missing for certain groups, consider adjusting filters or interpolation strategies
- Correlation strength depends on data variability; regions/variables with low variance may show spurious correlations

## Author

Master's thesis project on climate model gap-filling (GCM-RCM analysis)
