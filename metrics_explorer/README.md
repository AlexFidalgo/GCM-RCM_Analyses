# Metrics Explorer

Visualizations and analysis for error metrics dispersion across regions and physical variables.

## Overview

This toolkit queries the `climate_data` database to retrieve error metrics and generates comprehensive visualizations to explore the distribution of metric values across:
- **8 Regions**: AL, BI, EA, FR, IP, MD, ME, SC
- **2 Physical Variables**: ppt (precipitation), tas (temperature)
- **6 Target Metrics**: H10 (MAHE), d, dr, NED, MV, KGE (2009)

## Setup

### Prerequisites

1. **Python Environment**: Python 3.7+
2. **Required Packages**:
   ```bash
   pip install pandas sqlalchemy matplotlib seaborn numpy psycopg2-binary
   ```

3. **Database Connection**: Set the following environment variables:
   ```bash
   set DB_NAME=climate_data
   set DB_USER=your_username
   set DB_PASSWORD=your_password
   set DB_HOST=localhost
   set DB_PORT=5432
   ```

## Usage

### Step 1: Fetch and Filter Data from Database

The database contains 28M+ rows, so we first query and filter it down to only the target metrics:

```bash
python fetch_data.py
```

**What it does:**
- Queries the `error` and `metrics` tables from the database
- **Filters server-side** for only the 6 target metrics (reduces 28M rows to a manageable size)
- Saves filtered data to CSV files for analysis

**Output files:**
```
output/
├── error_metrics_full.csv      # Filtered dataset (target metrics only)
├── error_metrics_summary.csv   # Summary statistics by metric/region/variable
├── H10_MAHE_data.csv           # Individual metric files
├── d_data.csv
├── dr_data.csv
├── NED_data.csv
├── MV_data.csv
└── KGE_2009_data.csv
```

**Important:** This step only needs to be run once, or when you want to refresh the data from the database.

### Step 2: Generate Visualizations

Once the CSV files are created, generate all visualizations:

```bash
python visualize_metrics.py
```

**What it does:**
- Loads the filtered CSV data (much faster than querying 28M rows)
- Generates all visualizations
- Saves summary statistics

**What it generates:**

For each metric (e.g., `H10_(MAHE)/`):
```
output/
└── H10_MAHE/
    ├── ppt_histograms.png      # 8 histograms (one per region)
    ├── ppt_boxplot.png          # Box plot comparing all regions
    ├── ppt_violin.png           # Violin plot comparing all regions
    ├── ppt_combined.png         # Combined view (all above)
    ├── tas_histograms.png
    ├── tas_boxplot.png
    ├── tas_violin.png
    ├── tas_combined.png
    └── summary_statistics.csv   # Detailed stats by region/variable
```

## Visualization Types

### 1. Histogram Grid
- **Layout**: 2 rows × 4 columns (8 regions)
- **Shows**: Distribution shape, frequency
- **Includes**: Count, mean, std, median for each region

### 2. Box Plot Comparison
- **Layout**: Side-by-side box plots for all 8 regions
- **Shows**: Median, quartiles, outliers, range
- **Features**: Mean marked with red diamond

### 3. Violin Plot Comparison
- **Layout**: Side-by-side violin plots for all 8 regions
- **Shows**: Full distribution shape (density)
- **Features**: Median and mean lines included

### 4. Combined Visualization
- **Layout**: 3-row comprehensive view
  - Rows 1-2: Histogram grid (8 regions)
  - Row 3: Box plot (left) + Violin plot (right)
- **Purpose**: Complete overview on a single figure

## Understanding the Outputs

### Summary Statistics (`summary_statistics.csv`)

For each metric, region, and physical variable:
- **count**: Number of data points
- **mean**: Average mat_vector value
- **std**: Standard deviation
- **min/max**: Range of values
- **q25/median/q75**: Quartiles
- **skew**: Distribution asymmetry
- **kurtosis**: Distribution tail heaviness

### Interpretation Tips

1. **Comparing Regions**: 
   - Box plots are best for quick median/quartile comparisons
   - Violin plots show if distributions are unimodal or multimodal
   - Histograms reveal detailed distribution shapes

2. **Assessing Dispersion**:
   - Wide boxes/violins = high variability
   - Tall narrow violins = concentrated values
   - Long whiskers = presence of outliers

3. **Physical Variable Differences**:
   - Compare `ppt` vs `tas` visualizations for same metric
   - Check if patterns are consistent across variables

## Database Query

The data is fetched using:

```sql
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
    metrics.metric_name IN ('H10 (MAHE)', 'd', 'dr', 'NED', 'MV', 'KGE (2009)')
ORDER BY
    metrics.metric_name, error.region, error.physical_variable, error.gridpoint;
```

## Customization

### Adding More Metrics

Edit `fetch_data.py` and modify the `TARGET_METRICS` list:

```python
TARGET_METRICS = [
    'H10 (MAHE)',
    'd',
    'dr',
    'NED',
    'MV',
    'KGE (2009)',
    'Your New Metric'  # Add here
]
```

### Adjusting Plot Styles

In `visualize_metrics.py`, modify:
- `REGION_COLORS`: Change region color scheme
- `plt.rcParams`: Adjust font sizes, DPI
- Histogram bins: Change `bins=30` parameter
- Figure sizes: Modify `figsize=(width, height)`

### Running for Specific Metrics Only

Edit the metric directories you want to process in `visualize_metrics.py`:

```python
# In main(), after loading data:
metrics_to_process = ['H10 (MAHE)', 'KGE (2009)']  # Specify subset
for metric in metrics_to_process:
    process_metric(df, metric)
```

## Troubleshooting

### Database Connection Issues
- Verify environment variables are set correctly
- Check database is running and accessible
- Confirm user has read permissions on `error` and `metrics` tables

### Missing Data
- Ensure the database contains data for all target metrics
- Check metric names match exactly (including parentheses and spaces)

### Memory Issues with Large Datasets
- Process one metric at a time
- Reduce figure DPI: `plt.rcParams['savefig.dpi'] = 150`
- Use fewer histogram bins

## Output Structure

```
metrics_explorer/
├── fetch_data.py              # Data retrieval script
├── visualize_metrics.py       # Visualization generation script
├── README.md                  # This file
└── output/                    # Generated outputs
    ├── error_metrics_full.csv
    ├── error_metrics_summary.csv
    ├── H10_MAHE/
    │   ├── ppt_histograms.png
    │   ├── ppt_boxplot.png
    │   ├── ppt_violin.png
    │   ├── ppt_combined.png
    │   ├── tas_histograms.png
    │   ├── tas_boxplot.png
    │   ├── tas_violin.png
    │   ├── tas_combined.png
    │   └── summary_statistics.csv
    ├── d/
    ├── dr/
    ├── NED/
    ├── MV/
    └── KGE_2009/
```

## Next Steps

1. Run `fetch_data.py` to query and filter the database (28M+ rows → target metrics only)
2. Run `visualize_metrics.py` to generate all visualizations from the filtered CSV
3. Review the combined plots first for overview
4. Examine individual histograms for detailed distribution shapes
5. Check summary statistics CSV for numerical analysis
6. Compare patterns across metrics and physical variables

## Why Two Steps?

The database has **28M+ rows** in the error table. Instead of loading all that data every time:
1. **fetch_data.py** filters on the database side (WHERE clause) to get only the 6 target metrics
2. **visualize_metrics.py** works with the much smaller filtered dataset
3. You only need to run step 1 once (or when data updates)

## Notes

- All visualizations use consistent color schemes for regions
- High-resolution outputs (300 DPI) suitable for publications
- Statistics include skewness and kurtosis for advanced analysis
- Data is sorted consistently across all outputs for reproducibility
