#!/usr/bin/env python
"""
Analyze the distribution of `mat_vector` for one or many metrics
of a given physical_variable.

Configuration is done via global variables at the top of the file.

Results are saved into:
    results/<physical_variable>/<metric_slug>_results.txt
    + optional summary CSV when running all metrics.

Plots saved into:
    figs/<physical_variable>/<metric_slug>_hist.png
    figs/<physical_variable>/<metric_slug>_qq.png
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from scipy import stats
import matplotlib.pyplot as plt
from tqdm import tqdm


# ---------------------------------------------------------------------------
# GLOBAL CONFIGURATION
# ---------------------------------------------------------------------------

PHYSICAL_VARIABLE = "ppt"     # "ppt" or "tas"
METRIC_NAME = None            # e.g. "ACC" or None for ALL metrics
MAKE_PLOTS = True             # Save histogram + QQ

BASE_DIR = Path(__file__).resolve().parent
FIGS_BASE_DIR = BASE_DIR / "figs"
RESULTS_BASE_DIR = BASE_DIR / "results"


# ---------------------------------------------------------------------------
# DB CONNECTION
# ---------------------------------------------------------------------------

def get_engine():
    DB_NAME = os.getenv("DB_NAME")
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = os.getenv("DB_PORT")

    if not all([DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT]):
        raise RuntimeError("Missing DB_* environment variables.")

    return create_engine(
        f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

def load_mat_vector(engine, physical_variable: str, metric_name: str) -> pd.DataFrame:
    query = text("""
        SELECT
            e.region,
            e.gridpoint,
            e.physical_variable,
            e.model,
            e.rcm_id,
            e.gcm_id,
            m.metric_name,
            e.mat_vector
        FROM error e
        LEFT JOIN metrics m ON m.id = e.metric_id
        WHERE e.physical_variable = :physical_variable
          AND m.metric_name = :metric_name
    """)

    df = pd.read_sql_query(
        query,
        con=engine,
        params={"physical_variable": physical_variable, "metric_name": metric_name},
    )

    return df.dropna(subset=["mat_vector"])


def get_metric_names_for_physical_variable(engine, physical_variable: str):
    query = text("""
        SELECT DISTINCT m.metric_name
        FROM error e
        LEFT JOIN metrics m ON m.id = e.metric_id
        WHERE e.physical_variable = :physical_variable
          AND m.metric_name IS NOT NULL
        ORDER BY m.metric_name
    """)
    df = pd.read_sql_query(query, con=engine, params={"physical_variable": physical_variable})
    return df["metric_name"].tolist()


# ---------------------------------------------------------------------------
# NORMALITY + SYMMETRY
# ---------------------------------------------------------------------------

def check_normality_and_symmetry(series: pd.Series, max_shapiro_n: int = 5000):
    x = series.to_numpy()
    n = len(x)

    desc = {
        "n": n,
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "q1": float(np.quantile(x, 0.25)),
        "median": float(np.median(x)),
        "q3": float(np.quantile(x, 0.75)),
    }

    skew = stats.skew(x, bias=False)
    kurtosis = stats.kurtosis(x, fisher=True, bias=False)

    skew_test = stats.skewtest(x) if n >= 8 else None
    normal_test = stats.normaltest(x) if n >= 8 else None

    if n > max_shapiro_n:
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(n, size=max_shapiro_n, replace=False)
        x_shapiro = x[sample_idx]
    else:
        x_shapiro = x

    shapiro = stats.shapiro(x_shapiro) if len(x_shapiro) >= 3 else None

    # KS vs fitted N(μ,σ)
    if desc["std"] > 0:
        ks = stats.kstest((x - desc["mean"]) / desc["std"], "norm")
    else:
        ks = None

    return {
        "desc": desc,
        "skew": float(skew),
        "kurtosis": float(kurtosis),
        "skew_test": skew_test,
        "normal_test": normal_test,
        "shapiro": shapiro,
        "ks": ks,
    }


# ---------------------------------------------------------------------------
# HOMOSCEDASTICITY
# ---------------------------------------------------------------------------

def _group_values_for_test(df: pd.DataFrame, group_col: str, value_col: str,
                           min_group_size: int = 5, max_groups: int = 20):
    grouped = df.groupby(group_col)[value_col]
    sizes = grouped.size().sort_values(ascending=False)

    selected_ids = sizes[sizes >= min_group_size].head(max_groups).index.tolist()
    groups = [grouped.get_group(g).to_numpy() for g in selected_ids]
    return selected_ids, groups


def check_homoscedasticity(df: pd.DataFrame, value_col: str = "mat_vector"):
    out = {}

    for factor in ["gcm_id", "rcm_id"]:
        ids, groups = _group_values_for_test(df, factor, value_col)

        if len(groups) < 2:
            out[factor] = {
                "n_groups": len(groups),
                "used_levels": ids,
                "levene": None,
                "bartlett": None,
                "note": "Not enough groups for tests.",
            }
            continue

        levene = stats.levene(*groups, center="median")
        bartlett = stats.bartlett(*groups)

        out[factor] = {
            "n_groups": len(groups),
            "used_levels": ids,
            "levene": levene,
            "bartlett": bartlett,
            "note": f"Used {len(groups)} groups with >=5 observations.",
        }

    return out


# ---------------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------------

def slugify_metric_name(metric_name: str) -> str:
    slug = "".join(
        ch.lower() if ch.isalnum() else "_"
        for ch in metric_name
    )
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def make_plots(series: pd.Series, physical_variable: str, metric_name: str):
    metric_slug = slugify_metric_name(metric_name)
    figs_dir = FIGS_BASE_DIR / physical_variable
    figs_dir.mkdir(parents=True, exist_ok=True)

    x = series.to_numpy()
    prefix = f"{physical_variable} · {metric_name}"

    # Histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(x, bins=60, alpha=0.8)
    ax.set_title(f"{prefix} – Histogram")
    fig.tight_layout()
    fig.savefig(figs_dir / f"{metric_slug}_hist.png")
    plt.close(fig)

    # QQ plot
    fig, ax = plt.subplots(figsize=(8, 5))
    stats.probplot(x, dist="norm", plot=ax)
    ax.set_title(f"{prefix} – QQ Plot")
    fig.tight_layout()
    fig.savefig(figs_dir / f"{metric_slug}_qq.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# SAVE RESULTS TO FILE
# ---------------------------------------------------------------------------

def save_results_text(out_path: Path, physical_variable: str, metric_name: str,
                      norm_sym, homo):
    with open(out_path, "w", encoding="utf-8") as f:

        f.write(f"Physical variable: {physical_variable}\n")
        f.write(f"Metric name:       {metric_name}\n\n")

        f.write("=== Descriptive statistics ===\n")
        for k, v in norm_sym["desc"].items():
            f.write(f"{k:>8}: {v}\n")
        f.write("\n")

        f.write("=== Symmetry ===\n")
        f.write(f"skewness        : {norm_sym['skew']}\n")
        f.write(f"kurtosis        : {norm_sym['kurtosis']}\n")

        st = norm_sym["skew_test"]
        if st is not None:
            f.write(f"skew_test       : stat={st.statistic}, p={st.pvalue}\n")
        else:
            f.write("skew_test       : not computed\n")
        f.write("\n")

        f.write("=== Normality tests ===\n")
        nt = norm_sym["normal_test"]
        if nt is not None:
            f.write(f"D’Agostino-Pearson: stat={nt.statistic}, p={nt.pvalue}\n")
        else:
            f.write("D’Agostino-Pearson: not computed\n")

        sh = norm_sym["shapiro"]
        if sh is not None:
            f.write(f"Shapiro-Wilk      : stat={sh.statistic}, p={sh.pvalue}\n")
        else:
            f.write("Shapiro-Wilk      : not computed\n")

        ks = norm_sym["ks"]
        if ks is not None:
            f.write(f"KS fitted normal   : stat={ks.statistic}, p={ks.pvalue}\n")
        else:
            f.write("KS fitted normal   : not computed\n")

        f.write("\n=== Homoscedasticity ===\n")
        for factor, res in homo.items():
            f.write(f"\n-- Across {factor.upper()} --\n")
            f.write(f"used_levels: {res['used_levels']}\n")
            f.write(f"n_groups   : {res['n_groups']}\n")

            if res["levene"] is not None:
                lv = res["levene"]
                f.write(f"Levene     : stat={lv.statistic}, p={lv.pvalue}\n")
            else:
                f.write("Levene     : not computed\n")

            if res["bartlett"] is not None:
                bt = res["bartlett"]
                f.write(f"Bartlett   : stat={bt.statistic}, p={bt.pvalue}\n")
            else:
                f.write("Bartlett   : not computed\n")

            f.write(f"Note: {res['note']}\n")


# ---------------------------------------------------------------------------
# ANALYZE ONE METRIC
# ---------------------------------------------------------------------------

def analyze_one_metric(engine, physical_variable: str, metric_name: str):
    df = load_mat_vector(engine, physical_variable, metric_name)
    if df.empty:
        return None

    x = df["mat_vector"]

    norm_sym = check_normality_and_symmetry(x)
    homo = check_homoscedasticity(df)

    # Save results to text file
    metric_slug = slugify_metric_name(metric_name)
    out_dir = RESULTS_BASE_DIR / physical_variable
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{metric_slug}_results.txt"
    save_results_text(out_path, physical_variable, metric_name, norm_sym, homo)

    # Save plots
    if MAKE_PLOTS:
        make_plots(x, physical_variable, metric_name)

    # Return summary row for later CSV aggregation
    return {
        "metric_name": metric_name,
        "n": norm_sym["desc"]["n"],
        "skew": norm_sym["skew"],
        "kurtosis": norm_sym["kurtosis"],
        "normal_test_p": norm_sym["normal_test"].pvalue if norm_sym["normal_test"] else None,
        "shapiro_p": norm_sym["shapiro"].pvalue if norm_sym["shapiro"] else None,
        "ks_p": norm_sym["ks"].pvalue if norm_sym["ks"] else None,
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    engine = get_engine()

    if METRIC_NAME is None:
        # run for all metrics
        metric_names = get_metric_names_for_physical_variable(engine, PHYSICAL_VARIABLE)
        summary_rows = []

        for m in tqdm(metric_names, desc=f"Metrics for {PHYSICAL_VARIABLE}"):
            row = analyze_one_metric(engine, PHYSICAL_VARIABLE, m)
            if row:
                summary_rows.append(row)

        # Save summary CSV
        if summary_rows:
            out_dir = RESULTS_BASE_DIR / PHYSICAL_VARIABLE
            out_dir.mkdir(parents=True, exist_ok=True)
            df_sum = pd.DataFrame(summary_rows)
            df_sum.to_csv(out_dir / "summary.csv", index=False)

    else:
        analyze_one_metric(engine, PHYSICAL_VARIABLE, METRIC_NAME)


if __name__ == "__main__":
    main()
