"""Location performance analysis for Pastas R Us, Inc.

Answers the two questions the executive team asked:

1. Which site selection criteria actually predict annual sales?
2. Does the loyalty card program earn its budget?

Usage:
    python src/analysis.py --data data/pastas_r_us_locations.csv --out reports
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from scipy import stats  # noqa: E402

TARGET = "annual_sales_k"

PREDICTORS = [
    "median_income",
    "median_age",
    "pct_bachelors",
    "loyalty_pct",
    "avg_spend",
    "sqft",
]

LABELS = {
    "sqft": "Square feet",
    "avg_spend": "Average spend per person ($)",
    "sales_growth_pct": "Sales growth over prior year (%)",
    "loyalty_pct": "Loyalty card share of net sales (%)",
    "sales_per_sqft": "Annual sales per sq ft ($)",
    "median_income": "Median household income, 3 mi ($)",
    "median_age": "Median age, 3 mi",
    "pct_bachelors": "Bachelor's degree share, 3 mi (%)",
    "annual_sales_k": "Annual sales ($000)",
}


def load(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in PREDICTORS + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def descriptives(df: pd.DataFrame) -> pd.DataFrame:
    """Mean, median, spread, quartiles, and IQR for every numeric column."""
    num = df.drop(columns=["location_id"], errors="ignore").select_dtypes("number")
    out = pd.DataFrame(
        {
            "n": num.count(),
            "mean": num.mean(),
            "median": num.median(),
            "std_dev": num.std(),
            "min": num.min(),
            "q1": num.quantile(0.25),
            "q3": num.quantile(0.75),
            "max": num.max(),
        }
    )
    out["iqr"] = out["q3"] - out["q1"]
    return out.round(2)


def iqr_outliers(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Rows outside the 1.5 x IQR fence for one column."""
    q1, q3 = df[column].quantile([0.25, 0.75])
    fence = 1.5 * (q3 - q1)
    return df[(df[column] < q1 - fence) | (df[column] > q3 + fence)]


def simple_regressions(df: pd.DataFrame) -> pd.DataFrame:
    """One predictor at a time against annual sales."""
    rows = []
    for col in PREDICTORS:
        res = stats.linregress(df[col], df[TARGET])
        rows.append(
            {
                "predictor": col,
                "r": res.rvalue,
                "r_squared": res.rvalue**2,
                "slope": res.slope,
                "p_value": res.pvalue,
                "significant_at_05": res.pvalue < 0.05,
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("r_squared", ascending=False)
        .reset_index(drop=True)
        .round(4)
    )


def multiple_regression(df: pd.DataFrame, predictors: list[str] | None = None):
    """Ordinary least squares with all predictors entered together."""
    predictors = predictors or PREDICTORS
    X = sm.add_constant(df[predictors])
    return sm.OLS(df[TARGET], X).fit()


def loyalty_test(df: pd.DataFrame) -> dict:
    """Compare sales for high versus low loyalty share locations (median split)."""
    cut = df["loyalty_pct"].median()
    high = df.loc[df["loyalty_pct"] > cut, TARGET]
    low = df.loc[df["loyalty_pct"] <= cut, TARGET]
    t = stats.ttest_ind(high, low, equal_var=False)
    return {
        "median_split": round(float(cut), 2),
        "high_loyalty_mean_sales_k": round(float(high.mean()), 1),
        "low_loyalty_mean_sales_k": round(float(low.mean()), 1),
        "welch_t": round(float(t.statistic), 3),
        "p_value": round(float(t.pvalue), 4),
    }


def plot_histogram(df: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df[TARGET], bins=10, color="#1f4e79", edgecolor="white")
    ax.axvline(df[TARGET].mean(), color="#c9a227", lw=2, label="Mean")
    ax.axvline(df[TARGET].median(), color="#2a9d8f", lw=2, ls="--", label="Median")
    ax.set(title="Distribution of annual sales (74 locations)", xlabel=LABELS[TARGET], ylabel="Locations")
    ax.legend()
    path = out / "sales_histogram.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_boxplot(df: pd.DataFrame, out: Path) -> Path:
    """Built from the raw rows, not from summary statistics."""
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.boxplot(df[TARGET], orientation="horizontal", widths=0.6, patch_artist=True,
               boxprops={"facecolor": "#d6e4f0"}, medianprops={"color": "#1f4e79", "lw": 2})
    ax.set(title="Annual sales boxplot with 1.5 x IQR whiskers", xlabel=LABELS[TARGET], yticks=[])
    path = out / "sales_boxplot.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_scatter_grid(df: pd.DataFrame, out: Path) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, col in zip(axes.flat, PREDICTORS):
        x, y = df[col], df[TARGET]
        res = stats.linregress(x, y)
        ax.scatter(x, y, s=22, color="#1f4e79", alpha=0.75)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, res.intercept + res.slope * xs, color="#c9a227", lw=2)
        ax.set(xlabel=LABELS[col], ylabel=LABELS[TARGET] if col in ("median_income", "loyalty_pct") else "")
        ax.set_title(f"r = {res.rvalue:.2f}, R² = {res.rvalue**2:.2f}", fontsize=10)
    fig.suptitle("Site selection criteria versus annual sales", fontsize=14)
    path = out / "predictor_scatterplots.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_correlation(df: pd.DataFrame, out: Path) -> Path:
    cols = PREDICTORS + [TARGET]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)), cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Correlation matrix")
    path = out / "correlation_matrix.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run(data: str | Path, out: str | Path) -> dict:
    out = Path(out)
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    df = load(data)
    desc = descriptives(df)
    simple = simple_regressions(df)
    model = multiple_regression(df)
    loyalty = loyalty_test(df)
    outliers = iqr_outliers(df, TARGET)

    desc.to_csv(out / "descriptive_statistics.csv")
    simple.to_csv(out / "simple_regressions.csv", index=False)
    (out / "multiple_regression.txt").write_text(model.summary().as_text())

    for plot in (plot_histogram, plot_boxplot, plot_scatter_grid, plot_correlation):
        plot(df, figs)

    return {
        "n_locations": len(df),
        "top_predictor": simple.iloc[0]["predictor"],
        "top_r_squared": float(simple.iloc[0]["r_squared"]),
        "multiple_r_squared": round(float(model.rsquared), 3),
        "multiple_adj_r_squared": round(float(model.rsquared_adj), 3),
        "sales_outliers": outliers["location_id"].tolist(),
        "loyalty": loyalty,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="data/pastas_r_us_locations.csv")
    parser.add_argument("--out", default="reports")
    args = parser.parse_args()
    summary = run(args.data, args.out)
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
