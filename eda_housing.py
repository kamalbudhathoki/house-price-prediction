import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

DATA_PATH = Path("data/train.csv")
OUT_DIR = Path("plots")
OUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")
TARGET = "SalePrice"


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def plot_target_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df[TARGET], kde=True, ax=axes[0], color="#2c7fb8")
    axes[0].set_title("SalePrice distribution (raw)")
    log_price = np.log1p(df[TARGET])
    sns.histplot(log_price, kde=True, ax=axes[1], color="#d95f02")
    axes[1].set_title("SalePrice distribution (log1p)")
    axes[1].set_xlabel("log1p(SalePrice)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "1_target_distribution.png", dpi=120)
    plt.close(fig)
    skew = df[TARGET].skew()
    print(f"\n[1] SalePrice: mean=${df[TARGET].mean():,.0f}, "
          f"median=${df[TARGET].median():,.0f}, skew={skew:.2f}, "
          f"n_outliers_above_90th_band=None")
    print(f"    Raw skew = {skew:.2f} (heavy right tail); log1p skew = {log_price.skew():.2f}")


def plot_correlation_heatmap(df):
    num_df = df.select_dtypes(include=np.number).drop(columns=["Id"], errors="ignore")
    corr = num_df.corr()
    fig, ax = plt.subplots(figsize=(16, 13))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=False, cmap="coolwarm", center=0,
                square=True, linewidths=0.4, ax=ax,
                cbar_kws={"shrink": 0.7})
    ax.set_title(f"Correlation heatmap of {corr.shape[0]} numeric features")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "2_correlation_heatmap.png", dpi=120)
    plt.close(fig)
    corr_target = corr[TARGET].sort_values(ascending=False)
    print(f"\n[2] Correlation heatmap saved. Top 10 features vs {TARGET}:")
    print(corr_target.drop(TARGET).head(10).round(3).to_string())


def plot_top_correlated(df):
    num_df = df.select_dtypes(include=np.number).drop(columns=["Id"], errors="ignore")
    corr_target = num_df.corr()[TARGET].drop(TARGET).sort_values(ascending=False)
    top5 = corr_target.head(5).index.tolist()
    print(f"\n[3] Top 5 features correlated with {TARGET}: {top5}")
    print(corr_target.head(5).round(3).to_string())
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    for i, col in enumerate(top5):
        ax = axes[i]
        sns.regplot(x=df[col], y=df[TARGET], ax=ax, scatter_kws={"s": 12, "alpha": 0.4},
                    line_kws={"color": "crimson"})
        r = corr_target[col]
        ax.set_title(f"{col} vs {TARGET} (r={r:.2f})")
    axes[5].axis("off")
    fig.suptitle("Relationships between top 5 correlated features and SalePrice", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "3_top5_features_vs_target.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_outliers(df):
    num_df = df.select_dtypes(include=np.number).drop(columns=["Id"], errors="ignore")
    q1 = num_df.quantile(0.25)
    q3 = num_df.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outlier_count = ((num_df < lower) | (num_df > upper)).sum()

    total = len(df)
    outlier_summary = pd.DataFrame({
        "n_outliers": outlier_count,
        "pct_outliers": (outlier_count / total * 100).round(2),
    }).sort_values("n_outliers", ascending=False)
    outlier_summary = outlier_summary[outlier_summary["n_outliers"] > 0]

    print(f"\n[4] IQR outlier detection (factor 1.5) across {num_df.shape[1]} numeric columns:")
    print(outlier_summary.head(15).to_string())
    print(f"    Columns with outliers: {len(outlier_summary)} "
          f"(of {num_df.shape[1]} numeric)")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cols_for_box = outlier_summary.index[:8].tolist()
    if TARGET not in cols_for_box:
        cols_for_box = [TARGET] + cols_for_box[:7]
    df_box = df[cols_for_box]
    z = (df_box - df_box.mean()) / df_box.std()
    z.boxplot(ax=axes[0], rot=90)
    axes[0].set_title("Standardized boxplots (Z-score, top outlier cols)")
    axes[1].scatter(df["GrLivArea"], df[TARGET], s=12, alpha=0.5)
    gr_lower = df["GrLivArea"].quantile(0.25) - 1.5 * (df["GrLivArea"].quantile(0.75) - df["GrLivArea"].quantile(0.25))
    gr_upper = df["GrLivArea"].quantile(0.75) + 1.5 * (df["GrLivArea"].quantile(0.75) - df["GrLivArea"].quantile(0.25))
    for bound, lab in ((gr_lower, "lower IQR bound"), (gr_upper, "upper IQR bound")):
        axes[1].axvline(bound, color="crimson", ls="--", lw=1, label=lab)
    axes[1].set_xlabel("GrLivArea")
    axes[1].set_ylabel(TARGET)
    axes[1].set_title("GrLivArea vs SalePrice with IQR outlier bounds")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "4_outliers.png", dpi=120)
    plt.close(fig)

    n_saleprice_out = int(np.sum((df[TARGET] < lower[TARGET]) | (df[TARGET] > upper[TARGET])))
    print(f"    Outliers in {TARGET}: {n_saleprice_out} "
          f"({n_saleprice_out / total * 100:.1f}% of rows)")


NA_MEANS_ABSENT = {
    "Alley": "no alley access",
    "PoolQC": "no pool / no pool quality data",
    "Fence": "no fence",
    "MiscFeature": "none of the misc features",
    "FireplaceQu": "no fireplace",
    "GarageType": "no garage",
    "GarageFinish": "no garage",
    "GarageQual": "no garage",
    "GarageCond": "no garage",
    "GarageYrBlt": "no garage built year",
    "BsmtQual": "no basement",
    "BsmtCond": "no basement",
    "BsmtExposure": "no basement",
    "BsmtFinType1": "no basement",
    "BsmtFinType2": "no basement",
    "MasVnrType": "no masonry veneer",
}


def build_missing_plan(df):
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        print("\n[5] No missing values.")
        return

    plan_rows = []
    for col in missing.index:
        n_miss = int(missing[col])
        pct = n_miss / len(df) * 100
        dtype = "numeric" if pd.api.types.is_numeric_dtype(df[col]) else "categorical"
        if col == "LotFrontage":
            action = "Impute with median LotFrontage per Neighborhood (spatial feature)"
        elif col == "MasVnrArea":
            action = "Fill 0 (NA pairs with MasVnrType='None')"
        elif col == "Electrical":
            action = "Impute with mode ('SBrkr') - single missing value"
        elif col == "GarageYrBlt":
            action = "Fill with YearBuilt where garage exists (NA rows have no garage)"
        elif col in NA_MEANS_ABSENT:
            action = f"Fill with placeholder meaning '{NA_MEANS_ABSENT[col]}': for categoricals 'None'; missingness is informative - keep an indicator"
        elif dtype == "numeric":
            action = "Impute with median (low missing %) or median by Neighborhood"
        else:
            action = "Impute with mode or 'Unknown' category"
        plan_rows.append({
            "Column": col,
            "Dtype": dtype,
            "n_missing": n_miss,
            "pct_missing": round(pct, 2),
            "why_missing": NA_MEANS_ABSENT.get(col, "genuine missing / non-responsive"),
            "plan": action,
        })

    plan = pd.DataFrame(plan_rows)
    print(f"\n[5] Missing value summary ({plan['Column'].nunique()} columns with NaN):")
    print(plan.to_string(index=False))

    plan.to_csv(OUT_DIR / "5_missing_value_plan.csv", index=False)
    print(f"    Plan saved to {OUT_DIR / '5_missing_value_plan.csv'}")

    by_dtype = plan.groupby("Dtype")["Column"].apply(list)
    for k, v in by_dtype.items():
        print(f"    {k} ({len(v)}): {', '.join(v)}")


def main():
    df = load_data()
    plot_target_distribution(df)
    plot_correlation_heatmap(df)
    plot_top_correlated(df)
    plot_outliers(df)
    build_missing_plan(df)
    print(f"\nAll figures saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()