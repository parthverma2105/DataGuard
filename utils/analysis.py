from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_dataset(uploaded_file) -> pd.DataFrame:
    """Load one of DataGuard's supported file formats."""
    suffix = Path(uploaded_file.name).suffix.lower()
    uploaded_file.seek(0)

    if suffix == ".csv":
        return pd.read_csv(uploaded_file)
    if suffix == ".xlsx":
        return pd.read_excel(uploaded_file, engine="openpyxl")
    if suffix == ".xls":
        return pd.read_excel(uploaded_file, engine="xlrd")
    raise ValueError("Unsupported file type. Upload a CSV, XLSX, or XLS file.")


def format_memory_usage(df: pd.DataFrame) -> str:
    memory_bytes = int(df.memory_usage(deep=True).sum())
    if memory_bytes < 1024:
        return f"{memory_bytes} B"
    if memory_bytes < 1024**2:
        return f"{memory_bytes / 1024:.1f} KB"
    return f"{memory_bytes / 1024**2:.1f} MB"


def _is_datetime_like(series: pd.Series) -> tuple[bool, float]:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True, 1.0

    values = series.dropna()
    if values.empty or not pd.api.types.is_object_dtype(series):
        return False, 0.0

    parsed = pd.to_datetime(values, errors="coerce")
    ratio = float(parsed.notna().mean())
    name_hint = any(
        token in str(series.name).lower()
        for token in ("date", "time", "year", "month", "day")
    )
    return (ratio >= 0.8 and name_hint) or ratio >= 0.95, ratio


def classify_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    numeric = list(df.select_dtypes(include="number").columns)
    datetime_columns: list[str] = []
    for column in df.columns:
        is_date, _ = _is_datetime_like(df[column])
        if is_date and column not in numeric:
            datetime_columns.append(column)

    categorical = [
        column
        for column in df.columns
        if column not in numeric and column not in datetime_columns
    ]
    return {
        "numeric": numeric,
        "categorical": categorical,
        "datetime": datetime_columns,
    }


def profile_dataset(df: pd.DataFrame) -> dict[str, Any]:
    columns = classify_columns(df)
    numeric_stats = pd.DataFrame()
    if columns["numeric"]:
        numeric_stats = (
            df[columns["numeric"]]
            .describe()
            .T.rename(
                columns={
                    "count": "Count",
                    "mean": "Mean",
                    "50%": "Median",
                    "std": "Std. dev.",
                    "min": "Minimum",
                    "max": "Maximum",
                }
            )
        )
        numeric_stats = numeric_stats[
            ["Count", "Mean", "Median", "Std. dev.", "Minimum", "Maximum"]
        ].round(3)

    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "missing_total": int(df.isna().sum().sum()),
        "duplicate_count": int(df.duplicated().sum()),
        "memory_usage": format_memory_usage(df),
        "column_types": columns,
        "numeric_stats": numeric_stats,
    }


def missing_analysis(df: pd.DataFrame) -> pd.DataFrame:
    missing = pd.DataFrame(
        {
            "Column": df.columns.astype(str),
            "Missing values": df.isna().sum().astype(int).values,
            "Missing percentage": (
                (df.isna().mean() * 100).round(2).values
            ),
        }
    )
    return missing.sort_values(
        ["Missing values", "Column"], ascending=[False, True]
    ).reset_index(drop=True)


def duplicate_analysis(df: pd.DataFrame) -> dict[str, Any]:
    count = int(df.duplicated().sum())
    return {
        "count": count,
        "percentage": round((count / len(df)) * 100, 2) if len(df) else 0.0,
        "preview": df[df.duplicated(keep=False)].head(10),
    }


def _examples(values: list[Any], limit: int = 4) -> str:
    return ", ".join(repr(value) for value in values[:limit])


def consistency_analysis(df: pd.DataFrame) -> dict[str, Any]:
    """Find explainable format concerns without changing the uploaded data."""
    confirmed: list[dict[str, str]] = []
    potential: list[dict[str, str]] = []

    for column in df.columns:
        series = df[column]
        non_null = series.dropna()
        if non_null.empty:
            continue

        type_names = {type(value).__name__ for value in non_null}
        if len(type_names) > 1:
            potential.append(
                {
                    "Column": str(column),
                    "Issue": "Mixed Python value types",
                    "Details": f"Observed types: {', '.join(sorted(type_names))}.",
                }
            )

        if pd.api.types.is_object_dtype(series):
            numeric_values = pd.to_numeric(non_null, errors="coerce")
            numeric_ratio = float(numeric_values.notna().mean())
            if numeric_ratio >= 0.8:
                potential.append(
                    {
                        "Column": str(column),
                        "Issue": "Possible numeric values stored as text",
                        "Details": f"{numeric_ratio * 100:.1f}% of non-empty values look numeric.",
                    }
                )

            normalized = non_null.astype(str).str.strip().str.casefold()
            if normalized.nunique() < non_null.astype(str).nunique():
                examples = non_null.astype(str).drop_duplicates().tolist()
                potential.append(
                    {
                        "Column": str(column),
                        "Issue": "Inconsistent category formatting",
                        "Details": f"Labels that differ by case or whitespace may include {_examples(examples)}.",
                    }
                )

            is_date_like, parse_ratio = _is_datetime_like(series)
            if parse_ratio >= 0.5 and parse_ratio < 1.0:
                potential.append(
                    {
                        "Column": str(column),
                        "Issue": "Date-like values need review",
                        "Details": f"{parse_ratio * 100:.1f}% parse as dates; remaining values may be invalid or use another format.",
                    }
                )
            elif is_date_like and parse_ratio == 1.0:
                potential.append(
                    {
                        "Column": str(column),
                        "Issue": "Date-like column stored as text",
                        "Details": "All non-empty values parse as dates, but the source dtype is text.",
                    }
                )

    return {
        "confirmed": confirmed,
        "potential": potential,
        "total_issues": len(confirmed) + len(potential),
    }


def outlier_analysis(df: pd.DataFrame, numeric_columns: list[str]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    masks: dict[str, pd.Series] = {}

    for column in numeric_columns:
        values = df[column].dropna()
        if len(values) < 4:
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (df[column] < lower) | (df[column] > upper)
        count = int(mask.sum())
        masks[column] = mask
        rows.append(
            {
                "Column": str(column),
                "Outlier count": count,
                "Affected percentage": round((count / len(values)) * 100, 2),
                "Lower bound": round(float(lower), 3),
                "Upper bound": round(float(upper), 3),
            }
        )

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(
            ["Outlier count", "Column"], ascending=[False, True]
        ).reset_index(drop=True)
    return {"summary": summary, "masks": masks}


def anomaly_analysis(
    df: pd.DataFrame, numeric_columns: list[str], contamination: float
) -> dict[str, Any]:
    """Prepare numeric features for Isolation Forest and label unusual rows."""
    from sklearn.ensemble import IsolationForest

    usable = [
        column for column in numeric_columns if df[column].notna().any()
    ]
    if len(df) < 10 or not usable:
        return {
            "available": False,
            "reason": "Isolation Forest needs at least 10 rows and one usable numerical column.",
            "count": 0,
            "percentage": 0.0,
            "preview": pd.DataFrame(),
            "mask": pd.Series(False, index=df.index),
            "features": [],
        }

    features = df[usable].copy()
    features = features.fillna(features.median(numeric_only=True))
    features = features.dropna(axis=1, how="all")
    if features.empty:
        return {
            "available": False,
            "reason": "There are no usable numerical values after handling missing data.",
            "count": 0,
            "percentage": 0.0,
            "preview": pd.DataFrame(),
            "mask": pd.Series(False, index=df.index),
            "features": [],
        }

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
    )
    labels = model.fit_predict(features)
    mask = pd.Series(labels == -1, index=df.index)
    count = int(mask.sum())
    return {
        "available": True,
        "reason": "",
        "count": count,
        "percentage": round((count / len(df)) * 100, 2),
        "preview": df.loc[mask].head(10),
        "mask": mask,
        "features": list(features.columns),
    }


def build_recommendations(
    profile: dict[str, Any],
    missing: pd.DataFrame,
    duplicates: dict[str, Any],
    consistency: dict[str, Any],
    outliers: dict[str, Any],
    anomalies: dict[str, Any],
) -> list[str]:
    recommendations: list[str] = []
    high_missing = missing[missing["Missing percentage"] >= 50]
    if profile["missing_total"] > 0:
        recommendations.append(
            "Review missing-value patterns before imputation; use median imputation for skewed numerical columns and mode or an Unknown category for categorical data."
        )
    if not high_missing.empty:
        recommendations.append(
            f"Review whether {len(high_missing)} column(s) with at least 50% missing values should be retained."
        )
    if duplicates["count"]:
        recommendations.append(
            "Review duplicate rows and available identifiers before removing them so valid repeated observations are not lost."
        )
    if consistency["total_issues"]:
        recommendations.append(
            "Standardize labels, numeric text, or date formats only after confirming the intended format with the dataset owner."
        )
    if not outliers["summary"].empty and int(
        outliers["summary"]["Outlier count"].sum()
    ):
        recommendations.append(
            "Investigate IQR outliers in context; an outlier is not automatically an incorrect record."
        )
    if anomalies["available"] and anomalies["count"]:
        recommendations.append(
            "Review Isolation Forest anomalies as unusual observations requiring investigation, not as confirmed errors."
        )
    if not recommendations:
        recommendations.append(
            "No major issues were detected by the current checks; confirm results against the dataset's business context."
        )
    return recommendations


def numeric_value_outlier_percentage(
    outliers: dict[str, Any], df: pd.DataFrame, numeric_columns: list[str]
) -> float:
    if not numeric_columns:
        return 0.0
    total_values = int(df[numeric_columns].notna().sum().sum())
    if total_values == 0:
        return 0.0
    outlier_values = int(
        sum(int(mask.sum()) for mask in outliers["masks"].values())
    )
    return round((outlier_values / total_values) * 100, 2)