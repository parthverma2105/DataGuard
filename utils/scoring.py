from __future__ import annotations

from typing import Any


def calculate_quality_score(
    *,
    missing_percentage: float,
    duplicate_percentage: float,
    consistency_issues: int,
    outlier_percentage: float,
    anomaly_percentage: float,
) -> dict[str, Any]:
    """Calculate an explainable 0–100 score using capped issue penalties.

    The five penalties have maximum weights of 30, 15, 20, 20, and 15.
    Missing values and duplicates use their observed percentages. Consistency
    issues use five points per issue, while outliers and anomalies use their
    observed affected percentages. This is a project-specific indicator, not
    an industry-standard metric.
    """
    penalties = {
        "Missing data": min(30.0, max(0.0, missing_percentage) * 0.6),
        "Duplicate records": min(15.0, max(0.0, duplicate_percentage) * 0.5),
        "Type and format issues": min(20.0, max(0, consistency_issues) * 5.0),
        "Statistical outliers": min(20.0, max(0.0, outlier_percentage) * 0.5),
        "ML anomalies": min(15.0, max(0.0, anomaly_percentage) * 0.75),
    }
    score = round(max(0.0, min(100.0, 100.0 - sum(penalties.values()))), 1)
    if score >= 85:
        category = "Excellent"
    elif score >= 70:
        category = "Good"
    elif score >= 50:
        category = "Fair"
    else:
        category = "Poor"
    return {"score": score, "category": category, "penalties": penalties}