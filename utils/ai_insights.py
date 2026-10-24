from __future__ import annotations

import json
import os
from typing import Any

from groq import Groq


# Groq's currently available high-capability text model for this workspace.
GROQ_MODEL = "openai/gpt-oss-120b"


def build_analysis_summary(
    *,
    dataset_name: str,
    profile: dict[str, Any],
    missing,
    duplicates: dict[str, Any],
    consistency: dict[str, Any],
    outliers: dict[str, Any],
    anomalies: dict[str, Any],
    score: dict[str, Any],
    numeric_stats,
) -> dict[str, Any]:
    """Create a compact, dataset-free summary for the analyst model.

    Only derived metrics, column names, and small examples are included. The
    uploaded rows themselves never leave the local Streamlit application.
    """
    missing_rows = missing[missing["Missing values"] > 0].head(12)
    outlier_rows = outliers["summary"].head(12)
    return {
        "dataset_name": dataset_name,
        "dimensions": {"rows": profile["rows"], "columns": profile["columns"]},
        "column_types": {
            key: [str(value) for value in values]
            for key, values in profile["column_types"].items()
        },
        "missing_values": missing_rows.to_dict(orient="records"),
        "duplicate_count": duplicates["count"],
        "duplicate_percentage": duplicates["percentage"],
        "consistency_issues": {
            "confirmed": consistency["confirmed"][:10],
            "potential": consistency["potential"][:10],
        },
        "outlier_summary": outlier_rows.to_dict(orient="records"),
        "anomaly_count": anomalies["count"],
        "anomaly_percentage": anomalies["percentage"],
        "numeric_statistics": numeric_stats.head(12).reset_index().to_dict(
            orient="records"
        )
        if not numeric_stats.empty
        else [],
        "quality_score": score,
    }


def build_analyst_prompt(summary: dict[str, Any]) -> str:
    """Build a focused prompt that limits the model to local findings."""
    compact_summary = json.dumps(summary, indent=2, default=str)
    return f"""You are a professional data analyst reviewing a dataset quality assessment.

Analyze only the information provided. Do not invent findings, values,
relationships, causes, or patterns.

Prioritize the most important findings instead of repeating every metric.
Use actual column names, counts, percentages, and statistics from the provided
summary when available.
Explain the likely impact of important data quality issues on analytics or
machine learning.
Provide practical recommendations.
Clearly distinguish confirmed findings from recommendations requiring domain
review.

Do not behave like a chatbot.
Do not greet the user.
Do not use emojis.
Do not use exclamation marks.
Do not say "I hope this helps".
Do not use motivational or promotional language.
Do not make generic statements such as "Data quality is important".
Do not invent information.

Format the response with these concise sections:
Key Findings
Impact Assessment
Recommendations

Assessment:
{compact_summary}
"""


def generate_insights(summary: dict[str, Any]) -> tuple[str | None, str | None]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return (
            None,
            "AI Insights are unavailable until a Groq API key is configured. "
            "All local dataset analysis features remain available.",
        )

    try:
        client = Groq(api_key=api_key, timeout=30.0)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a data analyst reviewing a dataset quality assessment. "
                        "Use only the provided local analysis."
                    ),
                },
                {"role": "user", "content": build_analyst_prompt(summary)},
            ],
            temperature=0.2,
            max_tokens=700,
        )
        content = response.choices[0].message.content
        if not content:
            return None, "Groq returned an empty response. Please try again."
        return content.strip(), None
    except Exception as exc:
        message = str(exc).lower()
        if (
            "401" in message
            or "403" in message
            or "authentication" in message
            or "invalid api key" in message
        ):
            return None, "The Groq API key was rejected. Check the configured secret and try again."
        if "rate" in message or "429" in message:
            return None, "Groq rate limit reached. Please wait and try again."
        if "timeout" in message or "timed out" in message:
            return None, "Groq request timed out. Please try again."
        return None, "Groq could not generate insights right now. Please try again."