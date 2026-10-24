from __future__ import annotations

from dotenv import load_dotenv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from utils.ai_insights import build_analysis_summary, generate_insights
from utils.analysis import (
    anomaly_analysis,
    build_recommendations,
    classify_columns,
    consistency_analysis,
    duplicate_analysis,
    load_dataset,
    missing_analysis,
    numeric_value_outlier_percentage,
    outlier_analysis,
    profile_dataset,
)
from utils.report_generator import build_pdf_report
from utils.scoring import calculate_quality_score

def load_css():
    css_file = Path(__file__).parent / "assets" / "style.css"

    with open(css_file, "r", encoding="utf-8") as file:
        st.markdown(
            f"<style>{file.read()}</style>",
            unsafe_allow_html=True
        )

NAVY = "#102a43"
TEAL = "#168aad"
BLUE = "#2f80b7"
MUTED = "#627d98"
LINE = "#d9e2ec"
PAPER = "#f3f6fc"

load_dotenv()

st.set_page_config(
    page_title="DataGuard | Data quality workspace",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_css()


def figure_missing_values(missing: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    visible = missing[missing["Missing values"] > 0].head(20)
    if visible.empty:
        ax.text(
            0.5,
            0.5,
            "No missing values detected",
            ha="center",
            va="center",
            color=MUTED,
        )
        ax.set_axis_off()
        return fig
    ax.barh(visible["Column"][::-1], visible["Missing values"][::-1], color=TEAL)
    ax.set_xlabel("Missing values")
    ax.set_title("Missing values by column")
    fig.tight_layout()
    return fig


def figure_histogram(df: pd.DataFrame, column: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    values = df[column].dropna()
    if values.empty:
        ax.text(0.5, 0.5, "No values available", ha="center", va="center")
        ax.set_axis_off()
        return fig
    bins = max(6, min(30, int(np.sqrt(len(values)) * 2)))
    ax.hist(values, bins=bins, color=BLUE, edgecolor="white")
    ax.set_title(f"Distribution of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Records")
    fig.tight_layout()
    return fig


def figure_boxplot(df: pd.DataFrame, columns: list[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    values = [df[column].dropna() for column in columns]
    ax.boxplot(values, tick_labels=columns, patch_artist=True)
    for patch in ax.patches:
        patch.set_facecolor(TEAL)
        patch.set_alpha(0.75)
    ax.set_title("Numerical distributions and IQR ranges")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


def figure_correlation(df: pd.DataFrame, columns: list[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    correlation = df[columns].corr()
    image = ax.imshow(correlation, cmap="Blues", vmin=-1, vmax=1)
    ax.set_xticks(range(len(columns)), labels=columns, rotation=35, ha="right")
    ax.set_yticks(range(len(columns)), labels=columns)
    ax.set_title("Numerical correlation heatmap")
    fig.colorbar(image, ax=ax, shrink=0.8, label="Correlation")
    fig.tight_layout()
    return fig


def figure_score(score: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = list(score["penalties"].keys())
    values = list(score["penalties"].values())
    ax.barh(labels[::-1], values[::-1], color=TEAL)
    ax.set_xlabel("Score points deducted")
    ax.set_title(f"Data Quality Score: {score['score']:.1f} ({score['category']})")
    ax.set_xlim(0, max(20, max(values, default=0) + 5))
    fig.tight_layout()
    return fig


def close_figure(fig: plt.Figure) -> None:
    st.pyplot(fig, clear_figure=False)
    plt.close(fig)


def build_report(
    *,
    dataset_name: str,
    profile: dict,
    missing: pd.DataFrame,
    duplicates: dict,
    consistency: dict,
    outliers: dict,
    anomalies: dict,
    score: dict,
    recommendations: list[str],
    ai_insights: str | None,
) -> str:
    lines = [
        "DATAGUARD DATA QUALITY REPORT",
        "=" * 32,
        f"Dataset: {dataset_name}",
        "",
        "DATASET OVERVIEW",
        f"Rows: {profile['rows']}",
        f"Columns: {profile['columns']}",
        f"Memory usage: {profile['memory_usage']}",
        f"Numerical columns: {', '.join(map(str, profile['column_types']['numeric'])) or 'None'}",
        f"Categorical columns: {', '.join(map(str, profile['column_types']['categorical'])) or 'None'}",
        f"Datetime columns: {', '.join(map(str, profile['column_types']['datetime'])) or 'None'}",
        "",
        "MISSING-VALUE FINDINGS",
        f"Total missing values: {profile['missing_total']}",
        missing[missing["Missing values"] > 0].to_string(index=False)
        if profile["missing_total"]
        else "No missing values detected.",
        "",
        "DUPLICATE FINDINGS",
        f"Duplicate rows: {duplicates['count']} ({duplicates['percentage']:.2f}%)",
        "",
        "DATA CONSISTENCY FINDINGS",
        f"Confirmed issues: {len(consistency['confirmed'])}",
        f"Potential issues requiring review: {len(consistency['potential'])}",
    ]
    for issue in consistency["confirmed"] + consistency["potential"]:
        lines.append(f"- {issue['Column']}: {issue['Issue']} — {issue['Details']}")

    lines.extend(
        [
            "",
            "OUTLIER SUMMARY",
            outliers["summary"].to_string(index=False)
            if not outliers["summary"].empty
            else "No numerical outliers could be calculated.",
            "",
            "ANOMALY SUMMARY",
            (
                f"Isolation Forest anomalies: {anomalies['count']} "
                f"({anomalies['percentage']:.2f}%)"
                if anomalies["available"]
                else f"Skipped: {anomalies['reason']}"
            ),
            "",
            "DATA QUALITY SCORE",
            f"Score: {score['score']:.1f}/100",
            f"Category: {score['category']}",
        ]
    )
    for name, penalty in score["penalties"].items():
        lines.append(f"- {name}: -{penalty:.1f}")

    lines.extend(["", "KEY RECOMMENDATIONS"])
    lines.extend(f"- {recommendation}" for recommendation in recommendations)
    lines.extend(["", "AI INSIGHTS"])
    lines.append(ai_insights or "No AI insights generated.")
    return "\n".join(lines)


def render_ai_insights(text: str) -> None:
    headings = {"Key Findings", "Impact Assessment", "Recommendations"}
    sections: dict[str, list[str]] = {"Key Findings": []}
    current = "Key Findings"
    for line in text.splitlines():
        title = line.strip().rstrip(":")
        if title in headings:
            current = title
            sections.setdefault(current, [])
        elif line.strip():
            sections.setdefault(current, []).append(line.strip())

    for heading in ("Key Findings", "Impact Assessment", "Recommendations"):
        content = sections.get(heading, [])
        if content:
            st.subheader(heading)
            st.markdown("\n\n".join(content))


st.markdown(
    '<section class="hero">'
    '<div class="hero-kicker">Data quality control room</div>'
    '<div class="hero-title">DataGuard</div>'
    '<p class="hero-subtitle">Inspect dataset quality, identify inconsistencies, '
    'and detect unusual records before analysis.</p>'
    '</section>',
    unsafe_allow_html=True,
)

st.markdown('<div class="section-label">Navigate</div>', unsafe_allow_html=True)
page = st.radio(
    "Analysis sections",
    [
        "Overview",
        "Data Quality",
        "Outliers & Anomalies",
        "Visualizations",
        "Smart Insights",
        "Report",
    ],
    horizontal=True,
    label_visibility="collapsed",
)
st.divider()

with st.sidebar:
    st.markdown("### Workspace")
    uploaded_file = st.file_uploader(
        "Upload a dataset",
        type=["csv", "xlsx", "xls"],
        help="Supported formats: CSV, XLSX, and XLS.",
    )
    st.divider()
    st.caption("Isolation Forest setting")
    contamination = st.slider(
        "Expected unusual rows",
        min_value=0.01,
        max_value=0.25,
        value=0.05,
        step=0.01,
        help="A review setting for the model, not a claim that these rows are incorrect.",
    )
   

st.markdown('<div class="section-label">Analysis workspace</div>', unsafe_allow_html=True)

if not uploaded_file:
    st.header("Start with a dataset")
    st.write(
        "Upload a CSV or Excel workbook to profile its structure, assess quality, "
        "inspect unusual records, and prepare a concise report."
    )
    st.markdown(
        '<div class="empty-state"><strong>No dataset loaded yet</strong><br>'
        "Choose a CSV, XLSX, or XLS file from the sidebar to begin.</div>",
        unsafe_allow_html=True,
    )
    st.stop()

try:
    df = load_dataset(uploaded_file)
except (ValueError, pd.errors.ParserError, OSError, ImportError) as exc:
    st.error(f"Data could not be loaded: {exc}")
    st.stop()
except Exception:
    st.error(
        "Data could not be loaded. Check that the file is not corrupted and uses a "
        "supported CSV or Excel format."
    )
    st.stop()

if df.empty or df.shape[1] == 0:
    st.error(
        "This file loaded successfully, but it does not contain any rows or columns "
        "to analyze."
    )
    st.stop()

dataset_token = f"{uploaded_file.name}:{getattr(uploaded_file, 'size', len(df))}"
if st.session_state.get("dataset_token") != dataset_token:
    st.session_state.dataset_token = dataset_token
    st.session_state.ai_insights = None
    st.session_state.ai_error = None

profile = profile_dataset(df)
columns = classify_columns(df)
missing = missing_analysis(df)
duplicates = duplicate_analysis(df)
consistency = consistency_analysis(df)
outliers = outlier_analysis(df, columns["numeric"])
anomalies = anomaly_analysis(df, columns["numeric"], contamination)
outlier_percentage = numeric_value_outlier_percentage(
    outliers, df, columns["numeric"]
)
score = calculate_quality_score(
    missing_percentage=(profile["missing_total"] / (len(df) * len(df.columns))) * 100,
    duplicate_percentage=duplicates["percentage"],
    consistency_issues=consistency["total_issues"],
    outlier_percentage=outlier_percentage,
    anomaly_percentage=anomalies["percentage"],
)
recommendations = build_recommendations(
    profile, missing, duplicates, consistency, outliers, anomalies
)
ai_summary = build_analysis_summary(
    dataset_name=uploaded_file.name,
    profile=profile,
    missing=missing,
    duplicates=duplicates,
    consistency=consistency,
    outliers=outliers,
    anomalies=anomalies,
    score=score,
    numeric_stats=profile["numeric_stats"],
)

st.success(f"{uploaded_file.name} is ready for analysis.")

if page == "Overview":
    st.header("Dataset overview")
    st.write(
        "A compact view of the uploaded dataset before inspecting individual quality issues."
    )
    row1 = st.columns(3)
    row1[0].metric("Rows", f"{profile['rows']:,}")
    row1[1].metric("Columns", f"{profile['columns']:,}")
    row1[2].metric("Quality score", f"{score['score']:.1f}/100")

    row2 = st.columns(2)
    row2[0].metric("Missing values", f"{profile['missing_total']:,}")
    row2[1].metric("Duplicate rows", f"{duplicates['count']:,}")

    left, right = st.columns(2)
    with left:
        st.subheader("Detected column groups")
        st.write(f"**Numerical ({len(columns['numeric'])})**")
        st.caption(", ".join(map(str, columns["numeric"])) or "None detected")
        st.write(f"**Categorical ({len(columns['categorical'])})**")
        st.caption(", ".join(map(str, columns["categorical"])) or "None detected")
        st.write(f"**Datetime ({len(columns['datetime'])})**")
        st.caption(", ".join(map(str, columns["datetime"])) or "None detected")
    with right:
        st.subheader("Column structure")
        structure = pd.DataFrame(
            {
                "Column": df.columns.astype(str),
                "Detected type": [str(dtype) for dtype in df.dtypes],
                "Non-null values": [int(value) for value in df.notna().sum()],
                "Missing values": [int(value) for value in df.isna().sum()],
            }
        )
        st.dataframe(structure, width="stretch", hide_index=True)

    st.subheader("Numerical statistics")
    if profile["numeric_stats"].empty:
        st.info("No numerical columns were detected for descriptive statistics.")
    else:
        st.dataframe(profile["numeric_stats"], width="stretch")

elif page == "Data Quality":
    st.header("Data quality checks")
    st.write(
        "These checks describe the uploaded data without modifying or deleting any records."
    )

    st.subheader("Missing-value analysis")
    missing_left, missing_right = st.columns([1.05, 1.4])
    with missing_left:
        st.metric("Total missing values", f"{profile['missing_total']:,}")
        high_missing = missing[missing["Missing percentage"] >= 50]
        if high_missing.empty:
            st.caption("No column exceeds the 50% review threshold.")
        else:
            st.warning(
                f"{len(high_missing)} column(s) have at least 50% missing values and need review."
            )
        st.dataframe(missing, width="stretch", hide_index=True)
    with missing_right:
        close_figure(figure_missing_values(missing))

    st.subheader("Duplicate rows")
    if duplicates["count"]:
        st.warning(
            f"{duplicates['count']:,} duplicate rows affect {duplicates['percentage']:.2f}% of the dataset."
        )
        st.dataframe(duplicates["preview"], width="stretch", hide_index=True)
    else:
        st.success("No exact duplicate rows detected.")

    st.subheader("Consistency checks")
    confirmed_tab, potential_tab = st.tabs(
        ["Confirmed issues", "Potential issues requiring review"]
    )
    with confirmed_tab:
        if consistency["confirmed"]:
            st.dataframe(
                pd.DataFrame(consistency["confirmed"]),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info(
                "No confirmed type or format issues were produced by the current checks."
            )
    with potential_tab:
        if consistency["potential"]:
            st.dataframe(
                pd.DataFrame(consistency["potential"]),
                width="stretch",
                hide_index=True,
            )
        else:
            st.success("No potential type or format inconsistencies detected.")

    st.subheader("Recommended next steps")
    for recommendation in recommendations:
        st.markdown(f"- {recommendation}")

elif page == "Outliers & Anomalies":
    st.header("Outliers and unusual observations")
    st.write(
        "Statistical outliers and model anomalies are signals for review, not proof that a record is incorrect."
    )

    st.subheader("IQR outlier analysis")
    if outliers["summary"].empty:
        st.info(
            "There is not enough usable numerical data for IQR analysis. Each numerical column needs at least four values."
        )
    else:
        st.dataframe(outliers["summary"], width="stretch", hide_index=True)
        selected_outlier_columns = st.multiselect(
            "Columns for box plot",
            options=list(outliers["summary"]["Column"]),
            default=list(outliers["summary"]["Column"].head(4)),
        )
        if selected_outlier_columns:
            close_figure(figure_boxplot(df, selected_outlier_columns))

    st.subheader("Isolation Forest")
    if not anomalies["available"]:
        st.info(anomalies["reason"])
    else:
        anomaly_left, anomaly_right = st.columns(2)
        with anomaly_left:
            st.metric("Unusual observations", f"{anomalies['count']:,}")
        with anomaly_right:
            st.metric("Affected rows", f"{anomalies['percentage']:.2f}%")
        st.caption(
            f"Model features: {', '.join(map(str, anomalies['features']))}. "
            "Review these records in context before taking action."
        )
        if anomalies["count"]:
            st.dataframe(anomalies["preview"], width="stretch", hide_index=True)
        else:
            st.success("No unusual observations were identified at this setting.")

elif page == "Visualizations":
    st.header("Visualizations")
    st.write("Charts are generated locally with Matplotlib from the uploaded dataset.")

    st.subheader("Missing values")
    close_figure(figure_missing_values(missing))

    if columns["numeric"]:
        selected_column = st.selectbox(
            "Numerical column for distribution",
            options=columns["numeric"],
        )
        chart_left, chart_right = st.columns(2)
        with chart_left:
            close_figure(figure_histogram(df, selected_column))
        with chart_right:
            selected_box_columns = st.multiselect(
                "Numerical columns for box plot",
                options=columns["numeric"],
                default=columns["numeric"][: min(4, len(columns["numeric"]))],
            )
            if selected_box_columns:
                close_figure(figure_boxplot(df, selected_box_columns))
    else:
        st.info("Add numerical columns to the dataset to see distributions and box plots.")

    if len(columns["numeric"]) >= 2:
        st.subheader("Correlation")
        close_figure(figure_correlation(df, columns["numeric"]))

    st.subheader("Quality score breakdown")
    close_figure(figure_score(score))
    st.caption(
        "This project-specific score is an explainable indicator, not an official industry standard."
    )

elif page == "Smart Insights":
    st.header("AI analyst insights")
    st.write(
    "Get a concise summary of the main findings, their possible impact, "
    "and recommended next steps based on the dataset analysis."
)

    api_key_available = bool(os.getenv("GROQ_API_KEY"))

    if not api_key_available:
        st.info(
           "Smart Insights are unavailable until the API key is configured."
            "All local dataset analysis features remain available."
        )
        st.caption("Add GROQ_API_KEY to enable Smart Insights.")

    else:
        if st.button("Generate Smart Insights", type="primary"):
            with st.spinner("Generating a concise analyst review..."):
                insights, error = generate_insights(ai_summary)

            st.session_state.ai_insights = insights
            st.session_state.ai_error = error

        if st.session_state.get("ai_error"):
            st.error(st.session_state.ai_error)

        if st.session_state.get("ai_insights"):
            render_ai_insights(st.session_state.ai_insights)
        else:
            st.caption("Choose Generate AI Insights to request a review.")


elif page == "Report":
    st.header("Analysis report")
    st.write(
        "Generate a concise PDF summary of the dataset quality analysis."
    )

    pdf_report = build_pdf_report(
        dataset_name=uploaded_file.name,
        profile=profile,
        missing=missing,
        duplicates=duplicates,
        consistency=consistency,
        outliers=outliers,
        anomalies=anomalies,
        score=score,
        recommendations=recommendations,
        ai_insights=st.session_state.get("ai_insights"),
    )

    report_left, report_right = st.columns([1, 2])

    with report_left:
        st.metric("Quality score", f"{score['score']:.1f}/100")
        st.caption(
            f"{score['category']} · {profile['rows']:,} rows analyzed"
        )

    with report_right:
        st.markdown("**Included in the PDF**")
        st.markdown(
            "Dataset information · score breakdown · overview · "
            "missing values · duplicates · consistency findings · "
            "outliers · anomalies · recommendations"
        )

    st.download_button(
        "Download PDF Report",
        data=pdf_report,
        file_name=f"{Path(uploaded_file.name).stem}_dataguard_report.pdf",
        mime="application/pdf",
        type="primary",
    )

    st.caption(
        "The PDF includes page numbers, a generated timestamp, "
        "structured tables, and the DataGuard footer."
    )