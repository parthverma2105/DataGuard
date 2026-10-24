from __future__ import annotations

import html
from datetime import datetime
from io import BytesIO
from typing import Any

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


NAVY = colors.HexColor("#102a43")
TEAL = colors.HexColor("#168aad")
BLUE = colors.HexColor("#2f80b7")
INK = colors.HexColor("#243b53")
MUTED = colors.HexColor("#627d98")
LINE = colors.HexColor("#d9e2ec")
PALE_BLUE = colors.HexColor("#edf5f9")


def _text(value: Any) -> str:
    return html.escape(str(value)).replace("\n", "<br/>")


def _paragraph(value: Any, style: ParagraphStyle) -> Paragraph:
    return Paragraph(_text(value), style)


def _table(
    data: list[list[Any]],
    widths: list[float],
    styles: dict[str, ParagraphStyle],
) -> Table:
    rows = [
        [_paragraph(cell, styles["table_header"]) for cell in data[0]],
        *[
            [_paragraph(cell, styles["table_cell"]) for cell in row]
            for row in data[1:]
        ],
    ]
    table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE_BLUE]),
                ("GRID", (0, 0), (-1, -1), 0.35, LINE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def _section(title: str, styles: dict[str, ParagraphStyle]) -> list[Any]:
    return [
        Spacer(1, 7 * mm),
        Paragraph(_text(title), styles["section"]),
        Spacer(1, 2 * mm),
    ]


def _bullet_items(
    items: list[str], styles: dict[str, ParagraphStyle]
) -> list[Paragraph]:
    if not items:
        return [_paragraph("None reported.", styles["body"])]
    return [
        Paragraph(f"• {_text(item)}", styles["body"])
        for item in items
    ]


def build_pdf_report(
    *,
    dataset_name: str,
    profile: dict[str, Any],
    missing: pd.DataFrame,
    duplicates: dict[str, Any],
    consistency: dict[str, Any],
    outliers: dict[str, Any],
    anomalies: dict[str, Any],
    score: dict[str, Any],
    recommendations: list[str],
    ai_insights: str | None,
) -> bytes:
    """Build a structured, multi-page PDF report from local analysis results."""
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title="DataGuard Data Quality Report",
        author="DataGuard",
    )

    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=24,
            leading=28,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=3 * mm,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=11,
            leading=15,
            textColor=MUTED,
            spaceAfter=8 * mm,
        ),
        "section": ParagraphStyle(
            "Section",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=NAVY,
            spaceBefore=3 * mm,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=14,
            textColor=INK,
            spaceAfter=1.5 * mm,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=12,
            textColor=MUTED,
        ),
        "score": ParagraphStyle(
            "Score",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=30,
            leading=34,
            textColor=TEAL,
            alignment=TA_CENTER,
        ),
        "score_label": ParagraphStyle(
            "ScoreLabel",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=13,
            textColor=NAVY,
            alignment=TA_CENTER,
        ),
        "table_header": ParagraphStyle(
            "TableHeader",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            textColor=colors.white,
        ),
        "table_cell": ParagraphStyle(
            "TableCell",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            textColor=INK,
        ),
    }

    story: list[Any] = [
        Paragraph("DATA GUARD", styles["title"]),
        Paragraph(
            "Intelligent Data Quality and Anomaly Detection Report",
            styles["subtitle"],
        ),
        _table(
            [
                ["Dataset information", "Value"],
                ["File name", dataset_name],
                [
                    "Generated",
                    datetime.now().strftime("%d %B %Y, %H:%M"),
                ],
                ["Rows", f"{profile['rows']:,}"],
                ["Columns", f"{profile['columns']:,}"],
                ["Memory usage", profile["memory_usage"]],
            ],
            [55 * mm, 115 * mm],
            styles,
        ),
    ]

    story.extend(_section("Data Quality Score", styles))
    score_table = Table(
        [
            [
                Paragraph(f"{score['score']:.1f}/100", styles["score"]),
                Paragraph(_text(score["category"]), styles["score_label"]),
            ]
        ],
        colWidths=[55 * mm, 55 * mm],
        rowHeights=[25 * mm],
    )
    score_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PALE_BLUE),
                ("BOX", (0, 0), (-1, -1), 0.6, LINE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ]
        )
    )
    story.append(score_table)
    story.append(Spacer(1, 4 * mm))
    story.append(
        _table(
            [["Score component", "Deduction"]]
            + [
                [name, f"-{penalty:.1f}"]
                for name, penalty in score["penalties"].items()
            ],
            [125 * mm, 45 * mm],
            styles,
        )
    )
    story.append(
        Paragraph(
            "This is a project-specific, explainable indicator and is not an official industry-standard metric.",
            styles["small"],
        )
    )

    story.extend(_section("Dataset Overview", styles))
    story.append(
        _table(
            [
                ["Column group", "Columns"],
                [
                    "Numerical",
                    ", ".join(map(str, profile["column_types"]["numeric"]))
                    or "None detected",
                ],
                [
                    "Categorical",
                    ", ".join(map(str, profile["column_types"]["categorical"]))
                    or "None detected",
                ],
                [
                    "Datetime",
                    ", ".join(map(str, profile["column_types"]["datetime"]))
                    or "None detected",
                ],
            ],
            [35 * mm, 135 * mm],
            styles,
        )
    )

    story.extend(_section("Missing Value Analysis", styles))
    story.append(
        Paragraph(
            f"Total missing values: <b>{profile['missing_total']:,}</b>",
            styles["body"],
        )
    )
    affected_missing = missing[missing["Missing values"] > 0].head(20)
    if affected_missing.empty:
        story.append(Paragraph("No missing values detected.", styles["body"]))
    else:
        story.append(
            _table(
                [["Column", "Missing values", "Missing percentage"]]
                + [
                    [
                        row["Column"],
                        f"{int(row['Missing values']):,}",
                        f"{float(row['Missing percentage']):.2f}%",
                    ]
                    for _, row in affected_missing.iterrows()
                ],
                [75 * mm, 45 * mm, 50 * mm],
                styles,
            )
        )

    story.extend(_section("Duplicate Analysis", styles))
    story.append(
        Paragraph(
            f"Duplicate rows: <b>{duplicates['count']:,}</b> "
            f"({duplicates['percentage']:.2f}% of the dataset).",
            styles["body"],
        )
    )

    story.extend(_section("Data Consistency Findings", styles))
    consistency_rows = [["Type", "Column", "Finding", "Details"]]
    for issue in consistency["confirmed"]:
        consistency_rows.append(
            ["Confirmed", issue["Column"], issue["Issue"], issue["Details"]]
        )
    for issue in consistency["potential"]:
        consistency_rows.append(
            ["Potential review", issue["Column"], issue["Issue"], issue["Details"]]
        )
    if len(consistency_rows) == 1:
        story.append(Paragraph("No type or format concerns detected.", styles["body"]))
    else:
        story.append(
            _table(
                consistency_rows,
                [28 * mm, 32 * mm, 47 * mm, 63 * mm],
                styles,
            )
        )

    story.extend(_section("Outlier Analysis", styles))
    if outliers["summary"].empty:
        story.append(
            Paragraph(
                "No numerical outlier analysis was available for this dataset.",
                styles["body"],
            )
        )
    else:
        story.append(
            _table(
                [["Column", "Outlier count", "Affected percentage"]]
                + [
                    [
                        row["Column"],
                        f"{int(row['Outlier count']):,}",
                        f"{float(row['Affected percentage']):.2f}%",
                    ]
                    for _, row in outliers["summary"].head(15).iterrows()
                ],
                [80 * mm, 45 * mm, 45 * mm],
                styles,
            )
        )

    story.extend(_section("Anomaly Detection", styles))
    if anomalies["available"]:
        story.append(
            Paragraph(
                f"Isolation Forest identified <b>{anomalies['count']:,}</b> "
                f"unusual observations ({anomalies['percentage']:.2f}%).",
                styles["body"],
            )
        )
    else:
        story.append(Paragraph(f"Analysis skipped: {anomalies['reason']}", styles["body"]))
    story.append(
        Paragraph(
            "Anomalies are unusual observations requiring review; they are not automatically incorrect data.",
            styles["small"],
        )
    )

    if ai_insights:
        story.extend(_section("AI Insights", styles))
        for line in ai_insights.splitlines():
            if line.strip():
                story.append(Paragraph(_text(line), styles["body"]))

    story.extend(_section("Recommendations", styles))
    story.extend(_bullet_items(recommendations, styles))

    def draw_footer(canvas, document) -> None:
        canvas.saveState()
        canvas.setStrokeColor(LINE)
        canvas.line(16 * mm, 12 * mm, 194 * mm, 12 * mm)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(16 * mm, 7 * mm, "Generated by DataGuard")
        canvas.drawRightString(194 * mm, 7 * mm, f"Page {document.page}")
        canvas.restoreState()

    document.build(story, onFirstPage=draw_footer, onLaterPages=draw_footer)
    return buffer.getvalue()