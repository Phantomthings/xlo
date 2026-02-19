from fastapi import APIRouter, Request, Query, Body, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse, Response
from datetime import date, datetime
from typing import Any, Optional
from io import BytesIO
import base64
import pandas as pd
import numpy as np

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, Image, PageBreak
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

from db import query_df, LOGV0_BASE_URL
from routers.sessions import _build_conditions, _apply_status_filters, SITE_COLOR_PALETTE, EVI_MOMENT, EVI_CODE, DS_PC, _map_moment_label, _build_pivot_table

router = APIRouter(tags=["report"])
templates = Jinja2Templates(directory="templates")

# ── ReportLab PDF helpers ──────────────────────────────────────────────────────

_PDF_PRIMARY = colors.HexColor("#1d4ed8")
_PDF_LIGHT   = colors.HexColor("#f1f5f9")
_PDF_BORDER  = colors.HexColor("#e2e8f0")
_PDF_TEXT    = colors.HexColor("#1e293b")
_PDF_MUTED   = colors.HexColor("#64748b")


def _pdf_styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "rptTitle", parent=base["Title"],
            fontSize=20, textColor=_PDF_PRIMARY,
            spaceAfter=4, fontName="Helvetica-Bold",
        ),
        "subtitle": ParagraphStyle(
            "rptSubtitle", parent=base["Normal"],
            fontSize=10, textColor=_PDF_MUTED, spaceAfter=2,
        ),
        "section": ParagraphStyle(
            "rptSection", parent=base["Heading2"],
            fontSize=12, textColor=_PDF_PRIMARY,
            fontName="Helvetica-Bold", spaceBefore=12, spaceAfter=5,
        ),
        "normal": ParagraphStyle(
            "rptNormal", parent=base["Normal"],
            fontSize=9, textColor=_PDF_TEXT,
        ),
        "small": ParagraphStyle(
            "rptSmall", parent=base["Normal"],
            fontSize=7, textColor=_PDF_MUTED,
        ),
        "bold": ParagraphStyle(
            "rptBold", parent=base["Normal"],
            fontSize=9, fontName="Helvetica-Bold", textColor=_PDF_TEXT,
        ),
        "kpi_val": ParagraphStyle(
            "rptKpiVal", parent=base["Normal"],
            fontSize=18, fontName="Helvetica-Bold",
            alignment=TA_CENTER, textColor=_PDF_PRIMARY,
        ),
        "kpi_lbl": ParagraphStyle(
            "rptKpiLbl", parent=base["Normal"],
            fontSize=7, alignment=TA_CENTER, textColor=_PDF_MUTED,
        ),
    }


def _pdf_kpi_table(kpis, styles):
    """Build a KPI table with two rows: values (large) on top, labels (small) below."""
    n = len(kpis)
    col_w = (A4[0] - 3 * cm) / n
    # Row 0: values, Row 1: labels  — NO lists inside cells
    row_vals = [Paragraph(str(v), styles["kpi_val"]) for v, lbl in kpis]
    row_lbls = [Paragraph(str(lbl), styles["kpi_lbl"]) for v, lbl in kpis]
    t = Table([row_vals, row_lbls], colWidths=[col_w] * n)
    t.setStyle(TableStyle([
        ("BOX",           (0, 0), (-1, -1), 0.5, _PDF_BORDER),
        ("INNERGRID",     (0, 0), (-1, -1), 0.5, _PDF_BORDER),
        ("BACKGROUND",    (0, 0), (-1, -1), _PDF_LIGHT),
        ("TOPPADDING",    (0, 0), (-1, 0), 12),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 4),
        ("TOPPADDING",    (0, 1), (-1, 1), 4),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 12),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


def _pdf_data_table(headers, rows_data, styles, col_widths=None):
    total_w = A4[0] - 3 * cm
    if col_widths is None:
        col_widths = [total_w / len(headers)] * len(headers)
    data = [[Paragraph(h, styles["bold"]) for h in headers]]
    for row in rows_data:
        data.append([Paragraph(str(c), styles["normal"]) for c in row])
    t = Table(data, colWidths=col_widths, repeatRows=1)
    style_cmds = [
        ("BACKGROUND",    (0, 0), (-1, 0), _PDF_PRIMARY),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0), 9),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
        ("TOPPADDING",    (0, 0), (-1, 0), 6),
        ("GRID",          (0, 0), (-1, -1), 0.4, _PDF_BORDER),
        ("FONTSIZE",      (0, 1), (-1, -1), 8),
        ("TOPPADDING",    (0, 1), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 4),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            style_cmds.append(("BACKGROUND", (0, i), (-1, i), _PDF_LIGHT))
    t.setStyle(TableStyle(style_cmds))
    return t


def _build_pdf_export(
    date_debut_str, date_fin_str, sites_list, date_generation,
    total_charges, ok_charges, nok_charges, taux_reussite, taux_echec,
    energy_total, sites_actifs,
    site_performance, top_errors, moment_distribution,
    points_forts, points_attention, recommandations,
):
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=1.5*cm, rightMargin=1.5*cm,
        topMargin=1.5*cm, bottomMargin=1.5*cm,
    )
    styles = _pdf_styles()
    elems = []

    def _safe(text):
        """Escape XML special chars so Paragraph won't fail."""
        return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    elems.append(Paragraph("Rapport de performance de recharge", styles["title"]))
    period = f"Periode : {date_debut_str} au {date_fin_str}"
    if sites_list:
        period += f"  |  Sites : {', '.join(sites_list)}"
    elems.append(Paragraph(_safe(period), styles["subtitle"]))
    elems.append(Paragraph(f"Genere le {date_generation}", styles["small"]))
    elems.append(HRFlowable(width="100%", thickness=1, color=_PDF_BORDER, spaceAfter=8))

    elems.append(Paragraph("Resume global", styles["section"]))
    taux_color = "#16a34a" if taux_reussite >= 90 else ("#d97706" if taux_reussite >= 80 else "#dc2626")
    kpis = [
        (f'<font color="{taux_color}">{taux_reussite}%</font>', "Taux reussite"),
        (str(total_charges), "Total charges"),
        (f'<font color="#16a34a">{ok_charges}</font>', "OK"),
        (f'<font color="#dc2626">{nok_charges}</font>', "NOK"),
        (f"{energy_total:.1f} kWh", "Energie"),
        (str(sites_actifs), "Sites"),
    ]
    elems.append(_pdf_kpi_table(kpis, styles))
    elems.append(Spacer(1, 10))

    if site_performance:
        elems.append(Paragraph("Performance par site", styles["section"]))
        col_w = A4[0] - 3 * cm
        widths = [col_w*0.32, col_w*0.11, col_w*0.11, col_w*0.11, col_w*0.18, col_w*0.17]
        rows_data = []
        for sp in sorted(site_performance, key=lambda x: x.get("Site", "")):
            ev = sp.get("energy_total", 0)
            rows_data.append([
                _safe(sp.get("Site", "")),
                str(int(sp.get("total", 0))),
                str(int(sp.get("ok", 0))),
                str(int(sp.get("nok", 0))),
                f"{float(sp.get('taux_ok', 0)):.1f}%",
                f"{float(ev):.1f}" if pd.notna(ev) else "0.0",
            ])
        elems.append(_pdf_data_table(
            ["Site", "Total", "OK", "NOK", "Taux OK", "Energie (kWh)"],
            rows_data, styles, widths,
        ))
        elems.append(Spacer(1, 8))

    if top_errors:
        elems.append(Paragraph("Top erreurs", styles["section"]))
        col_w = A4[0] - 3 * cm
        widths = [col_w*0.06, col_w*0.54, col_w*0.20, col_w*0.20]
        rows_data = [
            [str(e["rank"]), _safe(str(e["type"])), str(e["count"]), f"{e['percentage']}%"]
            for e in top_errors
        ]
        elems.append(_pdf_data_table(
            ["#", "Type erreur", "Occurrences", "% erreurs"],
            rows_data, styles, widths,
        ))
        elems.append(Spacer(1, 8))

    if moment_distribution:
        elems.append(Paragraph("Repartition des erreurs par moment", styles["section"]))
        col_w = A4[0] - 3 * cm
        widths = [col_w*0.50, col_w*0.25, col_w*0.25]
        rows_data = [
            [_safe(m["moment"]), str(m["count"]), f"{m['percentage']}%"]
            for m in moment_distribution
        ]
        elems.append(_pdf_data_table(
            ["Moment", "Occurrences", "% erreurs"],
            rows_data, styles, widths,
        ))
        elems.append(Spacer(1, 8))

    if points_forts or points_attention or recommandations:
        elems.append(Paragraph("Insights et Recommandations", styles["section"]))
        if points_forts:
            elems.append(Paragraph("<b>Points forts</b>", styles["normal"]))
            for p in points_forts:
                elems.append(Paragraph(f'<font color="#16a34a">+</font>  {_safe(p)}', styles["normal"]))
            elems.append(Spacer(1, 4))
        if points_attention:
            elems.append(Paragraph("<b>Points attention</b>", styles["normal"]))
            for p in points_attention:
                elems.append(Paragraph(f'<font color="#d97706">!</font>  {_safe(p)}', styles["normal"]))
            elems.append(Spacer(1, 4))
        if recommandations:
            elems.append(Paragraph("<b>Recommandations</b>", styles["normal"]))
            for r in recommandations:
                elems.append(Paragraph(f'&gt;&gt;  {_safe(r)}', styles["normal"]))

    elems.append(Spacer(1, 16))
    elems.append(HRFlowable(width="100%", thickness=0.5, color=_PDF_BORDER))
    elems.append(Paragraph(f"Document genere automatiquement le {date_generation}", styles["small"]))

    doc.build(elems)
    buf.seek(0)
    return buf


def _build_pdf_from_report_pages(images_data_urls: list[str]) -> BytesIO:
    if not images_data_urls:
        raise HTTPException(status_code=400, detail="Aucune page de rapport fournie")

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=1.2 * cm,
        rightMargin=1.2 * cm,
        topMargin=1.2 * cm,
        bottomMargin=1.2 * cm,
    )

    max_width = A4[0] - (doc.leftMargin + doc.rightMargin)
    max_height = A4[1] - (doc.topMargin + doc.bottomMargin)
    story = []

    for idx, data_url in enumerate(images_data_urls):
        if not data_url.startswith("data:image"):
            continue

        try:
            _, raw_data = data_url.split(",", 1)
            image_bytes = base64.b64decode(raw_data)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Image de page invalide (index {idx})") from exc

        img = Image(BytesIO(image_bytes))
        ratio = min(max_width / img.imageWidth, max_height / img.imageHeight)
        img.drawWidth = img.imageWidth * ratio
        img.drawHeight = img.imageHeight * ratio
        story.append(img)

        if idx < len(images_data_urls) - 1:
            story.append(PageBreak())

    if not story:
        raise HTTPException(status_code=400, detail="Aucune image exploitable pour générer le PDF")

    doc.build(story)
    buf.seek(0)
    return buf


# ── HTML report (existing) ─────────────────────────────────────────────────────

@router.get("/report")
async def get_report(
    request: Request,
    sites: str = Query(default=""),
    date_debut: date = Query(default=None),
    date_fin: date = Query(default=None),
    error_types: str | None = Query(default=None),
    moments: str | None = Query(default=None),
):
    date_debut_parsed = date_debut
    date_fin_parsed = date_fin

    error_type_list = None if error_types is None else [e.strip() for e in error_types.split(",") if e.strip()]
    moment_list = None if moments is None else [m.strip() for m in moments.split(",") if m.strip()]

    where_clause, params = _build_conditions(
        sites, date_debut_parsed, date_fin_parsed
    )

    sql_count = f"""
        SELECT COUNT(*) as total_charges_count
        FROM kpi_sessions
        WHERE {where_clause}
    """

    df_count = query_df(sql_count, params)
    total_charges_from_count = int(df_count.iloc[0]['total_charges_count']) if not df_count.empty else 0

    sql_base = f"""
        SELECT
            ID,
            Site,
            PDC,
            `Datetime start`,
            `State of charge(0:good, 1:error)` as state,
            warning,
            type_erreur,
            moment
        FROM kpi_sessions
        WHERE {where_clause}
    """

    df_base = query_df(sql_base, params)

    if df_base.empty:
        return templates.TemplateResponse(
            "partials/report.html",
            {
                "request": request,
                "error": "Aucune donnée disponible pour la période sélectionnée",
                "date_debut": date_debut or "",
                "date_fin": date_fin or "",
                "sites": sites,
                # Variables nécessaires au <script> hors bloc {% if error %}
                "selected_date_debut": date_debut.isoformat() if date_debut else "",
                "selected_date_fin": date_fin.isoformat() if date_fin else "",
                "selected_sites": sites,
                "selected_error_types": error_types,
                "selected_moments": moments,
            },
        )

    total_from_base = len(df_base)

    if total_from_base > 0:
        ids = df_base['ID'].dropna().unique().tolist()

        if ids:
            if len(ids) > 1000:
                ids = ids[:1000]

            placeholders = ",".join([f":id_{i}" for i in range(len(ids))])
            params_extra = {f"id_{i}": str(ids[i]) for i in range(len(ids))}

            sql_extra = f"""
                SELECT
                    ID,
                    `Datetime start`,
                    `Datetime end`,
                    `Energy (Kwh)`,
                    `Max Power (Kw)`,
                    `SOC Start`,
                    `SOC End`,
                    Vehicle,
                    moment_avancee,
                    `{EVI_MOMENT}`,
                    `{EVI_CODE}`,
                    `{DS_PC}`
                FROM kpi_sessions
                WHERE ID IN ({placeholders})
            """

            df_extra = query_df(sql_extra, params_extra)

            cols_extra = [col for col in df_extra.columns if col != 'ID' and col != 'Datetime start']
            df = df_base.merge(df_extra[['ID'] + cols_extra], on='ID', how='left')
        else:
            df = df_base
    else:
        df = df_base

    if "Datetime start" not in df.columns:
        df = df_base.copy()

    total_charges_explicit = total_from_base

    df["is_ok"] = pd.to_numeric(df["state"], errors="coerce").fillna(0).astype(int).eq(0)
    df = _apply_status_filters(df, error_type_list, moment_list)

    total_charges = total_charges_explicit

    ok_charges = int(df["is_ok_filt"].sum())
    nok_charges = total_charges - ok_charges
    taux_reussite = round(ok_charges / total_charges * 100, 1) if total_charges else 0
    taux_echec = round(nok_charges / total_charges * 100, 1) if total_charges else 0

    energy_total = float(df["Energy (Kwh)"].sum()) if "Energy (Kwh)" in df.columns else 0.0
    energy_avg = float(df["Energy (Kwh)"].mean()) if "Energy (Kwh)" in df.columns and not df["Energy (Kwh)"].isna().all() else 0.0

    sites_actifs = df["Site"].nunique() if "Site" in df.columns else 0

    if "Datetime start" in df.columns and "Datetime end" in df.columns:
        df["duration"] = (pd.to_datetime(df["Datetime end"], errors="coerce") -
                         pd.to_datetime(df["Datetime start"], errors="coerce")).dt.total_seconds() / 60
        duree_moyenne = float(df["duration"].mean()) if not df["duration"].isna().all() else 0.0
    else:
        duree_moyenne = 0.0

    message_cle = ""
    if taux_reussite >= 90:
        message_cle = f"Excellent taux de réussite de {taux_reussite}%. Le réseau fonctionne de manière optimale."
    elif taux_reussite >= 80:
        message_cle = f"Bon taux de réussite de {taux_reussite}%. Quelques améliorations possibles sur les erreurs."
    else:
        message_cle = f"Taux de réussite de {taux_reussite}%. Attention requise sur les points d'erreur identifiés."

    stats_site = (
        df.groupby("Site")
        .agg(
            total=("is_ok_filt", "count"),
            ok=("is_ok_filt", "sum"),
        )
        .reset_index()
    )
    stats_site["nok"] = stats_site["total"] - stats_site["ok"]
    stats_site["taux_ok"] = np.where(
        stats_site["total"] > 0,
        (stats_site["ok"] / stats_site["total"] * 100).round(1),
        0,
    )
    stats_site["energy_total"] = df.groupby("Site")["Energy (Kwh)"].sum().values if "Energy (Kwh)" in df.columns else [0.0] * len(stats_site)

    site_performance = stats_site.to_dict("records")

    if "Datetime start" not in df.columns:
        df["date"] = pd.to_datetime(df.get("Datetime start", pd.Series()), errors="coerce").dt.date
    else:
        df["date"] = pd.to_datetime(df["Datetime start"], errors="coerce").dt.date

    evolution_daily = df.groupby("date").agg(
        total=("is_ok_filt", "count"),
        ok=("is_ok_filt", "sum"),
    ).reset_index()
    evolution_daily["nok"] = evolution_daily["total"] - evolution_daily["ok"]
    evolution_daily["taux_reussite"] = (evolution_daily["ok"] / evolution_daily["total"] * 100).round(1)

    if date_debut_parsed and date_fin_parsed:
        all_dates_evo = pd.date_range(date_debut_parsed, date_fin_parsed, freq="D").date
        evolution_daily["date_str"] = evolution_daily["date"].astype(str)
        evolution_dict = evolution_daily.set_index("date_str").to_dict("index")
        evolution_complete = []
        for d in all_dates_evo:
            d_str = str(d)
            if d_str in evolution_dict:
                row = evolution_dict[d_str]
                evolution_complete.append({
                    "date": d_str,
                    "total": int(row["total"]),
                    "ok": int(row["ok"]),
                    "nok": int(row["nok"]),
                    "taux_reussite": float(row["taux_reussite"])
                })
            else:
                evolution_complete.append({
                    "date": d_str,
                    "total": 0,
                    "ok": 0,
                    "nok": 0,
                    "taux_reussite": 0.0
                })
        evolution_timeline = evolution_complete
    elif not evolution_daily.empty:
        min_date_evo = evolution_daily["date"].min()
        max_date_evo = evolution_daily["date"].max()
        if pd.notna(min_date_evo) and pd.notna(max_date_evo):
            all_dates_evo = pd.date_range(min_date_evo, max_date_evo, freq="D").date
            evolution_dict = evolution_daily.set_index("date").to_dict("index")
            evolution_complete = []
            for d in all_dates_evo:
                if d in evolution_dict:
                    row = evolution_dict[d]
                    evolution_complete.append({
                        "date": str(d),
                        "total": int(row["total"]),
                        "ok": int(row["ok"]),
                        "nok": int(row["nok"]),
                        "taux_reussite": float(row["taux_reussite"])
                    })
                else:
                    evolution_complete.append({
                        "date": str(d),
                        "total": 0,
                        "ok": 0,
                        "nok": 0,
                        "taux_reussite": 0.0
                    })
            evolution_timeline = evolution_complete
        else:
            evolution_daily["date"] = evolution_daily["date"].astype(str)
            evolution_timeline = evolution_daily.to_dict("records")
    else:
        evolution_timeline = []

    err_df = df[~df["is_ok_filt"]].copy()

    top_errors = []
    if "type_erreur" in err_df.columns and not err_df["type_erreur"].isna().all():
        error_counts = err_df["type_erreur"].value_counts().head(5)
        for idx, (error_type, count) in enumerate(error_counts.items(), 1):
            top_errors.append({
                "rank": idx,
                "type": str(error_type) if pd.notna(error_type) else "Non spécifié",
                "count": int(count),
                "percentage": round(count / len(err_df) * 100, 1) if len(err_df) > 0 else 0
            })

    moment_distribution = []
    if "moment" in err_df.columns:
        moment_counts = err_df["moment"].value_counts()
        moment_groups = {
            "Avant charge": ["Init", "Lock Connector", "CableCheck"],
            "Charge": ["Charge"],
            "Fin de charge": ["Fin de charge"]
        }

        for group_name, group_values in moment_groups.items():
            count = sum(moment_counts.get(val, 0) for val in group_values)
            if count > 0:
                moment_distribution.append({
                    "moment": group_name,
                    "count": int(count),
                    "percentage": round(count / len(err_df) * 100, 1) if len(err_df) > 0 else 0
                })

    err_df["date"] = pd.to_datetime(err_df["Datetime start"], errors="coerce").dt.date
    error_evolution = err_df.groupby("date").size().reset_index(name="count")
    error_evolution = error_evolution.sort_values("date")

    if date_debut_parsed and date_fin_parsed:
        all_dates_err = pd.date_range(date_debut_parsed, date_fin_parsed, freq="D").date
        if not error_evolution.empty:
            error_evolution["date_str"] = error_evolution["date"].astype(str)
            error_dict = error_evolution.set_index("date_str")["count"].to_dict()
        else:
            error_dict = {}
        error_evolution_complete = []
        for d in all_dates_err:
            d_str = str(d)
            error_evolution_complete.append({
                "date": d_str,
                "count": int(error_dict.get(d_str, 0))
            })
        error_evolution_timeline = error_evolution_complete
    elif not error_evolution.empty:
        min_date_err = error_evolution["date"].min()
        max_date_err = error_evolution["date"].max()
        if pd.notna(min_date_err) and pd.notna(max_date_err):
            all_dates_err = pd.date_range(min_date_err, max_date_err, freq="D").date
            error_dict = error_evolution.set_index("date")["count"].to_dict()
            error_evolution_complete = []
            for d in all_dates_err:
                error_evolution_complete.append({
                    "date": str(d),
                    "count": int(error_dict.get(d, 0))
                })
            error_evolution_timeline = error_evolution_complete
        else:
            error_evolution["date"] = error_evolution["date"].astype(str)
            error_evolution_timeline = error_evolution.to_dict("records")
    else:
        if date_debut_parsed and date_fin_parsed:
            all_dates_err = pd.date_range(date_debut_parsed, date_fin_parsed, freq="D").date
            error_evolution_timeline = [{"date": str(d), "count": 0} for d in all_dates_err]
        else:
            error_evolution_timeline = []

    top_all_errors: list[dict[str, Any]] = []
    detail_all_errors: list[dict[str, Any]] = []
    detail_all_pivot: dict[str, Any] = {"columns": [], "rows": []}

    by_site = (
        df.groupby("Site", as_index=False)
        .agg(Total_Charges=("is_ok_filt", "count"), Charges_OK=("is_ok_filt", "sum"))
        .assign(Charges_NOK=lambda d: d["Total_Charges"] - d["Charges_OK"])
    )

    if not err_df.empty and EVI_MOMENT in err_df.columns:
        evi_step = pd.to_numeric(err_df.get(EVI_MOMENT, pd.Series(np.nan, index=err_df.index)), errors="coerce")
        evi_code = pd.to_numeric(err_df.get(EVI_CODE, pd.Series(np.nan, index=err_df.index)), errors="coerce").fillna(0).astype(int)
        ds_pc = pd.to_numeric(err_df.get(DS_PC, pd.Series(np.nan, index=err_df.index)), errors="coerce").fillna(0).astype(int)
        moment_raw = err_df.get("moment", pd.Series(None, index=err_df.index))

        def resolve_moment_label(idx: int) -> str:
            label = None
            step_val = evi_step.loc[idx] if idx in evi_step.index else np.nan
            raw_val = moment_raw.loc[idx] if idx in moment_raw.index else None

            if pd.notna(step_val):
                label = _map_moment_label(step_val)
            if (not label or label == "Unknown") and isinstance(raw_val, str) and raw_val.strip():
                label = raw_val.strip()
            return label or "Unknown"

        err_df["moment_label"] = [resolve_moment_label(i) for i in err_df.index]

        sub_evi_mask = (ds_pc.eq(8192)) | (ds_pc.eq(0) & evi_code.ne(0))
        sub_ds_mask = ds_pc.ne(0) & ds_pc.ne(8192)

        sub_evi = err_df.loc[sub_evi_mask].copy()
        sub_evi["step"] = evi_step.loc[sub_evi.index]
        sub_evi["code"] = evi_code.loc[sub_evi.index]
        sub_evi["type"] = "Erreur_EVI"

        sub_ds = err_df.loc[sub_ds_mask].copy()
        sub_ds["step"] = evi_step.loc[sub_ds.index]
        sub_ds["code"] = ds_pc.loc[sub_ds.index]
        sub_ds["type"] = "Erreur_DownStream"

        all_err = pd.concat([sub_evi, sub_ds], ignore_index=True)

        if not all_err.empty:
            tbl_all = (
                all_err.groupby(["moment_label", "step", "code", "type"])
                .size()
                .reset_index(name="Occurrences")
                .sort_values("Occurrences", ascending=False)
            )

            total_err = int(tbl_all["Occurrences"].sum()) or 1
            tbl_all["percent"] = (tbl_all["Occurrences"] / total_err * 100).round(2)

            top3_all = tbl_all.head(3)
            top_all_errors = top3_all.to_dict("records")

            top_keys = top3_all[["moment_label", "step", "code", "type"]].to_records(index=False).tolist()
            detail_all_df = all_err[
                all_err[["moment_label", "step", "code", "type"]].apply(tuple, axis=1).isin(top_keys)
            ]
            detail_all_errors = (
                detail_all_df.groupby(["moment_label", "step", "code", "type", "Site"])
                .size()
                .reset_index(name="Occurrences")
                .sort_values(
                    ["type", "moment_label", "step", "code", "Occurrences"],
                    ascending=[True, True, True, True, False],
                )
                .to_dict("records")
            )
            detail_all_pivot = _build_pivot_table(detail_all_df, by_site)

    site_performance = []
    if "Site" in df.columns:
        site_stats = df.groupby("Site").agg(
            total=("is_ok_filt", "count"),
            ok=("is_ok_filt", "sum"),
        ).reset_index()
        site_stats["nok"] = site_stats["total"] - site_stats["ok"]
        site_stats["taux_ok"] = (site_stats["ok"] / site_stats["total"] * 100).round(1)
        site_stats["energy_total"] = df.groupby("Site")["Energy (Kwh)"].sum().values if "Energy (Kwh)" in df.columns else [0.0] * len(site_stats)
        site_performance = site_stats.to_dict("records")

    pdc_performance_by_site: dict[str, list[dict[str, Any]]] = {}
    if "PDC" in df.columns and "Site" in df.columns:
        for site_name in df["Site"].unique():
            df_site = df[df["Site"] == site_name].copy()
            if "PDC" in df_site.columns:
                pdc_stats = df_site.groupby("PDC").agg(
                    total=("is_ok_filt", "count"),
                    ok=("is_ok_filt", "sum"),
                ).reset_index()
                pdc_stats["nok"] = pdc_stats["total"] - pdc_stats["ok"]
                pdc_stats["taux_ok"] = (pdc_stats["ok"] / pdc_stats["total"] * 100).round(1)
                pdc_stats["site"] = site_name
                pdc_performance_by_site[site_name] = pdc_stats.to_dict("records")

    top_day_by_site: dict[str, dict[str, Any]] = {}
    charges_list_by_site: dict[str, list[dict[str, Any]]] = {}

    if "Site" in df.columns and "Datetime start" in df.columns:
        df["date"] = pd.to_datetime(df["Datetime start"], errors="coerce").dt.date
        df["date_str"] = df["date"].astype(str)

        for site_name in df["Site"].unique():
            df_site = df[df["Site"] == site_name].copy()

            if not df_site.empty:
                daily_counts = df_site.groupby("date_str").agg(
                    total=("is_ok_filt", "count"),
                    ok=("is_ok_filt", "sum"),
                ).reset_index()
                daily_counts["nok"] = daily_counts["total"] - daily_counts["ok"]
                daily_counts = daily_counts.sort_values("total", ascending=False)

                if not daily_counts.empty:
                    top_day_row = daily_counts.iloc[0]
                    top_day_date = top_day_row["date_str"]
                    top_day_total = int(top_day_row["total"])
                    top_day_ok = int(top_day_row["ok"])
                    top_day_nok = int(top_day_row["nok"])

                    hourly_active_sessions = []
                    hourly_sessions_count = []
                    usage_duration_by_level = []

                    try:
                        query_15min = """
                            SELECT
                                Times,
                                active_sessions as count,
                                HOUR(Times) as hour,
                                MINUTE(Times) as minute
                            FROM indicator.kpi_concurrency_15min
                            WHERE Site = :site
                              AND DATE(Times) = :date_focus
                            ORDER BY Times
                        """
                        df_15min = query_df(query_15min, {"site": site_name, "date_focus": top_day_date})

                        if not df_15min.empty:
                            df_15min["Times"] = pd.to_datetime(df_15min["Times"])
                            day_start = pd.Timestamp.combine(pd.to_datetime(top_day_date).date(), pd.Timestamp.min.time())

                            for _, row in df_15min.iterrows():
                                time_point = row["Times"]
                                hour = int(row["hour"])
                                minute = int(row["minute"])
                                count = int(row["count"])
                                hour_label = f"{hour:02d}:{minute:02d}"
                                minutes_from_start = int((time_point - day_start).total_seconds() / 60)

                                hourly_active_sessions.append({
                                    "time": time_point.strftime("%H:%M"),
                                    "hour": hour,
                                    "minute": minute,
                                    "minutes_from_start": minutes_from_start,
                                    "hour_label": hour_label,
                                    "count": count,
                                })
                        else:
                            for hour in range(24):
                                hourly_active_sessions.append({
                                    "time": f"{hour:02d}:00",
                                    "hour": hour,
                                    "minute": 0,
                                    "minutes_from_start": hour * 60,
                                    "hour_label": f"{hour:02d}:00",
                                    "count": 0,
                                })

                        query_daily = """
                            SELECT
                                NbPriseLive as level,
                                Duration_H
                            FROM indicator.kpi_concurrency_Daily
                            WHERE Site = :site
                              AND Date = :date_focus
                            ORDER BY NbPriseLive
                        """
                        df_daily = query_df(query_daily, {"site": site_name, "date_focus": top_day_date})

                        if not df_daily.empty:
                            for _, row in df_daily.iterrows():
                                level = int(row["level"])
                                duration_h = float(row["Duration_H"])
                                duration_minutes = duration_h * 60
                                hours = int(duration_minutes // 60)
                                minutes = int(duration_minutes % 60)

                                usage_duration_by_level.append({
                                    "level": level,
                                    "duration_minutes": duration_minutes,
                                    "duration_hours": hours,
                                    "duration_minutes_remainder": minutes,
                                    "duration_formatted": f"{hours}h{minutes:02d}" if hours > 0 else f"{minutes}min",
                                })

                        query_sessions_hourly = """
                            SELECT
                                HOUR(`Datetime start`) as hour,
                                COUNT(DISTINCT ID) as session_count
                            FROM kpi_sessions
                            WHERE Site = :site
                              AND DATE(`Datetime start`) = :date_focus
                            GROUP BY HOUR(`Datetime start`)
                            ORDER BY hour
                        """
                        df_sessions_hourly = query_df(query_sessions_hourly, {"site": site_name, "date_focus": top_day_date})

                        if not df_sessions_hourly.empty:
                            sessions_by_hour = {int(row["hour"]): int(row["session_count"]) for _, row in df_sessions_hourly.iterrows()}
                            cumulative_count = 0
                            for hour in range(24):
                                cumulative_count += sessions_by_hour.get(hour, 0)
                                hourly_sessions_count.append({
                                    "hour": hour,
                                    "minutes_from_start": hour * 60,
                                    "hour_label": f"{hour:02d}:00",
                                    "count": cumulative_count,
                                })
                        else:
                            for hour in range(24):
                                hourly_sessions_count.append({
                                    "hour": hour,
                                    "minutes_from_start": hour * 60,
                                    "hour_label": f"{hour:02d}:00",
                                    "count": 0,
                                })
                    except Exception:
                        pass

                    top_day_by_site[site_name] = {
                        "date": top_day_date,
                        "total": top_day_total,
                        "ok": top_day_ok,
                        "nok": top_day_nok,
                        "hourly_active_sessions": hourly_active_sessions,
                        "hourly_sessions_count": hourly_sessions_count,
                        "usage_duration_by_level": usage_duration_by_level,
                    }

            if not df_site.empty:
                site_ids = df_site["ID"].dropna().unique().tolist()

                if site_ids:
                    site_ids_limited = site_ids[:100]

                    placeholders = ",".join([f":id_{i}" for i in range(len(site_ids_limited))])
                    params_charges = {f"id_{i}": str(site_ids_limited[i]) for i in range(len(site_ids_limited))}

                    sql_charges = f"""
                        SELECT
                            ID,
                            Site,
                            PDC,
                            `Datetime start`,
                            `Datetime end`,
                            `Energy (Kwh)`,
                            `Max Power (Kw)`,
                            `SOC Start`,
                            `SOC End`,
                            Vehicle,
                            `State of charge(0:good, 1:error)` as state,
                            warning,
                            type_erreur,
                            moment
                        FROM kpi_sessions
                        WHERE ID IN ({placeholders})
                        ORDER BY `Datetime start` DESC
                    """

                    df_charges = query_df(sql_charges, params_charges)

                    charges_list = []
                    if not df_charges.empty:
                        for _, row in df_charges.iterrows():
                            energy_val = row.get("Energy (Kwh)")
                            max_power_val = row.get("Max Power (Kw)")
                            soc_start_val = row.get("SOC Start")
                            soc_end_val = row.get("SOC End")
                            vehicle_val = row.get("Vehicle")
                            state_val = row.get("state", 0)
                            warning_val = row.get("warning", 0)
                            is_warning = int(warning_val) == 1 if pd.notna(warning_val) else False

                            is_ok = (int(state_val) == 0 if pd.notna(state_val) else True) or is_warning

                            session_id = row.get("ID")
                            logv0_url = None
                            if pd.notna(session_id) and LOGV0_BASE_URL:
                                logv0_url = f"{LOGV0_BASE_URL}?session_id={str(session_id).strip()}&project=ELTO"

                            charges_list.append({
                                "ID": str(session_id) if pd.notna(session_id) else "",
                                "Site": str(row.get("Site", "")) if pd.notna(row.get("Site")) else "",
                                "PDC": str(row.get("PDC", "")) if pd.notna(row.get("PDC")) else "",
                                "Datetime start": str(row.get("Datetime start", "")) if pd.notna(row.get("Datetime start")) else "",
                                "Datetime end": str(row.get("Datetime end", "")) if pd.notna(row.get("Datetime end")) else "",
                                "Energy (Kwh)": round(float(energy_val), 1) if pd.notna(energy_val) and energy_val != 0 else None,
                                "Max Power (Kw)": round(float(max_power_val), 1) if pd.notna(max_power_val) and max_power_val != 0 else None,
                                "SOC Start": round(float(soc_start_val), 1) if pd.notna(soc_start_val) and soc_start_val != 0 else None,
                                "SOC End": round(float(soc_end_val), 1) if pd.notna(soc_end_val) and soc_end_val != 0 else None,
                                "Vehicle": str(vehicle_val) if pd.notna(vehicle_val) and str(vehicle_val).strip() else None,
                                "is_ok": 1 if is_ok else 0,
                                "is_warning": 1 if is_warning else 0,
                                "statut": "✅ OK" if is_ok else "🔴 NOK",
                                "type_erreur": str(row.get("type_erreur", "")) if pd.notna(row.get("type_erreur")) else "",
                                "moment": str(row.get("moment", "")) if pd.notna(row.get("moment")) else "",
                                "url": f"https://elto.nidec-asi-online.com/Charge/detail?id={session_id}" if pd.notna(session_id) else "",
                                "logv0_url": logv0_url,
                            })

                    charges_list_by_site[site_name] = charges_list

    avg_usage_duration_by_level_by_site: dict[str, list[dict[str, Any]]] = {}
    if date_debut_parsed and date_fin_parsed and "Site" in df.columns:
        try:
            all_dates_period = pd.date_range(date_debut_parsed, date_fin_parsed, freq="D").strftime("%Y-%m-%d")
            sites_list_period = df["Site"].unique().tolist()

            for site_name in sites_list_period:
                all_duration_data = []
                for date_str in all_dates_period:
                    query_avg = """
                        SELECT
                            NbPriseLive as level,
                            Duration_H
                        FROM indicator.kpi_concurrency_Daily
                        WHERE Site = :site
                          AND Date = :date_focus
                    """
                    df_avg = query_df(query_avg, {"site": site_name, "date_focus": date_str})
                    if not df_avg.empty:
                        for _, row in df_avg.iterrows():
                            all_duration_data.append({
                                "level": int(row["level"]),
                                "duration_h": float(row["Duration_H"]),
                            })

                if all_duration_data:
                    df_avg_site = pd.DataFrame(all_duration_data)
                    avg_by_level = df_avg_site.groupby("level")["duration_h"].mean().reset_index()
                    avg_by_level = avg_by_level.sort_values("level")

                    site_avg_list = []
                    for _, row in avg_by_level.iterrows():
                        level = int(row["level"])
                        duration_h_avg = float(row["duration_h"])
                        duration_minutes_avg = duration_h_avg * 60
                        hours_avg = int(duration_minutes_avg // 60)
                        minutes_avg = int(duration_minutes_avg % 60)

                        site_avg_list.append({
                            "level": level,
                            "duration_minutes": duration_minutes_avg,
                            "duration_hours": hours_avg,
                            "duration_minutes_remainder": minutes_avg,
                            "duration_formatted": f"{hours_avg}h{minutes_avg:02d}" if hours_avg > 0 else f"{minutes_avg}min",
                        })

                    if site_avg_list:
                        avg_usage_duration_by_level_by_site[site_name] = site_avg_list
        except Exception:
            pass

    site_daily_stats: list[dict[str, Any]] = []
    if "Site" in df.columns and "Datetime start" in df.columns:
        df["date"] = pd.to_datetime(df["Datetime start"], errors="coerce").dt.date
        for site_name in df["Site"].unique():
            df_site = df[df["Site"] == site_name].copy()
            total_charges_site = len(df_site)
            unique_days = df_site["date"].nunique()
            avg_per_day = round(total_charges_site / unique_days, 1) if unique_days > 0 else 0.0
            site_daily_stats.append({
                "site": site_name,
                "total_charges": total_charges_site,
                "avg_per_day": avg_per_day
            })
        site_daily_stats = sorted(site_daily_stats, key=lambda x: x["site"])

    daily_detail_global: list[dict[str, Any]] = []
    if "Datetime start" in df.columns:
        ok_global = df[df["is_ok_filt"]].copy()
        nok_global = df[~df["is_ok_filt"]].copy()

        if date_debut_parsed and date_fin_parsed:
            all_days = pd.date_range(date_debut_parsed, date_fin_parsed, freq="D").strftime("%Y-%m-%d")
        elif not df.empty:
            min_date = pd.to_datetime(df["Datetime start"], errors="coerce").min()
            max_date = pd.to_datetime(df["Datetime start"], errors="coerce").max()
            if pd.notna(min_date) and pd.notna(max_date):
                all_days = pd.date_range(min_date.date(), max_date.date(), freq="D").strftime("%Y-%m-%d")
            else:
                all_days = []
        else:
            all_days = []

        if len(all_days) > 0:
            ok_global["day"] = pd.to_datetime(ok_global["Datetime start"], errors="coerce").dt.strftime("%Y-%m-%d")
            nok_global["day"] = pd.to_datetime(nok_global["Datetime start"], errors="coerce").dt.strftime("%Y-%m-%d")

            g_ok = ok_global.groupby("day").size().reset_index(name="Nb")
            g_nok = nok_global.groupby("day").size().reset_index(name="Nb")

            g_ok_dict = g_ok.set_index("day")["Nb"].to_dict()
            g_nok_dict = g_nok.set_index("day")["Nb"].to_dict()

            for day in all_days:
                ok_val = int(g_ok_dict.get(day, 0))
                nok_val = int(g_nok_dict.get(day, 0))
                daily_detail_global.append({
                    "day": day,
                    "ok": ok_val,
                    "nok": nok_val,
                    "total": ok_val + nok_val
                })

    daily_detail_by_site: dict[str, list[dict[str, Any]]] = {}
    if "Site" in df.columns and "Datetime start" in df.columns:
        for site_name in df["Site"].unique():
            df_site = df[df["Site"] == site_name].copy()
            ok_site = df_site[df_site["is_ok_filt"]].copy()
            nok_site = df_site[~df_site["is_ok_filt"]].copy()

            if date_debut_parsed and date_fin_parsed:
                all_days = pd.date_range(date_debut_parsed, date_fin_parsed, freq="D").strftime("%Y-%m-%d")
            elif not df_site.empty:
                min_date = pd.to_datetime(df_site["Datetime start"], errors="coerce").min()
                max_date = pd.to_datetime(df_site["Datetime start"], errors="coerce").max()
                if pd.notna(min_date) and pd.notna(max_date):
                    all_days = pd.date_range(min_date.date(), max_date.date(), freq="D").strftime("%Y-%m-%d")
                else:
                    all_days = []
            else:
                all_days = []

            if len(all_days) > 0:
                ok_site["day"] = pd.to_datetime(ok_site["Datetime start"], errors="coerce").dt.strftime("%Y-%m-%d")
                nok_site["day"] = pd.to_datetime(nok_site["Datetime start"], errors="coerce").dt.strftime("%Y-%m-%d")

                g_ok = ok_site.groupby("day").size().reset_index(name="Nb")
                g_nok = nok_site.groupby("day").size().reset_index(name="Nb")

                g_ok_dict = g_ok.set_index("day")["Nb"].to_dict()
                g_nok_dict = g_nok.set_index("day")["Nb"].to_dict()

                daily_rows = []
                for day in all_days:
                    ok_val = int(g_ok_dict.get(day, 0))
                    nok_val = int(g_nok_dict.get(day, 0))
                    daily_rows.append({
                        "day": day,
                        "ok": ok_val,
                        "nok": nok_val,
                        "total": ok_val + nok_val
                    })

                daily_detail_by_site[site_name] = daily_rows

    heatmap_data = []
    if "Datetime start" in df.columns:
        df["hour"] = pd.to_datetime(df["Datetime start"], errors="coerce").dt.hour
        heatmap_hourly = df.groupby("hour").agg(
            total=("is_ok_filt", "count"),
            ok=("is_ok_filt", "sum"),
        ).reset_index()
        heatmap_hourly["nok"] = heatmap_hourly["total"] - heatmap_hourly["ok"]
        heatmap_data = heatmap_hourly.to_dict("records")

    puissance_max = float(df["Max Power (Kw)"].max()) if "Max Power (Kw)" in df.columns and not df["Max Power (Kw)"].isna().all() else 0.0

    energy_distribution = []
    if "Energy (Kwh)" in df.columns:
        energy_col = df["Energy (Kwh)"].dropna()
        if len(energy_col) > 0:
            bins = [0, 10, 20, 30, 40, 50, 100, float('inf')]
            labels = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-100", "100+"]
            energy_col_binned = pd.cut(energy_col, bins=bins, labels=labels, include_lowest=True)
            energy_dist = energy_col_binned.value_counts().sort_index()
            for bin_label, count in energy_dist.items():
                energy_distribution.append({
                    "range": str(bin_label),
                    "count": int(count),
                    "percentage": round(count / len(energy_col) * 100, 1)
                })

    if "duration" not in df.columns:
        if "Datetime start" in df.columns and "Datetime end" in df.columns:
            df["duration"] = (pd.to_datetime(df["Datetime end"], errors="coerce") -
                           pd.to_datetime(df["Datetime start"], errors="coerce")).dt.total_seconds() / 60
        else:
            df["duration"] = pd.Series([0.0] * len(df))

    duration_col = df["duration"].dropna()
    duree_totale = float(duration_col.sum()) if len(duration_col) > 0 else 0.0
    duree_moyenne_calc = float(duration_col.mean()) if len(duration_col) > 0 else 0.0
    duree_max = float(duration_col.max()) if len(duration_col) > 0 else 0.0
    duree_min = float(duration_col.min()) if len(duration_col) > 0 else 0.0

    duration_distribution = []
    if len(duration_col) > 0:
        bins = [0, 15, 30, 45, 60, 90, 120, 180, float('inf')]
        labels = ["0-15", "15-30", "30-45", "45-60", "60-90", "90-120", "120-180", "180+"]
        duration_col_binned = pd.cut(duration_col, bins=bins, labels=labels, include_lowest=True)
        duration_dist = duration_col_binned.value_counts().sort_index()
        for bin_label, count in duration_dist.items():
            duration_distribution.append({
                "range": str(bin_label),
                "count": int(count),
                "percentage": round(count / len(duration_col) * 100, 1)
            })

    points_forts = []
    points_attention = []
    recommandations = []

    if taux_reussite >= 85:
        points_forts.append(f"Taux de réussite global excellent ({taux_reussite}%)")
    if energy_total > 0:
        points_forts.append(f"Énergie totale délivrée importante : {energy_total:.1f} kWh")
    if sites_actifs > 0:
        points_forts.append(f"{sites_actifs} site(s) actif(s) sur la période")

    if taux_echec > 15:
        points_attention.append(f"Taux d'échec élevé ({taux_echec}%) nécessitant une analyse approfondie")
    if top_errors:
        top_error = top_errors[0]
        points_attention.append(f"Type d'erreur le plus fréquent : {top_error['type']} ({top_error['count']} occurrences)")
    if moment_distribution:
        max_moment = max(moment_distribution, key=lambda x: x["count"])
        points_attention.append(f"Concentration des erreurs au moment : {max_moment['moment']} ({max_moment['count']} erreurs)")

    if taux_echec > 20:
        recommandations.append("Analyser en détail les causes des erreurs pour améliorer le taux de réussite")
    if pdc_performance_by_site:
        worst_pdc_info = None
        worst_taux = 100.0
        for site_name, pdc_list in pdc_performance_by_site.items():
            for pdc in pdc_list:
                taux = pdc.get("taux_ok", 100.0)
                if taux < worst_taux:
                    worst_taux = taux
                    worst_pdc_info = {"site": site_name, "pdc": pdc.get("PDC", "N/A"), "taux": taux}
        if worst_pdc_info and worst_taux < 100.0:
            recommandations.append(f"Intervention recommandée sur le PDC {worst_pdc_info['pdc']} du site {worst_pdc_info['site']} (taux: {worst_taux}%)")
    if heatmap_data:
        max_error_hour = max(heatmap_data, key=lambda x: x.get("nok", 0))
        if max_error_hour.get("nok", 0) > 0:
            recommandations.append(f"Surveillance renforcée recommandée aux heures {max_error_hour['hour']}h (période de pointe d'erreurs)")

    date_debut_str = date_debut_parsed.strftime("%d/%m/%Y") if date_debut_parsed else "Toutes les dates"
    date_fin_str = date_fin_parsed.strftime("%d/%m/%Y") if date_fin_parsed else "Toutes les dates"
    date_generation = datetime.now().strftime("%d/%m/%Y %H:%M")

    sites_list = [s.strip() for s in sites.split(",") if s.strip()] if sites else []
    if not sites_list:
        sites_list = sorted(df["Site"].unique().tolist()) if "Site" in df.columns else []

    site_success_cards = [
        {
            "site": row["Site"],
            "ok": int(row["ok"]),
            "total": int(row["total"]),
            "taux_ok": float(row["taux_ok"]),
            "color": SITE_COLOR_PALETTE[idx % len(SITE_COLOR_PALETTE)],
        }
        for idx, row in stats_site.sort_values("Site").iterrows()
    ]

    site_success_bars = [
        {
            "site": row["Site"],
            "taux_ok": float(row["taux_ok"]),
            "total": int(row["total"]),
            "color": SITE_COLOR_PALETTE[idx % len(SITE_COLOR_PALETTE)],
        }
        for idx, row in stats_site.sort_values("taux_ok", ascending=False).iterrows()
    ]

    return templates.TemplateResponse(
        "partials/report.html",
        {
            "request": request,
            "selected_sites": sites,
            "selected_date_debut": date_debut.isoformat() if date_debut else "",
            "selected_date_fin": date_fin.isoformat() if date_fin else "",
            "selected_error_types": error_types,
            "selected_moments": moments,
            "date_debut": date_debut_str,
            "date_fin": date_fin_str,
            "date_generation": date_generation,
            "sites_list": sites_list,
            "total_charges": total_charges,
            "ok_charges": ok_charges,
            "nok_charges": nok_charges,
            "taux_reussite": taux_reussite,
            "taux_echec": taux_echec,
            "energy_total": energy_total,
            "energy_avg": energy_avg,
            "sites_actifs": sites_actifs,
            "duree_moyenne": duree_moyenne,
            "message_cle": message_cle,
            "site_success_cards": site_success_cards,
            "site_success_bars": site_success_bars,
            "site_performance": site_performance,
            "evolution_timeline": evolution_timeline,
            "top_errors": top_errors,
            "moment_distribution": moment_distribution,
            "error_evolution_timeline": error_evolution_timeline,
            "top_all_errors": top_all_errors,
            "detail_all_errors": detail_all_errors,
            "detail_all_pivot": detail_all_pivot,
            "pdc_performance_by_site": pdc_performance_by_site,
            "site_daily_stats": site_daily_stats,
            "daily_detail_global": daily_detail_global,
            "daily_detail_by_site": daily_detail_by_site,
            "heatmap_data": heatmap_data,
            "top_day_by_site": top_day_by_site,
            "charges_list_by_site": charges_list_by_site,
            "avg_usage_duration_by_level_by_site": avg_usage_duration_by_level_by_site,
            "puissance_max": puissance_max,
            "energy_distribution": energy_distribution,
            "duree_totale": duree_totale,
            "duree_moyenne_calc": duree_moyenne_calc,
            "duree_max": duree_max,
            "duree_min": duree_min,
            "duration_distribution": duration_distribution,
            "points_forts": points_forts,
            "points_attention": points_attention,
            "recommandations": recommandations,
        },
    )


# ── PDF export ─────────────────────────────────────────────────────────────────

@router.get("/report/export-pdf")
async def export_report_pdf(
    request: Request,
    sites: str = Query(default=""),
    date_debut: date = Query(default=None),
    date_fin: date = Query(default=None),
    error_types: str | None = Query(default=None),
    moments: str | None = Query(default=None),
):
    error_type_list = None if error_types is None else [e.strip() for e in error_types.split(",") if e.strip()]
    moment_list = None if moments is None else [m.strip() for m in moments.split(",") if m.strip()]

    where_clause, params = _build_conditions(sites, date_debut, date_fin)

    sql = f"""
        SELECT
            ID, Site, PDC,
            `Datetime start`,
            `Energy (Kwh)`,
            `State of charge(0:good, 1:error)` as state,
            warning, type_erreur, moment
        FROM kpi_sessions
        WHERE {where_clause}
    """
    df = query_df(sql, params)

    if df.empty:
        return StreamingResponse(
            iter([b"Aucune donnee disponible."]),
            media_type="text/plain",
            status_code=404,
        )

    df["is_ok"] = pd.to_numeric(df["state"], errors="coerce").fillna(0).astype(int).eq(0)
    df = _apply_status_filters(df, error_type_list, moment_list)

    total_charges = len(df)
    ok_charges = int(df["is_ok_filt"].sum())
    nok_charges = total_charges - ok_charges
    taux_reussite = round(ok_charges / total_charges * 100, 1) if total_charges else 0.0
    taux_echec = round(nok_charges / total_charges * 100, 1) if total_charges else 0.0
    energy_total = float(df["Energy (Kwh)"].sum()) if "Energy (Kwh)" in df.columns else 0.0
    sites_actifs = int(df["Site"].nunique()) if "Site" in df.columns else 0

    site_performance = []
    if "Site" in df.columns:
        site_stats = df.groupby("Site").agg(
            total=("is_ok_filt", "count"),
            ok=("is_ok_filt", "sum"),
        ).reset_index()
        site_stats["nok"] = site_stats["total"] - site_stats["ok"]
        site_stats["taux_ok"] = (site_stats["ok"] / site_stats["total"] * 100).round(1)
        if "Energy (Kwh)" in df.columns:
            energy_by_site = df.groupby("Site")["Energy (Kwh)"].sum().reset_index()
            energy_by_site.columns = ["Site", "energy_total"]
            site_stats = site_stats.merge(energy_by_site, on="Site", how="left")
        else:
            site_stats["energy_total"] = 0.0
        site_performance = site_stats.to_dict("records")

    err_df = df[~df["is_ok_filt"]].copy()

    top_errors = []
    if "type_erreur" in err_df.columns and not err_df["type_erreur"].isna().all():
        for idx, (error_type, count) in enumerate(err_df["type_erreur"].value_counts().head(5).items(), 1):
            top_errors.append({
                "rank": idx,
                "type": str(error_type) if pd.notna(error_type) else "Non spécifié",
                "count": int(count),
                "percentage": round(count / len(err_df) * 100, 1) if len(err_df) > 0 else 0,
            })

    moment_distribution = []
    if "moment" in err_df.columns:
        moment_counts = err_df["moment"].value_counts()
        for group_name, group_values in {
            "Avant charge": ["Init", "Lock Connector", "CableCheck"],
            "Charge": ["Charge"],
            "Fin de charge": ["Fin de charge"],
        }.items():
            count = sum(moment_counts.get(val, 0) for val in group_values)
            if count > 0:
                moment_distribution.append({
                    "moment": group_name,
                    "count": int(count),
                    "percentage": round(count / len(err_df) * 100, 1) if len(err_df) > 0 else 0,
                })

    points_forts = []
    points_attention = []
    recommandations = []

    if taux_reussite >= 85:
        points_forts.append(f"Taux de réussite global excellent ({taux_reussite}%)")
    if energy_total > 0:
        points_forts.append(f"Énergie totale délivrée : {energy_total:.1f} kWh")
    if sites_actifs > 0:
        points_forts.append(f"{sites_actifs} site(s) actif(s) sur la période")
    if taux_echec > 15:
        points_attention.append(f"Taux d'échec élevé ({taux_echec}%) — analyse approfondie recommandée")
    if top_errors:
        points_attention.append(f"Erreur la plus fréquente : {top_errors[0]['type']} ({top_errors[0]['count']} occurrences)")
    if moment_distribution:
        max_moment = max(moment_distribution, key=lambda x: x["count"])
        points_attention.append(f"Concentration des erreurs : {max_moment['moment']} ({max_moment['count']} erreurs)")
    if taux_echec > 20:
        recommandations.append("Analyser les causes des erreurs pour améliorer le taux de réussite")
    if site_performance:
        worst = min(site_performance, key=lambda x: float(x.get("taux_ok", 100)))
        if float(worst.get("taux_ok", 100)) < 100:
            recommandations.append(f"Intervention recommandée sur le site {worst['Site']} (taux OK : {worst['taux_ok']}%)")

    date_debut_str = date_debut.strftime("%d/%m/%Y") if date_debut else "Toutes les dates"
    date_fin_str = date_fin.strftime("%d/%m/%Y") if date_fin else "Toutes les dates"
    date_generation = datetime.now().strftime("%d/%m/%Y %H:%M")

    sites_list = [s.strip() for s in sites.split(",") if s.strip()] if sites else []
    if not sites_list and "Site" in df.columns:
        sites_list = sorted(df["Site"].dropna().unique().tolist())

    pdf_buf = _build_pdf_export(
        date_debut_str=date_debut_str,
        date_fin_str=date_fin_str,
        sites_list=sites_list,
        date_generation=date_generation,
        total_charges=total_charges,
        ok_charges=ok_charges,
        nok_charges=nok_charges,
        taux_reussite=taux_reussite,
        taux_echec=taux_echec,
        energy_total=energy_total,
        sites_actifs=sites_actifs,
        site_performance=site_performance,
        top_errors=top_errors,
        moment_distribution=moment_distribution,
        points_forts=points_forts,
        points_attention=points_attention,
        recommandations=recommandations,
    )

    filename = f"rapport_recharge_{date_debut or 'all'}_{date_fin or 'all'}.pdf"
    return Response(
        content=pdf_buf.read(),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/report/export-pdf")
async def export_report_pdf_from_page(
    payload: dict = Body(default={}),
):
    images = payload.get("images", []) if isinstance(payload, dict) else []
    if not isinstance(images, list):
        raise HTTPException(status_code=400, detail="Le champ images doit être une liste")

    pdf_buf = _build_pdf_from_report_pages(images)
    filename = f"rapport_recharge_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
    return StreamingResponse(
        pdf_buf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
