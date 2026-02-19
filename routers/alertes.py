
from fastapi import APIRouter, Request, Query
from fastapi.templating import Jinja2Templates
from datetime import date
import pandas as pd

from db import query_df

router = APIRouter(tags=["alertes"])
templates = Jinja2Templates(directory="templates")


@router.get("/alertes")
async def get_alertes(
    request: Request,
    sites: str = Query(default=""),
    date_debut: date = Query(default=None),
    date_fin: date = Query(default=None),
    error_types: str = Query(default=""),
    moments: str = Query(default=""),
):
    sql = """
        SELECT
            Site,
            PDC,
            type_erreur,
            detection,
            occurrences_12h,
            moment,
            evi_code,
            downstream_code_pc
        FROM kpi_alertes
        ORDER BY detection DESC
    """
    
    df = query_df(sql)

    if not df.empty:
        df["detection"] = pd.to_datetime(df["detection"], errors="coerce")

        if date_debut:
            df = df[df["detection"] >= pd.Timestamp(date_debut)]
        if date_fin:
            df = df[df["detection"] < pd.Timestamp(date_fin) + pd.Timedelta(days=1)]
        
        if sites:
            site_list = [s.strip() for s in sites.split(",") if s.strip()]
            if site_list:
                df = df[df["Site"].isin(site_list)]

        if error_types and "type_erreur" in df.columns:
            type_list = [t.strip() for t in error_types.split(",") if t.strip()]
            if type_list:
                df = df[df["type_erreur"].isin(type_list)]

        if moments and "moment" in df.columns:
            moment_list = [m.strip() for m in moments.split(",") if m.strip()]
            if moment_list:
                df = df[df["moment"].isin(moment_list)]

        if "detection" in df.columns:
            df = df.dropna(subset=["detection"]).sort_values("detection", ascending=False)

    nb_alertes = len(df)
    
    if nb_alertes > 10:
        status = "danger"
    elif nb_alertes > 0:
        status = "warning"
    else:
        status = "success"
    
    top_sites = []
    if not df.empty:
        top = df.groupby("Site").size().sort_values(ascending=False).head(5)
        top_sites = [{"site": site, "count": count} for site, count in top.items()]

    rows = []
    if not df.empty:
        for _, row in df.iterrows():
            detection_value = row.get("detection")
            if pd.notna(detection_value):
                detection_value = pd.to_datetime(detection_value, errors="coerce")
                detection_value = (
                    detection_value.strftime("%Y-%m-%d %H:%M")
                    if not pd.isna(detection_value)
                    else ""
                )
            else:
                detection_value = ""

            occurrences_val = pd.to_numeric(row.get("occurrences_12h", 0), errors="coerce")
            occurrences = int(occurrences_val) if pd.notna(occurrences_val) else 0

            rows.append(
                {
                    "site": row.get("Site", ""),
                    "pdc": row.get("PDC", ""),
                    "type": row.get("type_erreur", ""),
                    "detection": detection_value,
                    "occurrences": occurrences,
                    "moment": row.get("moment", ""),
                    "evi_code": row.get("evi_code", ""),
                    "downstream_code_pc": row.get("downstream_code_pc", ""),
                }
            )

    return templates.TemplateResponse(
        "partials/alertes.html",
        {
            "request": request,
            "nb_alertes": nb_alertes,
            "status": status,
            "top_sites": top_sites,
            "rows": rows,
        }
    )