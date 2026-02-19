from fastapi import APIRouter, Request, Query
from fastapi.templating import Jinja2Templates
from datetime import date
import pandas as pd

from db import query_df, table_exists

router = APIRouter(tags=["kpis"])
templates = Jinja2Templates(directory="templates")
BASE_CHARGE_URL = "https://elto.nidec-asi-online.com/Charge/detail?id="


@router.get("/kpi/suspicious")
async def get_suspicious(
    request: Request,
    sites: str = Query(default=""),
    date_debut: date = Query(default=None),
    date_fin: date = Query(default=None),
):
    sql = """
        SELECT s.*,
               k.type_erreur,
               k.moment,
               k.warning,
               k.duration
        FROM kpi_suspicious_under_1kwh s
        LEFT JOIN kpi_sessions k ON s.ID = k.ID
    """
    df = query_df(sql)

    if not df.empty:
        if "Datetime start" in df.columns:
            df["Datetime start"] = pd.to_datetime(df["Datetime start"], errors="coerce")

            if date_debut:
                df = df[df["Datetime start"] >= pd.Timestamp(date_debut)]
            if date_fin:
                df = df[df["Datetime start"] < pd.Timestamp(date_fin) + pd.Timedelta(days=1)]

        if sites and "Site" in df.columns:
            site_list = [s.strip() for s in sites.split(",") if s.strip()]
            if site_list:
                df = df[df["Site"].isin(site_list)]

        if "Datetime start" in df.columns:
            df = df.sort_values("Datetime start")

    def format_ts(value):
        if pd.isna(value):
            return ""
        try:
            ts = pd.to_datetime(value, errors="coerce")
            return ts.strftime("%Y-%m-%d %H:%M") if not pd.isna(ts) else ""
        except Exception:
            return str(value)

    def to_str(value):
        if pd.isna(value):
            return ""
        return str(value)

    def to_float(value):
        if pd.isna(value):
            return ""
        try:
            return round(float(value), 3)
        except Exception:
            return value

    rows = []
    if not df.empty:
        for idx, (_, row) in enumerate(df.iterrows(), start=1):
            charge_id = to_str(row.get("ID", ""))
            rows.append(
                {
                    "rank": idx,
                    "id": charge_id,
                    "url": f"{BASE_CHARGE_URL}{charge_id}" if charge_id else "",
                    "site": row.get("Site", ""),
                    "pdc": to_str(row.get("PDC", "")),
                    "mac": row.get("MAC Address", ""),
                    "vehicle": row.get("Vehicle", ""),
                    "start": format_ts(row.get("Datetime start")),
                    "end": format_ts(row.get("Datetime end")),
                    "energy": to_float(row.get("Energy (Kwh)")),
                    "soc_start": to_float(row.get("SOC Start")),
                    "soc_end": to_float(row.get("SOC End")),
                    "erreur": to_str(row.get("type_erreur", "")),
                    "moment": to_str(row.get("moment", "")),
                    "is_warning": 1 if pd.notna(row.get("warning")) and int(row.get("warning", 0)) == 1 else 0,
                    "duration": to_float(row.get("duration")),
                }
            )

    return templates.TemplateResponse(
        "partials/suspicious.html",
        {
            "request": request,
            "rows": rows,
        },
    )


@router.get("/kpi/multi-attempts")
async def get_multi_attempts(
    request: Request,
    sites: str = Query(default=""),
    date_debut: date = Query(default=None),
    date_fin: date = Query(default=None),
):
    sql = "SELECT * FROM kpi_multi_attempts_hour"
    df = query_df(sql)

    if not df.empty and "Date_heure" in df.columns:
        df["Date_heure"] = pd.to_datetime(df["Date_heure"], errors="coerce")

        if date_debut:
            df = df[df["Date_heure"] >= pd.Timestamp(date_debut)]
        if date_fin:
            df = df[df["Date_heure"] < pd.Timestamp(date_fin) + pd.Timedelta(days=1)]

        if sites and "Site" in df.columns:
            site_list = [s.strip() for s in sites.split(",") if s.strip()]
            if site_list:
                df = df[df["Site"].isin(site_list)]

    if not df.empty:
        sort_cols = []
        sort_order = []
        if "Date_heure" in df.columns:
            sort_cols.append("Date_heure")
            sort_order.append(True)
        if "Site" in df.columns:
            sort_cols.append("Site")
            sort_order.append(True)
        if "tentatives" in df.columns:
            sort_cols.append("tentatives")
            sort_order.append(False)
        if sort_cols:
            df = df.sort_values(sort_cols, ascending=sort_order)

    soc_columns = [
        col
        for col in ["SOC start min", "SOC start max", "SOC end min", "SOC end max"]
        if col in df.columns
    ]

    def format_ts(value):
        if pd.isna(value):
            return ""
        try:
            ts = pd.to_datetime(value, errors="coerce")
            return ts.strftime("%Y-%m-%d %H:%M") if not pd.isna(ts) else ""
        except Exception:
            return str(value)

    def parse_ids(value):
        if not isinstance(value, str):
            value = "" if pd.isna(value) else str(value)
        ids = [v.strip() for v in value.split(",") if v.strip()]
        return [{"id": iid, "url": f"{BASE_CHARGE_URL}{iid}"} for iid in ids]

    table_rows = []
    if not df.empty:
        for idx, (_, row) in enumerate(df.iterrows(), start=1):
            date_value = row.get("Date_heure")
            hour_value = row.get("Heure")
            if (pd.isna(hour_value) or hour_value == "") and not pd.isna(date_value):
                ts_hour = pd.to_datetime(date_value, errors="coerce")
                hour_value = ts_hour.strftime("%Y-%m-%d %H:%M") if not pd.isna(ts_hour) else ""

            tentatives_val = pd.to_numeric(row.get("tentatives", 0), errors="coerce")
            tentatives = int(tentatives_val) if pd.notna(tentatives_val) else 0

            table_rows.append(
                {
                    "rank": idx,
                    "site": row.get("Site", ""),
                    "hour": hour_value or "",
                    "mac": row.get("MAC", ""),
                    "vehicle": row.get("Vehicle", ""),
                    "tentatives": tentatives,
                    "pdc": row.get("PDC(s)", ""),
                    "first_attempt": format_ts(row.get("1ère tentative")),
                    "last_attempt": format_ts(row.get("Dernière tentative")),
                    "ids": parse_ids(row.get("ID(s)")),
                    "soc_values": {col: row.get(col, "") for col in soc_columns},
                }
            )

    return templates.TemplateResponse(
        "partials/multi_attempts.html",
        {
            "request": request,
            "rows": table_rows,
            "soc_columns": soc_columns,
        },
    )


@router.get("/kpi/evolution")
async def get_kpi_evolution(
    request: Request,
    sites: str = Query(default=""),
    date_debut: date = Query(default=None),
    date_fin: date = Query(default=None),
    error_types: str = Query(default=""),
    moments: str = Query(default=""),
):

    table_name = "kpi_evo"
    if not table_exists(table_name):
        return templates.TemplateResponse(
            "partials/evolution.html",
            {
                "request": request,
                "error_message": "La table `kpi_evo` est introuvable dans la base.",
            },
        )

    try:
        df = query_df(f"SELECT * FROM {table_name}")
    except Exception as exc: 
        return templates.TemplateResponse(
            "partials/evolution.html",
            {
                "request": request,
                "error_message": f"Impossible de charger les données : {exc}",
            },
        )

    if df.empty:
        return templates.TemplateResponse(
            "partials/evolution.html",
            {
                "request": request,
                "no_data": True,
            },
        )

    month = ["mois"]
    rate = ["tr"]

    month_col = next((col for col in month if col in df.columns), None)
    rate_col = next((col for col in rate  if col in df.columns), None)

    if not month_col or not rate_col:
        return templates.TemplateResponse(
            "partials/evolution.html",
            {
                "request": request,
                "error_message": "Colonnes `mois` et `taux de réussite` manquantes dans `kpi_evo`.",
            },
        )

    df[month_col] = pd.to_datetime(df[month_col], errors="coerce")
    df = df.dropna(subset=[month_col])
    df = df.sort_values(month_col)
    df["mois_affiche"] = df[month_col].dt.strftime("%Y-%m")

    df["taux_val"] = pd.to_numeric(df[rate_col], errors="coerce")
    df = df.dropna(subset=["taux_val"])

    if df.empty:
        return templates.TemplateResponse(
            "partials/evolution.html",
            {
                "request": request,
                "no_data": True,
            },
        )

    taux_series = df["taux_val"]
    if taux_series.between(0, 1).all():
        df["taux_pct"] = taux_series * 100
    else:
        df["taux_pct"] = taux_series

    df["taux_pct"] = df["taux_pct"].round(1)

    rows = df[["mois_affiche", "taux_pct"]].to_dict("records")
    chart_data = {
        "months": [r["mois_affiche"] for r in rows],
        "values": [r["taux_pct"] for r in rows],
    }

    latest_rate = rows[-1]["taux_pct"] if rows else None
    previous_rate = rows[-2]["taux_pct"] if len(rows) > 1 else None
    variation = latest_rate - previous_rate if latest_rate is not None and previous_rate is not None else None

    return templates.TemplateResponse(
        "partials/evolution.html",
        {
            "request": request,
            "rows": rows,
            "chart_data": chart_data,
            "latest_rate": latest_rate,
            "variation": variation,
            "previous_rate": previous_rate,
        },
    )