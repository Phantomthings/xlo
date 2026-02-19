from fastapi import APIRouter, Form, Query, Request
from fastapi.templating import Jinja2Templates
from datetime import date
import pandas as pd
import numpy as np
import re

from db import query_df, table_exists

router = APIRouter(tags=["mac_address"])
templates = Jinja2Templates(directory="templates")

BASE_CHARGE_URL = "https://elto.nidec-asi-online.com/Charge/detail?id="


def _fmt_mac(mac: str) -> str:
    if pd.isna(mac) or not mac:
        return ""
    s = str(mac).strip().lower().replace("0x", "")
    s = re.sub(r"[^0-9a-f]", "", s)
    if len(s) >= 12:
        return ":".join([s[i:i+2] for i in range(0, 12, 2)]).upper()
    return mac.upper()


def _format_soc_evolution(s0, s1):
    if pd.notna(s0) and pd.notna(s1):
        try:
            return f"{int(round(s0))}% → {int(round(s1))}%"
        except Exception:
            return ""
    return ""


def _build_conditions(
    sites: str,
    date_debut: date | None,
    date_fin: date | None,
    table_alias: str = "",
    *,
    error_alias: str | None = None,
    error_types: str = "",
    moments: str = "",
):
    conditions = ["1=1"]
    params: dict[str, str] = {}
    prefix = f"{table_alias}." if table_alias else ""
    error_prefix = f"{error_alias or table_alias}." if (error_alias or table_alias) else ""

    def _parse_filter_values(values: str) -> list[str]:
        return [v.strip() for v in values.split(",") if v.strip()]

    if date_debut:
        conditions.append(f"{prefix}`Datetime start` >= :date_debut")
        params["date_debut"] = str(date_debut)
    if date_fin:
        conditions.append(f"{prefix}`Datetime start` < DATE_ADD(:date_fin, INTERVAL 1 DAY)")
        params["date_fin"] = str(date_fin)
    if sites:
        site_list = [s.strip() for s in sites.split(",") if s.strip()]
        if site_list:
            placeholders = ",".join([f":site_{i}" for i in range(len(site_list))])
            conditions.append(f"{prefix}Site IN ({placeholders})")
            for i, s in enumerate(site_list):
                params[f"site_{i}"] = s

    if error_types:
        type_list = _parse_filter_values(error_types)
        if type_list and error_prefix:
            placeholders = ",".join([f":type_{i}" for i in range(len(type_list))])
            conditions.append(f"{error_prefix}`type_erreur` IN ({placeholders})")
            params.update({f"type_{i}": t for i, t in enumerate(type_list)})

    if moments:
        moment_list = _parse_filter_values(moments)
        if moment_list and error_prefix:
            placeholders = ",".join([f":moment_{i}" for i in range(len(moment_list))])
            conditions.append(f"{error_prefix}`moment` IN ({placeholders})")
            params.update({f"moment_{i}": m for i, m in enumerate(moment_list)})

    return " AND ".join(conditions), params


@router.get("/mac-address/search")
async def search_mac(
    request: Request,
    sites: str = Query(default=""),
    date_debut: date = Query(default=None),
    date_fin: date = Query(default=None),
    error_types: str = Query(default=""),
    moments: str = Query(default=""),
    mac_query: str = Query(default=""),
):
    if not mac_query or len(mac_query.strip()) < 2:
        return templates.TemplateResponse(
            "partials/mac_search.html",
            {
                "request": request,
                "prompt": "Saisissez au moins 2 caractères d'une adresse MAC",
                "mac_query": mac_query,
            }
        )

    mac_norm = mac_query.strip().lower().replace("0x", "")
    mac_norm = re.sub(r"[^0-9a-f]", "", mac_norm)

    where_clause, params = _build_conditions(
        sites,
        date_debut,
        date_fin,
        "s",
        error_alias="s",
        error_types=error_types,
        moments=moments,
    )

    sql = f"""
        SELECT
            s.ID,
            s.Site,
            s.PDC,
            s.`Datetime start`,
            s.`Datetime end`,
            s.`Energy (Kwh)`,
            s.`MAC Address` as mac,
            s.Vehicle,
            s.`SOC Start`,
            s.`SOC End`,
            s.`State of charge(0:good, 1:error)` as state,
            s.warning,
            s.type_erreur,
            s.moment,
            s.duration
        FROM kpi_sessions s
        WHERE {where_clause}
    """

    df = query_df(sql, params)

    if df.empty:
        return templates.TemplateResponse(
            "partials/mac_search.html",
            {
                "request": request,
                "no_data": True,
                "mac_query": mac_query,
            }
        )

    df["mac_norm"] = (
        df["mac"].astype(str).str.lower()
        .str.replace("0x", "", regex=False)
        .str.replace(r"[^0-9a-f]", "", regex=True)
    )

    df = df[df["mac_norm"].str.contains(mac_norm, na=False)].copy()

    if df.empty:
        return templates.TemplateResponse(
            "partials/mac_search.html",
            {
                "request": request,
                "no_results": True,
                "mac_query": mac_query,
            }
        )

    for col in ["Datetime start", "Datetime end"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    for col in ["Energy (Kwh)", "SOC Start", "SOC End"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["is_ok"] = pd.to_numeric(df["state"], errors="coerce").fillna(0).astype(int).eq(0)
    df["is_warning"] = pd.to_numeric(df.get("warning", 0), errors="coerce").fillna(0).astype(int).eq(1)
    df["is_ok"] = (df["is_ok"] | df["is_warning"]).astype(bool)
    df["mac_formatted"] = df["mac"].apply(_fmt_mac)
    df["evolution_soc"] = df.apply(
        lambda r: _format_soc_evolution(r.get("SOC Start"), r.get("SOC End")), axis=1
    )
    df["elto_link"] = BASE_CHARGE_URL + df["ID"].astype(str)
    
    df["erreur"] = df.apply(
        lambda r: f"{r['type_erreur']} — {r['moment']}" 
        if pd.notna(r.get("type_erreur")) and pd.notna(r.get("moment")) 
        else (r.get("type_erreur") or ""),
        axis=1
    )

    total = len(df)
    ok_count = int(df["is_ok"].sum())
    nok_count = total - ok_count
    success_rate = round(ok_count / total * 100, 1) if total else 0

    df_ok = df[df["is_ok"]].copy()
    df_nok = df[~df["is_ok"]].copy()

    if "Datetime start" in df_ok.columns and not df_ok.empty:
        df_ok = df_ok.sort_values("Datetime start", ascending=False)
    if "Datetime start" in df_nok.columns and not df_nok.empty:
        df_nok = df_nok.sort_values("Datetime start", ascending=False)

    display_cols = [
        "Site", "PDC", "Datetime start", "Datetime end",
        "evolution_soc", "mac_formatted", "Vehicle", "Energy (Kwh)", "erreur", "elto_link",
        "is_warning", "duration"
    ]
    display_cols = [c for c in display_cols if c in df.columns]

    ok_rows = df_ok[display_cols].to_dict("records") if not df_ok.empty else []
    nok_rows = df_nok[display_cols].to_dict("records") if not df_nok.empty else []

    return templates.TemplateResponse(
        "partials/mac_search.html",
        {
            "request": request,
            "mac_query": mac_query,
            "total": total,
            "ok_count": ok_count,
            "nok_count": nok_count,
            "success_rate": success_rate,
            "ok_rows": ok_rows,
            "nok_rows": nok_rows,
        }
    )


@router.get("/mac-address/top10")
async def get_top10_unidentified(
    request: Request,
    sites: str = Query(default=""),
    date_debut: date = Query(default=None),
    date_fin: date = Query(default=None),
):
    if not table_exists("kpi_mac_id"):
        return templates.TemplateResponse(
            "partials/mac_top10.html",
            {
                "request": request,
                "error": "Table kpi_mac_id non disponible",
            }
        )

    sql = """
        SELECT Mac, nombre_de_charges, taux_reussite
        FROM kpi_mac_id
        ORDER BY nombre_de_charges DESC
        LIMIT 10
    """

    df = query_df(sql)

    if df.empty:
        return templates.TemplateResponse(
            "partials/mac_top10.html",
            {
                "request": request,
                "no_data": True,
            }
        )

    df["Mac"] = df["Mac"].apply(_fmt_mac)
    df.insert(0, "Rang", range(1, len(df) + 1))

    rows = df.to_dict("records")

    return templates.TemplateResponse(
        "partials/mac_top10.html",
        {
            "request": request,
        "rows": rows,
    }
)


@router.get("/mac-address/code-analysis")
async def get_code_analysis_tab(
    request: Request,
    sites: str = Query(default=""),
    date_debut: date = Query(default=None),
    date_fin: date = Query(default=None),
):
    return templates.TemplateResponse(
        "partials/code_analysis.html",
        {
            "request": request,
        },
    )


@router.post("/mac-address/code-analysis/search")
async def search_by_codes(
    request: Request,
    codes: str = Form(...),
    code_type: str = Form(default="Tous"),
    sites: str = Form(default=""),
    date_debut: str | None = Form(default=None),
    date_fin: str | None = Form(default=None),
    error_types: str = Form(default=""),
    moments: str = Form(default=""),
):
    parts = re.split(r"[,\s;]+", codes.strip())
    try:
        code_list = [int(p) for p in parts if p.strip()]
    except Exception:
        code_list = []

    if not code_list:
        return templates.TemplateResponse(
            "partials/code_results.html",
            {"request": request, "error": "Aucun code valide"},
        )

    def _parse_date_field(val: str | None) -> date | None:
        if not val:
            return None
        try:
            return pd.to_datetime(val).date()
        except Exception:
            return None

    date_debut_val = _parse_date_field(date_debut)
    date_fin_val = _parse_date_field(date_fin)

    where_clause, params = _build_conditions(
        sites,
        date_debut_val,
        date_fin_val,
        "s",
        error_types=error_types,
        moments=moments,
    )

    error_scope_clause = where_clause
    error_scope_params = dict(params)

    placeholders = ", ".join([f":code_{i}" for i in range(len(code_list))])
    code_filter = code_type if code_type in {"Erreur_EVI", "Erreur_DownStream"} else "Tous"

    if code_filter == "Erreur_EVI":
        where_clause += f" AND s.`EVI Error Code` IN ({placeholders})"
    elif code_filter == "Erreur_DownStream":
        where_clause += f" AND s.`Downstream Code PC` IN ({placeholders})"
    else:
        where_clause += f" AND (s.`EVI Error Code` IN ({placeholders}) OR s.`Downstream Code PC` IN ({placeholders}))"

    params.update({f"code_{i}": c for i, c in enumerate(code_list)})

    error_condition = "s.`State of charge(0:good, 1:error)` = 1 AND (s.warning IS NULL OR s.warning = 0)"

    sql = f"""
        SELECT
            s.ID,
            s.Site,
            s.PDC,
            s.`Datetime start`,
            s.`Datetime end`,
            s.`Energy (Kwh)`,
            s.`MAC Address`,
            s.Vehicle,
            s.`SOC Start`,
            s.`SOC End`,
            s.type_erreur,
            s.moment,
            s.`EVI Error Code`,
            s.`Downstream Code PC`,
            s.duration
        FROM kpi_sessions s
        WHERE {error_condition} AND {where_clause}
    """

    df = query_df(sql, params)

    total_error_sql = f"""
        SELECT COUNT(*) AS total_errors
        FROM kpi_sessions s
        WHERE {error_condition} AND {error_scope_clause}
    """

    total_errors_df = query_df(total_error_sql, error_scope_params)
    total_error_count = (
        int(total_errors_df["total_errors"].iloc[0]) if not total_errors_df.empty else 0
    )

    if df.empty:
        return templates.TemplateResponse(
            "partials/code_results.html",
            {"request": request, "error": "Aucune charge en erreur trouvée"},
        )

    df["Datetime start"] = pd.to_datetime(df["Datetime start"], errors="coerce")
    df["Datetime end"] = pd.to_datetime(df["Datetime end"], errors="coerce")
    df["MAC Address"] = df["MAC Address"].apply(_fmt_mac)

    df["Évolution SOC"] = df.apply(
        lambda r: _format_soc_evolution(r.get("SOC Start"), r.get("SOC End")), axis=1
    )
    df["Erreur"] = df.apply(
        lambda r: f"{r['type_erreur']} — {r['moment']}"
        if pd.notna(r.get("moment")) and str(r.get("moment")).strip()
        else (r.get("type_erreur") or ""),
        axis=1,
    )

    if "Energy (Kwh)" in df.columns:
        df["Energy (Kwh)"] = pd.to_numeric(df["Energy (Kwh)"], errors="coerce")

    df = df.sort_values("Datetime start", ascending=False)

    occ_site_pdc = (
        df.groupby(["Site", "PDC"])
        .size()
        .reset_index(name="Occurrences")
        .sort_values("Occurrences", ascending=False)
    )

    monthly_hist = []
    if {"Datetime start", "Site"}.issubset(df.columns):
        monthly_df = df.dropna(subset=["Datetime start", "Site"]).copy()
        monthly_df["month"] = monthly_df["Datetime start"].dt.to_period("M").astype(str)

        monthly_counts = (
            monthly_df.groupby(["month", "Site"])
            .size()
            .reset_index(name="Occurrences")
            .sort_values(["month", "Site"])
        )

        if not monthly_counts.empty:
            max_occ = monthly_counts["Occurrences"].max()

            for month, group in monthly_counts.groupby("month"):
                monthly_hist.append(
                    {
                        "month": month,
                        "sites": [
                            {
                                "Site": row["Site"],
                                "Occurrences": int(row["Occurrences"]),
                                "occ_pct": (row["Occurrences"] / max_occ * 100) if max_occ else 0,
                            }
                            for _, row in group.iterrows()
                        ],
                    }
                )

    vehicle_counts = None
    if "Vehicle" in df.columns:
        vehicle_series = df["Vehicle"].astype(str).str.strip()
        vehicle_series = vehicle_series.replace(
            {"": np.nan, "nan": np.nan, "none": np.nan, "NULL": np.nan}, regex=False
        )
        vehicle_df = df.copy()
        vehicle_df["Vehicle"] = vehicle_series
        vehicle_df = vehicle_df[vehicle_df["Vehicle"].notna()]
        vehicle_df = vehicle_df[vehicle_df["Vehicle"].str.len().gt(0)]
        vehicle_df = vehicle_df[vehicle_df["Vehicle"].str.lower() != "unknown"]

        if not vehicle_df.empty:
            vehicle_counts = (
                vehicle_df.groupby("Vehicle")
                .size()
                .reset_index(name="Occurrences")
            )

    occ_vehicle = []
    if vehicle_counts is not None and not vehicle_counts.empty:
        total_where, total_params = _build_conditions(
            sites,
            date_debut_val,
            date_fin_val,
            "cs",
            error_types=error_types,
            moments=moments,
        )
        total_sql = f"""
            SELECT
                cs.Vehicle,
                COUNT(*) AS total_charges
            FROM kpi_sessions cs
            WHERE {total_where}
            GROUP BY cs.Vehicle
        """

        vehicle_totals = query_df(total_sql, total_params)

        if not vehicle_totals.empty:
            vehicle_totals["Vehicle"] = vehicle_totals["Vehicle"].astype(str).str.strip()
            vehicle_totals = vehicle_totals[vehicle_totals["Vehicle"].str.len().gt(0)]
            vehicle_totals = vehicle_totals[vehicle_totals["Vehicle"].str.lower() != "unknown"]

        merged_vehicle = vehicle_counts.merge(
            vehicle_totals, how="left", on="Vehicle"
        ) if vehicle_totals is not None else vehicle_counts

        merged_vehicle["total_charges"] = merged_vehicle["total_charges"].fillna(0).astype(int)
        merged_vehicle["vehicle_label"] = merged_vehicle.apply(
            lambda r: f"{r['Vehicle']} ({r['total_charges']})" if r.get("total_charges") else str(r["Vehicle"]),
            axis=1,
        )
        merged_vehicle = merged_vehicle.sort_values("Occurrences", ascending=True)

        max_occ = merged_vehicle["Occurrences"].max()
        merged_vehicle["occ_pct"] = (
            merged_vehicle["Occurrences"] / max_occ * 100 if max_occ else 0
        )

        occ_vehicle = merged_vehicle.to_dict("records")

    df["month"] = df["Datetime start"].dt.to_period("M").astype(str)
    df["day"] = df["Datetime start"].dt.date.astype(str)
    df["hour"] = df["Datetime start"].dt.hour

    site_options = sorted(df["Site"].dropna().unique().tolist())

    daily_counts = (
        df.groupby(["Site", "month", "day", "PDC"])
        .size()
        .reset_index(name="Occurrences")
    )

    hourly_counts = (
        df.groupby(["Site", "day", "hour", "PDC"])
        .size()
        .reset_index(name="Occurrences")
    )

    charges_rows = df.to_dict("records")

    error_share_pct = round(len(df) / total_error_count * 100, 1) if total_error_count else 0

    return templates.TemplateResponse(
        "partials/code_results.html",
        {
            "request": request,
            "codes_str": ", ".join(str(c) for c in code_list),
            "charges": charges_rows,
            "total_error_count": total_error_count,
            "error_share_pct": error_share_pct,
            "occ_site_pdc": occ_site_pdc.to_dict("records"),
            "occ_vehicle": occ_vehicle,
            "monthly_hist": monthly_hist,
            "base_url": BASE_CHARGE_URL,
            "site_options": site_options,
            "daily_counts": daily_counts.to_dict("records"),
            "hourly_counts": hourly_counts.to_dict("records"),
        },
    )


@router.get("/mac-address")
async def get_mac_address_tab(
    request: Request,
    sites: str = Query(default=""),
    date_debut: date = Query(default=None),
    date_fin: date = Query(default=None),
):
    return templates.TemplateResponse(
        "partials/mac_address.html",
        {
            "request": request,
        }
    )