from fastapi import APIRouter, Request, Query
from datetime import date
import pandas as pd
import numpy as np

from db import query_df, LOGV0_BASE_URL
from routers.filters import MOMENT_ORDER
from .common import _calculate_period_dates, _build_conditions, _get_vehicle_strategy, _apply_super_filters, templates

router = APIRouter(tags=["sessions"])


@router.get("/sessions/tableau-charges")
async def get_tableau_charges(
    request: Request,
    sites: str = Query(default=""),
    period_type: str = Query(default="semaine-1"),
    date_debut: date = Query(default=None),
    date_fin: date = Query(default=None),
    error_types: str | None = Query(default=None),
    moments: str | None = Query(default=None),
    moment_avancee: str = Query(default=""),
    charge_type: str = Query(default="Toutes"),
    energy_max: str = Query(default=""),
    energy_operator: str = Query(default="<="),
    vehicle_type: str = Query(default="Tous"),
    voltage_800v: str = Query(default="Tous"),
    pdc_filter: str = Query(default=""),
    mac_address_filter: str = Query(default=""),
    pmax_min: str = Query(default=""),
    pmax_operator: str = Query(default=">="),
    duration: str = Query(default=""),
    duration_operator: str = Query(default=">="),
    error_moment: str = Query(default=""),
    visible_columns: str = Query(default=""),
):
    calculated_date_debut, calculated_date_fin = _calculate_period_dates(
        period_type, date_debut, date_fin
    )

    final_date_debut = calculated_date_debut if calculated_date_debut else date_debut
    final_date_fin = calculated_date_fin if calculated_date_fin else date_fin

    error_type_list = None if error_types is None else [e.strip() for e in error_types.split(",") if e.strip()]
    moment_list = None if moments is None else [m.strip() for m in moments.split(",") if m.strip()]
    moment_avancee_list = [m.strip() for m in moment_avancee.split(",") if m.strip()] if moment_avancee else []
    site_list = [s.strip() for s in sites.split(",") if s.strip()] if sites else []

    query_params = dict(request.query_params)
    energy_operator_from_query = query_params.get('energy_operator', None)

    if energy_operator_from_query:
        energy_operator = energy_operator_from_query
    elif not energy_operator or not energy_operator.strip():
        energy_operator = "<="

    if energy_operator and energy_operator.strip():
        from urllib.parse import unquote
        if "%3E" in energy_operator or "%3C" in energy_operator:
            energy_operator = unquote(energy_operator)
        energy_operator = energy_operator.strip()
        if energy_operator not in (">=", "<="):
            energy_operator = "<="
    else:
        energy_operator = "<="

    where_clause, params = _build_conditions(
        sites, final_date_debut, final_date_fin,
        error_types=error_type_list if error_type_list else None,
        moments=moment_list if moment_list else None
    )

    if moment_avancee_list:
        moment_avancee_placeholders = ",".join([f":moment_avancee_{i}" for i in range(len(moment_avancee_list))])
        where_clause += f" AND moment_avancee IN ({moment_avancee_placeholders})"
        for i, ma in enumerate(moment_avancee_list):
            params[f"moment_avancee_{i}"] = ma

    sites_sql = "SELECT DISTINCT Site FROM kpi_sessions WHERE Site IS NOT NULL ORDER BY Site"
    sites_df = query_df(sites_sql)
    site_options = sites_df["Site"].tolist() if not sites_df.empty else []

    error_types_sql = "SELECT DISTINCT type_erreur FROM kpi_sessions WHERE type_erreur IS NOT NULL ORDER BY type_erreur"
    error_types_df = query_df(error_types_sql)
    error_type_options = sorted(error_types_df["type_erreur"].dropna().unique().tolist()) if not error_types_df.empty else []

    moments_sql = "SELECT DISTINCT moment FROM kpi_sessions WHERE moment IS NOT NULL"
    moments_df = query_df(moments_sql)
    raw_moments = moments_df["moment"].dropna().unique().tolist() if not moments_df.empty else []
    moment_options = [m for m in MOMENT_ORDER if m in raw_moments]
    moment_options.extend([m for m in raw_moments if m not in MOMENT_ORDER])

    moment_avancee_sql = "SELECT DISTINCT moment_avancee FROM kpi_sessions WHERE moment_avancee IS NOT NULL ORDER BY moment_avancee"
    moment_avancee_df = query_df(moment_avancee_sql)
    moment_avancee_options = sorted(moment_avancee_df["moment_avancee"].dropna().unique().tolist()) if not moment_avancee_df.empty else []

    sql = f"""
        SELECT
            ID,
            Site,
            PDC,
            `Datetime start`,
            `Datetime end`,
            `Energy (Kwh)`,
            `Mean Power (Kw)`,
            `Max Power (Kw)`,
            `SOC Start`,
            `SOC End`,
            Vehicle,
            `State of charge(0:good, 1:error)` as state,
            CASE WHEN `State of charge(0:good, 1:error)` = 0 THEN 1 ELSE 0 END as is_ok,
            warning,
            type_erreur,
            moment,
            moment_avancee,
            `Downstream Code PC`,
            `EVI Error Code`,
            `EVI Status during error`,
            `MAC Address`,
            charge_900V,
            duration
        FROM kpi_sessions
        WHERE {where_clause}
        ORDER BY `Datetime start` DESC
        LIMIT 10000
    """

    df = query_df(sql, params)

    if df.empty:
        return templates.TemplateResponse(
            "partials/tableau_charges.html",
            {
                "request": request,
                "rows": [],
                "total": 0,
                "ok": 0,
                "nok": 0,
                "taux_reussite": 0,
                "taux_echec": 0,
                "super_filters": {
                    "sites": sites,
                    "period_type": period_type,
                    "date_debut": str(final_date_debut) if final_date_debut else "",
                    "date_fin": str(final_date_fin) if final_date_fin else "",
                    "error_types": error_types,
                    "moments": moments,
                    "moment_avancee": moment_avancee,
                    "charge_type": charge_type,
                    "energy_max": energy_max,
                    "energy_operator": energy_operator if energy_operator else "<=",
                    "vehicle_type": vehicle_type,
                    "voltage_800v": voltage_800v,
                    "pdc_filter": pdc_filter,
                    "mac_address_filter": mac_address_filter,
                    "pmax_min": pmax_min,
                    "pmax_operator": pmax_operator,
                    "duration": duration,
                    "duration_operator": duration_operator,
                    "error_moment": error_moment,
                },
                "site_options": site_options,
                "error_type_options": error_type_options,
                "moment_options": moment_options,
                "moment_avancee_options": moment_avancee_options,
                "vehicle_options": ["Tous"],
                "visible_columns": ["Site", "PDC", "Datetime start", "Datetime end", "Energy (Kwh)", "SOC Start", "SOC End", "Vehicle", "is_ok", "Lien", "Actions", "Erreur"],
                "available_columns": {
                    "Site": "Site",
                    "PDC": "PDC",
                    "Datetime start": "Datetime start",
                    "Datetime end": "Datetime end",
                    "Energy (Kwh)": "Energy (Kwh)",
                    "Mean Power (Kw)": "Mean Power (Kw)",
                    "Max Power (Kw)": "Max Power (Kw)",
                    "SOC Start": "SOC Start",
                    "SOC End": "SOC End",
                    "Vehicle": "Vehicle",
                    "is_ok": "is_ok",
                    "type_erreur": "Type erreur",
                    "moment": "Moment",
                    "moment_avancee": "Moment avancé",
                    "evi_error_code": "EVI Error Code",
                    "downstream_code_pc": "Downstream Code PC",
                    "evi_status_during_error": "EVI Status during error",
                    "mac_address": "MAC Address",
                    "charge_900V": "Charge 900V",
                    "duration": "Durée",
                    "Lien": "Lien",
                    "Actions": "Actions",
                    "Erreur": "Erreur"
                },
            },
        )

    if "Energy (Kwh)" in df.columns:
        df["Energy (Kwh)"] = pd.to_numeric(df["Energy (Kwh)"], errors="coerce")
    if "Max Power (Kw)" in df.columns:
        df["Max Power (Kw)"] = pd.to_numeric(df["Max Power (Kw)"], errors="coerce")

    df["is_ok"] = pd.to_numeric(df["state"], errors="coerce").fillna(0).astype(int).eq(0)
    df["is_warning"] = pd.to_numeric(df.get("warning", 0), errors="coerce").fillna(0).astype(int).eq(1)
    df["is_ok"] = df["is_ok"] | df["is_warning"]

    df = _apply_super_filters(
        df,
        charge_type=charge_type,
        energy_max=energy_max,
        energy_operator=energy_operator,
        vehicle_type=vehicle_type,
        voltage_800v=voltage_800v,
        pdc_filter=pdc_filter,
        mac_address_filter=mac_address_filter,
        pmax_min=pmax_min,
        pmax_operator=pmax_operator,
        duration=duration,
        duration_operator=duration_operator,
        error_moment=error_moment,
    )

    rows = []
    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        row_dict = {
            "rank": idx,
            "id": str(row.get("ID", "")) if pd.notna(row.get("ID")) else "",
            "Site": str(row.get("Site", "")) if pd.notna(row.get("Site")) else "",
            "PDC": str(row.get("PDC", "")) if pd.notna(row.get("PDC")) else "",
            "Datetime start": str(row.get("Datetime start", "")) if pd.notna(row.get("Datetime start")) else "",
            "Datetime end": str(row.get("Datetime end", "")) if pd.notna(row.get("Datetime end")) else "",
            "Energy (Kwh)": round(float(row.get("Energy (Kwh)", 0)), 1) if pd.notna(row.get("Energy (Kwh)")) else 0.0,
            "Mean Power (Kw)": round(float(row.get("Mean Power (Kw)", 0)), 2) if pd.notna(row.get("Mean Power (Kw)")) else 0.0,
            "Max Power (Kw)": float(row.get("Max Power (Kw)", 0)) if pd.notna(row.get("Max Power (Kw)")) else 0.0,
            "SOC Start": float(row.get("SOC Start", 0)) if pd.notna(row.get("SOC Start")) else 0.0,
            "SOC End": float(row.get("SOC End", 0)) if pd.notna(row.get("SOC End")) else 0.0,
            "Vehicle": str(row.get("Vehicle", "")) if pd.notna(row.get("Vehicle")) else "",
            "is_ok": int(row.get("is_ok", 1)) if pd.notna(row.get("is_ok")) else 1,
            "is_warning": int(row.get("is_warning", 0)) if pd.notna(row.get("is_warning")) else 0,
            "statut": "✅ OK" if (int(row.get("is_ok", 1)) == 1) else "🔴 NOK",
            "type_erreur": str(row.get("type_erreur", "")) if pd.notna(row.get("type_erreur")) else "",
            "moment": str(row.get("moment", "")) if pd.notna(row.get("moment")) else "",
            "moment_avancee": str(row.get("moment_avancee", "")) if pd.notna(row.get("moment_avancee")) else "",
            "downstream_code_pc": str(row.get("Downstream Code PC", "")) if pd.notna(row.get("Downstream Code PC")) else "",
            "evi_error_code": str(row.get("EVI Error Code", "")) if pd.notna(row.get("EVI Error Code")) else "",
            "evi_status_during_error": str(row.get("EVI Status during error", "")) if pd.notna(row.get("EVI Status during error")) else "",
            "mac_address": str(row.get("MAC Address", "")) if pd.notna(row.get("MAC Address")) else "",
            "charge_900V": str(row.get("charge_900V", "")) if pd.notna(row.get("charge_900V")) else "",
            "duration": str(row.get("duration")) if pd.notna(row.get("duration")) else None,
            "url": f"https://elto.nidec-asi-online.com/Charge/detail?id={row.get('ID', '')}" if pd.notna(row.get("ID")) else "",
            "logv0_url": f"{LOGV0_BASE_URL}?session_id={row.get('ID', '')}&project=ELTO" if LOGV0_BASE_URL and pd.notna(row.get("ID")) else None,
        }
        rows.append(row_dict)

    total = len(rows)
    ok = sum(1 for r in rows if r.get("is_ok", 1) == 1)
    nok = total - ok
    taux_reussite = round(ok / total * 100, 1) if total else 0
    taux_echec = round(nok / total * 100, 1) if total else 0

    vehicle_options = ["Tous"]
    if "Vehicle" in df.columns:
        vehicles = sorted(df["Vehicle"].dropna().unique().tolist())
        vehicle_options.extend(vehicles)

    default_columns = [
        "Site", "PDC", "Datetime start", "Datetime end", "Energy (Kwh)",
        "SOC Start", "SOC End", "Vehicle", "is_ok", "Lien", "Actions", "Erreur"
    ]

    available_columns = {
        "Site": "Site",
        "PDC": "PDC",
        "Datetime start": "Datetime start",
        "Datetime end": "Datetime end",
        "Energy (Kwh)": "Energy (Kwh)",
        "Mean Power (Kw)": "Mean Power (Kw)",
        "Max Power (Kw)": "Max Power (Kw)",
        "SOC Start": "SOC Start",
        "SOC End": "SOC End",
        "Vehicle": "Vehicle",
        "is_ok": "is_ok",
        "type_erreur": "Type erreur",
        "moment": "Moment",
        "moment_avancee": "Moment avancé",
        "evi_error_code": "EVI Error Code",
        "downstream_code_pc": "Downstream Code PC",
        "evi_status_during_error": "EVI Status during error",
        "mac_address": "MAC Address",
        "charge_900V": "Charge 900V",
        "Lien": "Lien",
        "Actions": "Actions",
        "Erreur": "Erreur"
    }

    if visible_columns and visible_columns.strip():
        visible_columns_list = [col.strip() for col in visible_columns.split(",") if col.strip()]
    else:
        visible_columns_list = default_columns.copy()

    essential_columns = ["Site", "PDC", "Datetime start", "is_ok", "Lien", "Actions", "Erreur"]
    for col in essential_columns:
        if col not in visible_columns_list:
            visible_columns_list.append(col)

    return templates.TemplateResponse(
        "partials/tableau_charges.html",
        {
            "request": request,
            "rows": rows,
            "total": total,
            "ok": ok,
            "nok": nok,
            "taux_reussite": taux_reussite,
            "taux_echec": taux_echec,
            "super_filters": {
                "sites": sites,
                "period_type": period_type,
                "date_debut": str(final_date_debut) if final_date_debut else "",
                "date_fin": str(final_date_fin) if final_date_fin else "",
                "error_types": error_types,
                "moments": moments,
                "moment_avancee": moment_avancee,
                "charge_type": charge_type,
                "energy_max": energy_max,
                "energy_operator": energy_operator if energy_operator else "<=",
                "vehicle_type": vehicle_type,
                "voltage_800v": voltage_800v,
                "pdc_filter": pdc_filter,
                "mac_address_filter": mac_address_filter,
                "pmax_min": pmax_min,
                "pmax_operator": pmax_operator,
                "duration": duration,
                "duration_operator": duration_operator,
                "error_moment": error_moment,
            },
            "site_options": site_options,
            "error_type_options": error_type_options,
            "moment_options": moment_options,
            "moment_avancee_options": moment_avancee_options,
            "vehicle_options": vehicle_options,
            "visible_columns": visible_columns_list,
            "available_columns": available_columns,
        },
    )
