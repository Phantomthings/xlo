from fastapi import APIRouter, Request, Query
from datetime import date
import pandas as pd
import numpy as np

from .common import (
    _get_vehicle_strategy,
    _build_conditions,
    _apply_status_filters,
    templates,
)
from db import query_df

router = APIRouter(tags=["sessions"])


@router.get("/sessions/stats")
async def get_sessions_stats(
    request: Request,
    sites: str = Query(default=""),
    date_debut: date = Query(default=None),
    date_fin: date = Query(default=None),
    error_types: str | None = Query(default=None),
    moments: str | None = Query(default=None),
):
    error_type_list = None if error_types is None else [e.strip() for e in error_types.split(",") if e.strip()]
    moment_list = None if moments is None else [m.strip() for m in moments.split(",") if m.strip()]

    where_clause, params = _build_conditions(sites, date_debut, date_fin, table_alias="k")

    vehicle_select, join_clause = _get_vehicle_strategy()

    sql = f"""
        SELECT
            k.ID,
            k.Site,
            k.PDC,
            k.`Datetime start`,
            k.`Datetime end`,
            k.`Energy (Kwh)`,
            k.`Mean Power (Kw)`,
            k.`Max Power (Kw)`,
            k.`SOC Start`,
            k.`SOC End`,
            k.`MAC Address`,
            k.`State of charge(0:good, 1:error)` as state,
            k.type_erreur,
            k.moment,
            {vehicle_select}
        FROM kpi_sessions k
        {join_clause}
        WHERE {where_clause}
    """

    df = query_df(sql, params)

    if df.empty:
        return templates.TemplateResponse(
            "partials/sessions_stats.html",
            {
                "request": request,
                "no_data": True,
            }
        )

    for col in ["Datetime start", "Datetime end"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    for col in ["Energy (Kwh)", "Mean Power (Kw)", "Max Power (Kw)", "SOC Start", "SOC End"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["is_ok_raw"] = pd.to_numeric(df["state"], errors="coerce").fillna(0).astype(int).eq(0)

    df = _apply_status_filters(df, error_type_list, moment_list)

    ok_mask = df["is_ok_filt"]
    nok_mask = ~df["is_ok_filt"]

    ok_df = df[ok_mask].copy()
    nok_df = df[nok_mask].copy()

    energy_all = pd.to_numeric(df.get("Energy (Kwh)", pd.Series(dtype=float)), errors="coerce")
    e_total_all = round(float(energy_all.sum(skipna=True)), 3) if energy_all.notna().any() else 0

    energy_ok = pd.to_numeric(ok_df.get("Energy (Kwh)", pd.Series(dtype=float)), errors="coerce")
    e_mean = round(float(energy_ok.mean(skipna=True)), 3) if energy_ok.notna().any() else 0
    e_max = round(float(energy_ok.max(skipna=True)), 3) if energy_ok.notna().any() else 0

    pmean_ok = pd.to_numeric(ok_df.get("Mean Power (Kw)", pd.Series(dtype=float)), errors="coerce")
    pm_mean = round(float(pmean_ok.mean(skipna=True)), 3) if pmean_ok.notna().any() else 0
    pm_max = round(float(pmean_ok.max(skipna=True)), 3) if pmean_ok.notna().any() else 0

    pmax_ok = pd.to_numeric(ok_df.get("Max Power (Kw)", pd.Series(dtype=float)), errors="coerce")
    px_mean = round(float(pmax_ok.mean(skipna=True)), 3) if pmax_ok.notna().any() else 0
    px_max = round(float(pmax_ok.max(skipna=True)), 3) if pmax_ok.notna().any() else 0

    soc_start = pd.to_numeric(ok_df.get("SOC Start", pd.Series(dtype=float)), errors="coerce")
    soc_end = pd.to_numeric(ok_df.get("SOC End", pd.Series(dtype=float)), errors="coerce")
    soc_start_mean = round(float(soc_start.mean(skipna=True)), 2) if soc_start.notna().any() else 0
    soc_end_mean = round(float(soc_end.mean(skipna=True)), 2) if soc_end.notna().any() else 0

    if soc_start.notna().any() and soc_end.notna().any():
        soc_gain_mean = round(float((soc_end - soc_start).mean(skipna=True)), 2)
    else:
        soc_gain_mean = 0

    dt_start = pd.to_datetime(ok_df.get("Datetime start"), errors="coerce")
    dt_end = pd.to_datetime(ok_df.get("Datetime end"), errors="coerce")
    durations = (dt_end - dt_start).dt.total_seconds() / 60
    dur_mean = round(float(durations.mean(skipna=True)), 1) if durations.notna().any() else 0

    if not ok_df.empty:
        ok_df["day"] = pd.to_datetime(ok_df["Datetime start"]).dt.date
        charges_by_site_day = (
            ok_df.groupby(["Site", "day"])
            .size()
            .reset_index(name="Nb")
        )

        daily_stats = charges_by_site_day.groupby("day")["Nb"].sum().reset_index()
        nb_days = len(daily_stats)
        mean_day = round(float(daily_stats["Nb"].mean()), 2) if nb_days else 0
        med_day = round(float(daily_stats["Nb"].median()), 2) if nb_days else 0

        if not charges_by_site_day.empty:
            max_row = charges_by_site_day.loc[charges_by_site_day["Nb"].idxmax()]
            max_day_site = str(max_row["Site"])
            max_day_date = str(max_row["day"])
            max_day_nb = int(max_row["Nb"])
        else:
            max_day_site = "—"
            max_day_date = "—"
            max_day_nb = 0
    else:
        nb_days = 0
        mean_day = 0
        med_day = 0
        max_day_site = "—"
        max_day_date = "—"
        max_day_nb = 0

    dur_source = ok_df.copy()

    if moment_list is not None and "moment" in dur_source.columns:
        dur_source = dur_source[dur_source["moment"].isin(moment_list)]

    if not dur_source.empty and "Datetime start" in dur_source.columns and "Datetime end" in dur_source.columns:
        dur_df = dur_source[["Site", "PDC", "Datetime start", "Datetime end"]].copy()
        dur_df = dur_df.dropna(subset=["Datetime start", "Datetime end"])
        dur_df["dur_min"] = (
            pd.to_datetime(dur_df["Datetime end"]) - pd.to_datetime(dur_df["Datetime start"])
        ).dt.total_seconds() / 60

        by_site_dur = (
            dur_df.groupby("Site")["dur_min"]
            .sum()
            .reset_index()
            .assign(Heures=lambda d: (d["dur_min"] / 60).round(1))
            .sort_values("Heures", ascending=False)
        )
        durations_by_site = by_site_dur[["Site", "Heures"]].to_dict("records")

        by_pdc_dur = (
            dur_df.groupby(["Site", "PDC"])["dur_min"]
            .sum()
            .reset_index()
            .assign(Heures=lambda d: (d["dur_min"] / 60).round(1))
        )
        durations_by_pdc_raw = by_pdc_dur.to_dict("records")
    else:
        durations_by_site = []
        durations_by_pdc_raw = []

    durations_by_site_dict = {}
    for row in durations_by_pdc_raw:
        site = row["Site"]
        if site not in durations_by_site_dict:
            durations_by_site_dict[site] = []
        durations_by_site_dict[site].append({
            "PDC": row["PDC"],
            "Heures": row["Heures"]
        })

    for site in durations_by_site_dict:
        durations_by_site_dict[site] = sorted(
            durations_by_site_dict[site],
            key=lambda x: x["Heures"],
            reverse=True
        )

    site_options_order = [row["Site"] for row in durations_by_site]

    vehicle_stats = []
    vehicle_debug_info = {
        "has_column": "Vehicle" in df.columns,
        "total_rows": len(df),
        "non_null_count": 0,
        "valid_count": 0,
        "unknown_count": 0,
    }

    if "Vehicle" in df.columns and not df.empty:
        df_vehicle = df.copy()
        df_vehicle["Vehicle"] = df_vehicle["Vehicle"].astype(str).str.strip()

        vehicle_debug_info["non_null_count"] = int(df_vehicle["Vehicle"].notna().sum())

        df_vehicle["Vehicle"] = df_vehicle["Vehicle"].replace(
            {"": "Unknown", "nan": "Unknown", "none": "Unknown", "NULL": "Unknown", "None": "Unknown"},
            regex=False
        )
        df_vehicle["Vehicle"] = df_vehicle["Vehicle"].fillna("Unknown")

        vehicle_debug_info["unknown_count"] = int((df_vehicle["Vehicle"] == "Unknown").sum())

        df_vehicle = df_vehicle[df_vehicle["Vehicle"] != "Unknown"]
        vehicle_debug_info["valid_count"] = len(df_vehicle)

        if not df_vehicle.empty:
            vehicle_grouped = (
                df_vehicle.groupby("Vehicle", dropna=False)["is_ok_filt"]
                .agg(total="size", ok="sum")
                .reset_index()
            )
            vehicle_grouped["nok"] = vehicle_grouped["total"] - vehicle_grouped["ok"]
            vehicle_grouped["percent_ok"] = np.where(
                vehicle_grouped["total"] > 0,
                (vehicle_grouped["ok"] / vehicle_grouped["total"] * 100).round(2),
                0.0
            )
            vehicle_grouped["percent_nok"] = 100 - vehicle_grouped["percent_ok"]

            vehicle_grouped = vehicle_grouped.sort_values(
                ["percent_ok", "total"],
                ascending=[False, False]
            ).reset_index(drop=True)

            vehicle_stats = vehicle_grouped.to_dict("records")

    return templates.TemplateResponse(
        "partials/sessions_stats.html",
        {
            "request": request,
            "no_data": False,
            "total_charges": len(df),
            "total_ok": len(ok_df),
            "total_nok": len(nok_df),
            "e_total_all": e_total_all,
            "e_mean": e_mean,
            "e_max": e_max,
            "pm_mean": pm_mean,
            "pm_max": pm_max,
            "px_mean": px_mean,
            "px_max": px_max,
            "soc_start_mean": soc_start_mean,
            "soc_end_mean": soc_end_mean,
            "soc_gain_mean": soc_gain_mean,
            "dur_mean": dur_mean,
            "nb_days": nb_days,
            "mean_day": mean_day,
            "med_day": med_day,
            "max_day_site": max_day_site,
            "max_day_date": max_day_date,
            "max_day_nb": max_day_nb,
            "durations_by_site": durations_by_site,
            "durations_by_site_dict": durations_by_site_dict,
            "site_options_dur": site_options_order if site_options_order else list(durations_by_site_dict.keys()),
            "vehicle_stats": vehicle_stats,
            "vehicle_debug_info": vehicle_debug_info,
        }
    )
