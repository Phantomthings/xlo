from fastapi import APIRouter, Request, Query
from datetime import date
from urllib.parse import urlencode
import pandas as pd
import numpy as np

from db import query_df, LOGV0_BASE_URL
from routers.filters import MOMENT_ORDER
from .common import (_calculate_period_dates, _build_conditions, _get_vehicle_strategy,
                     _apply_status_filters, _map_moment_label, _map_phase_label, templates)

router = APIRouter(tags=["sessions"])


def _format_soc(s0, s1):
    if pd.notna(s0) and pd.notna(s1):
        try:
            return f"{int(round(s0))}% → {int(round(s1))}%"
        except Exception:
            return ""
    return ""


def _prepare_query_params(request: Request) -> str:
    allowed = {"sites", "date_debut", "date_fin", "error_types", "moments"}
    data = {k: v for k, v in request.query_params.items() if k in allowed and v}
    return urlencode(data)


@router.get("/sessions/site-details")
async def get_sessions_site_details(
    request: Request,
    sites: str = Query(default=""),
    date_debut: date = Query(default=None),
    date_fin: date = Query(default=None),
    error_types: str | None = Query(default=None),
    moments: str | None = Query(default=None),
    site_focus: str = Query(default=""),
    month_focus: str = Query(default=""),
    day_focus: str = Query(default=""),
    pdc: str = Query(default=""),
):
    error_type_list = None if error_types is None else [e.strip() for e in error_types.split(",") if e.strip()]
    moment_list = None if moments is None else [m.strip() for m in moments.split(",") if m.strip()]

    where_clause, params = _build_conditions(sites, date_debut, date_fin)

    sql = f"""
        SELECT
            Site,
            PDC,
            ID,
            `Datetime start`,
            `Datetime end`,
            `Energy (Kwh)`,
            `MAC Address`,
            Vehicle,
            type_erreur,
            moment,
            moment_avancee,
            `SOC Start`,
            `SOC End`,
            `Downstream Code PC`,
            `EVI Error Code`,
            `State of charge(0:good, 1:error)` as state,
            warning,
            duration
        FROM kpi_sessions
        WHERE {where_clause}
    """

    df = query_df(sql, params)

    if df.empty:
        return templates.TemplateResponse(
            "partials/sessions_site_details.html",
            {
                "request": request,
                "site_options": [],
                "base_query": _prepare_query_params(request),
                "site_success_rate": 0.0,
                "site_total_charges": 0,
                "site_charges_ok": 0,
            },
        )

    df["PDC"] = df["PDC"].astype(str)
    df["is_ok"] = pd.to_numeric(df["state"], errors="coerce").fillna(0).astype(int).eq(0)
    df["is_warning"] = pd.to_numeric(df.get("warning", 0), errors="coerce").fillna(0).astype(int).eq(1)
    df["is_ok"] = df["is_ok"] | df["is_warning"]

    mask_type = df["type_erreur"].isin(error_type_list) if error_type_list is not None and "type_erreur" in df.columns else True
    mask_moment = df["moment"].isin(moment_list) if moment_list is not None and "moment" in df.columns else True
    mask_nok = ~df["is_ok"]
    mask_filtered_error = mask_nok & mask_type & mask_moment
    df["is_ok_filt"] = np.where(mask_filtered_error, False, True)

    site_options = sorted(df["Site"].dropna().unique().tolist())
    site_value = site_focus if site_focus in site_options else (site_options[0] if site_options else "")

    df_site = df[df["Site"] == site_value].copy()
    if df_site.empty:
        return templates.TemplateResponse(
            "partials/sessions_site_details.html",
            {
                "request": request,
                "site_options": site_options,
                "site_focus": site_value,
                "pdc_options": [],
                "selected_pdc": [],
                "base_query": _prepare_query_params(request),
                "site_success_rate": 0.0,
                "site_total_charges": 0,
                "site_charges_ok": 0,
            },
        )

    pdc_options = sorted(df_site["PDC"].dropna().unique().tolist())
    selected_pdc = [p.strip() for p in pdc.split(",") if p.strip()] if pdc else pdc_options
    selected_pdc = [p for p in selected_pdc if p in pdc_options] or pdc_options

    df_site = df_site[df_site["PDC"].isin(selected_pdc)].copy()

    mask_type_site = (
        df_site["type_erreur"].isin(error_type_list)
        if error_type_list is not None and "type_erreur" in df_site.columns
        else pd.Series(True, index=df_site.index)
    )
    mask_moment_site = (
        df_site["moment"].isin(moment_list)
        if moment_list is not None and "moment" in df_site.columns
        else pd.Series(True, index=df_site.index)
    )
    df_filtered = df_site[mask_type_site & mask_moment_site].copy()

    for col in ["Datetime start", "Datetime end"]:
        if col in df_filtered.columns:
            df_filtered[col] = pd.to_datetime(df_filtered[col], errors="coerce")
    for col in ["Energy (Kwh)", "SOC Start", "SOC End"]:
        if col in df_filtered.columns:
            df_filtered[col] = pd.to_numeric(df_filtered[col], errors="coerce")

    err_rows = df_filtered[~df_filtered["is_ok"]].copy()
    err_rows["evolution_soc"] = err_rows.apply(lambda r: _format_soc(r.get("SOC Start"), r.get("SOC End")), axis=1)
    err_rows["elto"] = err_rows["ID"].apply(lambda x: f"https://elto.nidec-asi-online.com/Charge/detail?id={str(x).strip()}" if pd.notna(x) else "") if "ID" in err_rows.columns else ""
    err_rows["logv0_url"] = err_rows["ID"].apply(lambda x: f"{LOGV0_BASE_URL}?session_id={str(x).strip()}&project=ELTO" if LOGV0_BASE_URL and pd.notna(x) else None) if "ID" in err_rows.columns else None
    err_display_cols = [
        "ID",
        "Datetime start",
        "Datetime end",
        "PDC",
        "Energy (Kwh)",
        "MAC Address",
        "Vehicle",
        "type_erreur",
        "moment",
        "evolution_soc",
        "elto",
        "logv0_url",
        "Downstream Code PC",
        "EVI Error Code",
        "duration",
    ]
    err_table = err_rows[err_display_cols].copy() if not err_rows.empty else pd.DataFrame(columns=err_display_cols)
    if "Datetime start" in err_table.columns:
        err_table = err_table.sort_values("Datetime start", ascending=False)

    ok_rows = df_filtered[df_filtered["is_ok"]].copy()
    ok_rows["evolution_soc"] = ok_rows.apply(lambda r: _format_soc(r.get("SOC Start"), r.get("SOC End")), axis=1)
    ok_rows["elto"] = ok_rows["ID"].apply(lambda x: f"https://elto.nidec-asi-online.com/Charge/detail?id={str(x).strip()}" if pd.notna(x) else "") if "ID" in ok_rows.columns else ""
    ok_rows["logv0_url"] = ok_rows["ID"].apply(lambda x: f"{LOGV0_BASE_URL}?session_id={str(x).strip()}&project=ELTO" if LOGV0_BASE_URL and pd.notna(x) else None) if "ID" in ok_rows.columns else None
    ok_display_cols = [
        "ID",
        "Datetime start",
        "Datetime end",
        "PDC",
        "Energy (Kwh)",
        "MAC Address",
        "Vehicle",
        "evolution_soc",
        "elto",
        "logv0_url",
        "is_warning",
        "duration",
    ]
    ok_table = ok_rows[ok_display_cols].copy() if not ok_rows.empty else pd.DataFrame(columns=ok_display_cols)
    if "Datetime start" in ok_table.columns:
        ok_table = ok_table.sort_values("Datetime start", ascending=False)

    site_total_charges = int(df_site["is_ok_filt"].count())
    site_charges_ok = int(df_site["is_ok_filt"].sum())
    site_success_rate = round(site_charges_ok / site_total_charges * 100, 2) if site_total_charges else 0.0

    by_pdc = (
        df_site.groupby("PDC", as_index=False)
        .agg(Total_Charges=("is_ok_filt", "count"), Charges_OK=("is_ok_filt", "sum"))
        .assign(Charges_NOK=lambda d: d["Total_Charges"] - d["Charges_OK"])
    )
    by_pdc["% Réussite"] = np.where(
        by_pdc["Total_Charges"].gt(0),
        (by_pdc["Charges_OK"] / by_pdc["Total_Charges"] * 100).round(2),
        0.0,
    )
    by_pdc = by_pdc.sort_values(["% Réussite", "PDC"], ascending=[True, True])

    error_moment: list[dict] = []
    error_moment_grouped: list[dict] = []
    error_moment_adv: list[dict] = []
    error_type_distribution: list[dict] = []
    error_type_total = 0
    if not err_rows.empty:
        if "moment" in err_rows.columns:
            counts = err_rows.groupby("moment").size().reset_index(name="Nb")
            total = counts["Nb"].sum()
            if total:
                error_moment = (
                    counts.assign(percent=lambda d: (d["Nb"] / total * 100).round(2))
                    .sort_values("percent", ascending=False)
                    .to_dict("records")
                )
                error_moment_grouped = error_moment

        if "moment_avancee" in err_rows.columns:
            counts_adv = (
                err_rows.groupby("moment_avancee")
                .size()
                .reset_index(name="Nb")
                .sort_values("Nb", ascending=False)
            )

            total_adv = counts_adv["Nb"].sum()
            if total_adv:
                error_moment_adv = (
                    counts_adv.assign(percent=lambda d: (d["Nb"] / total_adv * 100).round(2))
                    .to_dict("records")
                )

        error_type_order = ["Erreur_EVI", "Erreur_DownStream", "Erreur_Unknow_S"]
        error_type_labels = {
            "Erreur_EVI": "EVI",
            "Erreur_DownStream": "Downstream",
            "Erreur_Unknow_S": "Erreur_Unknow_S",
        }

        type_counts = (
            err_rows[err_rows["type_erreur"].isin(error_type_order)]
            .groupby("type_erreur")
            .size()
            .reindex(error_type_order, fill_value=0)
            .reset_index(name="count")
        )

        error_type_total = int(type_counts["count"].sum())

        error_type_distribution = [
            {
                "type_erreur": row["type_erreur"],
                "label": error_type_labels.get(row["type_erreur"], row["type_erreur"]),
                "count": int(row["count"]),
                "percent": round(row["count"] / error_type_total * 100, 1) if error_type_total else 0,
            }
            for _, row in type_counts.iterrows()
            if row["count"] > 0
        ]

    downstream_occ: list[dict] = []
    downstream_moments: list[str] = []
    if not err_rows.empty:
        need_cols_ds = {"Downstream Code PC", "moment"}
        if need_cols_ds.issubset(err_rows.columns):
            ds_num = pd.to_numeric(err_rows["Downstream Code PC"], errors="coerce").fillna(0).astype(int)
            mask_downstream = (ds_num != 0) & (ds_num != 8192)
            sub = err_rows.loc[mask_downstream, ["Downstream Code PC", "moment"]].copy()

            if not sub.empty:
                sub["Code_PC"] = pd.to_numeric(sub["Downstream Code PC"], errors="coerce").fillna(0).astype(int)
                tmp = sub.groupby(["Code_PC", "moment"]).size().reset_index(name="Occurrences")
                downstream_moments = [m for m in MOMENT_ORDER if m in tmp["moment"].unique()]
                downstream_moments += [m for m in sorted(tmp["moment"].unique()) if m not in downstream_moments]

                table = (
                    tmp.pivot(index="Code_PC", columns="moment", values="Occurrences")
                    .reindex(columns=downstream_moments, fill_value=0)
                    .reset_index()
                )

                table[downstream_moments] = table[downstream_moments].fillna(0).astype(int)
                table["Total"] = table[downstream_moments].sum(axis=1).astype(int)
                table = table.sort_values("Total", ascending=False).reset_index(drop=True)

                total_all = int(table["Total"].sum())
                table["Percent"] = np.where(
                    total_all > 0,
                    (table["Total"] / total_all * 100).round(2),
                    0.0,
                )

                table.insert(0, "Rank", range(1, len(table) + 1))

                total_row = {
                    "Rank": "",
                    "Code_PC": "Total",
                    **{m: int(table[m].sum()) for m in downstream_moments},
                }
                total_row["Total"] = int(table["Total"].sum())
                total_row["Percent"] = 100.0 if total_all else 0.0

                downstream_occ = table.to_dict("records") + [total_row]

    evi_occ: list[dict] = []
    evi_occ_moments: list[str] = []
    if not err_rows.empty:
        need_cols_evi = {"EVI Error Code", "moment"}
        if need_cols_evi.issubset(err_rows.columns):
            ds_num = pd.to_numeric(err_rows.get("Downstream Code PC", 0), errors="coerce").fillna(0).astype(int)
            evi_code = pd.to_numeric(err_rows["EVI Error Code"], errors="coerce").fillna(0).astype(int)

            mask_evi = (ds_num == 8192) | ((ds_num == 0) & (evi_code != 0))
            sub = err_rows.loc[mask_evi, ["EVI Error Code", "moment"]].copy()

            if not sub.empty:
                sub["EVI_Code"] = pd.to_numeric(sub["EVI Error Code"], errors="coerce").astype(int)
                tmp = sub.groupby(["EVI_Code", "moment"]).size().reset_index(name="Occurrences")
                evi_occ_moments = [m for m in MOMENT_ORDER if m in tmp["moment"].unique()]
                evi_occ_moments += [m for m in sorted(tmp["moment"].unique()) if m not in evi_occ_moments]

                table = (
                    tmp.pivot(index="EVI_Code", columns="moment", values="Occurrences")
                    .reindex(columns=evi_occ_moments, fill_value=0)
                    .reset_index()
                )

                table[evi_occ_moments] = table[evi_occ_moments].fillna(0).astype(int)
                table["Total"] = table[evi_occ_moments].sum(axis=1).astype(int)
                table = table.sort_values("Total", ascending=False).reset_index(drop=True)

                total_all = int(table["Total"].sum())
                table["Percent"] = np.where(
                    total_all > 0,
                    (table["Total"] / total_all * 100).round(2),
                    0.0,
                )

                table.insert(0, "Rank", range(1, len(table) + 1))

                total_row = {
                    "Rank": "",
                    "EVI_Code": "Total",
                    **{m: int(table[m].sum()) for m in evi_occ_moments},
                }
                total_row["Total"] = int(table["Total"].sum())
                total_row["Percent"] = 100.0 if total_all else 0.0

                evi_occ = table.to_dict("records") + [total_row]

    monthly_rows = []
    daily_rows = []
    month_options: list[str] = []
    month_focus_value = ""

    if site_value:
        base_site = df_filtered[df_filtered["Site"] == site_value].copy()
        if "Datetime start" in base_site.columns:
            base_site["Datetime start"] = pd.to_datetime(base_site["Datetime start"], errors="coerce")

        ok_focus = base_site[base_site["is_ok_filt"]].copy()
        nok_focus = base_site[~base_site["is_ok_filt"]].copy()

        if not ok_focus.empty or not nok_focus.empty:
            ok_focus["month"] = pd.to_datetime(ok_focus["Datetime start"], errors="coerce").dt.to_period("M").astype(str)
            nok_focus["month"] = pd.to_datetime(nok_focus["Datetime start"], errors="coerce").dt.to_period("M").astype(str)

            g_ok_m = ok_focus.groupby("month").size().reset_index(name="Nb").assign(Status="OK")
            g_nok_m = nok_focus.groupby("month").size().reset_index(name="Nb").assign(Status="NOK")

            g_both_m = pd.concat([g_ok_m, g_nok_m], ignore_index=True)
            g_both_m["month"] = pd.to_datetime(g_both_m["month"], errors="coerce")
            g_both_m = g_both_m.dropna(subset=["month"]).sort_values("month")
            g_both_m["month"] = g_both_m["month"].dt.strftime("%Y-%m")

            if not g_both_m.empty:
                piv_m = g_both_m.pivot(index="month", columns="Status", values="Nb").fillna(0).sort_index()
                month_options = piv_m.index.tolist()
                month_focus_value = month_focus if month_focus in month_options else (month_options[-1] if month_options else "")
                for month in month_options:
                    ok_val = int(piv_m.at[month, "OK"]) if "OK" in piv_m.columns else 0
                    nok_val = int(piv_m.at[month, "NOK"]) if "NOK" in piv_m.columns else 0
                    total_val = ok_val + nok_val
                    ok_pct = round(ok_val / total_val * 100, 1) if total_val else 0
                    nok_pct = round(nok_val / total_val * 100, 1) if total_val else 0
                    monthly_rows.append(
                        {
                            "month": month,
                            "ok": ok_val,
                            "nok": nok_val,
                            "ok_pct": ok_pct,
                            "nok_pct": nok_pct,
                        }
                    )

                if month_focus_value:
                    ok_month = ok_focus[ok_focus["month"] == month_focus_value].copy()
                    nok_month = nok_focus[nok_focus["month"] == month_focus_value].copy()

                    ok_month["day"] = pd.to_datetime(ok_month["Datetime start"], errors="coerce").dt.strftime("%Y-%m-%d")
                    nok_month["day"] = pd.to_datetime(nok_month["Datetime start"], errors="coerce").dt.strftime("%Y-%m-%d")

                    per = pd.Period(month_focus_value, freq="M")
                    days = pd.date_range(per.to_timestamp(how="start"), per.to_timestamp(how="end"), freq="D").strftime("%Y-%m-%d")

                    g_ok_d = ok_month.groupby("day").size().reindex(days, fill_value=0).reset_index()
                    g_ok_d.columns = ["day", "Nb"]
                    g_ok_d["Status"] = "OK"
                    g_nok_d = nok_month.groupby("day").size().reindex(days, fill_value=0).reset_index()
                    g_nok_d.columns = ["day", "Nb"]
                    g_nok_d["Status"] = "NOK"

                    g_both_d = pd.concat([g_ok_d, g_nok_d], ignore_index=True)
                    piv_d = g_both_d.pivot(index="day", columns="Status", values="Nb").fillna(0)
                    for day in piv_d.index.tolist():
                        ok_val = int(piv_d.at[day, "OK"]) if "OK" in piv_d.columns else 0
                        nok_val = int(piv_d.at[day, "NOK"]) if "NOK" in piv_d.columns else 0
                        total_val = ok_val + nok_val
                        ok_pct = round(ok_val / total_val * 100, 1) if total_val else 0
                        nok_pct = round(nok_val / total_val * 100, 1) if total_val else 0
                        daily_rows.append(
                            {
                                "day": day,
                                "ok": ok_val,
                                "nok": nok_val,
                                "ok_pct": ok_pct,
                                "nok_pct": nok_pct,
                            }
                        )

    hourly_active_sessions = []
    hourly_sessions_count = []
    day_focus_value = ""
    day_options: list[str] = []
    usage_duration_by_level: list[dict] = []

    if site_value:
        try:
            query_dates = """
                SELECT DISTINCT DATE(Times) as date_str
                FROM indicator.kpi_concurrency_15min
                WHERE Site = :site
                ORDER BY date_str DESC
            """
            df_dates = query_df(query_dates, {"site": site_value})
            if not df_dates.empty:
                day_options = df_dates["date_str"].astype(str).tolist()

                if day_focus and day_focus in day_options:
                    day_focus_value = day_focus
                elif day_options:
                    day_focus_value = day_options[0]
        except Exception:
            pass

    if site_value and day_focus_value:
        try:
            day_focus_date = pd.to_datetime(day_focus_value).date()
            day_start = pd.Timestamp.combine(day_focus_date, pd.Timestamp.min.time())

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
            df_15min = query_df(query_15min, {"site": site_value, "date_focus": day_focus_value})

            if not df_15min.empty:
                df_15min["Times"] = pd.to_datetime(df_15min["Times"])

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
            df_daily = query_df(query_daily, {"site": site_value, "date_focus": day_focus_value})

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
            df_sessions_hourly = query_df(query_sessions_hourly, {"site": site_value, "date_focus": day_focus_value})

            if not df_sessions_hourly.empty:
                sessions_by_hour = {int(row["hour"]): int(row["session_count"]) for _, row in df_sessions_hourly.iterrows()}

                cumulative_count = 0
                for hour in range(24):
                    cumulative_count += sessions_by_hour.get(hour, 0)
                    minutes_from_start = hour * 60
                    hourly_sessions_count.append({
                        "hour": hour,
                        "minutes_from_start": minutes_from_start,
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
        except Exception as e:
            for hour in range(24):
                hourly_active_sessions.append({
                    "time": f"{hour:02d}:00",
                    "hour": hour,
                    "minute": 0,
                    "minutes_from_start": hour * 60,
                    "hour_label": f"{hour:02d}:00",
                    "count": 0,
                })
                hourly_sessions_count.append({
                    "hour": hour,
                    "minutes_from_start": hour * 60,
                    "hour_label": f"{hour:02d}:00",
                    "count": 0,
                })

    return templates.TemplateResponse(
        "partials/sessions_site_details.html",
        {
            "request": request,
            "site_options": site_options,
            "site_focus": site_value,
            "pdc_options": pdc_options,
            "selected_pdc": selected_pdc,
            "err_rows": err_table.to_dict("records"),
            "ok_rows": ok_table.to_dict("records"),
            "by_pdc": by_pdc.to_dict("records"),
            "site_success_rate": site_success_rate,
            "site_total_charges": site_total_charges,
            "site_charges_ok": site_charges_ok,
            "error_moment": error_moment,
            "error_moment_grouped": error_moment_grouped,
            "error_moment_adv": error_moment_adv,
            "error_type_distribution": error_type_distribution,
            "error_type_total": error_type_total,
            "downstream_occ": downstream_occ,
            "downstream_moments": downstream_moments,
            "evi_occ": evi_occ,
            "evi_occ_moments": evi_occ_moments,
            "monthly_rows": monthly_rows,
            "daily_rows": daily_rows,
            "month_options": month_options,
            "month_focus": month_focus_value,
            "hourly_active_sessions": hourly_active_sessions,
            "hourly_sessions_count": hourly_sessions_count,
            "day_focus": day_focus_value,
            "day_options": day_options,
            "usage_duration_by_level": usage_duration_by_level,
            "base_query": _prepare_query_params(request),
        },
    )
