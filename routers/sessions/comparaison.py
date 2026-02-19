from fastapi import APIRouter, Request, Query
from datetime import date
import pandas as pd
import numpy as np

from db import query_df
from .common import (_calculate_period_dates, _build_conditions, _get_vehicle_strategy,
                     _apply_status_filters, _comparaison_base_context, SITE_COLOR_PALETTE, templates)

router = APIRouter(tags=["sessions"])


@router.get("/sessions/comparaison")
async def get_sessions_comparaison(
    request: Request,
    sites: str = Query(default=""),
    date_debut: date = Query(default=None),
    date_fin: date = Query(default=None),
    error_types: str | None = Query(default=None),
    moments: str | None = Query(default=None),
    site_focus: str = Query(default=""),
    month_focus: str = Query(default=""),
):
    error_type_list = None if error_types is None else [e.strip() for e in error_types.split(",") if e.strip()]
    moment_list = None if moments is None else [m.strip() for m in moments.split(",") if m.strip()]

    filters = {
        "sites": sites,
        "date_debut": str(date_debut) if date_debut else "",
        "date_fin": str(date_fin) if date_fin else "",
        "error_types": error_types,
        "moments": moments,
    }

    where_clause, params = _build_conditions(sites, date_debut, date_fin)

    sql = f"""
        SELECT
            Site,
            `Datetime start`,
            `State of charge(0:good, 1:error)` as state,
            type_erreur,
            moment
        FROM kpi_sessions
        WHERE {where_clause}
    """

    try:
        df = query_df(sql, params)
    except Exception as exc:
        return templates.TemplateResponse(
            "partials/sessions_comparaison.html",
            _comparaison_base_context(
                request,
                filters,
                site_focus="",
                month_focus="",
                error_message=str(exc),
            ),
        )

    if df.empty:
        return templates.TemplateResponse(
            "partials/sessions_comparaison.html",
            _comparaison_base_context(request, filters),
        )

    df = _apply_status_filters(df, error_type_list, moment_list)

    if "Datetime start" in df.columns:
        df["Datetime start"] = pd.to_datetime(df["Datetime start"], errors="coerce")

    site_col = "Site"

    by_site = (
        df.groupby(site_col, as_index=False)
        .agg(
            Total_Charges=("is_ok_filt", "count"),
            Charges_OK=("is_ok_filt", "sum"),
        )
    )
    by_site["Charges_NOK"] = by_site["Total_Charges"] - by_site["Charges_OK"]
    by_site["% Réussite"] = np.where(
        by_site["Total_Charges"].gt(0),
        (by_site["Charges_OK"] / by_site["Total_Charges"] * 100).round(2),
        0.0,
    )
    by_site["% Échec"] = np.where(
        by_site["Total_Charges"].gt(0),
        (by_site["Charges_NOK"] / by_site["Total_Charges"] * 100).round(2),
        0.0,
    )
    by_site = by_site.reset_index(drop=True)

    site_rows = by_site.to_dict("records")
    by_site_sorted = by_site.sort_values("Total_Charges", ascending=False)
    max_total = int(by_site_sorted["Total_Charges"].max()) if not by_site_sorted.empty else 0

    count_bars = [
        {
            "site": row[site_col],
            "ok": int(row["Charges_OK"]),
            "nok": int(row["Charges_NOK"]),
            "total": int(row["Total_Charges"]),
        }
        for _, row in by_site_sorted.iterrows()
    ]

    percent_bars = [
        {
            "site": row[site_col],
            "ok_pct": float(row["% Réussite"]),
            "nok_pct": float(row["% Échec"]),
        }
        for _, row in by_site_sorted.iterrows()
    ]

    base = df.copy()
    base["hour"] = pd.to_datetime(base["Datetime start"], errors="coerce").dt.hour

    g = (
        base.dropna(subset=["hour"])
        .groupby([site_col, "hour"])
        .size()
        .reset_index(name="Nb")
    )

    peak_rows = []
    heatmap_rows = []
    heatmap_hours: list[int] = []
    heatmap_max = 0

    if not g.empty:
        peak = g.loc[g.groupby(site_col)["Nb"].idxmax()][[site_col, "hour", "Nb"]].rename(
            columns={"hour": "Heure de pic", "Nb": "Nb au pic"}
        )

        def _w_median_hours(dfh: pd.DataFrame) -> int:
            s = dfh.sort_values("hour")
            c = s["Nb"].cumsum()
            half = s["Nb"].sum() / 2.0
            return int(s.loc[c >= half, "hour"].iloc[0])

        med = (
            g.groupby(site_col)[["hour", "Nb"]]
            .apply(_w_median_hours)
            .reset_index(name="Heure médiane")
        )
        summ = peak.merge(med, on=site_col, how="left")

        for _, row in summ.sort_values(site_col).iterrows():
            peak_rows.append(
                {
                    "site": row[site_col],
                    "peak_hour": f"{int(row['Heure de pic']):02d}:00",
                    "peak_nb": int(row["Nb au pic"]),
                    "median_hour": f"{int(row['Heure médiane']):02d}:00",
                }
            )

        heatmap = g.pivot(index=site_col, columns="hour", values="Nb").fillna(0)
        heatmap_hours = sorted(heatmap.columns.tolist())
        heatmap_max = int(heatmap.values.max()) if heatmap.size else 0
        for idx in heatmap.index:
            heatmap_rows.append(
                {
                    "site": idx,
                    "values": [int(heatmap.at[idx, h]) if h in heatmap.columns else 0 for h in heatmap_hours],
                }
            )

    site_options = by_site_sorted[site_col].tolist()
    site_focus_value = site_focus if site_focus and site_focus in site_options else (site_options[0] if site_options else "")

    monthly_rows = []
    daily_rows = []
    month_options: list[str] = []
    month_focus_value = ""

    if site_focus_value:
        base_site = base[base[site_col] == site_focus_value].copy()
        ok_focus = base_site[base_site["is_ok_filt"]].copy()
        nok_focus = base_site[~base_site["is_ok_filt"]].copy()

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

    context = _comparaison_base_context(
        request,
        filters,
        site_focus=site_focus_value,
        month_focus=month_focus_value,
    )
    context.update(
        {
            "site_rows": site_rows,
            "count_bars": count_bars,
            "percent_bars": percent_bars,
            "max_total": max_total,
            "peak_rows": peak_rows,
            "heatmap_rows": heatmap_rows,
            "heatmap_hours": heatmap_hours,
            "heatmap_max": heatmap_max,
            "site_options": site_options,
            "month_options": month_options,
            "monthly_rows": monthly_rows,
            "daily_rows": daily_rows,
        }
    )

    return templates.TemplateResponse("partials/sessions_comparaison.html", context)
