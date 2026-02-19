from fastapi import APIRouter, Request, Query
from datetime import date
from typing import Any
import pandas as pd
import numpy as np

from db import query_df, LOGV0_BASE_URL
from routers.filters import MOMENT_ORDER
from .common import _build_conditions, _get_vehicle_strategy, _apply_status_filters, templates, SITE_COLOR_PALETTE

router = APIRouter(tags=["sessions"])


@router.get("/sessions/general")
async def get_sessions_general(
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
            Site,
            PDC,
            `State of charge(0:good, 1:error)` as state,
            type_erreur,
            moment
        FROM kpi_sessions
        WHERE {where_clause}
    """

    df = query_df(sql, params)

    if df.empty:
        return templates.TemplateResponse(
            "partials/sessions_general.html",
            {
                "request": request,
                "total": 0,
                "ok": 0,
                "nok": 0,
                "taux_reussite": 0,
                "taux_echec": 0,
                "recap_columns": [],
                "recap_rows": [],
                "moment_distribution": [],
                "moment_total_errors": 0,
                "error_type_distribution": [],
                "error_type_total": 0,
                "site_success_cards": [],
                "site_success_bars": [],
            },
        )

    df = _apply_status_filters(df, error_type_list, moment_list)

    total = len(df)
    ok = int(df["is_ok_filt"].sum())
    nok = total - ok
    taux_reussite = round(ok / total * 100, 1) if total else 0
    taux_echec = round(nok / total * 100, 1) if total else 0

    df["PDC"] = df.get("PDC", "").astype(str)

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

    stats_pdc = (
        df.groupby(["Site", "PDC"])
        .agg(total=("is_ok_filt", "count"), ok=("is_ok_filt", "sum"))
        .reset_index()
    )
    stats_pdc["nok"] = stats_pdc["total"] - stats_pdc["ok"]
    stats_pdc["taux_ok"] = np.where(
        stats_pdc["total"] > 0,
        (stats_pdc["ok"] / stats_pdc["total"] * 100).round(1),
        0,
    )

    stat_global = stats_site.rename(columns={"Site": "Site", "total": "Total", "ok": "Total_OK"})
    stat_global["Total_NOK"] = stat_global["Total"] - stat_global["Total_OK"]
    stat_global["% OK"] = np.where(
        stat_global["Total"] > 0,
        (stat_global["Total_OK"] / stat_global["Total"] * 100).round(2),
        0,
    )
    stat_global["% NOK"] = np.where(
        stat_global["Total"] > 0,
        (stat_global["Total_NOK"] / stat_global["Total"] * 100).round(2),
        0,
    )

    err = df[~df["is_ok_filt"]].copy()

    recap_columns: list[str] = []
    recap_rows: list[dict] = []
    moment_distribution = []
    moment_total_errors = 0
    error_type_distribution: list[dict[str, int | float | str]] = []
    error_type_total = 0

    if not err.empty:
        err_grouped = (
            err.groupby(["Site", "moment"])
            .size()
            .reset_index(name="Nb")
            .pivot(index="Site", columns="moment", values="Nb")
            .fillna(0)
            .astype(int)
            .reset_index()
        )

        err_pdc_grouped = (
            err.groupby(["Site", "PDC", "moment"])
            .size()
            .reset_index(name="Nb")
            .pivot(index=["Site", "PDC"], columns="moment", values="Nb")
            .fillna(0)
            .astype(int)
            .reset_index()
        )

        err_moment_cols = [c for c in err_grouped.columns if c not in ["Site"]]
        err_pdc_moment_cols = [c for c in err_pdc_grouped.columns if c not in ["Site", "PDC"]]

        moment_cols = [m for m in MOMENT_ORDER if m in err_moment_cols or m in err_pdc_moment_cols]
        extra_moment_cols = [c for c in err_moment_cols + err_pdc_moment_cols if c not in moment_cols]
        moment_cols += [c for c in extra_moment_cols if c not in moment_cols]

        recap = (
            stat_global
            .merge(err_grouped, on="Site", how="left")
            .fillna(0)
            .sort_values("Total_NOK", ascending=False)
            .reset_index(drop=True)
        )

        recap_columns = [
            "Site / PDC",
            "Total",
            "Total_OK",
            "Total_NOK",
        ] + moment_cols + ["% OK", "% NOK"]

        recap["Site / PDC"] = recap["Site"]

        for col in moment_cols:
            if col not in recap.columns:
                recap[col] = 0

        numeric_moment_cols = [c for c in moment_cols if c in recap.columns]
        if numeric_moment_cols:
            recap[numeric_moment_cols] = recap[numeric_moment_cols].astype(int)

        pdc_recap = (
            stats_pdc.rename(
                columns={
                    "total": "Total",
                    "ok": "Total_OK",
                    "nok": "Total_NOK",
                    "taux_ok": "% OK",
                }
            )
            .assign(**{"% NOK": lambda d: np.where(d["Total"] > 0, (d["Total_NOK"] / d["Total"] * 100).round(2), 0)})
            .merge(err_pdc_grouped, on=["Site", "PDC"], how="left")
            .fillna(0)
        )

        numeric_moment_cols_pdc = [c for c in moment_cols if c in pdc_recap.columns]
        if numeric_moment_cols_pdc:
            pdc_recap[numeric_moment_cols_pdc] = pdc_recap[numeric_moment_cols_pdc].astype(int)

        for col in moment_cols:
            if col not in pdc_recap.columns:
                pdc_recap[col] = 0

        pdc_recap["Site / PDC"] = "↳ PDC " + pdc_recap["PDC"].astype(str)
        pdc_recap_display = pdc_recap[[c for c in recap_columns if c in pdc_recap.columns]].copy()

        recap_rows = []
        for _, row in recap.iterrows():
            row_dict = row.to_dict()
            label = row_dict.get("Site / PDC", "")
            row_dict["Site / PDC"] = f"{label} (Total)" if label else label
            row_dict.update({"row_type": "site", "site_key": row_dict.get("Site", "")})
            recap_rows.append(row_dict)

            site_pdcs = pdc_recap[pdc_recap["Site"].eq(row["Site"])].copy()
            site_pdcs = site_pdcs.sort_values("Total_NOK", ascending=False)

            for _, pdc_row in site_pdcs.iterrows():
                pdc_dict = pdc_row.to_dict()
                display_dict = {k: pdc_dict.get(k) for k in pdc_recap_display.columns}
                display_dict.update({"row_type": "pdc", "site_key": row_dict.get("Site", "")})
                recap_rows.append(display_dict)

        counts_moment = (
            err.groupby("moment")
            .size()
            .reindex(MOMENT_ORDER, fill_value=0)
            .reset_index(name="count")
        )
        counts_moment = counts_moment[counts_moment["count"] > 0]

        total_err = len(err)
        moment_total_errors = int(total_err)
        moment_distribution = [
            {
                "moment": row["moment"],
                "count": int(row["count"]),
                "percent": round(row["count"] / total_err * 100, 1) if total_err else 0,
            }
            for _, row in counts_moment.iterrows()
        ]

        error_type_order = ["Erreur_EVI", "Erreur_DownStream", "Erreur_Unknow_S"]
        error_type_labels = {
            "Erreur_EVI": "EVI",
            "Erreur_DownStream": "Downstream",
            "Erreur_Unknow_S": "Erreur_Unknow_S",
        }

        type_counts = (
            err[err["type_erreur"].isin(error_type_order)]
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

    return templates.TemplateResponse(
        "partials/sessions_general.html",
        {
            "request": request,
            "total": total,
            "ok": ok,
            "nok": nok,
            "taux_reussite": taux_reussite,
            "taux_echec": taux_echec,
            "recap_columns": recap_columns,
            "recap_rows": recap_rows,
            "moment_distribution": moment_distribution,
            "moment_total_errors": moment_total_errors,
            "error_type_distribution": error_type_distribution,
            "error_type_total": error_type_total,
            "site_success_cards": site_success_cards,
            "site_success_bars": site_success_bars,
        },
    )
