from fastapi import APIRouter, Request, Query
from datetime import date
from typing import Any
import pandas as pd
import numpy as np

from db import query_df, LOGV0_BASE_URL
from routers.filters import MOMENT_ORDER
from .common import (_build_conditions, _get_vehicle_strategy, _apply_status_filters,
                     _build_pivot_table, _map_moment_label, _map_phase_label,
                     EVI_MOMENT, EVI_CODE, DS_PC, templates)

router = APIRouter(tags=["sessions"])


@router.get("/sessions/error-analysis")
async def get_error_analysis(
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

    sql = f"""
        SELECT
            k.Site,
            k.`State of charge(0:good, 1:error)` as state,
            k.type_erreur,
            k.moment,
            k.moment_avancee,
            k.`{EVI_MOMENT}`,
            k.`{EVI_CODE}`,
            k.`{DS_PC}`
        FROM kpi_sessions k
        WHERE {where_clause}
    """

    df = query_df(sql, params)

    if df.empty:
        return templates.TemplateResponse(
            "partials/error_analysis.html",
            {"request": request, "no_data": True},
        )

    df["is_ok"] = pd.to_numeric(df["state"], errors="coerce").fillna(0).astype(int).eq(0)
    df = _apply_status_filters(df, error_type_list, moment_list)
    df["Site"] = df.get("Site", "").fillna("")

    err = df[~df["is_ok_filt"]].copy()

    if err.empty:
        return templates.TemplateResponse(
            "partials/error_analysis.html",
            {"request": request, "no_errors": True},
        )

    evi_step = pd.to_numeric(err.get(EVI_MOMENT, pd.Series(np.nan, index=err.index)), errors="coerce")
    evi_code = pd.to_numeric(err.get(EVI_CODE, pd.Series(np.nan, index=err.index)), errors="coerce").fillna(0).astype(int)
    ds_pc = pd.to_numeric(err.get(DS_PC, pd.Series(np.nan, index=err.index)), errors="coerce").fillna(0).astype(int)
    moment_raw = err.get("moment", pd.Series(None, index=err.index))

    def resolve_moment_label(idx: int) -> str:
        label = None
        step_val = evi_step.loc[idx] if idx in evi_step.index else np.nan
        raw_val = moment_raw.loc[idx] if idx in moment_raw.index else None

        if pd.notna(step_val):
            label = _map_moment_label(step_val)
        if (not label or label == "Unknown") and isinstance(raw_val, str) and raw_val.strip():
            label = raw_val.strip()
        return label or "Unknown"

    err["moment_label"] = [resolve_moment_label(i) for i in err.index]

    sub_evi_mask = (ds_pc.eq(8192)) | (ds_pc.eq(0) & evi_code.ne(0))
    sub_ds_mask = ds_pc.ne(0) & ds_pc.ne(8192)

    sub_evi = err.loc[sub_evi_mask].copy()
    sub_evi["step"] = evi_step.loc[sub_evi.index]
    sub_evi["code"] = evi_code.loc[sub_evi.index]
    sub_evi["type"] = "Erreur_EVI"

    sub_ds = err.loc[sub_ds_mask].copy()
    sub_ds["step"] = evi_step.loc[sub_ds.index]
    sub_ds["code"] = ds_pc.loc[sub_ds.index]
    sub_ds["type"] = "Erreur_DownStream"

    evi_moment_code: list[dict[str, Any]] = []
    evi_moment_code_site: list[dict[str, Any]] = []
    if not sub_evi.empty:
        evi_moment_code_df = (
            sub_evi.groupby(["moment_label", "step", "code"])
            .size()
            .reset_index(name="Somme de Charge_NOK")
            .sort_values("Somme de Charge_NOK", ascending=False)
        )
        evi_total = int(evi_moment_code_df["Somme de Charge_NOK"].sum())
        total_row = pd.DataFrame(
            [
                {
                    "moment_label": "Total",
                    "step": "",
                    "code": "",
                    "Somme de Charge_NOK": evi_total,
                }
            ]
        )
        evi_moment_code_df = pd.concat([evi_moment_code_df, total_row], ignore_index=True)
        evi_moment_code_df.rename(
            columns={"moment_label": "Moment", "step": "Step", "code": "Code"}, inplace=True
        )
        evi_moment_code = evi_moment_code_df.to_dict("records")

        evi_moment_code_site_df = (
            sub_evi.groupby(["Site", "moment_label", "step", "code"])
            .size()
            .reset_index(name="Somme de Charge_NOK")
            .sort_values(["Site", "Somme de Charge_NOK"], ascending=[True, False])
        )
        evi_moment_code_site_df.rename(
            columns={"moment_label": "Moment", "step": "Step", "code": "Code"}, inplace=True
        )
        evi_moment_code_site = evi_moment_code_site_df.to_dict("records")

    ds_moment_code: list[dict[str, Any]] = []
    ds_moment_code_site: list[dict[str, Any]] = []
    if not sub_ds.empty:
        ds_moment_code_df = (
            sub_ds.groupby(["moment_label", "step", "code"])
            .size()
            .reset_index(name="Somme de Charge_NOK")
            .sort_values("Somme de Charge_NOK", ascending=False)
        )
        ds_total = int(ds_moment_code_df["Somme de Charge_NOK"].sum())
        ds_total_row = pd.DataFrame(
            [
                {
                    "moment_label": "Total",
                    "step": "",
                    "code": "",
                    "Somme de Charge_NOK": ds_total,
                }
            ]
        )
        ds_moment_code_df = pd.concat([ds_moment_code_df, ds_total_row], ignore_index=True)
        ds_moment_code_df.rename(
            columns={"moment_label": "Moment", "step": "Step", "code": "Code PC"}, inplace=True
        )
        ds_moment_code = ds_moment_code_df.to_dict("records")

        ds_moment_code_site_df = (
            sub_ds.groupby(["Site", "moment_label", "step", "code"])
            .size()
            .reset_index(name="Somme de Charge_NOK")
            .sort_values(["Site", "Somme de Charge_NOK"], ascending=[True, False])
        )
        ds_moment_code_site_df.rename(
            columns={"moment_label": "Moment", "step": "Step", "code": "Code PC"}, inplace=True
        )
        ds_moment_code_site = ds_moment_code_site_df.to_dict("records")

    by_site = (
        df.groupby("Site", as_index=False)
        .agg(Total_Charges=("is_ok_filt", "count"), Charges_OK=("is_ok_filt", "sum"))
        .assign(Charges_NOK=lambda d: d["Total_Charges"] - d["Charges_OK"])
    )

    all_err = pd.concat([sub_evi, sub_ds], ignore_index=True)

    top_all: list[dict[str, Any]] = []
    detail_all: list[dict[str, Any]] = []
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
        top_all = top3_all.to_dict("records")

        top_keys = top3_all[["moment_label", "step", "code", "type"]].to_records(index=False).tolist()
        detail_all_df = all_err[
            all_err[["moment_label", "step", "code", "type"]].apply(tuple, axis=1).isin(top_keys)
        ]
        detail_all = (
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

    top_evi: list[dict[str, Any]] = []
    detail_evi: list[dict[str, Any]] = []
    if not sub_evi.empty:
        tbl_evi = (
            sub_evi.groupby(["moment_label", "step", "code"])
            .size()
            .reset_index(name="Occurrences")
            .sort_values("Occurrences", ascending=False)
        )
        total_evi = int(tbl_evi["Occurrences"].sum()) or 1
        tbl_evi["percent"] = (tbl_evi["Occurrences"] / total_evi * 100).round(2)
        top_evi = tbl_evi.head(3).to_dict("records")

        top_keys_evi = tbl_evi.head(3)[["moment_label", "step", "code"]].to_records(index=False).tolist()
        detail_evi_df = sub_evi[
            sub_evi[["moment_label", "step", "code"]].apply(tuple, axis=1).isin(top_keys_evi)
        ]
        detail_evi = (
            detail_evi_df.groupby(["moment_label", "step", "code", "Site"])
            .size()
            .reset_index(name="Occurrences")
            .sort_values(["moment_label", "step", "code", "Occurrences"], ascending=[True, True, True, False])
            .to_dict("records")
        )
        detail_evi_pivot = _build_pivot_table(detail_evi_df, by_site)


    top_ds: list[dict[str, Any]] = []
    detail_ds: list[dict[str, Any]] = []
    if not sub_ds.empty:
        tbl_ds = (
            sub_ds.groupby(["moment_label", "step", "code"])
            .size()
            .reset_index(name="Occurrences")
            .sort_values("Occurrences", ascending=False)
        )
        total_ds = int(tbl_ds["Occurrences"].sum()) or 1
        tbl_ds["percent"] = (tbl_ds["Occurrences"] / total_ds * 100).round(2)
        top_ds = tbl_ds.head(3).to_dict("records")

        top_keys_ds = tbl_ds.head(3)[["moment_label", "step", "code"]].to_records(index=False).tolist()
        detail_ds_df = sub_ds[
            sub_ds[["moment_label", "step", "code"]].apply(tuple, axis=1).isin(top_keys_ds)
        ]
        detail_ds = (
            detail_ds_df.groupby(["moment_label", "step", "code", "Site"])
            .size()
            .reset_index(name="Occurrences")
            .sort_values(["moment_label", "step", "code", "Occurrences"], ascending=[True, True, True, False])
            .to_dict("records")
        )
        detail_ds_pivot = _build_pivot_table(detail_ds_df, by_site)

    detail_all_pivot = locals().get("detail_all_pivot", {"columns": [], "rows": []})
    detail_evi_pivot = locals().get("detail_evi_pivot", {"columns": [], "rows": []})
    detail_ds_pivot = locals().get("detail_ds_pivot", {"columns": [], "rows": []})

    err_phase = err.copy()
    err_phase["Phase"] = err_phase["moment_label"].map(_map_phase_label)

    err_by_phase = (
        err_phase.groupby(["Site", "Phase"])
        .size()
        .unstack("Phase", fill_value=0)
        .reset_index()
    )

    df_final = by_site.merge(err_by_phase, on="Site", how="left").fillna(0)

    for col in ["Avant charge", "Charge", "Fin de charge", "Unknown"]:
        if col not in df_final.columns:
            df_final[col] = 0

    df_final["% Réussite"] = np.where(
        df_final["Total_Charges"] > 0,
        (df_final["Charges_OK"] / df_final["Total_Charges"] * 100).round(2),
        0.0,
    )
    df_final["% Erreurs"] = np.where(
        df_final["Total_Charges"] > 0,
        (
            (
                df_final["Avant charge"]
                + df_final["Charge"]
                + df_final["Fin de charge"]
                + df_final["Unknown"]
            )
            / df_final["Total_Charges"]
            * 100
        ).round(2),
        0.0,
    )

    site_summary = df_final[
        [
            "Site",
            "Total_Charges",
            "Charges_OK",
            "Charges_NOK",
            "% Réussite",
            "% Erreurs",
            "Avant charge",
            "Charge",
            "Fin de charge",
            "Unknown",
        ]
    ].to_dict("records")

    error_type_counts: list[dict[str, Any]] = []
    err_nonempty = err.loc[err["type_erreur"].notna() & err["type_erreur"].ne("")].copy()
    if not err_nonempty.empty:
        counts_t = (
            err_nonempty.groupby("type_erreur")
            .size()
            .reset_index(name="Nb")
            .sort_values("Nb", ascending=False)
        )
        counts_t = pd.concat(
            [counts_t, pd.DataFrame([{"type_erreur": "Total", "Nb": int(counts_t["Nb"].sum())}])],
            ignore_index=True,
        )
        error_type_counts = counts_t.to_dict("records")

    moment_counts: list[dict[str, Any]] = []
    if "moment" in err.columns:
        counts_moment = (
            err.groupby("moment")
            .size()
            .reindex(MOMENT_ORDER, fill_value=0)
            .reset_index(name="Somme de Charge_NOK")
        )
        counts_moment = counts_moment[counts_moment["Somme de Charge_NOK"] > 0]
        if not counts_moment.empty:
            counts_moment = pd.concat(
                [
                    counts_moment,
                    pd.DataFrame(
                        [
                            {
                                "moment": "Total",
                                "Somme de Charge_NOK": int(counts_moment["Somme de Charge_NOK"].sum()),
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
            moment_counts = counts_moment.to_dict("records")

    moment_adv_counts: list[dict[str, Any]] = []
    if "moment_avancee" in err.columns:
        counts_av = (
            err.groupby("moment_avancee")
            .size()
            .reset_index(name="Somme de Charge_NOK")
            .sort_values("Somme de Charge_NOK", ascending=False)
        )
        if not counts_av.empty:
            counts_av = pd.concat(
                [
                    counts_av,
                    pd.DataFrame(
                        [
                            {
                                "moment_avancee": "Total",
                                "Somme de Charge_NOK": int(counts_av["Somme de Charge_NOK"].sum()),
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
            moment_adv_counts = counts_av.to_dict("records")

    err_evi = err[err["type_erreur"] == "Erreur_EVI"].copy()
    evi_moment_distribution: list[dict[str, Any]] = []
    evi_moment_adv_distribution: list[dict[str, Any]] = []
    if not err_evi.empty and "moment" in err_evi.columns:
        counts_moment = (
            err_evi.groupby("moment")
            .size()
            .reindex(MOMENT_ORDER, fill_value=0)
            .reset_index(name="Nb")
        )
        total_evi_err = int(counts_moment["Nb"].sum())
        if total_evi_err > 0:
            counts_moment["%"] = (counts_moment["Nb"] / total_evi_err * 100).round(2)
            counts_moment = counts_moment[counts_moment["Nb"] > 0]
            counts_moment = pd.concat(
                [
                    counts_moment,
                    pd.DataFrame(
                        [
                            {
                                "moment": "Total",
                                "Nb": total_evi_err,
                                "%": 100.0,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
            evi_moment_distribution = counts_moment.to_dict("records")

        if "moment_avancee" in err_evi.columns:
            counts_ma = (
                err_evi.groupby("moment_avancee")
                .size()
                .reset_index(name="Nb")
                .sort_values("Nb", ascending=False)
            )
            if not counts_ma.empty:
                counts_ma = pd.concat(
                    [
                        counts_ma,
                        pd.DataFrame([
                            {"moment_avancee": "Total", "Nb": int(counts_ma["Nb"].sum())}
                        ]),
                    ],
                    ignore_index=True,
                )
                evi_moment_adv_distribution = counts_ma.to_dict("records")

    err_ds = err[err["type_erreur"] == "Erreur_DownStream"].copy()
    ds_moment_distribution: list[dict[str, Any]] = []
    ds_moment_adv_distribution: list[dict[str, Any]] = []
    if not err_ds.empty and "moment" in err_ds.columns:
        counts_moment_ds = (
            err_ds.groupby("moment")
            .size()
            .reindex(MOMENT_ORDER, fill_value=0)
            .reset_index(name="Nb")
        )
        total_ds_err = int(counts_moment_ds["Nb"].sum())
        if total_ds_err > 0:
            counts_moment_ds["%"] = (counts_moment_ds["Nb"] / total_ds_err * 100).round(2)
            counts_moment_ds = counts_moment_ds[counts_moment_ds["Nb"] > 0]
            counts_moment_ds = pd.concat(
                [
                    counts_moment_ds,
                    pd.DataFrame(
                        [
                            {
                                "moment": "Total",
                                "Nb": total_ds_err,
                                "%": 100.0,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
            ds_moment_distribution = counts_moment_ds.to_dict("records")

        if "moment_avancee" in err_ds.columns:
            counts_ma_ds = (
                err_ds.groupby("moment_avancee")
                .size()
                .reset_index(name="Nb")
                .sort_values("Nb", ascending=False)
            )
            if not counts_ma_ds.empty:
                counts_ma_ds = pd.concat(
                    [
                        counts_ma_ds,
                        pd.DataFrame(
                            [
                                {
                                    "moment_avancee": "Total",
                                    "Nb": int(counts_ma_ds["Nb"].sum()),
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )
                ds_moment_adv_distribution = counts_ma_ds.to_dict("records")

    return templates.TemplateResponse(
        "partials/error_analysis.html",
        {
            "request": request,
        "top_all": top_all,
        "detail_all": detail_all,
        "detail_all_pivot": detail_all_pivot,
        "top_evi": top_evi,
        "detail_evi": detail_evi,
        "detail_evi_pivot": detail_evi_pivot,
        "top_ds": top_ds,
        "detail_ds": detail_ds,
        "detail_ds_pivot": detail_ds_pivot,
            "evi_moment_code": evi_moment_code,
            "evi_moment_code_site": evi_moment_code_site,
            "ds_moment_code": ds_moment_code,
            "ds_moment_code_site": ds_moment_code_site,
            "site_summary": site_summary,
            "error_type_counts": error_type_counts,
            "moment_counts": moment_counts,
            "moment_adv_counts": moment_adv_counts,
            "evi_moment_distribution": evi_moment_distribution,
            "evi_moment_adv_distribution": evi_moment_adv_distribution,
            "ds_moment_distribution": ds_moment_distribution,
            "ds_moment_adv_distribution": ds_moment_adv_distribution,
        },
    )
