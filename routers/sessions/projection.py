from fastapi import APIRouter, Request, Query
from datetime import date
from typing import Any
import pandas as pd
import numpy as np

from db import query_df, LOGV0_BASE_URL
from routers.filters import MOMENT_ORDER
from .common import _build_conditions, _get_vehicle_strategy, _apply_status_filters, _map_moment_label, templates, SITE_COLOR_PALETTE

router = APIRouter(tags=["sessions"])


@router.get("/sessions/projection")
async def get_sessions_projection(
    request: Request,
    sites: str = Query(default=""),
    date_debut: date = Query(default=None),
    date_fin: date = Query(default=None),
    error_types: str | None = Query(default=None),
    moments: str | None = Query(default=None),
    hide_empty: bool = Query(default=False),
):
    error_type_list = None if error_types is None else [e.strip() for e in error_types.split(",") if e.strip()]
    moment_list = None if moments is None else [m.strip() for m in moments.split(",") if m.strip()]

    site_options_df = query_df(
        """
        SELECT DISTINCT Site
        FROM kpi_sessions
        WHERE Site IS NOT NULL
        ORDER BY Site
        """
    )
    site_options = site_options_df["Site"].tolist() if not site_options_df.empty else []

    selected_sites = [s.strip() for s in sites.split(",") if s.strip()] if sites else []

    if not selected_sites:
        return templates.TemplateResponse(
            "partials/projection.html",
            {
                "request": request,
                "site_options": site_options,
                "selected_sites": [],
                "hide_empty": hide_empty,
                "show_prompt": True,
            },
        )

    where_clause, params = _build_conditions(",".join(selected_sites), date_debut, date_fin, table_alias="k")

    sql = f"""
        SELECT
            k.Site,
            k.PDC,
            k.`State of charge(0:good, 1:error)` as state,
            k.type_erreur,
            k.moment,
            k.`EVI Error Code`,
            k.`Downstream Code PC`,
            k.`EVI Status during error`
        FROM kpi_sessions k
        WHERE {where_clause}
    """

    df = query_df(sql, params)

    if df.empty:
        return templates.TemplateResponse(
            "partials/projection.html",
            {
                "request": request,
                "no_data": True,
                "site_options": site_options,
                "selected_sites": selected_sites,
                "hide_empty": hide_empty,
            },
        )

    df["is_ok"] = pd.to_numeric(df["state"], errors="coerce").fillna(0).astype(int).eq(0)
    df = _apply_status_filters(df, error_type_list, moment_list)

    err = df[~df["is_ok_filt"]].copy()
    if err.empty:
        return templates.TemplateResponse(
            "partials/projection.html",
            {
                "request": request,
                "no_errors": True,
                "site_options": site_options,
                "selected_sites": selected_sites,
                "hide_empty": hide_empty,
            },
        )

    evi_step = pd.to_numeric(
        err.get("EVI Status during error", pd.Series(np.nan, index=err.index)),
        errors="coerce",
    )
    evi_code = pd.to_numeric(
        err.get("EVI Error Code", pd.Series(np.nan, index=err.index)), errors="coerce"
    ).fillna(0).astype(int)
    ds_pc = pd.to_numeric(
        err.get("Downstream Code PC", pd.Series(np.nan, index=err.index)), errors="coerce"
    ).fillna(0).astype(int)
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
    sub_evi["step_num"] = evi_step.loc[sub_evi.index]
    sub_evi["code_num"] = evi_code.loc[sub_evi.index]

    sub_ds = err.loc[sub_ds_mask].copy()
    sub_ds["step_num"] = evi_step.loc[sub_ds.index]
    sub_ds["code_num"] = ds_pc.loc[sub_ds.index]

    evi_long = pd.concat([sub_evi, sub_ds], ignore_index=True)

    if evi_long.empty:
        return templates.TemplateResponse(
            "partials/projection.html",
            {
                "request": request,
                "no_errors": True,
                "site_options": site_options,
                "selected_sites": selected_sites,
                "hide_empty": hide_empty,
            },
        )

    evi_long["Site"] = evi_long.get("Site", "").fillna("")
    evi_long["PDC"] = evi_long.get("PDC", "").fillna("").astype(str)
    evi_long["moment_label"] = evi_long["moment_label"].fillna("Unknown")

    unique_moments = evi_long["moment_label"].dropna().unique().tolist()
    moments_sorted = [m for m in MOMENT_ORDER if m in unique_moments]
    moments_sorted += [m for m in sorted(unique_moments) if m not in moments_sorted]

    columns: list[tuple[str, int]] = []
    for m in moments_sorted:
        codes = (
            evi_long.loc[evi_long["moment_label"].eq(m), "code_num"]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )
        for code in sorted(codes):
            columns.append((m, int(code)))

    if not columns:
        return templates.TemplateResponse(
            "partials/projection.html",
            {
                "request": request,
                "no_errors": True,
                "site_options": site_options,
                "selected_sites": selected_sites,
                "hide_empty": hide_empty,
            },
        )

    column_template = pd.MultiIndex.from_tuples(columns, names=["moment", "code"])

    sites_payload: list[dict] = []
    site_list = sorted(evi_long["Site"].dropna().unique().tolist())

    for site in site_list:
        site_rows = evi_long[evi_long["Site"].eq(site)].copy()
        if site_rows.empty:
            continue

        base_site = df[df["Site"] == site].copy()
        total_site = len(base_site)
        ok_site = int(base_site["is_ok"].sum()) if not base_site.empty else 0
        success_rate = round(ok_site / total_site * 100, 1) if total_site else 0.0

        has_pdc = "PDC" in site_rows.columns and site_rows["PDC"].replace("", np.nan).notna().any()

        if has_pdc:
            g_pdc = (
                site_rows.groupby(["PDC", "moment_label", "code_num"]).size().rename("Nb").reset_index()
            )
            g_tot = site_rows.groupby(["moment_label", "code_num"]).size().rename("Nb").reset_index()
            g_tot["PDC"] = "__TOTAL__"
            full = pd.concat([g_tot, g_pdc], ignore_index=True)

            pv = full.pivot_table(
                index="PDC",
                columns=["moment_label", "code_num"],
                values="Nb",
                fill_value=0,
                aggfunc="sum",
            )

            pv = pv.reindex(columns=column_template, fill_value=0)

            pdcs = sorted(pv.index.tolist(), key=str)
            if "__TOTAL__" in pdcs:
                pdcs.remove("__TOTAL__")
                pdcs = ["__TOTAL__"] + pdcs
            pv = pv.reindex(pdcs)

            df_disp = pv.reset_index()
            df_disp["label"] = np.where(
                df_disp["PDC"].eq("__TOTAL__"), f"{site} (TOTAL)", "   " + df_disp["PDC"].astype(str)
            )
            if isinstance(df_disp.columns, pd.MultiIndex):
                df_disp = df_disp.drop(columns=["PDC"], level=0, errors="ignore")
            else:
                df_disp = df_disp.drop(columns=["PDC"], errors="ignore")
        else:
            g_site = site_rows.groupby(["moment_label", "code_num"]).size().rename("Nb").reset_index()
            pv = g_site.pivot_table(
                index=pd.Index([site], name="Site"),
                columns=["moment_label", "code_num"],
                values="Nb",
                fill_value=0,
                aggfunc="sum",
            )
            pv = pv.reindex(columns=column_template, fill_value=0)

            df_disp = pv.reset_index(drop=True)
            df_disp["label"] = f"{site} (TOTAL)"

        all_value_cols = list(column_template)
        numeric_values_all = df_disp[all_value_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

        value_cols = all_value_cols
        if hide_empty:
            value_cols = [
                col for col in all_value_cols
                if (numeric_values_all[col] != 0).any()
            ]

        numeric_values = df_disp[value_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        df_disp["row_total"] = numeric_values_all.sum(axis=1).astype(int)

        total_row_mask = df_disp["label"].astype(str).str.endswith("(TOTAL)")
        if total_row_mask.any():
            total_general_value = int(df_disp.loc[total_row_mask, "row_total"].iloc[0])
            df_disp["row_percent"] = np.where(
                total_general_value > 0,
                np.where(
                    total_row_mask,
                    100.0,
                    (df_disp["row_total"] / total_general_value * 100).round(1),
                ),
                0.0,
            )
        else:
            total_general_value = int(df_disp["row_total"].sum())
            df_disp["row_percent"] = np.where(
                total_general_value > 0,
                (df_disp["row_total"] / total_general_value * 100).round(1),
                0.0,
            )

        def _clean_label(val: Any) -> str:
            if isinstance(val, pd.Series):
                val = val.squeeze()
            if isinstance(val, (list, np.ndarray)):
                val = val[0] if len(val) else ""
            return str(val)

        def _get_scalar(val: Any) -> Any:
            if isinstance(val, pd.Series):
                val = val.iloc[0] if not val.empty else 0
            elif isinstance(val, (list, np.ndarray)):
                val = val[0] if len(val) else 0
            return val

        rows = []
        for _, r in df_disp.iterrows():
            values = [
                int(val) if pd.notna(val) else 0
                for val in (_get_scalar(r[col]) for col in value_cols)
            ]

            total_val = _get_scalar(r["row_total"])
            percent_val = _get_scalar(r["row_percent"])

            rows.append(
                {
                    "label": _clean_label(r["label"]),
                    "values": values,
                    "total": int(total_val) if pd.notna(total_val) else 0,
                    "percent": float(percent_val) if pd.notna(percent_val) else 0.0,
                }
            )

        column_headers = [
            {"moment": moment, "code": code}
            for moment, code in value_cols
        ]

        moment_headers: list[dict] = []
        for moment in moments_sorted:
            span = sum(1 for m, _ in value_cols if m == moment)
            if span:
                moment_headers.append({"moment": moment, "span": span})

        sites_payload.append(
            {
                "site": site,
                "success_rate": success_rate,
                "total_site": total_site,
                "ok_site": ok_site,
                "columns": column_headers,
                "moment_headers": moment_headers,
                "rows": rows,
            }
        )

    if not sites_payload:
        return templates.TemplateResponse(
            "partials/projection.html",
            {
                "request": request,
                "no_errors": True,
                "site_options": site_options,
                "selected_sites": selected_sites,
                "hide_empty": hide_empty,
            },
        )

    return templates.TemplateResponse(
        "partials/projection.html",
        {
            "request": request,
            "sites": sites_payload,
            "site_options": site_options,
            "selected_sites": selected_sites,
            "hide_empty": hide_empty,
        },
    )
