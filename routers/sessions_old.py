from fastapi import APIRouter, Request, Query
from fastapi.templating import Jinja2Templates
from datetime import date, datetime, timedelta
from typing import Any
from urllib.parse import urlencode
import pandas as pd
import numpy as np
import logging

from db import query_df, LOGV0_BASE_URL
from routers.filters import MOMENT_ORDER

EVI_MOMENT = "EVI Status during error"
EVI_CODE = "EVI Error Code"
DS_PC = "Downstream Code PC"

PHASE_MAP = {
    "Avant charge": {"Init", "Lock Connector", "CableCheck"},
    "Charge": {"Charge"},
    "Fin de charge": {"Fin de charge"},
    "Unknown": {"Unknown"},
}

SITE_COLOR_PALETTE = [
    "#f97316",
    "#a855f7",
    "#eab308",
    "#ef4444",
    "#6366f1",
    "#3b82f6",
    "#06b6d4",
    "#10b981",
    "#0ea5e9",
    "#f43f5e",
]

router = APIRouter(tags=["sessions"])
templates = Jinja2Templates(directory="templates")
logger = logging.getLogger(__name__)

_vehicle_strategy_cache = None


def _get_vehicle_strategy():
    global _vehicle_strategy_cache

    if _vehicle_strategy_cache is not None:
        return _vehicle_strategy_cache

    _vehicle_strategy_cache = ("k.Vehicle", "")

    return _vehicle_strategy_cache


def _calculate_period_dates(period_type: str, date_debut: date | None = None, date_fin: date | None = None):
    """
    Calcule les dates de début et fin en fonction du type de période.
    
    Args:
        period_type: Type de période ("focus_jours", "mois", "j-1", "semaine-1", "toute_periode", "manuel")
        date_debut: Date de début (utilisée uniquement pour "manuel")
        date_fin: Date de fin (utilisée uniquement pour "manuel")
    
    Returns:
        Tuple (date_debut, date_fin)
    """
    today = date.today()
    
    if period_type == "focus_jours":
        # Focus jours : aujourd'hui
        return today, today
    elif period_type == "mois":
        # Mois : premier jour du mois en cours jusqu'à aujourd'hui
        return date(today.year, today.month, 1), today
    elif period_type == "j-1":
        # J-1 : hier
        yesterday = today - timedelta(days=1)
        return yesterday, yesterday
    elif period_type == "semaine-1":
        # Semaine -1 : semaine dernière (lundi à dimanche)
        today_weekday = today.weekday()  # 0 = lundi, 6 = dimanche
        days_since_monday = today_weekday
        last_monday = today - timedelta(days=days_since_monday + 7)
        last_sunday = last_monday + timedelta(days=6)
        return last_monday, last_sunday
    elif period_type == "toute_periode":
        # Toute période : pas de filtre de date
        return None, None
    elif period_type == "manuel":
        # Manuel : utiliser les dates fournies
        return date_debut, date_fin
    else:
        # Par défaut : semaine -1
        today_weekday = today.weekday()
        days_since_monday = today_weekday
        last_monday = today - timedelta(days=days_since_monday + 7)
        last_sunday = last_monday + timedelta(days=6)
        return last_monday, last_sunday


def _build_conditions(
    sites: str,
    date_debut: date | None,
    date_fin: date | None,
    table_alias: str | None = None,
    error_types: list[str] | None = None,
    moments: list[str] | None = None,
):
    conditions = ["1=1"]
    params = {}

    alias = f"{table_alias}." if table_alias else ""
    datetime_col = f"{alias}`Datetime start`"
    site_col = f"{alias}Site"

    if date_debut:
        conditions.append(f"{datetime_col} >= :date_debut")
        params["date_debut"] = str(date_debut)
    if date_fin:
        conditions.append(f"{datetime_col} < DATE_ADD(:date_fin, INTERVAL 1 DAY)")
        params["date_fin"] = str(date_fin)
    if sites:
        site_list = [s.strip() for s in sites.split(",") if s.strip()]
        if site_list:
            placeholders = ",".join([f":site_{i}" for i in range(len(site_list))])
            conditions.append(f"{site_col} IN ({placeholders})")
            for i, s in enumerate(site_list):
                params[f"site_{i}"] = s
    
    # Filtres sur type_erreur
    if error_types:
        error_placeholders = ",".join([f":error_type_{i}" for i in range(len(error_types))])
        conditions.append(f"{alias}type_erreur IN ({error_placeholders})")
        for i, et in enumerate(error_types):
            params[f"error_type_{i}"] = et
    
    # Filtres sur moment
    if moments:
        moment_placeholders = ",".join([f":moment_{i}" for i in range(len(moments))])
        conditions.append(f"{alias}moment IN ({moment_placeholders})")
        for i, m in enumerate(moments):
            params[f"moment_{i}"] = m

    return " AND ".join(conditions), params


def _apply_status_filters(df: pd.DataFrame, error_type_list: list[str], moment_list: list[str]) -> pd.DataFrame:
    df["is_ok"] = pd.to_numeric(df["state"], errors="coerce").fillna(0).astype(int).eq(0)
    mask_nok = ~df["is_ok"]
    mask_type = (
        df["type_erreur"].isin(error_type_list)
        if error_type_list and "type_erreur" in df.columns
        else pd.Series(True, index=df.index)
    )
    mask_moment = (
        df["moment"].isin(moment_list)
        if moment_list and "moment" in df.columns
        else pd.Series(True, index=df.index)
    )
    df["is_ok_filt"] = np.where(mask_nok & mask_type & mask_moment, False, True)
    return df


def _apply_super_filters(
    df: pd.DataFrame,
    charge_type: str = "Toutes",
    energy_max: str = "",
    energy_operator: str = "<=",
    vehicle_type: str = "Tous",
    voltage_800v: str = "Tous",
    pdc_filter: str = "",
    mac_address_filter: str = "",
    pmax_min: str = "",
    pmax_operator: str = ">=",
    duration: str = "",
    duration_operator: str = ">=",
    error_moment: str = "",
) -> pd.DataFrame:
    """
    Applique les super filtres sur le DataFrame.
    
    Args:
        df: DataFrame à filtrer
        charge_type: "Toutes", "OK", "Non OK"
        energy_max: Valeur d'énergie
        energy_operator: ">=" ou "<=" pour l'énergie
        vehicle_type: Type de véhicule
        voltage_800v: "Tous", "800V", "Non 800V"
        pdc_filter: Liste de PDC séparés par virgule
        pmax_min: Valeur de puissance max
        pmax_operator: ">=" ou "<=" pour la puissance
        duration: Valeur de durée en minutes
        duration_operator: ">=" ou "<=" pour la durée
        error_moment: "Avant charge", "Charge", "Fin de charge"
    
    Returns:
        DataFrame filtré
    """
    if df.empty:
        return df
    
    mask = pd.Series(True, index=df.index)
    
    # 1. Filtre Type de charge (OK/NOK)
    if charge_type != "Toutes":
        if "is_ok" in df.columns:
            if charge_type == "OK":
                if df["is_ok"].dtype == bool:
                    mask = mask & (df["is_ok"] == True)
                else:
                    mask = mask & (df["is_ok"] == 1)
            elif charge_type == "Non OK":
                if df["is_ok"].dtype == bool:
                    mask = mask & (df["is_ok"] == False)
                else:
                    mask = mask & (df["is_ok"] == 0)
        elif "state" in df.columns:
            state_num = pd.to_numeric(df["state"], errors="coerce").fillna(1)
            if charge_type == "OK":
                mask = mask & (state_num == 0)
            elif charge_type == "Non OK":
                mask = mask & (state_num == 1)
    
    # 2. Filtre Énergie (>= ou <=)
    if energy_max and energy_max.strip():
        try:
            energy_val = float(energy_max.strip())
            # Normaliser l'opérateur
            # FastAPI décode automatiquement les paramètres URL, donc %3E%3D devient >=
            # Mais on vérifie aussi les versions encodées au cas où
            op = str(energy_operator).strip() if energy_operator else "<="
            
            # Décoder manuellement si nécessaire (au cas où FastAPI ne l'aurait pas fait)
            from urllib.parse import unquote
            op_decoded = unquote(op)
            
            if "Energy (Kwh)" in df.columns:
                energy_col = pd.to_numeric(df["Energy (Kwh)"], errors="coerce")
                
                # Appliquer le filtre selon l'opérateur (vérifier les deux versions)
                # On vérifie d'abord la version décodée, puis la version originale
                if op_decoded == ">=" or op == ">=" or op == "%3E%3D":
                    mask = mask & (energy_col >= energy_val) & energy_col.notna()
                elif op_decoded == "<=" or op == "<=" or op == "%3C%3D":
                    mask = mask & (energy_col <= energy_val) & energy_col.notna()
                else:
                    # Par défaut, utiliser <= pour compatibilité avec l'ancien comportement
                    mask = mask & (energy_col <= energy_val) & energy_col.notna()
            # Si la colonne n'existe pas, on ignore silencieusement
        except (ValueError, TypeError):
            # En cas d'erreur de conversion, on ignore le filtre
            pass
    
    # 3. Filtre Type de véhicule
    if vehicle_type != "Tous" and vehicle_type.strip():
        if "Vehicle" in df.columns:
            mask = mask & (df["Vehicle"] == vehicle_type.strip())
    
    # 4. Filtre 800V (optionnel - si colonne n'existe pas, on ignore)
    if voltage_800v != "Tous":
        voltage_cols = ["charge_900V", "Is800V", "800V", "VoltageClass", "EV_800V", "Voltage"]
        voltage_col = None
        for col in voltage_cols:
            if col in df.columns:
                voltage_col = col
                break
        
        if voltage_col:
            if voltage_800v == "800V":
                # Pour charge_900V: 1 = 900V (donc 800V+), 0 = non 800V
                if voltage_col == "charge_900V":
                    voltage_num = pd.to_numeric(df[voltage_col], errors="coerce")
                    mask = mask & (voltage_num == 1)
                elif df[voltage_col].dtype == bool:
                    mask = mask & (df[voltage_col] == True)
                else:
                    voltage_num = pd.to_numeric(df[voltage_col], errors="coerce")
                    mask = mask & (voltage_num >= 650)
            elif voltage_800v == "Non 800V":
                # Pour charge_900V: 0 = non 800V
                if voltage_col == "charge_900V":
                    voltage_num = pd.to_numeric(df[voltage_col], errors="coerce")
                    mask = mask & (voltage_num == 0)
                elif df[voltage_col].dtype == bool:
                    mask = mask & (df[voltage_col] == False)
                else:
                    voltage_num = pd.to_numeric(df[voltage_col], errors="coerce")
                    mask = mask & (voltage_num < 650)
    
    # 5. Filtre PDC (multi-select)
    if pdc_filter and pdc_filter.strip():
        try:
            pdc_list = [p.strip() for p in pdc_filter.split(",") if p.strip()]
            if pdc_list and "PDC" in df.columns:
                pdc_str = df["PDC"].astype(str)
                mask = mask & pdc_str.isin(pdc_list)
        except Exception:
            pass
    
    # 6. Filtre MAC Address (recherche partielle, insensible à la casse)
    if mac_address_filter and mac_address_filter.strip():
        mac_search = mac_address_filter.strip()
        # Chercher dans la colonne MAC Address (exactement comme dans la requête SQL)
        if "MAC Address" in df.columns:
            try:
                # Remplacer les NaN par des chaînes vides
                mac_series = df["MAC Address"].fillna("")
                # Convertir en string pour être sûr
                mac_str = mac_series.astype(str)
                # Normaliser : enlever les deux-points, tirets et espaces pour la comparaison
                # Cela permet de trouver "48c5" même si l'adresse est "48:C5:..." ou "48-C5-..."
                mac_str_normalized = mac_str.str.replace(":", "", regex=False).str.replace("-", "", regex=False).str.replace(" ", "", regex=False).str.upper()
                mac_search_normalized = mac_search.replace(":", "").replace("-", "").replace(" ", "").upper()
                
                # Recherche partielle (contient) - avec la version normalisée
                mac_mask = mac_str_normalized.str.contains(mac_search_normalized, na=False, regex=False)
                
                # Appliquer le filtre
                mask = mask & mac_mask
            except Exception as e:
                # En cas d'erreur, on ignore le filtre (ne pas bloquer l'affichage)
                pass
    
    # 7. Filtre Puissance max (>= ou <=)
    if pmax_min and pmax_min.strip():
        try:
            pmax_val = float(pmax_min.strip())
            if "Max Power (Kw)" in df.columns:
                pmax_col = pd.to_numeric(df["Max Power (Kw)"], errors="coerce")
                if pmax_operator == ">=":
                    mask = mask & (pmax_col >= pmax_val) & pmax_col.notna()
                else:  # <=
                    mask = mask & (pmax_col <= pmax_val) & pmax_col.notna()
        except (ValueError, TypeError):
            pass
    
    # 7. Filtre Durée (>= ou <=) - la colonne duration n'existe pas encore mais sera ajoutée plus tard
    if duration and duration.strip():
        try:
            duration_val = float(duration.strip())
            # Chercher la colonne duration (peut avoir différents noms)
            duration_cols = ["duration", "Duration", "duration_min", "Duration (min)"]
            duration_col = None
            for col in duration_cols:
                if col in df.columns:
                    duration_col = col
                    break
            
            if duration_col:
                duration_num = pd.to_numeric(df[duration_col], errors="coerce")
                if duration_operator == ">=":
                    mask = mask & (duration_num >= duration_val) & duration_num.notna()
                else:  # <=
                    mask = mask & (duration_num <= duration_val) & duration_num.notna()
            # Si la colonne n'existe pas encore, on ignore silencieusement (sera ajoutée plus tard)
        except (ValueError, TypeError):
            pass
    
    # 8. Filtre Moment d'erreur (uniquement pour les charges en erreur)
    if error_moment and error_moment.strip():
        moment_mapping = {
            "Avant charge": ["Init", "Lock Connector", "CableCheck"],
            "Charge": ["Charge"],
            "Fin de charge": ["Fin de charge"]
        }
        
        if error_moment in moment_mapping:
            moment_values = moment_mapping[error_moment]
            if "moment" in df.columns:
                mask = mask & df["moment"].isin(moment_values)
    
    return df[mask].copy()


def _map_moment_label(val: int) -> str:
    try:
        v = int(val)
    except Exception:
        return "Unknown"

    if v == 0:
        return "Fin de charge"
    if 1 <= v <= 2:
        return "Init"
    if 4 <= v <= 6:
        return "Lock Connector"
    if v == 7:
        return "CableCheck"
    if v == 8:
        return "Charge"
    if v > 8:
        return "Fin de charge"
    return "Unknown"


def _map_phase_label(moment: str | int | float | None) -> str:
    if pd.isna(moment):
        return "Unknown"

    if isinstance(moment, (list, tuple, set)):
        for value in moment:
            mapped = _map_phase_label(value)
            if mapped != "Unknown":
                return mapped
        return "Unknown"

    moment_str = str(moment)

    for phase, moments in PHASE_MAP.items():
        if moment_str in moments:
            return phase

    return "Unknown"


def _build_pivot_table(detail_df: pd.DataFrame, by_site: pd.DataFrame) -> dict[str, Any]:
    if detail_df.empty:
        return {"columns": [], "rows": []}

    # Ensure Site column exists
    if "Site" not in detail_df.columns:
        return {"columns": [], "rows": []}

    pivot_df = detail_df.assign(
        _site=detail_df["Site"],
        _type=detail_df.get("type", ""),
        _moment=detail_df["moment_label"],
        _step=detail_df["step"],
        _code=detail_df["code"],
    )

    pivot_table = pd.pivot_table(
        pivot_df,
        index="_site",
        columns=["_type", "_moment", "_step", "_code"],
        aggfunc="size",
        fill_value=0,
    ).sort_index(axis=1)

    # Reset index and handle column names properly
    pivot_table = pivot_table.reset_index()

    # Flatten column names if they are MultiIndex
    if isinstance(pivot_table.columns, pd.MultiIndex):
        pivot_table.columns = [
            col[0] if col[0] in ["_site", "Site"] else " | ".join(str(c) for c in col if c).strip()
            for col in pivot_table.columns
        ]

    # Rename _site to Site
    if "_site" in pivot_table.columns:
        pivot_table = pivot_table.rename(columns={"_site": "Site"})

    # Process column names to format tuples with " | " separator
    new_columns = []
    for col in pivot_table.columns:
        if col == "Site":
            new_columns.append("Site")
        elif isinstance(col, tuple):
            new_columns.append(" | ".join(map(str, col)).strip())
        else:
            new_columns.append(str(col))
    pivot_table.columns = new_columns

    # Verify Site column exists before merging
    if "Site" not in pivot_table.columns:
        return {"columns": [], "rows": []}

    if "Site" not in by_site.columns:
        by_site = by_site.reset_index()

    pivot_table = pivot_table.merge(
        by_site[["Site", "Total_Charges"]].rename(columns={"Total_Charges": "Total Charges"}),
        on="Site",
        how="left",
    )

    # 5. Réorganiser les colonnes
    ordered_columns = ["Site", "Total Charges"] + [
        col for col in pivot_table.columns if col not in {"Site", "Total Charges"}
    ]
    pivot_table = pivot_table[ordered_columns].fillna(0)

    # 6. Convertir les colonnes numériques en int
    numeric_cols = [col for col in pivot_table.columns if col != "Site"]
    pivot_table[numeric_cols] = pivot_table[numeric_cols].astype(int)

    return {
        "columns": pivot_table.columns.tolist(),
        "rows": pivot_table.to_dict("records"),
    }


def _comparaison_base_context(
    request: Request,
    filters: dict,
    site_focus: str = "",
    month_focus: str = "",
    error_message: str | None = None,
):
    return {
        "request": request,
        "site_rows": [],
        "count_bars": [],
        "percent_bars": [],
        "max_total": 0,
        "peak_rows": [],
        "heatmap_rows": [],
        "heatmap_hours": [],
        "heatmap_max": 0,
        "site_options": [],
        "site_focus": site_focus,
        "month_options": [],
        "month_focus": month_focus,
        "monthly_rows": [],
        "daily_rows": [],
        "filters": filters,
        "error_message": error_message,
    }


@router.get("/sessions/stats")
async def get_sessions_stats(
    request: Request,
    sites: str = Query(default=""),
    date_debut: date = Query(default=None),
    date_fin: date = Query(default=None),
    error_types: str = Query(default=""),
    moments: str = Query(default=""),
):
    """
    Retourne les statistiques complètes des sessions (énergie, puissance, SOC, durées, etc.)
    """
    error_type_list = [e.strip() for e in error_types.split(",") if e.strip()] if error_types else []
    moment_list = [m.strip() for m in moments.split(",") if m.strip()] if moments else []

    where_clause, params = _build_conditions(sites, date_debut, date_fin, table_alias="k")

    # Récupération complète des données avec toutes les colonnes nécessaires
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

    # Conversion des types
    for col in ["Datetime start", "Datetime end"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    for col in ["Energy (Kwh)", "Mean Power (Kw)", "Max Power (Kw)", "SOC Start", "SOC End"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Déterminer is_ok
    df["is_ok_raw"] = pd.to_numeric(df["state"], errors="coerce").fillna(0).astype(int).eq(0)

    # Filtrage par type d'erreur et moment
    df = _apply_status_filters(df, error_type_list, moment_list)

    # Séparer OK et NOK
    ok_mask = df["is_ok_filt"]
    nok_mask = ~df["is_ok_filt"]

    ok_df = df[ok_mask].copy()
    nok_df = df[nok_mask].copy()

    # === STATISTIQUES D'ÉNERGIE ===
    energy_all = pd.to_numeric(df.get("Energy (Kwh)", pd.Series(dtype=float)), errors="coerce")
    e_total_all = round(float(energy_all.sum(skipna=True)), 3) if energy_all.notna().any() else 0

    # Pour moyenne et max : utiliser uniquement les charges OK
    energy_ok = pd.to_numeric(ok_df.get("Energy (Kwh)", pd.Series(dtype=float)), errors="coerce")
    e_mean = round(float(energy_ok.mean(skipna=True)), 3) if energy_ok.notna().any() else 0
    e_max = round(float(energy_ok.max(skipna=True)), 3) if energy_ok.notna().any() else 0

    # === STATISTIQUES DE PUISSANCE MOYENNE ===
    pmean_ok = pd.to_numeric(ok_df.get("Mean Power (Kw)", pd.Series(dtype=float)), errors="coerce")
    pm_mean = round(float(pmean_ok.mean(skipna=True)), 3) if pmean_ok.notna().any() else 0
    pm_max = round(float(pmean_ok.max(skipna=True)), 3) if pmean_ok.notna().any() else 0

    # === STATISTIQUES DE PUISSANCE MAXIMALE ===
    pmax_ok = pd.to_numeric(ok_df.get("Max Power (Kw)", pd.Series(dtype=float)), errors="coerce")
    px_mean = round(float(pmax_ok.mean(skipna=True)), 3) if pmax_ok.notna().any() else 0
    px_max = round(float(pmax_ok.max(skipna=True)), 3) if pmax_ok.notna().any() else 0

    # === STATISTIQUES SOC ===
    soc_start = pd.to_numeric(ok_df.get("SOC Start", pd.Series(dtype=float)), errors="coerce")
    soc_end = pd.to_numeric(ok_df.get("SOC End", pd.Series(dtype=float)), errors="coerce")
    soc_start_mean = round(float(soc_start.mean(skipna=True)), 2) if soc_start.notna().any() else 0
    soc_end_mean = round(float(soc_end.mean(skipna=True)), 2) if soc_end.notna().any() else 0

    if soc_start.notna().any() and soc_end.notna().any():
        soc_gain_mean = round(float((soc_end - soc_start).mean(skipna=True)), 2)
    else:
        soc_gain_mean = 0

    # === DURÉES DE CHARGE ===
    dt_start = pd.to_datetime(ok_df.get("Datetime start"), errors="coerce")
    dt_end = pd.to_datetime(ok_df.get("Datetime end"), errors="coerce")
    durations = (dt_end - dt_start).dt.total_seconds() / 60  # minutes
    dur_mean = round(float(durations.mean(skipna=True)), 1) if durations.notna().any() else 0

    # === CHARGES PAR SITE ===
    if not ok_df.empty:
        ok_df["day"] = pd.to_datetime(ok_df["Datetime start"]).dt.date
        charges_by_site_day = (
            ok_df.groupby(["Site", "day"])
            .size()
            .reset_index(name="Nb")
        )

        # Statistiques par jour
        daily_stats = charges_by_site_day.groupby("day")["Nb"].sum().reset_index()
        nb_days = len(daily_stats)
        mean_day = round(float(daily_stats["Nb"].mean()), 2) if nb_days else 0
        med_day = round(float(daily_stats["Nb"].median()), 2) if nb_days else 0

        # Max par jour
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

    # === DURÉES DE FONCTIONNEMENT PAR SITE ===
    dur_source = ok_df.copy()

    # Appliquer le filtre moment pour les durées afin de respecter le filtrage côté interface
    if moment_list and "moment" in dur_source.columns:
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

    # Grouper par site pour le sélecteur
    durations_by_site_dict = {}
    for row in durations_by_pdc_raw:
        site = row["Site"]
        if site not in durations_by_site_dict:
            durations_by_site_dict[site] = []
        durations_by_site_dict[site].append({
            "PDC": row["PDC"],
            "Heures": row["Heures"]
        })

    # Trier les PDC par durée décroissante dans chaque site
    for site in durations_by_site_dict:
        durations_by_site_dict[site] = sorted(
            durations_by_site_dict[site],
            key=lambda x: x["Heures"],
            reverse=True
        )

    # Conserver l'ordre des sites affichés dans le tableau principal
    site_options_order = [row["Site"] for row in durations_by_site]

    # === STATISTIQUES PAR TYPE DE VÉHICULE ===
    vehicle_stats = []
    vehicle_debug_info = {
        "has_column": "Vehicle" in df.columns,
        "total_rows": len(df),
        "non_null_count": 0,
        "valid_count": 0,
        "unknown_count": 0,
    }

    if "Vehicle" in df.columns and not df.empty:
        # Nettoyer les données Vehicle
        df_vehicle = df.copy()
        df_vehicle["Vehicle"] = df_vehicle["Vehicle"].astype(str).str.strip()

        # Compter les non-NULL avant nettoyage
        vehicle_debug_info["non_null_count"] = int(df_vehicle["Vehicle"].notna().sum())

        df_vehicle["Vehicle"] = df_vehicle["Vehicle"].replace(
            {"": "Unknown", "nan": "Unknown", "none": "Unknown", "NULL": "Unknown", "None": "Unknown"},
            regex=False
        )
        df_vehicle["Vehicle"] = df_vehicle["Vehicle"].fillna("Unknown")

        # Compter les Unknown
        vehicle_debug_info["unknown_count"] = int((df_vehicle["Vehicle"] == "Unknown").sum())

        # Exclure les véhicules inconnus
        df_vehicle = df_vehicle[df_vehicle["Vehicle"] != "Unknown"]
        vehicle_debug_info["valid_count"] = len(df_vehicle)

        if not df_vehicle.empty:
            # Grouper par véhicule et calculer les statistiques
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

            # Trier par taux de réussite décroissant puis par volume décroissant
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
            # Énergie
            "e_total_all": e_total_all,
            "e_mean": e_mean,
            "e_max": e_max,
            # Puissance moyenne
            "pm_mean": pm_mean,
            "pm_max": pm_max,
            # Puissance maximale
            "px_mean": px_mean,
            "px_max": px_max,
            # SOC
            "soc_start_mean": soc_start_mean,
            "soc_end_mean": soc_end_mean,
            "soc_gain_mean": soc_gain_mean,
            # Durées
            "dur_mean": dur_mean,
            # Charges par jour
            "nb_days": nb_days,
            "mean_day": mean_day,
            "med_day": med_day,
            "max_day_site": max_day_site,
            "max_day_date": max_day_date,
            "max_day_nb": max_day_nb,
            # Durées de fonctionnement
            "durations_by_site": durations_by_site,
            "durations_by_site_dict": durations_by_site_dict,
            "site_options_dur": site_options_order if site_options_order else list(durations_by_site_dict.keys()),
            # Statistiques par véhicule
            "vehicle_stats": vehicle_stats,
            "vehicle_debug_info": vehicle_debug_info,
        }
    )


@router.get("/sessions/projection")
async def get_sessions_projection(
    request: Request,
    sites: str = Query(default=""),
    date_debut: date = Query(default=None),
    date_fin: date = Query(default=None),
    error_types: str = Query(default=""),
    moments: str = Query(default=""),
    hide_empty: bool = Query(default=False),
):
    error_type_list = [e.strip() for e in error_types.split(",") if e.strip()] if error_types else []
    moment_list = [m.strip() for m in moments.split(",") if m.strip()] if moments else []

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
            # Avoid MultiIndex drop performance warning by targeting the column level explicitly
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


@router.get("/sessions/error-analysis")
async def get_error_analysis(
    request: Request,
    sites: str = Query(default=""),
    date_debut: date = Query(default=None),
    date_fin: date = Query(default=None),
    error_types: str = Query(default=""),
    moments: str = Query(default=""),
):
    error_type_list = [e.strip() for e in error_types.split(",") if e.strip()] if error_types else []
    moment_list = [m.strip() for m in moments.split(",") if m.strip()] if moments else []

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


@router.get("/sessions/general")
async def get_sessions_general(
    request: Request,
    sites: str = Query(default=""),
    date_debut: date = Query(default=None),
    date_fin: date = Query(default=None),
    error_types: str = Query(default=""),
    moments: str = Query(default=""),
):
    error_type_list = [e.strip() for e in error_types.split(",") if e.strip()] if error_types else []
    moment_list = [m.strip() for m in moments.split(",") if m.strip()] if moments else []

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


@router.get("/sessions/comparaison")
async def get_sessions_comparaison(
    request: Request,
    sites: str = Query(default=""),
    date_debut: date = Query(default=None),
    date_fin: date = Query(default=None),
    error_types: str = Query(default=""),
    moments: str = Query(default=""),
    site_focus: str = Query(default=""),
    month_focus: str = Query(default=""),
):
    error_type_list = [e.strip() for e in error_types.split(",") if e.strip()] if error_types else []
    moment_list = [m.strip() for m in moments.split(",") if m.strip()] if moments else []

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
    except Exception as exc:  # pragma: no cover - defensive fallback for UI visibility
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
    error_types: str = Query(default=""),
    moments: str = Query(default=""),
    site_focus: str = Query(default=""),
    month_focus: str = Query(default=""),
    day_focus: str = Query(default=""),
    pdc: str = Query(default=""),
):
    error_type_list = [e.strip() for e in error_types.split(",") if e.strip()] if error_types else []
    moment_list = [m.strip() for m in moments.split(",") if m.strip()] if moments else []

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
            `State of charge(0:good, 1:error)` as state
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

    mask_type = df["type_erreur"].isin(error_type_list) if error_type_list and "type_erreur" in df.columns else True
    mask_moment = df["moment"].isin(moment_list) if moment_list and "moment" in df.columns else True
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
        if error_type_list and "type_erreur" in df_site.columns
        else pd.Series(True, index=df_site.index)
    )
    mask_moment_site = (
        df_site["moment"].isin(moment_list)
        if moment_list and "moment" in df_site.columns
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

                # Fill potential missing occurrences before casting to int to avoid
                # pandas IntCastingNaNError when moments are absent for a Code_PC.
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

                # Fill potential missing occurrences before casting to int to avoid
                # pandas IntCastingNaNError when moments are absent for an EVI code.
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

    # Zoom site et périodicité (même logique que comparaison)
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

    # Nombre de prises actives par heure - Utilisation des nouvelles tables KPI
    hourly_active_sessions = []
    hourly_sessions_count = []  # Nombre de sessions uniques par heure (nombre de clients)
    day_focus_value = ""
    day_options: list[str] = []
    usage_duration_by_level: list[dict] = []  # Temps cumulé par niveau d'utilisation
    
    # Récupérer les dates disponibles depuis kpi_concurrency_15min pour le site sélectionné
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
                
                # Sélectionner la date : celle fournie, ou la dernière disponible
                if day_focus and day_focus in day_options:
                    day_focus_value = day_focus
                elif day_options:
                    day_focus_value = day_options[0]  # Dernière date par défaut
        except Exception:
            pass
    
    # Récupérer les données depuis kpi_concurrency_15min pour la date sélectionnée
    if site_value and day_focus_value:
        try:
            day_focus_date = pd.to_datetime(day_focus_value).date()
            day_start = pd.Timestamp.combine(day_focus_date, pd.Timestamp.min.time())
            
            # Récupérer les données 15min depuis kpi_concurrency_15min
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
                # Si aucune donnée, créer des entrées vides toutes les heures
                for hour in range(24):
                    hourly_active_sessions.append({
                        "time": f"{hour:02d}:00",
                        "hour": hour,
                        "minute": 0,
                        "minutes_from_start": hour * 60,
                        "hour_label": f"{hour:02d}:00",
                        "count": 0,
                    })
            
            # Récupérer le temps cumulé par niveau depuis kpi_concurrency_Daily
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
            
            # Récupérer toutes les sessions avec leur heure de démarrage pour calculer le cumul
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
                # Créer un dictionnaire pour les sessions par heure
                sessions_by_hour = {int(row["hour"]): int(row["session_count"]) for _, row in df_sessions_hourly.iterrows()}
                
                # Calculer le cumul : pour chaque heure, compter toutes les sessions qui ont démarré jusqu'à cette heure
                cumulative_count = 0
                for hour in range(24):
                    # Ajouter les sessions qui ont démarré à cette heure
                    cumulative_count += sessions_by_hour.get(hour, 0)
                    minutes_from_start = hour * 60
                    hourly_sessions_count.append({
                        "hour": hour,
                        "minutes_from_start": minutes_from_start,
                        "hour_label": f"{hour:02d}:00",
                        "count": cumulative_count,
                    })
            else:
                # Si aucune donnée, créer des entrées vides toutes les heures
                for hour in range(24):
                    hourly_sessions_count.append({
                        "hour": hour,
                        "minutes_from_start": hour * 60,
                        "hour_label": f"{hour:02d}:00",
                        "count": 0,
                    })
        except Exception as e:
            # En cas d'erreur, créer des entrées vides
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


@router.get("/sessions/tableau-charges")
async def get_tableau_charges(
    request: Request,
    # Nouveaux paramètres pour les super filtres
    sites: str = Query(default=""),
    period_type: str = Query(default="semaine-1"),  # Par défaut: semaine -1
    date_debut: date = Query(default=None),
    date_fin: date = Query(default=None),
    error_types: str = Query(default=""),
    moments: str = Query(default=""),
    moment_avancee: str = Query(default=""),
    # Super filtres existants
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
    """Tableau chronologique de toutes les charges."""
    # Calculer les dates en fonction du type de période
    calculated_date_debut, calculated_date_fin = _calculate_period_dates(
        period_type, date_debut, date_fin
    )
    
    # Utiliser les dates calculées si elles sont disponibles, sinon utiliser celles fournies
    final_date_debut = calculated_date_debut if calculated_date_debut else date_debut
    final_date_fin = calculated_date_fin if calculated_date_fin else date_fin
    
    # Parser les listes de filtres
    error_type_list = [e.strip() for e in error_types.split(",") if e.strip()] if error_types else []
    moment_list = [m.strip() for m in moments.split(",") if m.strip()] if moments else []
    moment_avancee_list = [m.strip() for m in moment_avancee.split(",") if m.strip()] if moment_avancee else []
    site_list = [s.strip() for s in sites.split(",") if s.strip()] if sites else []
    
    # Décoder manuellement energy_operator si nécessaire
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

    # Construire les conditions SQL avec tous les filtres
    where_clause, params = _build_conditions(
        sites, final_date_debut, final_date_fin,
        error_types=error_type_list if error_type_list else None,
        moments=moment_list if moment_list else None
    )
    
    # Ajouter le filtre moment_avancee si nécessaire
    if moment_avancee_list:
        moment_avancee_placeholders = ",".join([f":moment_avancee_{i}" for i in range(len(moment_avancee_list))])
        where_clause += f" AND moment_avancee IN ({moment_avancee_placeholders})"
        for i, ma in enumerate(moment_avancee_list):
            params[f"moment_avancee_{i}"] = ma

    # Récupérer les listes pour les dropdowns (avant le test df.empty)
    # Sites
    sites_sql = "SELECT DISTINCT Site FROM kpi_sessions WHERE Site IS NOT NULL ORDER BY Site"
    sites_df = query_df(sites_sql)
    site_options = sites_df["Site"].tolist() if not sites_df.empty else []
    
    # Error types
    error_types_sql = "SELECT DISTINCT type_erreur FROM kpi_sessions WHERE type_erreur IS NOT NULL ORDER BY type_erreur"
    error_types_df = query_df(error_types_sql)
    error_type_options = sorted(error_types_df["type_erreur"].dropna().unique().tolist()) if not error_types_df.empty else []
    
    # Moments
    moments_sql = "SELECT DISTINCT moment FROM kpi_sessions WHERE moment IS NOT NULL"
    moments_df = query_df(moments_sql)
    raw_moments = moments_df["moment"].dropna().unique().tolist() if not moments_df.empty else []
    moment_options = [m for m in MOMENT_ORDER if m in raw_moments]
    moment_options.extend([m for m in raw_moments if m not in MOMENT_ORDER])
    
    # Moments avancés
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
            type_erreur,
            moment,
            moment_avancee,
            `Downstream Code PC`,
            `EVI Error Code`,
            `EVI Status during error`,
            `MAC Address`,
            charge_900V
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
                    "Lien": "Lien",
                    "Actions": "Actions",
                    "Erreur": "Erreur"
                },
            },
        )
    
    # Convertir les colonnes numériques
    if "Energy (Kwh)" in df.columns:
        df["Energy (Kwh)"] = pd.to_numeric(df["Energy (Kwh)"], errors="coerce")
    if "Max Power (Kw)" in df.columns:
        df["Max Power (Kw)"] = pd.to_numeric(df["Max Power (Kw)"], errors="coerce")
    
    # Créer is_ok pour les super filtres (les filtres error_types et moments sont déjà dans la requête SQL)
    df["is_ok"] = pd.to_numeric(df["state"], errors="coerce").fillna(0).astype(int).eq(0)
    
    # Appliquer les super filtres (qui incluent le filtre charge_type OK/NOK)
    # Les filtres globaux (error_types et moments) sont déjà appliqués dans la requête SQL
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
    
    # Préparer les lignes
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
            "statut": "✅ OK" if (int(row.get("is_ok", 1)) == 1) else "🔴 NOK",
            "type_erreur": str(row.get("type_erreur", "")) if pd.notna(row.get("type_erreur")) else "",
            "moment": str(row.get("moment", "")) if pd.notna(row.get("moment")) else "",
            "moment_avancee": str(row.get("moment_avancee", "")) if pd.notna(row.get("moment_avancee")) else "",
            "downstream_code_pc": str(row.get("Downstream Code PC", "")) if pd.notna(row.get("Downstream Code PC")) else "",
            "evi_error_code": str(row.get("EVI Error Code", "")) if pd.notna(row.get("EVI Error Code")) else "",
            "evi_status_during_error": str(row.get("EVI Status during error", "")) if pd.notna(row.get("EVI Status during error")) else "",
            "mac_address": str(row.get("MAC Address", "")) if pd.notna(row.get("MAC Address")) else "",
            "charge_900V": str(row.get("charge_900V", "")) if pd.notna(row.get("charge_900V")) else "",
            "url": f"https://elto.nidec-asi-online.com/Charge/detail?id={row.get('ID', '')}" if pd.notna(row.get("ID")) else "",
            "logv0_url": f"{LOGV0_BASE_URL}?session_id={row.get('ID', '')}&project=ELTO" if LOGV0_BASE_URL and pd.notna(row.get("ID")) else None,
        }
        rows.append(row_dict)
    
    total = len(rows)
    ok = sum(1 for r in rows if r.get("is_ok", 1) == 1)
    nok = total - ok
    taux_reussite = round(ok / total * 100, 1) if total else 0
    taux_echec = round(nok / total * 100, 1) if total else 0
    
    # Options de véhicules
    vehicle_options = ["Tous"]
    if "Vehicle" in df.columns:
        vehicles = sorted(df["Vehicle"].dropna().unique().tolist())
        vehicle_options.extend(vehicles)
    
    # Gérer les colonnes visibles
    # Colonnes par défaut (celles actuellement affichées)
    default_columns = [
        "Site", "PDC", "Datetime start", "Datetime end", "Energy (Kwh)",
        "SOC Start", "SOC End", "Vehicle", "is_ok", "Lien", "Actions", "Erreur"
    ]
    
    # Toutes les colonnes disponibles avec leurs labels d'affichage
    # Les clés doivent correspondre aux clés du row_dict
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
    
    # Parser visible_columns (format: "col1,col2,col3" ou vide pour utiliser les défauts)
    if visible_columns and visible_columns.strip():
        visible_columns_list = [col.strip() for col in visible_columns.split(",") if col.strip()]
    else:
        visible_columns_list = default_columns.copy()
    
    # S'assurer que les colonnes essentielles sont toujours présentes
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