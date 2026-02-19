
from fastapi import APIRouter, Request, Query
from fastapi.templating import Jinja2Templates
from datetime import datetime
import pandas as pd

from db import query_df

router = APIRouter(tags=["defauts"])
templates = Jinja2Templates(directory="templates")


@router.get("/defauts-actifs")
async def get_defauts_actifs(
    request: Request,
    sites: str = Query(default="", description="Sites séparés par virgule"),
):
    sql = """
        SELECT
            site,
            date_debut,
            defaut,
            eqp
        FROM kpi_defauts_log
        WHERE date_fin IS NULL
        ORDER BY date_debut DESC
    """
    
    df = query_df(sql)
    
    if sites:
        site_list = [s.strip() for s in sites.split(",") if s.strip()]
        if site_list:
            df = df[df["site"].isin(site_list)]
    
    nb_defauts = len(df)
    nb_sites = df["site"].nunique() if not df.empty else 0
    
    if nb_defauts > 5:
        status = "danger"
    elif nb_defauts > 0:
        status = "warning"
    else:
        status = "success"
    
    defauts_list = []
    if not df.empty:
        df["date_debut"] = pd.to_datetime(df["date_debut"], errors="coerce")
        now = pd.Timestamp.now()
        
        for _, row in df.iterrows():
            delta_days = (now - row["date_debut"]).days if pd.notna(row["date_debut"]) else 0
            is_recent = delta_days < 1
            
            defauts_list.append({
                "site": row["site"],
                "defaut": row["defaut"],
                "eqp": row["eqp"],
                "depuis_jours": delta_days,
                "is_recent": is_recent,
                "card_class": "critical" if delta_days > 7 else "warning",
            })
    
    sites_recent = list(set(d["site"] for d in defauts_list if d["is_recent"]))
    
    defauts_par_site = {}
    for d in defauts_list:
        site = d["site"]
        if site not in defauts_par_site:
            defauts_par_site[site] = []
        defauts_par_site[site].append(d)
    
    return templates.TemplateResponse(
        "partials/defauts_actifs.html",
        {
            "request": request,
            "nb_defauts": nb_defauts,
            "nb_sites": nb_sites,
            "status": status,
            "sites_recent": sites_recent,
        "defauts_par_site": defauts_par_site,
    }
    )


@router.get("/defauts-historique")
async def get_defauts_historique(
    request: Request,
    sites: str = Query(default="", description="Sites séparés par virgule"),
    date_debut: str | None = Query(default=None, description="Date de début (YYYY-MM-DD)"),
    date_fin: str | None = Query(default=None, description="Date de fin (YYYY-MM-DD)"),
):

    params: dict[str, object] = {}
    where_clauses: list[str] = []

    if date_debut and date_fin:
        start_date = pd.to_datetime(date_debut, errors="coerce")
        end_date = pd.to_datetime(date_fin, errors="coerce")

        if not pd.isna(start_date) and not pd.isna(end_date):
            d1 = start_date.normalize()
            d2 = (end_date + pd.Timedelta(days=1)).normalize()
            params["d1"] = d1
            params["d2"] = d2
            where_clauses.append(
                "((date_debut >= :d1 AND date_debut < :d2) "
                "OR (date_fin >= :d1 AND date_fin < :d2) "
                "OR (date_debut <= :d1 AND (date_fin >= :d2 OR date_fin IS NULL)))"
            )

    sql = """
        SELECT
            site,
            date_debut,
            date_fin,
            defaut,
            eqp
        FROM kpi_defauts_log
    """

    if where_clauses:
        sql += " WHERE " + " AND ".join(where_clauses)

    sql += " ORDER BY date_debut DESC"

    df = query_df(sql, params=params)

    if sites:
        site_list = [s.strip() for s in sites.split(",") if s.strip()]
        if site_list:
            df = df[df["site"].isin(site_list)]

    if not df.empty:
        df["date_debut"] = pd.to_datetime(df["date_debut"], errors="coerce")
        df["date_fin"] = pd.to_datetime(df["date_fin"], errors="coerce")
        now = pd.Timestamp.now()
        df["duree_jours"] = ((df["date_fin"].fillna(now)) - df["date_debut"]).dt.days
        df["statut"] = df["date_fin"].apply(lambda x: "En cours" if pd.isna(x) else "Résolu")

    nb_total = len(df)
    if not df.empty and "statut" in df.columns:
        nb_en_cours = int((df["statut"] == "En cours").sum())
        nb_resolus = int((df["statut"] == "Résolu").sum())
    else:
        nb_en_cours = 0
        nb_resolus = 0

    duree_moyenne = df["duree_jours"].mean() if not df.empty else 0

    rows = []
    if not df.empty:
        df_sorted = df.sort_values(by="date_debut", ascending=False)
        for _, row in df_sorted.iterrows():
            rows.append(
                {
                    "site": row.get("site") or "—",
                    "date_debut": row["date_debut"].strftime("%Y-%m-%d %H:%M") if pd.notna(row["date_debut"]) else "-",
                    "date_fin": row["date_fin"].strftime("%Y-%m-%d %H:%M") if pd.notna(row["date_fin"]) else "En cours",
                    "duree": int(row.get("duree_jours", 0)),
                    "statut": row.get("statut", "-"),
                    "defaut": row.get("defaut") or "-",
                    "eqp": row.get("eqp") or "-",
                }
            )

    top_equipements = []
    top_defauts = []
    if not df.empty:
        top_equipements = [
            {"eqp": eqp, "count": int(count)} for eqp, count in df["eqp"].value_counts().head(5).items()
        ]
        top_defauts = [
            {"defaut": defaut, "count": int(count)}
            for defaut, count in df["defaut"].value_counts().head(5).items()
        ]

    return templates.TemplateResponse(
        "partials/defauts_historique.html",
        {
            "request": request,
            "rows": rows,
            "nb_total": nb_total,
            "nb_en_cours": nb_en_cours,
            "nb_resolus": nb_resolus,
            "duree_moyenne": round(duree_moyenne, 1) if duree_moyenne else 0,
            "top_equipements": top_equipements,
            "top_defauts": top_defauts,
        },
    )