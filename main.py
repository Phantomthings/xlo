from fastapi import Depends, FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse, JSONResponse
from contextlib import asynccontextmanager

from db import engine, get_sites, get_date_range
from routers import defauts, alertes, kpis, overview, filters, mac_address, report
from routers.auth import get_current_user, router as auth_router
from routers.sessions import (
    projection_router,
    error_analysis_router,
    general_router,
    comparaison_router,
    site_details_router,
    tableau_charges_router,
)
from routers.sessions.stats import router as stats_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Démarrage ELTO Dashboard")
    yield
    engine.dispose()
    print("Arrêt")

app = FastAPI(title="ELTO Dashboard", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/assets", StaticFiles(directory="assets"), name="assets")

@app.exception_handler(HTTPException)
async def redirect_unauthorized(request: Request, exc: HTTPException):
    if exc.status_code == 401 and request.url.path not in ["/", "/logout"]:
        return RedirectResponse(url="/", status_code=302)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


templates = Jinja2Templates(directory="templates")

app.include_router(auth_router)

protected = [Depends(get_current_user)]
app.include_router(filters.router, prefix="/api", dependencies=protected)
app.include_router(overview.router, prefix="/api", dependencies=protected)
app.include_router(defauts.router, prefix="/api", dependencies=protected)
app.include_router(alertes.router, prefix="/api", dependencies=protected)
app.include_router(stats_router, prefix="/api", dependencies=protected)
app.include_router(projection_router, prefix="/api", dependencies=protected)
app.include_router(error_analysis_router, prefix="/api", dependencies=protected)
app.include_router(general_router, prefix="/api", dependencies=protected)
app.include_router(comparaison_router, prefix="/api", dependencies=protected)
app.include_router(site_details_router, prefix="/api", dependencies=protected)
app.include_router(tableau_charges_router, prefix="/api", dependencies=protected)
app.include_router(kpis.router, prefix="/api", dependencies=protected)
app.include_router(mac_address.router, prefix="/api", dependencies=protected)
app.include_router(report.router, prefix="/api", dependencies=protected)
@app.get("/dashboard")

async def dashboard_redirect():
    return RedirectResponse(url="/dashboard/overview", status_code=302)

@app.get("/dashboard/{tab}")
async def dashboard_tab(
    request: Request,
    tab: str,
    current_user: dict = Depends(get_current_user)
):
    valid_tabs = ['overview', 'general', 'details-site', 'comparaison', 'stats', 
                  'analyse-erreur', 'code-analysis', 'projection', 'mac-address',
                  'tentatives', 'suspectes', 'alertes', 'evolution', 'historique', 'tableau-charges', 'rapport']
    
    if tab not in valid_tabs:
        return RedirectResponse(url="/dashboard/overview", status_code=302)
    
    sites = get_sites()
    date_range = get_date_range()
    
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "user": current_user,
            "sites": sites,
            "date_min": date_range["min"],
            "date_max": date_range["max"],
            "initial_tab": tab,
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)