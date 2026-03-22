"""
Health alert API routes.
Endpoints for evaluating cattle health, viewing alerts, and triggering notifications.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional
from app.auth import get_current_user, require_role, require_farm_access
from app.alert_models import EvaluationResult, AlertResponse, AlertSummaryResponse
from app.alert_services import (
    evaluate_cattle_health,
    evaluate_all_cattle,
    get_alerts_for_cattle,
    get_recent_alerts,
    get_alert_counter,
)
from app.services import get_cattle_metadata

alert_router = APIRouter(
    prefix="/api/v1/alerts",
    tags=["Health Alerts"],
    dependencies=[Depends(get_current_user)],
)


@alert_router.post("/evaluate/{cid}", response_model=EvaluationResult)
async def evaluate_single_cattle(
    cid: int,
    user: Optional[dict] = Depends(require_role(["admin"])),
):
    """
    Evaluate health status for a specific cattle.
    Processes latest sensor data, updates consecutive counter,
    and triggers alerts/emails if thresholds are met.
    Admin or API key required.
    """
    cattle = await get_cattle_metadata(cid)
    if not cattle:
        raise HTTPException(status_code=404, detail=f"Cattle with cid {cid} not found")
    if user:
        require_farm_access(user, cattle.get("farm_id", ""))

    result = await evaluate_cattle_health(cid)
    return result


@alert_router.post("/evaluate-all", response_model=AlertSummaryResponse)
async def evaluate_all(
    user: Optional[dict] = Depends(require_role(["admin"])),
):
    """
    Evaluate health for all registered cattle.
    Returns summary with individual results.
    Admin or API key required.
    """
    result = await evaluate_all_cattle()
    return result


@alert_router.get("/{cid}", response_model=list[AlertResponse])
async def get_cattle_alerts(
    cid: int,
    limit: int = Query(default=50, ge=1, le=500),
    user: Optional[dict] = Depends(get_current_user),
):
    """Get alert history for a specific cattle, newest first."""
    cattle = await get_cattle_metadata(cid)
    if not cattle:
        raise HTTPException(status_code=404, detail=f"Cattle with cid {cid} not found")
    if user and user.get("farm_ids"):
        require_farm_access(user, cattle.get("farm_id", ""))

    alerts = await get_alerts_for_cattle(cid, limit)
    if not alerts:
        raise HTTPException(status_code=404, detail=f"No alerts found for cattle {cid}")
    return alerts


@alert_router.get("/recent/all", response_model=list[AlertResponse])
async def get_recent_alert_list(
    limit: int = Query(default=50, ge=1, le=500),
    user: Optional[dict] = Depends(get_current_user),
):
    """Get recent alerts across all cattle, newest first. Dashboard alert panel."""
    alerts = await get_recent_alerts(limit)
    if not alerts:
        raise HTTPException(status_code=404, detail="No alerts found")
    return alerts


@alert_router.get("/{cid}/counter")
async def get_counter_status(
    cid: int,
    user: Optional[dict] = Depends(get_current_user),
):
    """Get the current consecutive bad-reading counter for a cattle."""
    counter = await get_alert_counter(cid)
    return counter
