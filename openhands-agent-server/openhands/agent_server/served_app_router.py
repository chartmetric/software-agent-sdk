from fastapi import APIRouter, Depends

from openhands.agent_server.served_app_service import (
    ServedApp,
    ServedAppService,
    get_served_app_service,
)


served_app_router = APIRouter(prefix="/served-apps", tags=["Served Apps"])


@served_app_router.get("")
async def list_served_apps(
    service: ServedAppService = Depends(get_served_app_service),
) -> list[ServedApp]:
    return service.list_apps()
