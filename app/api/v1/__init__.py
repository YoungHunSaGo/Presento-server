from fastapi import APIRouter

from app.api.v1 import analyses, auth, health, mypage, presentations

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(mypage.router)
api_router.include_router(presentations.router)
api_router.include_router(analyses.router)
