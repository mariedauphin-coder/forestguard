from .detection import router as detection_router
from .alerts import router as alerts_router

__all__ = ["detection_router", "alerts_router"]
