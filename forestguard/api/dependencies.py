"""Shared FastAPI dependencies: model singleton, in-memory alert store."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Annotated

import torch
from fastapi import Depends

from ..models.fusion_model import ForestGuardModel
from ..postprocessing.alerts import AlertGenerator

logger = logging.getLogger(__name__)

# In-memory alert registry (replace with a DB in production)
_alert_store: list[dict] = []
_model_ready: bool = False


@lru_cache(maxsize=1)
def _load_model() -> ForestGuardModel:
    global _model_ready
    checkpoint = os.environ.get("FORESTGUARD_CHECKPOINT", "")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if checkpoint and os.path.exists(checkpoint):
        logger.info("Loading model from %s on %s", checkpoint, device)
        model = ForestGuardModel.from_checkpoint(checkpoint)
    else:
        logger.warning("No checkpoint found — initialising untrained model.")
        model = ForestGuardModel(pretrained=False)

    model.eval()
    _model_ready = True
    return model.to(device)


def is_model_ready() -> bool:
    return _model_ready


@lru_cache(maxsize=1)
def _load_alert_generator() -> AlertGenerator:
    return AlertGenerator(
        confidence_threshold=float(os.environ.get("CONFIDENCE_THRESHOLD", "0.5")),
        min_area_ha=float(os.environ.get("MIN_AREA_HA", "1.0")),
    )


def get_model() -> ForestGuardModel:
    return _load_model()


def get_alert_generator() -> AlertGenerator:
    return _load_alert_generator()


def get_alert_store() -> list[dict]:
    return _alert_store


ModelDep = Annotated[ForestGuardModel, Depends(get_model)]
AlertGenDep = Annotated[AlertGenerator, Depends(get_alert_generator)]
AlertStoreDep = Annotated[list, Depends(get_alert_store)]
