"""Logs JSON de trazabilidad enviados a stdout para su captura por Fly.io."""
from __future__ import annotations

import logging
import sys
from typing import Any

import structlog


def configurar_logging() -> None:
    """Configuración idempotente: un objeto JSON por línea, nivel INFO."""
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp"),
            structlog.processors.JSONRenderer(ensure_ascii=False),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.WriteLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )


configurar_logging()
logger = structlog.get_logger("asistente_farmacias")


def log_consulta_info(
    evento: str,
    request_id: str,
    user_id: str,
    pregunta: str,
    **resultado: Any,
) -> None:
    """Emite el contrato demostrable en Fly: metadata + resultado del turno."""
    logger.info(
        evento,
        servicio="asistente_farmacias",
        metadata={
            "request_id": request_id,
            "user_id": user_id,
            "pregunta": str(pregunta).replace("\n", " ")[:1000],
        },
        **resultado,
    )
