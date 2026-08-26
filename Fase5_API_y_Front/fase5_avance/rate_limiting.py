"""Rate limiting de ventana deslizante, local al proceso y seguro entre hilos."""
from __future__ import annotations

import math
import threading
import time
from collections import defaultdict, deque
from typing import Callable, Deque, Dict

from fastapi import HTTPException


class RateLimiter:
    """Registra timestamps por clave y devuelve segundos de espera, o cero."""

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._eventos: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def comprobar(self, clave: str, limite: int, ventana_s: int) -> int:
        ahora = self._clock()
        umbral = ahora - ventana_s
        with self._lock:
            eventos = self._eventos[clave]
            while eventos and eventos[0] <= umbral:
                eventos.popleft()
            if len(eventos) >= limite:
                return max(1, math.ceil(ventana_s - (ahora - eventos[0])))
            eventos.append(ahora)
            return 0

    def olvidar(self, prefijo: str) -> None:
        """Elimina contadores asociados a una sesión borrada."""
        with self._lock:
            for clave in [k for k in self._eventos if k.startswith(prefijo)]:
                self._eventos.pop(clave, None)


def exigir_limite(limiter: RateLimiter, clave: str, limite: int, ventana_s: int) -> None:
    espera = limiter.comprobar(clave, limite, ventana_s)
    if espera:
        raise HTTPException(
            status_code=429,
            detail="Demasiadas solicitudes. Intenta nuevamente más tarde.",
            headers={"Retry-After": str(espera)},
        )
