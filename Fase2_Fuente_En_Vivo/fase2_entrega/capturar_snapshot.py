"""
capturar_snapshot.py — Captura el dataset COMPLETO del MINSAL para el respaldo.

Contexto (hallazgo del grupo, 17-08-2026): el WAF del MINSAL responde 403 a las
IPs de datacenter (verificado desde máquinas Fly en 2 regiones, incluso con
headers de navegador), pero NUNCA falla desde IPs residenciales/corporativas.
El servidor desplegado, por tanto, no puede consultar en vivo — pero cualquiera
del grupo SÍ puede, desde su computador.

Este script se corre desde un computador normal ANTES de cada `fly deploy`:
descarga todos los registros, los valida y los guarda en `snapshot_minsal.json`
con su fecha de captura. La imagen Docker lo incluye, y `tool_minsal.py` lo usa
como respaldo rotulado ("captura local del <fecha>") cuando la fuente en vivo
no responde. Así el respaldo cubre TODAS las comunas con datos recientes —
idealmente capturados la misma mañana de la demo.

Uso:
    python capturar_snapshot.py
"""
import json
from datetime import datetime
from pathlib import Path

import requests

URL_TURNOS = "https://midas.minsal.cl/farmacia_v2/WS/getLocalesTurnos.php"
DESTINO = Path(__file__).parent / "snapshot_minsal.json"

CAMPOS_MINIMOS = {
    "local_nombre", "comuna_nombre", "local_direccion",
    "funcionamiento_hora_apertura", "funcionamiento_hora_cierre", "fecha",
}


def main() -> None:
    respuesta = requests.get(URL_TURNOS, timeout=15)
    respuesta.raise_for_status()
    datos = respuesta.json()
    validos = [r for r in datos if isinstance(r, dict) and CAMPOS_MINIMOS <= set(r.keys())]
    if not validos:
        raise SystemExit("La respuesta no trae registros válidos — no se guarda nada.")

    contenido = {
        "fecha_captura": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "total": len(validos),
        "filas": validos,
    }
    DESTINO.write_text(json.dumps(contenido, ensure_ascii=False), encoding="utf-8")
    comunas = len({str(r.get("comuna_nombre", "")).strip().upper() for r in validos})
    print(f"Snapshot guardado: {len(validos)} registros · {comunas} comunas · {contenido['fecha_captura']}")
    print(f"→ {DESTINO.name} (recuerda hacer commit + fly deploy para que viaje en la imagen)")


if __name__ == "__main__":
    main()
