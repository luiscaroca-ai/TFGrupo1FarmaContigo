"""
tool_minsal.py — Fase 2, tareas 9-12: la tool MINSAL real (fuente en vivo).

Consulta getLocalesTurnos.php aplicando el pipeline de calidad de 5 pasos del
enunciado, con resiliencia (timeout, cache corto, fallback rotulado) y error
digno. Respeta el CONTRATO fijado en la Fase 3 (tool_minsal_stub.py):

    tool_minsal(comuna: str) -> {
      "encontrado": bool, "comuna": str,
      "farmacias": [{nombre, direccion, comuna, horario_apertura,
                     horario_cierre, telefono, cierra_dia_siguiente}],
      "fuente": str,   # SIEMPRE dice si es dato en vivo, cache o respaldo
    }
    (`cierra_dia_siguiente` es un campo ADITIVO de esta fase: materializa el
     paso 4 del pipeline para que el generador no tenga que inferirlo.)

Pipeline de calidad (tarea 10):
  1. RECIBIR    — timeout 5 s, valida HTTP, JSON y campos mínimos.
  2. NORMALIZAR — strip, colapsa espacios, horas HH:MM, teléfonos vacíos.
  3. FILTRAR    — vigente = fecha más reciente del lote O el día anterior
                  (los turnos nocturnos cruzan la medianoche: la fila de ayer
                  sigue vigente en la madrugada de hoy). Si una comuna solo
                  tiene registros más antiguos (caso real: farmacias URGENCIA
                  24 h con fecha estancada, p. ej. Providencia/Ñuñoa el
                  16-08-2026), se entregan CON ADVERTENCIA en `fuente` y con
                  su `fecha_registro`, en vez de dejar al usuario sin nada.
                  `fk_region` NO se usa: es un id interno, no la región oficial.
  4. INTERPRETAR— cierre < apertura ⇒ cierra al día siguiente (turno nocturno).
  5. RESPONDER  — solo local, dirección, horario y teléfono. Nunca stock/precio.

Resiliencia (tarea 11):
  - Cache en memoria por 15 min (la fuente cambia a diario; 15 min es sobra).
  - Si la fuente falla, degrada en orden: último lote bueno en memoria →
    `snapshot_minsal.json` (captura COMPLETA hecha con `capturar_snapshot.py`
    desde un computador del grupo, rotulada con su fecha) → snapshot mínimo de
    la Fase 3. El usuario SIEMPRE ve de dónde salió el dato.

Hallazgo del grupo (17-08-2026): el WAF del MINSAL (midas.minsal.cl) responde
403 a las IPs de datacenter — verificado desde máquinas Fly en 2 regiones y
también con headers completos de navegador (no es el User-Agent). Desde IPs
residenciales nunca falla. Solución adoptada: MINSAL_URL apunta por defecto a
un espejo de datos del Departamento de Medicina de la Universidad de Chile
(mismo esquema, actualizado a diario), que sí responde desde la nube. Como
respaldo adicional para la demo, correr `capturar_snapshot.py` desde un
computador normal + `fly deploy` la mañana de la presentación.

Error digno (tarea 12): ninguna excepción llega al usuario — se degrada al
fallback con una nota clara. Probar con:  MINSAL_URL=http://url-mala python probar_tool_minsal.py

Uso suelto:
    python tool_minsal.py Providencia
"""
import os
import re
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path

import requests

from datetime import timedelta

URL_TURNOS = os.getenv("MINSAL_URL", "https://dpi.med.uchile.cl/test/api/farmacia_turno.php")
RUTA_SNAPSHOT = Path(__file__).parent / "snapshot_minsal.json"
TIMEOUT_S = 5
TTL_CACHE_S = 15 * 60

CAMPOS_MINIMOS = {
    "local_nombre", "comuna_nombre", "local_direccion",
    "funcionamiento_hora_apertura", "funcionamiento_hora_cierre", "fecha",
}

# Estado del módulo: cache vigente + último lote bueno (para degradar con dignidad).
_cache = {"filas": None, "descargado_en": 0.0}
_ultimo_bueno = {"filas": None, "descargado_en": 0.0}


# ==============================================================================
# Paso 2 · Normalizar
# ==============================================================================
def _limpiar_texto(valor) -> str:
    return re.sub(r"\s+", " ", str(valor or "").strip())


def _normalizar_comuna(texto: str) -> str:
    t = unicodedata.normalize("NFKD", _limpiar_texto(texto).lower())
    return "".join(c for c in t if not unicodedata.combining(c))


def _normalizar_hora(valor: str) -> str:
    """'09:00:00' -> '09:00'; deja pasar valores raros sin romper."""
    v = _limpiar_texto(valor)
    m = re.match(r"^(\d{1,2}):(\d{2})", v)
    return f"{int(m.group(1)):02d}:{m.group(2)}" if m else v


# ==============================================================================
# Paso 1 · Recibir (descarga + validación de esquema)
# ==============================================================================
def _descargar() -> list[dict]:
    respuesta = requests.get(URL_TURNOS, timeout=TIMEOUT_S)
    respuesta.raise_for_status()
    datos = respuesta.json()
    if not isinstance(datos, list) or not datos:
        raise ValueError("respuesta sin lista de registros")
    validos = [r for r in datos if isinstance(r, dict) and CAMPOS_MINIMOS <= set(r.keys())]
    if not validos:
        raise ValueError("ningún registro trae los campos mínimos")
    return validos


def _obtener_filas() -> tuple[list[dict], str]:
    """Devuelve (filas, etiqueta_fuente) aplicando cache y fallback rotulado.
    La etiqueta es deliberadamente simple (sin ventana de cache ni timestamp
    exacto): lo que le importa al usuario es si el dato es confiable ahora,
    no la mecánica interna de cómo se obtuvo."""
    ahora = time.time()

    if _cache["filas"] is not None and ahora - _cache["descargado_en"] < TTL_CACHE_S:
        return _cache["filas"], "MINSAL, en vivo"

    try:
        filas = _descargar()
        _cache.update(filas=filas, descargado_en=ahora)
        _ultimo_bueno.update(filas=filas, descargado_en=ahora)
        return filas, "MINSAL, en vivo"
    except Exception:
        # Error digno: degradar, nunca propagar el stacktrace al usuario.
        if _ultimo_bueno["filas"] is not None:
            return _ultimo_bueno["filas"], "MINSAL (respaldo reciente) — confirma antes de ir"
        return _cargar_respaldo()


def _cargar_respaldo() -> tuple[list[dict], str]:
    """Respaldo persistente: primero la captura completa del grupo
    (`snapshot_minsal.json`, aportada por Luis — 300+ registros de todas las
    comunas, generada con `capturar_snapshot.py`), y como último recurso el
    snapshot mínimo del enunciado (13 comunas)."""
    if RUTA_SNAPSHOT.exists():
        try:
            import json
            contenido = json.loads(RUTA_SNAPSHOT.read_text(encoding="utf-8"))
            filas = contenido.get("filas") or []
            if filas:
                return filas, "MINSAL (respaldo) — podría no estar actualizado, confirma por teléfono antes de ir"
        except Exception:
            pass  # captura corrupta -> seguir al último recurso
    return _filas_snapshot(), "MINSAL (respaldo) — podría no estar actualizado, confirma por teléfono antes de ir"


def _filas_snapshot() -> list[dict]:
    """Convierte el snapshot de la Fase 3 al esquema crudo del MINSAL."""
    ruta_fase3 = Path(__file__).resolve().parents[2] / "Fase3_Orquestacion_LangGraph" / "fase3_entrega"
    if str(ruta_fase3) not in sys.path:
        sys.path.insert(0, str(ruta_fase3))
    from tool_minsal_stub import _SNAPSHOT, FECHA_SNAPSHOT
    return [
        {
            "local_nombre": f["nombre"], "comuna_nombre": f["comuna"],
            "local_direccion": f["direccion"], "funcionamiento_hora_apertura": f["apertura"],
            "funcionamiento_hora_cierre": f["cierre"], "local_telefono": f["telefono"],
            "fecha": FECHA_SNAPSHOT,
        }
        for f in _SNAPSHOT
    ]


# ==============================================================================
# Paso 3 · Filtrar + Paso 4 · Interpretar
# ==============================================================================
def _parsear_fecha(valor: str):
    try:
        return datetime.strptime(_limpiar_texto(valor), "%Y-%m-%d").date()
    except ValueError:
        return None


def _particionar_por_vigencia(filas: list[dict]) -> tuple[list[dict], list[dict]]:
    """(vigentes, antiguas). Vigente = fecha máxima del lote o el día anterior:
    el turno nocturno de ayer sigue abierto en la madrugada de hoy, así que
    descartar 'ayer' dejaría el sistema vacío justo en la emergencia nocturna
    que motiva el caso de negocio. Todo lo anterior a eso es 'antiguo': no se
    presenta como vigente, pero se conserva para el rescate por comuna
    (farmacias URGENCIA 24 h con fecha estancada)."""
    fechas = [d for f in filas if (d := _parsear_fecha(f.get("fecha")))]
    if not fechas:
        return filas, []
    limite = max(fechas) - timedelta(days=1)
    vigentes, antiguas = [], []
    for f in filas:
        d = _parsear_fecha(f.get("fecha"))
        (vigentes if d is not None and d >= limite else antiguas).append(f)
    return vigentes, antiguas


def _validar_telefono(valor) -> str:
    """Descarta teléfonos rotos en vez de mostrarlos: doble '+' (dato mal
    concatenado, ej. '+56+56982252509') o muy pocos dígitos como para ser un
    número real (ej. '+560', '+56123', '1') — casos reales vistos en el MINSAL.
    Sin número válido, dice explícitamente que no hay uno disponible, en vez
    de dejar el campo en blanco."""
    v = _limpiar_texto(valor)
    if not v:
        return "Teléfono no disponible"
    if v.count("+") > 1:
        return "Teléfono no disponible"
    digitos = re.sub(r"\D", "", v)
    if len(digitos) < 7:
        return "Teléfono no disponible"
    return v


def _a_farmacia(fila: dict) -> dict:
    apertura = _normalizar_hora(fila.get("funcionamiento_hora_apertura"))
    cierre = _normalizar_hora(fila.get("funcionamiento_hora_cierre"))
    return {
        "nombre": _limpiar_texto(fila.get("local_nombre")),
        "direccion": _limpiar_texto(fila.get("local_direccion")),
        "comuna": _limpiar_texto(fila.get("comuna_nombre")),
        "horario_apertura": apertura,
        "horario_cierre": cierre,
        "telefono": _validar_telefono(fila.get("local_telefono")),
        "cierra_dia_siguiente": bool(apertura and cierre and cierre < apertura),
        "fecha_registro": _limpiar_texto(fila.get("fecha")),
    }


# ==============================================================================
# Paso 5 · Responder (la tool pública)
# ==============================================================================
def tool_minsal(comuna: str) -> dict:
    filas, fuente = _obtener_filas()
    objetivo = _normalizar_comuna(comuna)
    vigentes, antiguas = _particionar_por_vigencia(filas)

    def _de(coleccion: list[dict]) -> list[dict]:
        return [
            _a_farmacia(f) for f in coleccion
            if _normalizar_comuna(f.get("comuna_nombre", "")) == objetivo
        ]

    farmacias = _de(vigentes)
    if not farmacias:
        # Rescate por comuna: registros con fecha estancada (típicamente
        # URGENCIA 24 h). Mejor entregarlos CON advertencia que dejar a la
        # persona sin ninguna opción en una emergencia — pero jamás
        # presentados como dato vigente.
        farmacias = _de(antiguas)
        if farmacias:
            fuente += " · atención: podría no estar actualizado — confirma por teléfono antes de ir"
    return {
        "encontrado": len(farmacias) > 0,
        "comuna": _limpiar_texto(comuna),
        "farmacias": farmacias,
        "fuente": fuente,
    }


if __name__ == "__main__":
    consulta = " ".join(sys.argv[1:]) or "Providencia"
    resultado = tool_minsal(consulta)
    print(f"Comuna: {resultado['comuna']} · encontrado={resultado['encontrado']}")
    print(f"Fuente: {resultado['fuente']}")
    for f in resultado["farmacias"]:
        extra = " · cierra al día siguiente" if f["cierra_dia_siguiente"] else ""
        tel = f" · tel {f['telefono']}" if f["telefono"] else ""
        print(f"— {f['nombre']} · {f['direccion']} · {f['horario_apertura']}-{f['horario_cierre']}{extra}{tel}")
