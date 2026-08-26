"""
historial.py — Fase 5/6: registro de auditoría de preguntas y respuestas.

El profesor lo pidió explícitamente: los logs de Fly (genéricos, pensados
para depurar) no bastan como evidencia de trazabilidad. Este módulo deja
un registro propio, con la estructura exacta que pidió:

    pregunta → tool llamada → respuesta → fuente

Cada turno queda con fecha/hora REAL del servidor, en un archivo aparte,
con un endpoint para verlo o descargarlo en cualquier momento (incluida
la demo misma).

USO (agregar a api_server.py, 2 líneas):

    from historial import registrar_turno, router_historial
    app.include_router(router_historial)

    # Dentro de cada endpoint, justo después de invocar el grafo:
    registrar_turno(
        user_id=p.user_id, pregunta=p.pregunta, respuesta=texto_respuesta,
        intencion=estado["intencion"], datos_tool=estado.get("datos_tool"),
    )

    # Si el mensaje fue bloqueado por el guardrail de ENTRADA (nunca llegó
    # al grafo), no hay "estado" — usa esta forma en su lugar:
    registrar_turno(
        user_id=p.user_id, pregunta=p.pregunta, respuesta=texto_rechazo,
        tool_llamada="ninguna (bloqueada en guardrail de entrada)", fuente="n/a",
    )

El módulo deriva solo "tool_llamada" y "fuente" a partir de la intención
y los datos crudos que ya devuelve el grafo — no hay que armarlos a mano
en cada punto de integración.

LIMITACIÓN HONESTA: el archivo vive en el disco del contenedor de Fly,
que es efímero — si la máquina se reinicia, el historial acumulado se
pierde (mismo problema que el historial de conversación en RAM, ya
documentado). Ver la nota al final del archivo para hacerlo persistente
con un Volume de Fly, si queda tiempo.
"""
from __future__ import annotations

import json
import os
import secrets
import threading
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse

RUTA_HISTORIAL = Path(__file__).parent / "historial_preguntas_respuestas.jsonl"
TTL_HISTORIAL_S = max(60, int(os.getenv("CONVERSATION_TTL_SECONDS", "86400")))
_lock = threading.Lock()  # evita que dos requests simultáneos se pisen al escribir

def _exigir_clave_admin(
    x_admin_key: Optional[str] = Header(default=None),
    clave: Optional[str] = None,
) -> None:
    """Acepta la clave por header (X-Admin-Key, para llamadas programáticas)
    o por parámetro de URL (?clave=..., para poder pegar el link directo en
    el navegador durante la demo)."""
    esperada = os.getenv("HISTORIAL_ADMIN_KEY", "")
    if not esperada:
        raise HTTPException(status_code=503, detail="Historial administrativo no habilitado")
    recibida = x_admin_key or clave
    if not recibida or not secrets.compare_digest(recibida, esperada):
        raise HTTPException(status_code=401, detail="Clave administrativa inválida")


router_historial = APIRouter(dependencies=[Depends(_exigir_clave_admin)])


def _tool_y_fuente(intencion: str, datos_tool: dict) -> tuple[str, str]:
    """A partir de la intención del router y los datos crudos que devolvió
    el grafo, determina qué tool se llamó y cuál fue la fuente citada."""
    datos_tool = datos_tool or {}

    if intencion == "turnos":
        return "tool_minsal", datos_tool.get("fuente", "sin datos")

    if intencion == "ficha":
        fichas = datos_tool.get("fichas", [])
        fuente = "; ".join(f.get("cita", "") for f in fichas) if fichas else "sin datos"
        return "tool_rag", fuente

    if intencion == "ambos":
        turnos = datos_tool.get("turnos", {}) or {}
        ficha = datos_tool.get("ficha", {}) or {}
        fuente_turnos = turnos.get("fuente", "sin datos")
        fichas = ficha.get("fichas", [])
        fuente_ficha = "; ".join(f.get("cita", "") for f in fichas) if fichas else "sin datos"
        return "tool_minsal + tool_rag", f"MINSAL: {fuente_turnos} | Vademécum: {fuente_ficha}"

    if intencion == "tratamiento":
        return "ninguna (rechazo determinista, sin LLM)", "n/a"

    # conversacion, fuera_de_dominio, u otra intención sin tool
    return "ninguna", "n/a"


def registrar_turno(
    user_id: str,
    pregunta: str,
    respuesta: str,
    intencion: str = "",
    datos_tool: dict | None = None,
    tool_llamada: str | None = None,
    fuente: str | None = None,
) -> None:
    """Agrega una línea al historial, con la estructura pregunta -> tool
    llamada -> respuesta -> fuente. Nunca lanza excepción hacia el
    llamador — si falla el registro, no debe romper la respuesta al usuario.

    Dos formas de uso:
    - Pasando intencion + datos_tool (turno normal, procesado por el grafo):
      tool_llamada y fuente se derivan automáticamente.
    - Pasando tool_llamada + fuente directo (ej. bloqueado por el guardrail
      de entrada, donde nunca hubo intención ni datos_tool)."""
    if tool_llamada is None or fuente is None:
        tool_llamada, fuente = _tool_y_fuente(intencion, datos_tool)

    entrada = {
        "fecha_hora": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "user_id": user_id,
        "pregunta": pregunta,
        "tool_llamada": tool_llamada,
        "respuesta": respuesta,
        "fuente": fuente,
    }
    try:
        with _lock:
            _depurar_expirados_sin_lock()
            with open(RUTA_HISTORIAL, "a", encoding="utf-8") as f:
                f.write(json.dumps(entrada, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[historial] no se pudo registrar el turno: {e!r}")


def _fecha_registro(valor: str) -> datetime | None:
    try:
        # Tolera también el formato antiguo incorrecto "+00:00Z".
        limpio = valor[:-1] if valor.endswith("Z") else valor
        fecha = datetime.fromisoformat(limpio)
        return fecha.replace(tzinfo=timezone.utc) if fecha.tzinfo is None else fecha
    except (TypeError, ValueError):
        return None


def _depurar_expirados_sin_lock() -> int:
    if not RUTA_HISTORIAL.exists():
        return 0
    limite = datetime.now(timezone.utc) - timedelta(seconds=TTL_HISTORIAL_S)
    vigentes = []
    total = 0
    with open(RUTA_HISTORIAL, encoding="utf-8") as f:
        for linea in f:
            try:
                turno = json.loads(linea)
                total += 1
                fecha = _fecha_registro(turno.get("fecha_hora", ""))
                if fecha is not None and fecha >= limite:
                    vigentes.append(turno)
            except json.JSONDecodeError:
                continue
    with open(RUTA_HISTORIAL, "w", encoding="utf-8") as f:
        for turno in vigentes:
            f.write(json.dumps(turno, ensure_ascii=False) + "\n")
    return total - len(vigentes)


def depurar_historial_expirado() -> int:
    """Elimina registros cuya antigüedad supera el TTL configurado."""
    with _lock:
        return _depurar_expirados_sin_lock()


def borrar_historial_usuario(user_id: str) -> int:
    """Elimina del registro de auditoría todos los turnos del usuario."""
    with _lock:
        _depurar_expirados_sin_lock()
        if not RUTA_HISTORIAL.exists():
            return 0
        turnos = _leer_todo_sin_lock()
        vigentes = [t for t in turnos if t.get("user_id") != user_id]
        with open(RUTA_HISTORIAL, "w", encoding="utf-8") as f:
            for turno in vigentes:
                f.write(json.dumps(turno, ensure_ascii=False) + "\n")
        return len(turnos) - len(vigentes)


def _leer_todo_sin_lock() -> list[dict]:
    if not RUTA_HISTORIAL.exists():
        return []
    turnos = []
    with open(RUTA_HISTORIAL, encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if linea:
                try:
                    turnos.append(json.loads(linea))
                except json.JSONDecodeError:
                    pass  # línea corrupta -> se ignora, no rompe el resto
    return turnos


def _leer_todo() -> list[dict]:
    with _lock:
        _depurar_expirados_sin_lock()
        return _leer_todo_sin_lock()


@router_historial.get("/historial")
def ver_historial():
    """Devuelve todos los turnos registrados, más recientes primero."""
    turnos = _leer_todo()
    return {"total": len(turnos), "turnos": list(reversed(turnos))}


@router_historial.get("/historial/descargar", response_class=PlainTextResponse)
def descargar_historial():
    """Devuelve el archivo crudo (JSON Lines), para guardarlo como evidencia."""
    if not RUTA_HISTORIAL.exists():
        return "Sin registros todavía."
    return RUTA_HISTORIAL.read_text(encoding="utf-8")


_COLOR_TOOL = {
    "tool_minsal": ("#e3edf9", "#2b5b8c", "#5b8bc4"),
    "tool_rag": ("#e3edf9", "#2b5b8c", "#5b8bc4"),
    "tool_minsal + tool_rag": ("#e8ddfb", "#5b3fa0", "#8b6cc9"),
}


def _color_tool(tool_llamada: str) -> tuple[str, str, str]:
    """(fondo, texto, borde) según el tipo de tool — gris/ámbar para lo bloqueado."""
    if tool_llamada in _COLOR_TOOL:
        return _COLOR_TOOL[tool_llamada]
    if "bloquead" in tool_llamada or "rechazo" in tool_llamada:
        return ("#fbeecd", "#8a5a12", "#b8860b")
    return ("#eef1f5", "#475569", "#94a3b8")


def _fuente_html_si_hace_falta(t: dict) -> str:
    """La respuesta de 'turnos' ya trae su propia línea '📄 Fuente: ...' (parte
    del texto redactado por el generador) — mostrarla de nuevo aparte sería
    duplicada. Para 'ficha' o casos bloqueados, el texto no siempre repite el
    mismo formato, así que ahí sí vale la pena mostrarla como campo aparte."""
    fuente = t.get("fuente", "")
    respuesta = t.get("respuesta", "")
    if not fuente or fuente == "n/a":
        return ""
    if "📄 Fuente:" in respuesta:
        return ""  # ya está en el texto de la respuesta, no repetir
    return f'<div class="fuente">📄 Fuente: {escape(fuente)}</div>'


@router_historial.get("/historial/ver", response_class=HTMLResponse)
def ver_historial_html():
    """Vista visual del historial — la misma evidencia que /historial, pero
    legible de un vistazo (para mostrar en la demo, no solo para exportar)."""
    turnos = list(reversed(_leer_todo()))

    filas = ""
    for t in turnos:
        bg, fg, bd = _color_tool(t.get("tool_llamada", ""))
        filas += f"""
        <div class="turno">
          <div class="cabecera">
            <span class="fecha">{escape(t.get('fecha_hora', ''))}</span>
            <span class="badge" style="background:{bg};color:{fg};border:1px solid {bd}">
              {escape(t.get('tool_llamada', ''))}
            </span>
          </div>
          <div class="pregunta">❓ {escape(t.get('pregunta', ''))}</div>
          <div class="respuesta">💊 {escape(t.get('respuesta', '')).replace(chr(10), '<br>')}</div>
          {_fuente_html_si_hace_falta(t)}
        </div>"""

    if not turnos:
        filas = '<p class="vacio">Todavía no hay preguntas registradas.</p>'

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="15">
<title>Historial · Asistente de farmacias</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, "Segoe UI", Arial, sans-serif; background: #f8fafc;
         margin: 0; padding: 0 0 40px; color: #1e293b; }}
  header {{ background: #0f766e; color: white; padding: 20px 32px; }}
  header h1 {{ margin: 0; font-size: 20px; }}
  header p {{ margin: 4px 0 0; font-size: 13px; opacity: 0.85; }}
  .contenedor {{ max-width: 820px; margin: 24px auto; padding: 0 16px; }}
  .total {{ color: #64748b; font-size: 13px; margin-bottom: 16px; }}
  .turno {{ background: white; border: 1px solid #e2e8f0; border-radius: 12px;
           padding: 16px 20px; margin-bottom: 14px; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }}
  .cabecera {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }}
  .fecha {{ font-size: 12px; color: #94a3b8; }}
  .badge {{ font-size: 11px; font-weight: 700; padding: 3px 10px; border-radius: 999px; }}
  .pregunta {{ font-weight: 700; margin-bottom: 8px; }}
  .respuesta {{ color: #334155; font-size: 14px; line-height: 1.5; margin-bottom: 8px; }}
  .fuente {{ font-size: 12px; color: #64748b; border-top: 1px dashed #e2e8f0; padding-top: 8px; }}
  .vacio {{ text-align: center; color: #94a3b8; padding: 60px 0; }}
</style>
</head>
<body>
  <header>
    <h1>📋 Historial de preguntas y respuestas</h1>
    <p>FarmaContigo — evidencia de trazabilidad, con fecha y hora reales del servidor</p>
  </header>
  <div class="contenedor">
    <div class="total">{len(turnos)} turnos registrados · se actualiza solo cada 15 s</div>
    {filas}
  </div>
</body>
</html>"""
    return html


# ==============================================================================
# NOTA — cómo hacerlo persistente de verdad (opcional, si queda tiempo):
#
# 1. Crear un volumen en Fly (una sola vez):
#      fly volumes create historial_data --size 1 --region gru
#
# 2. Agregar a fly.toml:
#      [mounts]
#        source = "historial_data"
#        destination = "/data"
#
# 3. Cambiar la línea de arriba:
#      RUTA_HISTORIAL = Path("/data/historial_preguntas_respuestas.jsonl")
#
# Con esto, el archivo sobrevive aunque la máquina se reinicie o redespliegue.
# ==============================================================================
