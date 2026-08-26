"""
api_server.py — Avance de la Fase 5 (tareas 20-21): API + front sobre el grafo de Fase 3.

Expone el contrato exigido por la rúbrica (criterio 6):
    POST /chat  {user_id, pregunta}  ->  {respuesta, intencion, motivo_ruteo}
y sirve un front conversacional mínimo en GET / que consume esa API.

Alcance de este avance (decidido con el grupo el 16-08):
  - Sirve para PROBAR el núcleo hasta la Fase 3 de forma conversacional.
  - El flujo de turnos usa el stub MINSAL (snapshot rotulado) hasta que exista la Fase 2.
  - Corre LOCAL. La tarea 22 (deploy cloud) queda pendiente: localhost NO acredita
    deploy para la rúbrica — este archivo es la base que después se dockeriza
    (mismo patrón FastAPI + Dockerfile + Fly.io de Tarea_Final_Deployment).

El historial vive en el checkpointer del grafo (MemorySaver): se conserva mientras
el servidor esté arriba, separado por user_id. Para persistir entre reinicios,
inyectar un SqliteSaver en construir_grafo() (ya soportado).

Uso:
    python -m uvicorn api_server:app --port 8010
    → front en http://127.0.0.1:8010 · Swagger en http://127.0.0.1:8010/docs
"""
import os
import sys
import threading
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv

# Las llaves y modelos viven en el .env de la Fase 3 — cargar ANTES de importar
# el grafo (los nombres de modelo se leen al importar el módulo). En cloud no hay
# .env: las mismas variables llegan como secretos del entorno y load_dotenv de un
# archivo inexistente es un no-op.
BASE = Path(__file__).resolve().parents[2]
RUTA_FASE3 = BASE / "Fase3_Orquestacion_LangGraph" / "fase3_entrega"
RUTA_FASE2 = BASE / "Fase2_Fuente_En_Vivo" / "fase2_entrega"
RUTA_FASE4 = BASE / "Fase4_Seguridad_Guardrails" / "fase4_entrega"
load_dotenv(RUTA_FASE3 / ".env")
for _ruta in (RUTA_FASE3, RUTA_FASE2, RUTA_FASE4):
    if str(_ruta) not in sys.path:
        sys.path.insert(0, str(_ruta))

import json  # noqa: E402

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.responses import FileResponse, StreamingResponse  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from grafo_farmacias import construir_grafo  # noqa: E402
from guardrails import conversar_seguro, revisar_entrada, revisar_salida  # noqa: E402
from tool_minsal import tool_minsal  # noqa: E402  (Fase 2: fuente en vivo)
from historial import (  # noqa: E402
    borrar_historial_usuario,
    depurar_historial_expirado,
    registrar_turno,
    router_historial,
)
from rate_limiting import RateLimiter, exigir_limite  # noqa: E402
from trazabilidad import log_consulta_info  # noqa: E402

VERSION_PRIVACIDAD = "2026-08-25"
TTL_CONVERSACION_S = max(60, int(os.getenv("CONVERSATION_TTL_SECONDS", "86400")))
_ultima_actividad: dict[str, float] = {}
_lock_actividad = threading.Lock()
_limiter = RateLimiter()
LIMITE_CHAT_SESION = max(1, int(os.getenv("RATE_LIMIT_CHAT_PER_MINUTE", "10")))
LIMITE_CHAT_IP = max(LIMITE_CHAT_SESION, int(os.getenv("RATE_LIMIT_CHAT_IP_PER_MINUTE", "30")))
LIMITE_BORRADO = max(1, int(os.getenv("RATE_LIMIT_DELETE_PER_MINUTE", "3")))

app = FastAPI(
    title="Asistente de farmacias — Trabajo Final M04",
    description=(
        "Fases integradas: RAG real (F1) · MINSAL en vivo con fallback rotulado (F2) · "
        "grafo LangGraph con historial por user_id (F3) · guardrails entrada/salida (F4)."
    ),
    version="1.0.0",
)

grafo = construir_grafo(tool_minsal=tool_minsal)  # MINSAL REAL inyectado
app.include_router(router_historial)


class Pregunta(BaseModel):
    user_id: str = Field(min_length=1, max_length=64, description="Hilo de conversación del usuario")
    pregunta: str = Field(min_length=1, max_length=1000, description="Mensaje en lenguaje natural")
    consentimiento_privacidad: bool = Field(description="Aceptación expresa del aviso de privacidad")
    version_privacidad: str = Field(description="Versión del aviso aceptado")


class Respuesta(BaseModel):
    respuesta: str
    intencion: str = Field(description="Ruta que tomó el grafo en este turno")
    motivo_ruteo: str = Field(description="Explicación corta del router (trazabilidad)")
    capa: str = Field(description="Qué capa respondió: input_guard, grafo u output_guard")


def _exigir_consentimiento(p: Pregunta) -> None:
    if not p.consentimiento_privacidad or p.version_privacidad != VERSION_PRIVACIDAD:
        raise HTTPException(
            status_code=428,
            detail="Debes aceptar la versión vigente del aviso de privacidad antes de consultar.",
        )


def _depurar_conversaciones() -> None:
    """Expira hilos inactivos tanto del checkpointer como del registro JSONL."""
    ahora = time.monotonic()
    with _lock_actividad:
        expirados = [u for u, ultimo in _ultima_actividad.items()
                     if ahora - ultimo >= TTL_CONVERSACION_S]
        for user_id in expirados:
            grafo.checkpointer.delete_thread(user_id)
            _ultima_actividad.pop(user_id, None)
    depurar_historial_expirado()


def _registrar_actividad(user_id: str) -> None:
    _depurar_conversaciones()
    with _lock_actividad:
        _ultima_actividad[user_id] = time.monotonic()


def _ip_cliente(request: Request) -> str:
    return request.client.host if request.client else "desconocida"


def _limitar_chat(request: Request, user_id: str) -> None:
    ip = _ip_cliente(request)
    exigir_limite(_limiter, f"chat-sesion:{user_id}", LIMITE_CHAT_SESION, 60)
    exigir_limite(_limiter, f"chat-ip:{ip}", LIMITE_CHAT_IP, 60)


@app.post("/chat", response_model=Respuesta)
def chat(p: Pregunta, request: Request) -> Respuesta:
    _exigir_consentimiento(p)
    _limitar_chat(request, p.user_id)
    _registrar_actividad(p.user_id)
    request_id = uuid.uuid4().hex
    inicio = time.perf_counter()
    log_consulta_info(
        "consulta_recibida", request_id, p.user_id, p.pregunta,
        endpoint="/chat", estado="recibida",
    )
    r = conversar_seguro(grafo, p.user_id, p.pregunta)
    if r["capa"] == "input_guard":
        registrar_turno(user_id=p.user_id, pregunta=p.pregunta, respuesta=r["respuesta"],
                         tool_llamada="ninguna (bloqueada en guardrail de entrada)", fuente="n/a")
    else:
        registrar_turno(user_id=p.user_id, pregunta=p.pregunta, respuesta=r["respuesta"],
                         intencion=r["intencion"])
    log_consulta_info(
        "consulta_completada", request_id, p.user_id, p.pregunta,
        endpoint="/chat", estado="completada", intencion=r["intencion"],
        capa=r["capa"], duracion_ms=round((time.perf_counter() - inicio) * 1000, 2),
    )
    return Respuesta(
        respuesta=r["respuesta"],
        intencion=r["intencion"],
        motivo_ruteo=r["motivo_ruteo"],
        capa=r["capa"],
    )


_ETAPA_POR_INTENCION = {
    "turnos": "📡 Consultando farmacias de turno en el MINSAL…",
    "ficha": "📚 Buscando en el vademécum…",
    "tratamiento": "🛡 Aplicando el protocolo de seguridad…",
    "conversacion": "✍️ Preparando la respuesta…",
    "fuera_de_dominio": "✍️ Preparando la respuesta…",
}


@app.post("/chat/stream")
def chat_stream(p: Pregunta, request: Request) -> StreamingResponse:
    """Igual que /chat, pero emite el AVANCE por etapas (SSE) mientras el grafo
    trabaja: el front muestra qué está pasando y la espera se siente corta.
    Los guardrails de la Fase 4 aplican igual (entrada antes, salida al final)."""

    _exigir_consentimiento(p)
    _limitar_chat(request, p.user_id)
    _registrar_actividad(p.user_id)
    request_id = uuid.uuid4().hex
    inicio = time.perf_counter()
    log_consulta_info(
        "consulta_recibida", request_id, p.user_id, p.pregunta,
        endpoint="/chat/stream", estado="recibida",
    )

    def eventos():
        def sse(objeto: dict) -> str:
            return f"data: {json.dumps(objeto, ensure_ascii=False)}\n\n"

        entrada = revisar_entrada(p.pregunta)
        if not entrada["permitido"]:
            registrar_turno(user_id=p.user_id, pregunta=p.pregunta, respuesta=entrada["respuesta"],
                             tool_llamada="ninguna (bloqueada en guardrail de entrada)", fuente="n/a")
            log_consulta_info(
                "consulta_completada", request_id, p.user_id, p.pregunta,
                endpoint="/chat/stream", estado="completada",
                intencion="bloqueada_en_entrada", capa="input_guard",
                duracion_ms=round((time.perf_counter() - inicio) * 1000, 2),
            )
            yield sse({"tipo": "final", "respuesta": entrada["respuesta"],
                       "intencion": "bloqueada_en_entrada",
                       "motivo_ruteo": entrada["motivo"], "capa": "input_guard"})
            return

        yield sse({"tipo": "etapa", "texto": "🔎 Entendiendo tu pregunta…"})
        config = {"configurable": {"thread_id": p.user_id}}
        intencion, motivo, texto = "desconocida", "", ""
        datos_tool = None
        for actualizacion in grafo.stream(
            {"messages": [HumanMessage(content=p.pregunta)]}, config, stream_mode="updates"
        ):
            for nodo, cambio in actualizacion.items():
                if nodo == "router":
                    intencion = (cambio or {}).get("intencion", "desconocida")
                    motivo = (cambio or {}).get("motivo_ruteo", "")
                    yield sse({"tipo": "etapa",
                               "texto": _ETAPA_POR_INTENCION.get(intencion, "✍️ Preparando la respuesta…")})
                elif nodo in ("turnos", "ficha", "ambos"):
                    datos_tool = (cambio or {}).get("datos_tool")
                    yield sse({"tipo": "etapa", "texto": "✍️ Redactando con los datos obtenidos…"})
                elif nodo in ("generador", "rechazo"):
                    mensajes = (cambio or {}).get("messages") or []
                    if mensajes:
                        texto = mensajes[-1].content

        salida = revisar_salida(texto)
        registrar_turno(user_id=p.user_id, pregunta=p.pregunta, respuesta=salida["respuesta_final"],
                         intencion=intencion, datos_tool=datos_tool)
        capa = "grafo" if salida["permitido"] else "output_guard"
        log_consulta_info(
            "consulta_completada", request_id, p.user_id, p.pregunta,
            endpoint="/chat/stream", estado="completada", intencion=intencion,
            capa=capa, duracion_ms=round((time.perf_counter() - inicio) * 1000, 2),
        )
        yield sse({"tipo": "final", "respuesta": salida["respuesta_final"],
                   "intencion": intencion, "motivo_ruteo": motivo,
                   "capa": capa})

    return StreamingResponse(eventos(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/salud")
def salud() -> dict:
    return {"ok": True}


@app.get("/privacidad")
def privacidad() -> dict:
    return {
        "version": VERSION_PRIVACIDAD,
        "ttl_segundos": TTL_CONVERSACION_S,
        "datos": "user_id seudónimo, preguntas, respuestas, herramienta y fuente consultada",
        "finalidad": "prestar el chat, mantener contexto temporal y generar trazabilidad educativa",
        "terceros": "OpenAI para clasificación/redacción, Qdrant para fichas y MINSAL para turnos",
        "borrado": "Usa el botón Borrar mis datos o DELETE /privacidad/datos/{user_id}",
    }


@app.delete("/privacidad/datos/{user_id}")
def borrar_datos(user_id: str, request: Request) -> dict:
    """Borra el contexto en RAM y los turnos de auditoría asociados al user_id."""
    if not user_id or len(user_id) > 64:
        raise HTTPException(status_code=400, detail="user_id inválido")
    exigir_limite(_limiter, f"borrado-ip:{_ip_cliente(request)}", LIMITE_BORRADO, 60)
    grafo.checkpointer.delete_thread(user_id)
    with _lock_actividad:
        _ultima_actividad.pop(user_id, None)
    borrados = borrar_historial_usuario(user_id)
    _limiter.olvidar(f"chat-sesion:{user_id}")
    return {"ok": True, "user_id": user_id, "registros_borrados": borrados}


@app.get("/", include_in_schema=False)
def portada():
    # no-store: el front evoluciona seguido y un navegador con caché mostraría
    # la versión vieja (nos pasó el 16-08). El HTML pesa poco; siempre fresco.
    return FileResponse(
        Path(__file__).parent / "front" / "index.html",
        headers={"Cache-Control": "no-store, must-revalidate"},
    )
