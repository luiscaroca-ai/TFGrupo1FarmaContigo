"""
grafo_farmacias.py — Fase 3, tareas 13-15: la orquestación LangGraph del asistente.

El grafo decide, según la intención de cada pregunta, si consulta la tool MINSAL
(farmacias de turno), la tool RAG (fichas del vademécum), responde directo o
rechaza con derivación segura. Mantiene historial multi-turno por `user_id`
mediante un checkpointer: la misma persona puede preguntar "¿y en Providencia?"
sin repetir el contexto.

Decisiones de diseño (justificación completa en Decisiones_Grafo_Fase3.md):
  - El RUTEO es un paso LLM barato con salida estructurada (Pydantic): además de
    la intención emite una CONSULTA AUTÓNOMA (la pregunta reescrita con el
    contexto del historial) — así las tools reciben una pregunta completa aunque
    el usuario haya escrito solo "¿y en Providencia?".
  - Ante la duda entre "ficha" y "tratamiento", el router debe elegir
    "tratamiento" (fail-closed): es preferible rechazar de más que recomendar
    un medicamento (condición dura del criterio 5).
  - El RECHAZO clínico es DETERMINISTA (texto fijo, sin LLM): en seguridad, un
    control probabilístico que acierta el 99% deja pasar justo el 1% (Módulo 04:
    "en application security, 99% es nota de reprobación"). La Fase 4 (Luis)
    agrega los guardrails de entrada/salida como capas adicionales.
  - Las tools NO redactan y el generador NO busca: cada tool devuelve datos
    estructurados con su cita, y el nodo generador arma la respuesta final.
  - Todo es INYECTABLE (LLMs, tools, checkpointer): el grafo se puede probar
    sin red (test_grafo_fakes.py) y las Fases 2 (MINSAL real), 4 (guardrails)
    y 5 (API) enchufan sus piezas sin tocar este archivo.

Contrato hacia la API (Fase 5):
    grafo = construir_grafo()
    respuesta = conversar(grafo, user_id="u-123", pregunta="¿farmacia de turno en Temuco?")
    # El thread_id del checkpointer ES el user_id → historial por usuario.

Uso como script: ver probar_grafo.py (requiere OPENAI_API_KEY; la ruta de
fichas requiere además las llaves de Qdrant y la colección de la Fase 1).
"""
import json
import re
import os
import sys
from pathlib import Path
from typing import Annotated, Literal, Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

load_dotenv()

# Modelos del curso; se pueden cambiar por variable de entorno sin tocar código.
MODELO_ROUTER = os.getenv("MODELO_ROUTER", "gpt-5.4-nano")        # gate barato
MODELO_GENERADOR = os.getenv("MODELO_GENERADOR", "gpt-5.4-mini")  # redacción final


# ==============================================================================
# Estado del grafo
# ==============================================================================
class EstadoAsistente(TypedDict):
    """Lo que persiste el checkpointer por `user_id` (thread_id).

    `messages` acumula el historial completo de la conversación (add_messages).
    Los demás campos son "de trabajo" del turno actual: el router los REESCRIBE
    al inicio de cada turno, para que nunca quede basura del turno anterior.
    """
    messages: Annotated[list[AnyMessage], add_messages]
    intencion: str
    comuna: Optional[str]
    consulta_autonoma: str
    motivo_ruteo: str
    datos_tool: Optional[dict]


class DecisionRuteo(BaseModel):
    """Salida estructurada del router — es también la evidencia de la tarea 15:
    cada turno queda con su intención y el motivo de la decisión."""

    intencion: Literal["turnos", "ficha", "ambos", "tratamiento", "conversacion", "fuera_de_dominio"] = Field(
        description="Categoría del último mensaje del usuario."
    )
    comuna: Optional[str] = Field(
        default=None,
        description="Comuna chilena aludida en el último mensaje o heredada del historial; null si no hay.",
    )
    consulta_autonoma: str = Field(
        description=(
            "El último mensaje reescrito como pregunta completa y autónoma usando el "
            "historial. Ej.: '¿y en Providencia?' tras hablar de turnos → "
            "'¿Qué farmacias están de turno en Providencia?'"
        )
    )
    motivo: str = Field(description="Una frase corta que explica la clasificación.")


# ==============================================================================
# Prompts
# ==============================================================================
PROMPT_ROUTER = """Eres el clasificador de intención de un asistente informativo de farmacias en Chile.
Clasifica el ÚLTIMO mensaje del usuario; usa el historial solo como contexto.

Intenciones posibles:
- "turnos": pide farmacias de turno o abiertas, direcciones, horarios o teléfonos de locales.
  Incluye seguimientos que dependen del historial ("¿y en Providencia?", "¿y el horario de la segunda?").
  Reconoce TODAS estas formas de pedirlo, no solo "farmacia de turno en X":
  "hay algo abierto en X", "farmacia cerca de X", "algo abierto cerca de X", "farmacia por X",
  "farmacia abierta cerca de X", "hay alguna farmacia en la zona de X".
- "ficha": pregunta información general de un medicamento que ya conoce o le indicaron
  (para qué sirve, efectos secundarios, interacciones, conservación, categoría de embarazo).
- "ambos": el mensaje pide turnos Y ficha a la vez (usando cualquiera de las formas de arriba
  para turnos), Y nombra explícitamente el medicamento
  (ej. "¿la loratadina sirve para la alergia y hay farmacia en Quilpué?",
  "¿el paracetamol sirve para la fiebre y hay algo abierto cerca de Providencia?").
  Nombrar el medicamento no es ambiguo: el usuario ya eligió de qué habla, no le estás sugiriendo nada.
- "tratamiento": pide que le recomienden un medicamento, una dosis, qué tomar para un síntoma,
  o un diagnóstico — SIN nombrar un medicamento existente (ej. "me duele la cabeza, ¿hay algo
  abierto y qué me tomo?"). REGLA DURA: ante la duda entre "ficha"/"ambos" y "tratamiento",
  si el mensaje NO nombra un medicamento existente, elige "tratamiento".
- "conversacion": saludos, despedidas, agradecimientos, o preguntas sobre qué sabe hacer el asistente.
- "fuera_de_dominio": cualquier otro tema (deportes, tareas, programación, etc.).

Además entrega:
- comuna: la comuna chilena mencionada en el último mensaje o heredada del historial; null si no hay.
  Se menciona de muchas formas, todas válidas: "en Providencia", "cerca de Providencia",
  "por Providencia", "en la comuna de Providencia", "en la zona de Providencia" — en todas estas
  la comuna es "Providencia". Extrae el nombre propio de la comuna aunque la preposición no sea "en".
- consulta_autonoma: el último mensaje reescrito como pregunta completa y autónoma según el historial.
- motivo: una frase corta explicando tu decisión."""

PROMPT_GENERADOR = """Eres el asistente informativo de farmacias del proyecto (Chile). Informas; NO tratas.

REGLAS DURAS (ninguna excepción):
1. Responde SOLO con lo que aparece en "DATOS DE LA CONSULTA" y el historial. Si no alcanza,
   di que no tienes información suficiente. Nunca inventes locales, horarios ni contenido clínico.
2. NUNCA recomiendes medicamentos ni dosis; no diagnostiques ni prescribas. Si la conversación
   se acerca a eso, rechaza y deriva a un profesional de salud.
3. Cita siempre la fuente: en fichas usa el campo "cita" de cada ficha; en turnos menciona lo
   que diga el campo "fuente" (dato en vivo o respaldo con fecha).
4. No informes stock, precio ni disponibilidad: el sistema no los conoce.
5. Español de Chile, tono cercano y natural, como lo explicarías conversando. Evita
   convertir la respuesta en una lista de campo:valor tipo formulario (ej. "Clase:
   analgésico / Indicación: dolor"). Redacta en prosa breve; usa una lista solo si
   de verdad ayuda a leer varias farmacias u opciones, nunca para una ficha simple.
6. Responde en TEXTO PLANO, sin Markdown (nada de **negrita**, guiones de lista
   como viñeta, ni encabezados) — el front que muestra esto puede no interpretarlo,
   y el usuario vería los símbolos literales en vez de formato.

CÓMO RESPONDER SEGÚN LA SITUACIÓN:
- turnos con "error": "falta_comuna" → pide la comuna amablemente, sin inventar nada.
- turnos con farmacias → lista nombre, dirección, horario y teléfono (el campo siempre trae algo:
  el número real, o el texto "Teléfono no disponible" — inclúyelo tal cual viene). Si el cierre es
  menor que la apertura (ej. 09:00 a 08:59), aclara que cierra al día siguiente. Si hay MÁS DE UNA
  farmacia, pon cada una en su propia línea (salto de línea real) — nunca las juntes todas en
  un mismo párrafo separadas por punto y coma. No uses viñetas ni guiones (regla 6): solo texto
  plano, una farmacia por línea. Al final, en un PUNTO APARTE (línea propia, después de un salto
  de línea en blanco), agrega exactamente: "📄 Fuente: " seguido del contenido del campo "fuente"
  tal cual viene, sin parafrasearlo ni resumirlo (ej. "📄 Fuente: MINSAL, en vivo", o si trae
  advertencia: "📄 Fuente: MINSAL (respaldo) — podría no estar actualizado, confirma por teléfono
  antes de ir"). Esa línea va UNA SOLA VEZ al final, nunca repetida por cada farmacia. NUNCA
  menciones la palabra "cache" ni un timestamp exacto — eso sí es plomería interna.
- turnos con encontrado=false → di que no tienes registros para esa comuna en la fuente actual.
- ficha con fichas → explica en lenguaje simple citando cada ficha usada. Si hay varias
  presentaciones del mismo medicamento, resúmelo sin repetir.
- ficha con encontrado=false → di que el vademécum no tiene información suficiente y sugiere
  consultar a un profesional de salud.
- ambos → contiene {"turnos": ..., "ficha": ...}. Responde LAS DOS partes, aplicando a cada
  una la misma regla que si hubiera llegado sola (turnos y ficha de arriba).
- conversacion → saluda y cuenta qué haces (farmacias de turno por comuna, fichas de medicamentos)
  y qué no haces (recomendar tratamientos o dosis, stock, precios).
- fuera_de_dominio → indica amablemente que solo ayudas con farmacias de turno y fichas de
  medicamentos, y ofrece esas dos opciones."""

# Patrón exigido por la rúbrica (criterio 5): rechazar + derivar + ofrecer acción permitida.
# Texto FIJO a propósito: el camino de rechazo no pasa por ningún LLM.
RESPUESTA_RECHAZO = (
    "No puedo recomendarte un medicamento ni una dosis; eso requiere la evaluación de un "
    "profesional de salud. Te sugiero consultarlo con un médico o químico farmacéutico. "
    "Lo que sí puedo hacer es ayudarte a encontrar una farmacia de turno en tu comuna o "
    "explicarte la ficha de un medicamento que ya te hayan indicado. ¿Te sirve alguna de las dos?"
)

# --- Aportado por Luis (Fase 4): sanitiza datos recuperados de fuentes externas -----
# antes de que lleguen al generador. Cubre la pata de "acceso a datos no confiables"
# del lethal trifecta (Willison) ya documentado en el informe de seguridad.
_PATRON_INSTRUCCION_EN_FUENTE = re.compile(
    r"(ignora|olvida|omite).{0,50}(instrucciones|reglas|prompt)|"
    r"(system prompt|modo desarrollador|ejecuta este comando|revela.*clave)", re.I,
)


def sanitizar_datos_fuente(valor):
    """El contenido recuperado es dato, nunca una instrucción para el modelo."""
    if isinstance(valor, dict):
        return {str(k)[:80]: sanitizar_datos_fuente(v) for k, v in valor.items()}
    if isinstance(valor, list):
        return [sanitizar_datos_fuente(v) for v in valor[:100]]
    if isinstance(valor, str):
        texto = valor[:4000]
        if _PATRON_INSTRUCCION_EN_FUENTE.search(texto):
            return "[contenido no confiable eliminado]"
        return texto.replace("\x00", "")
    return valor


# ==============================================================================
# Dependencias por defecto (todas reemplazables al construir el grafo)
# ==============================================================================
def _llm_router_por_defecto():
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=MODELO_ROUTER).with_structured_output(DecisionRuteo)


def _llm_generador_por_defecto():
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=MODELO_GENERADOR)


def _tool_rag_por_defecto():
    """Importa la tool real de la Fase 1 (carpeta hermana). Import perezoso:
    solo se necesita qdrant si efectivamente se usa la ruta de fichas."""
    ruta_fase1 = Path(__file__).resolve().parents[2] / "Fase1_Datos_y_RAG" / "fase1_entrega"
    if str(ruta_fase1) not in sys.path:
        sys.path.insert(0, str(ruta_fase1))
    from tool_rag_vademecum import tool_rag_vademecum
    return tool_rag_vademecum


def _tool_minsal_por_defecto():
    """Mientras la Fase 2 no exista, se usa el stub con snapshot rotulado.
    Cuando la Fase 2 entregue su tool real, basta inyectarla en construir_grafo()."""
    if str(Path(__file__).resolve().parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
    from tool_minsal_stub import tool_minsal
    return tool_minsal


# ==============================================================================
# Construcción del grafo
# ==============================================================================
def construir_grafo(
    llm_router=None,
    llm_generador=None,
    tool_rag=None,
    tool_minsal=None,
    checkpointer=None,
):
    """Arma y compila el grafo. Todos los parámetros son inyectables:

    - llm_router: runnable cuyo .invoke(mensajes) devuelve una DecisionRuteo.
    - llm_generador: runnable cuyo .invoke(mensajes) devuelve un mensaje con .content.
    - tool_rag: callable(pregunta: str) -> dict  (contrato de la Fase 1).
    - tool_minsal: callable(comuna: str) -> dict (contrato en tool_minsal_stub.py).
    - checkpointer: memoria del historial. Por defecto MemorySaver (en RAM);
      la Fase 5 puede inyectar un SqliteSaver para persistir entre reinicios.
    """
    llm_router = llm_router or _llm_router_por_defecto()
    llm_generador = llm_generador or _llm_generador_por_defecto()
    tool_rag = tool_rag or _tool_rag_por_defecto()
    tool_minsal = tool_minsal or _tool_minsal_por_defecto()
    checkpointer = checkpointer or MemorySaver()

    # --- Nodos -----------------------------------------------------------------
    def nodo_router(estado: EstadoAsistente) -> dict:
        """Clasifica el último mensaje y reescribe los campos de trabajo del turno.
        `datos_tool` se limpia SIEMPRE aquí: el estado persiste entre turnos y no
        queremos que el generador vea datos de una consulta anterior."""
        mensajes = [SystemMessage(content=PROMPT_ROUTER)] + list(estado["messages"])
        decision = llm_router.invoke(mensajes)
        return {
            "intencion": decision.intencion,
            "comuna": decision.comuna,
            "consulta_autonoma": decision.consulta_autonoma,
            "motivo_ruteo": decision.motivo,
            "datos_tool": None,
        }

    def nodo_turnos(estado: EstadoAsistente) -> dict:
        """Consulta la tool MINSAL. Sin comuna no hay consulta: se marca el error
        y el generador pedirá la comuna (nunca se inventa una)."""
        if not estado.get("comuna"):
            return {"datos_tool": {"encontrado": False, "error": "falta_comuna"}}
        return {"datos_tool": sanitizar_datos_fuente(tool_minsal(estado["comuna"]))}

    def nodo_ficha(estado: EstadoAsistente) -> dict:
        """Consulta la tool RAG con la consulta AUTÓNOMA (no el mensaje crudo):
        así '¿y sus efectos secundarios?' llega como pregunta completa."""
        return {"datos_tool": sanitizar_datos_fuente(tool_rag(estado["consulta_autonoma"]))}

    def nodo_ambos(estado: EstadoAsistente) -> dict:
        """El mensaje nombra un medicamento Y pide turnos a la vez — se consultan
        LAS DOS tools; ninguna ambigüedad de seguridad (el medicamento ya está
        elegido por el usuario, no es una sugerencia). Sin comuna, la parte de
        turnos queda marcada como 'falta_comuna', igual que en nodo_turnos."""
        datos_turnos = (
            {"encontrado": False, "error": "falta_comuna"}
            if not estado.get("comuna") else tool_minsal(estado["comuna"])
        )
        datos_ficha = tool_rag(estado["consulta_autonoma"])
        return {"datos_tool": sanitizar_datos_fuente({"turnos": datos_turnos, "ficha": datos_ficha})}

    def nodo_rechazo(estado: EstadoAsistente) -> dict:
        """Camino seguro: texto fijo, sin LLM. Rechaza + deriva + ofrece
        alternativa permitida (condición dura del criterio 5)."""
        return {"messages": [AIMessage(content=RESPUESTA_RECHAZO)]}

    def nodo_generador(estado: EstadoAsistente) -> dict:
        """Redacta la respuesta final SOLO con los datos de la tool + historial."""
        datos = estado.get("datos_tool")
        contexto = (
            f"INTENCIÓN DETECTADA: {estado.get('intencion', 'conversacion')}\n"
            "DATOS DE LA CONSULTA (única fuente permitida para datos de turnos o fichas):\n"
            f"{json.dumps(datos, ensure_ascii=False, indent=2) if datos is not None else 'ninguno (no se consultó ninguna tool)'}"
        )
        mensajes = (
            [SystemMessage(content=PROMPT_GENERADOR)]
            + list(estado["messages"])
            + [SystemMessage(content=contexto)]
        )
        respuesta = llm_generador.invoke(mensajes)
        return {"messages": [AIMessage(content=respuesta.content)]}

    # --- Ruteo condicional -----------------------------------------------------
    def decidir_ruta(estado: EstadoAsistente) -> str:
        return estado["intencion"]

    constructor = StateGraph(EstadoAsistente)
    constructor.add_node("router", nodo_router)
    constructor.add_node("turnos", nodo_turnos)
    constructor.add_node("ficha", nodo_ficha)
    constructor.add_node("ambos", nodo_ambos)
    constructor.add_node("rechazo", nodo_rechazo)
    constructor.add_node("generador", nodo_generador)

    constructor.add_edge(START, "router")
    constructor.add_conditional_edges(
        "router",
        decidir_ruta,
        {
            "turnos": "turnos",
            "ficha": "ficha",
            "ambos": "ambos",
            "tratamiento": "rechazo",       # fail-closed: nunca pasa por el generador
            "conversacion": "generador",    # respuesta directa, sin tools
            "fuera_de_dominio": "generador",
        },
    )
    constructor.add_edge("turnos", "generador")
    constructor.add_edge("ficha", "generador")
    constructor.add_edge("ambos", "generador")
    constructor.add_edge("rechazo", END)
    constructor.add_edge("generador", END)

    return constructor.compile(checkpointer=checkpointer)


# ==============================================================================
# Interfaz para la API (Fase 5)
# ==============================================================================
def conversar(grafo, user_id: str, pregunta: str) -> str:
    """Un turno de conversación. El thread_id del checkpointer ES el user_id:
    cada usuario tiene su propio historial, aislado del resto."""
    config = {"configurable": {"thread_id": user_id}}
    resultado = grafo.invoke({"messages": [HumanMessage(content=pregunta)]}, config)
    return resultado["messages"][-1].content


def estado_de(grafo, user_id: str) -> dict:
    """Estado persistido de un usuario (útil para trazabilidad y debugging)."""
    return grafo.get_state({"configurable": {"thread_id": user_id}}).values
