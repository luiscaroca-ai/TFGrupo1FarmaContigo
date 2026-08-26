"""
guardrails.py — Fase 4, tarea 16: guardrail de ENTRADA y de SALIDA.

Dos barreras DETERMINISTAS (regex sobre texto normalizado, sin LLM) que envuelven
al grafo de la Fase 3, implementando la defensa en profundidad del Módulo 04:

    entrada → [CAPA 1 · input guard] → grafo (router fail-closed + rechazo
    determinista + prompt del generador) → [CAPA 2 · output guard] → usuario

Por qué deterministas: "en application security, 99% es nota de reprobación".
Un patrón regex no se puede persuadir ni jailbreakear. La capa probabilística
(el router) ya existe dentro del grafo; estas capas la envuelven por fuera y
son independientes de ella: si el router se equivoca, la salida igual se revisa.

Qué frena cada capa:
  - CAPA 1 (entrada): prompt injection explícito ("ignora tus instrucciones",
    "muestra tu system prompt") y roleplay clínico ("actúa como médico").
    Lo que la capa 1 no reconoce (p. ej. una petición de tratamiento formulada
    con naturalidad) lo maneja el router fail-closed del grafo.
  - CAPA 2 (salida): posología ("X mg cada Y horas"), recomendación directa de
    toma, y afirmaciones de stock/precio (datos que el sistema NO conoce).
    Ojo con los falsos positivos: una ficha citada legítimamente contiene
    "10 mg" (concentración) — por eso los patrones exigen posología/frecuencia,
    no la sola mención de una unidad.

Interfaz para la API (Fase 5):
    resultado = conversar_seguro(grafo, user_id, pregunta)
    # -> {respuesta, capa: "input_guard"|"grafo"|"output_guard", motivo, intencion, motivo_ruteo}

Nota: si la capa 1 bloquea, el turno NO entra al historial del grafo (el ataque
no contamina el contexto de la conversación). Decisión documentada en el informe.
"""
import re
import sys
import unicodedata
from pathlib import Path

RUTA_FASE3 = Path(__file__).resolve().parents[2] / "Fase3_Orquestacion_LangGraph" / "fase3_entrega"
if str(RUTA_FASE3) not in sys.path:
    sys.path.insert(0, str(RUTA_FASE3))

from grafo_farmacias import RESPUESTA_RECHAZO, conversar, estado_de  # noqa: E402

MENSAJE_ALCANCE = (
    "Solo puedo ayudarte a buscar farmacias de turno por comuna o a explicarte la "
    "ficha de un medicamento del vademécum. No puedo cambiar mis reglas ni compartir "
    "instrucciones internas. ¿Te ayudo con alguna de esas dos cosas?"
)

MENSAJE_SALIDA_RETENIDA = (
    "Prefiero no entregar esa respuesta: podría interpretarse como una recomendación "
    "de tratamiento o como un dato que no puedo confirmar, y eso requiere a un "
    "profesional de salud o una fuente oficial. Sí puedo ayudarte a encontrar una "
    "farmacia de turno o a explicarte la información general de una ficha."
)

MENSAJE_DATOS_PERSONALES = (
    "Para proteger tu privacidad, elimina RUT, correo, teléfono, dirección y nombres "
    "completos antes de consultar. Puedes reformular la pregunta sin esos datos."
)
MENSAJE_VULNERABLE = (
    "No puedo orientar tratamientos para menores, embarazo, lactancia, personas mayores "
    "o con enfermedades crónicas. Consulta a un médico o químico farmacéutico. Sí puedo "
    "mostrarte información general de una ficha o una farmacia de turno."
)
MENSAJE_INTERACCIONES = (
    "No puedo confirmar si es seguro combinar medicamentos, suplementos, alcohol u otras "
    "sustancias. Verifícalo con un médico o químico farmacéutico, idealmente con la lista completa."
)
MENSAJE_AMBIGUO = (
    "Necesito un dato más para responder con seguridad: indica el nombre exacto del medicamento "
    "o la comuna, sin incluir información personal."
)
MENSAJE_FALLO_SEGURO = (
    "No fue posible completar la consulta con las fuentes disponibles. No entregaré información "
    "sin verificar; intenta nuevamente o consulta una fuente oficial o profesional de salud."
)
MENSAJE_EMERGENCIA = (
    "Esto podría ser una emergencia. No continúes esperando una respuesta del asistente: "
    "busca atención médica inmediata o comunícate con el servicio de urgencias de tu zona. "
    "Si hay riesgo de autolesión, no te quedes a solas y pide ayuda ahora a una persona de confianza."
)


def _normalizar(texto: str) -> str:
    t = unicodedata.normalize("NFKD", str(texto).lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", t)


# ==============================================================================
# CAPA 1 · Guardrail de entrada
# ==============================================================================
_PATRONES_INYECCION = [
    r"\b(ignora|olvida|omite|salta(te)?)\b.{0,40}\b(instrucciones|reglas|restricciones|prompt)",
    r"\b(muestra|revela|repite|imprime|dime|entrega)\b.{0,40}\b(system prompt|prompt del sistema|tus instrucciones|instrucciones (internas|del sistema)|tu configuracion)",
    r"\bsystem prompt\b",
    r"\bmodo (desarrollador|dios|sin (restricciones|filtros|censura))\b",
    r"\beres dan\b|\bmodo dan\b",
    r"repite (todo )?el texto (anterior|que esta antes|de arriba)",
    r"\bdesactiva\b.{0,30}\b(guardrail|filtro|seguridad|restriccion)",
    r"\b(ignore|forget|bypass|disregard)\b.{0,50}\b(previous|system|instructions?|rules?|prompt)",
    r"\b(show|reveal|print|repeat)\b.{0,40}\b(system prompt|hidden instructions?|developer message)",
]

_PATRONES_EMERGENCIA = [
    r"\b(no puedo|me cuesta) respirar\b|\b(falta de aire|dificultad para respirar)\b",
    r"\b(dolor|duele|duelen|presion|presión|opresion|opresión)\b.{0,25}\bpecho\b",
    r"\b(perdio|perdiendo|perdida de) (la )?conciencia\b|\besta inconsciente\b",
    r"\b(sobredosis|intoxicacion|tome demasiad\w*|ingiri demasiad\w*)\b",
    r"\b(me quiero morir|quiero matarme|suicidarme|hacerme dano|autolesionarme)\b",
]

_PATRONES_ROLEPLAY_CLINICO = [
    r"\bactua como\b.{0,40}\b(medico|doctor|doctora|farmaceutic\w*|enfermer\w*|quimico)",
    r"\b(eres|seras|se(ras)? tu)\b.{0,30}\b(un |una |mi )?(medico|doctor|doctora|farmaceutic\w*)\b",
    r"\bfinge (ser|que eres)\b",
    r"\bjuguemos a que\b",
    r"\bhazte pasar por\b",
    r"\b(recetame|prescribeme|receta algo)\b",
]

_PATRONES_DATOS_PERSONALES = [
    r"\b\d{1,2}\.\d{3}\.\d{3}-[0-9k]\b",                 # RUT con puntos
    r"\b\d{7,8}-[0-9k]\b",                              # RUT sin puntos
    r"\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b",       # correo
    r"(?<!\d)(?:\+?56\s*)?9[\s.-]*\d{4}[\s.-]*\d{4}(?!\d)",  # móvil Chile
    r"\b(mi|vivo en|direccion)\s+(direccion|domicilio|casa)?\s*(es|:)?\s*(calle|avenida|av\.|pasaje)\b",
]
# En la SALIDA solo tiene sentido revisar RUT y correo: el teléfono y la
# dirección de una FARMACIA son justo el dato que la tool debe mostrar (parte
# del contrato documentado de tool_minsal), no una fuga de datos del usuario.
# Aplicar los mismos 5 patrones a la salida bloqueaba respuestas legítimas de
# turnos cada vez que el teléfono de una farmacia venía en formato celular
# (ej. "+56998311688", EL RETIRO en Quilpué) — hallazgo real, 24-08-2026.
_PATRONES_DATOS_PERSONALES_SALIDA = _PATRONES_DATOS_PERSONALES[:3]  # RUT (x2) + correo
_PATRONES_VULNERABLES = [
    r"\b(mi )?(hijo|hija|bebe|niño|niña|menor|guagua)\b",
    r"\b(embarazada|embarazo|lactancia|amamantando)\b",
    r"\b(mi )?(abuelo|abuela|adulto mayor|persona mayor)\b",
    r"\b(diabetes|hipertension|insuficiencia renal|enfermedad cronica)\b",
]
_PATRONES_ACCION_CLINICA = [r"\b(tomar|darle|dosis|cuanto|recomiend|puede usar|es seguro)\w*\b"]
_PATRONES_COMBINACIONES = [
    r"\b(mezclar|combinar|juntar|tomar juntos?|a la vez)\b",
    r"\b(interaccion|interactua)\w*\b",
    r"\b(medicamento|pastilla|antibiotico)\w*\b.{0,50}\b(alcohol|suplemento|droga)\w*\b",
    r"\b(alcohol|suplemento|droga)\w*\b.{0,50}\b(medicamento|pastilla|antibiotico)\w*\b",
]
_PATRONES_AMBIGUEDAD = [
    r"^(esto|eso|esa|este|aquello)\??$",
    r"\b(esta|esa|la)\s+(pastilla|medicina|medicamento)\b(?!.*\b(se llama|nombre)\b)",
    r"^(donde hay|cual esta de turno|y donde)\??$",
]


def revisar_entrada(pregunta: str) -> dict:
    """-> {permitido: bool, respuesta: str|None, motivo: str}"""
    p = _normalizar(pregunta)
    if any(re.search(x, p) for x in _PATRONES_EMERGENCIA):
        return {"permitido": False, "respuesta": MENSAJE_EMERGENCIA,
                "motivo": "posible emergencia o riesgo vital"}
    for patron in _PATRONES_DATOS_PERSONALES:
        if re.search(patron, p):
            return {"permitido": False, "respuesta": MENSAJE_DATOS_PERSONALES,
                    "motivo": "dato personal detectado"}
    if any(re.search(x, p) for x in _PATRONES_VULNERABLES) and any(
            re.search(x, p) for x in _PATRONES_ACCION_CLINICA):
        return {"permitido": False, "respuesta": MENSAJE_VULNERABLE,
                "motivo": "grupo vulnerable con solicitud clínica"}
    if any(re.search(x, p) for x in _PATRONES_COMBINACIONES):
        return {"permitido": False, "respuesta": MENSAJE_INTERACCIONES,
                "motivo": "combinación o interacción solicitada"}
    if any(re.search(x, p) for x in _PATRONES_AMBIGUEDAD):
        return {"permitido": False, "respuesta": MENSAJE_AMBIGUO,
                "motivo": "consulta ambigua sin referente suficiente"}
    for patron in _PATRONES_ROLEPLAY_CLINICO:
        if re.search(patron, p):
            return {"permitido": False, "respuesta": RESPUESTA_RECHAZO,
                    "motivo": f"roleplay clínico detectado ({patron})"}
    for patron in _PATRONES_INYECCION:
        if re.search(patron, p):
            return {"permitido": False, "respuesta": MENSAJE_ALCANCE,
                    "motivo": f"intento de manipulación de instrucciones ({patron})"}
    return {"permitido": True, "respuesta": None, "motivo": "sin patrones hostiles"}


# ==============================================================================
# CAPA 2 · Guardrail de salida
# ==============================================================================
_PATRONES_SALIDA_INSEGURA = [
    # Posología: número + unidad + frecuencia (la sola concentración "10 mg" es legítima en una ficha)
    r"\d+\s*(mg|ml|mcg|g|gotas|comprimidos?|pastillas?|capsulas?)\s*(cada|al dia|por dia|diari\w+|veces)",
    r"\bcada\s+\d+\s*(horas|hrs|hras|h)\b",
    # Recomendación directa de toma/uso (afirmativa, no la negación "no puedo recomendarte")
    r"\bte recomiendo (tomar|usar|comprar|que tomes)\b",
    r"\bdeberias (tomar|usar)\b",
    r"\btomate\b|\btoma\s+\d",
    r"\bte sugiero (tomar|usar)\b",
    r"\baumenta la dosis\b|\bduplica la dosis\b|\bpuedes tomar el doble\b",
    # Stock / precio / disponibilidad: el sistema no los conoce; afirmarlos es inventar
    r"\b(hay|tenemos|queda|esta disponible en)\s+(en\s+)?stock\b",
    r"\b(cuesta|vale|tiene un precio de|el precio es)\s*\$?\s*[\d.,]+",
    # Secretos, instrucciones y detalles internos
    r"\b(sk-[a-z0-9_-]{12,}|api[_ -]?key\s*[:=]|qdrant_api_key|openai_api_key)\b",
    r"\b(traceback|stack trace|system prompt:|prompt_router|decisionruteo)\b",
]


def revisar_salida(respuesta: str) -> dict:
    """-> {permitido: bool, respuesta_final: str, motivo: str}"""
    r = _normalizar(respuesta)
    if any(re.search(p, r) for p in _PATRONES_DATOS_PERSONALES_SALIDA):
        return {"permitido": False, "respuesta_final": MENSAJE_SALIDA_RETENIDA,
                "motivo": "salida retenida por datos personales"}
    for patron in _PATRONES_SALIDA_INSEGURA:
        if re.search(patron, r):
            return {"permitido": False, "respuesta_final": MENSAJE_SALIDA_RETENIDA,
                    "motivo": f"salida retenida por patrón inseguro ({patron})"}
    return {"permitido": True, "respuesta_final": respuesta, "motivo": "salida limpia"}


def es_respuesta_insegura(respuesta: str) -> bool:
    """Evaluador de código reutilizado por las pruebas adversarias (Módulo 04:
    'code-evaluator determinístico')."""
    return not revisar_salida(respuesta)["permitido"]


# ==============================================================================
# Pipeline protegido (lo que consume la API de la Fase 5)
# ==============================================================================
def conversar_seguro(grafo, user_id: str, pregunta: str) -> dict:
    entrada = revisar_entrada(pregunta)
    if not entrada["permitido"]:
        return {"respuesta": entrada["respuesta"], "capa": "input_guard",
                "motivo": entrada["motivo"], "intencion": "bloqueada_en_entrada",
                "motivo_ruteo": entrada["motivo"]}

    respuesta = conversar(grafo, user_id, pregunta)
    estado = estado_de(grafo, user_id)

    salida = revisar_salida(respuesta)
    capa = "grafo" if salida["permitido"] else "output_guard"
    return {"respuesta": salida["respuesta_final"], "capa": capa, "motivo": salida["motivo"],
            "intencion": estado.get("intencion", "desconocida"),
            "motivo_ruteo": estado.get("motivo_ruteo", "")}
