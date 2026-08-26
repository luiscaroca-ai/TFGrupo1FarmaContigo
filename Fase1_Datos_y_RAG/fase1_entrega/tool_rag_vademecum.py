"""
tool_rag_vademecum.py — Tarea 8: la tool RAG del asistente de farmacias.

Recibe una pregunta, busca por significado en la colección `vademecum_farmacias`
(ya poblada por ingest_vademecum.py) y devuelve DATOS ESTRUCTURADOS + la cita de
cada ficha recuperada. No redacta la respuesta final: esa decisión es del nodo
generador dentro de LangGraph (Fase 3), para no pisarle el trabajo a Cristian.

Contrato de salida (lo que necesita conocer quien arme el grafo):
    tool_rag_vademecum(pregunta: str) -> dict
    {
      "encontrado": bool,
      "fichas": [
        {
          "nombre": str, "nombre_generico": str, "concentracion": str,
          "forma_farmaceutica": str, "clase": str, "indicaciones": str,
          "mecanismo_de_accion": str, "efectos_secundarios": str,
          "reacciones_adversas": str, "interacciones": str,
          "advertencias": str, "categoria_embarazo": str, "conservacion": str,
          "condicion_venta": str, "cita": str,   # ej. "Vademécum · ficha ID 50"
        },
        ...
      ],
    }

Si no hay ninguna ficha suficientemente relacionada, "encontrado" es False y
"fichas" es una lista vacía — el nodo generador debe usar la frase estándar del
guardrail ("No tengo información suficiente / consulta a un profesional").

Uso como script (prueba manual, no requiere LangGraph):
    python tool_rag_vademecum.py "¿para qué sirve la cetirizina?"
"""
from __future__ import annotations

import os
import re
import sys
import unicodedata

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

load_dotenv()

COLLECTION = "vademecum_farmacias"
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIMS = 256
TOP_K = 3                # cuántas fichas candidatas trae la búsqueda semántica
UMBRAL_SIMILITUD = 0.49  # calibrado con casos reales: separa "alergia" (Loratadina
                          # 0.53, Cetirizina 0.498) de un vecino semántico incorrecto
                          # (Asma 0.484); ver Fase1 - notas de calibración del umbral

# --- Campos que se devuelven en cada ficha (todos vienen de la metadata) --------
CAMPOS_FICHA = [
    "nombre", "nombre_generico", "concentracion", "forma_farmaceutica", "clase",
    "indicaciones", "mecanismo_de_accion", "efectos_secundarios",
    "reacciones_adversas", "interacciones", "advertencias",
    "categoria_embarazo", "conservacion", "condicion_venta",
]


def _normalizar(texto: str) -> str:
    """minúsculas y sin tildes, para comparar 'cetirizina' con 'Cetirizina' o 'cetirizína'."""
    t = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in t if not unicodedata.combining(c))


def _construir_vector_store() -> QdrantVectorStore:
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL, dimensions=EMBED_DIMS)
    client = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"])
    return QdrantVectorStore(client=client, collection_name=COLLECTION, embedding=embeddings)


_vector_store: QdrantVectorStore | None = None
_nombres_conocidos: list[str] | None = None  # cache: todos los nombres/genéricos del corpus


def _cargar_nombres_conocidos() -> list[str]:
    """Trae, una sola vez, todos los nombres y genéricos distintos que existen
    en la colección — para poder detectar si la pregunta menciona alguno."""
    global _vector_store
    client = _vector_store.client
    nombres = set()
    siguiente = None
    while True:
        puntos, siguiente = client.scroll(
            collection_name=COLLECTION, limit=256, offset=siguiente, with_payload=True,
        )
        for p in puntos:
            meta = p.payload.get("metadata", p.payload)
            for campo in ("nombre", "nombre_generico"):
                v = meta.get(campo, "").strip()
                if v:
                    nombres.add(v)
        if siguiente is None:
            break
    return sorted(nombres)


def _detectar_nombres_en_pregunta(pregunta: str) -> list[str]:
    """¿La pregunta menciona, tal cual, alguno de los medicamentos del corpus?"""
    global _nombres_conocidos
    if _nombres_conocidos is None:
        _nombres_conocidos = _cargar_nombres_conocidos()

    p_norm = _normalizar(pregunta)
    encontrados = []
    for nombre in _nombres_conocidos:
        # \b evita que "sal" matchee dentro de "salbutamol"
        if re.search(rf"\b{re.escape(_normalizar(nombre))}\b", p_norm):
            encontrados.append(nombre)
    return encontrados


def _buscar_por_nombre_exacto(nombres: list[str]) -> list[dict]:
    """Coincidencia exacta: trae TODAS las fichas de esos nombres (puede haber
    varias concentraciones del mismo medicamento — todas son relevantes)."""
    client = _vector_store.client
    fichas = []
    for nombre in nombres:
        condiciones = Filter(should=[
            FieldCondition(key="metadata.nombre", match=MatchValue(value=nombre)),
            FieldCondition(key="metadata.nombre_generico", match=MatchValue(value=nombre)),
        ])
        puntos, _ = client.scroll(
            collection_name=COLLECTION, scroll_filter=condiciones, limit=20, with_payload=True,
        )
        for p in puntos:
            meta = p.payload.get("metadata", p.payload)
            ficha = {campo: meta.get(campo, "") for campo in CAMPOS_FICHA}
            ficha["cita"] = f"Vademécum · ficha ID {meta.get('id', '?')}"
            fichas.append(ficha)
    return fichas


def tool_rag_vademecum(pregunta: str) -> dict:
    """Busca en el vademécum y devuelve fichas estructuradas + cita. No redacta.

    Estrategia en dos pasos:
      1) Si la pregunta nombra un medicamento del corpus -> coincidencia EXACTA
         por nombre (sin depender de similitud de texto).
      2) Si no nombra ninguno -> búsqueda semántica por significado (síntoma,
         uso, etc.), filtrada por el umbral de similitud.
    """
    global _vector_store
    if _vector_store is None:
        _vector_store = _construir_vector_store()

    nombres_detectados = _detectar_nombres_en_pregunta(pregunta)
    if nombres_detectados:
        fichas = _buscar_por_nombre_exacto(nombres_detectados)
        return {"encontrado": len(fichas) > 0, "fichas": fichas, "modo": "nombre_exacto"}

    # sin nombre reconocible -> búsqueda semántica (preguntas por síntoma/uso)
    # Se pide más candidatos de lo necesario (k=10) y se DIVERSIFICA por nombre,
    # para que 5 concentraciones del mismo medicamento no monopolicen el top-3
    # dejando fuera a otros medicamentos igual de relevantes.
    candidatos = _vector_store.similarity_search_with_score(pregunta, k=10)
    fichas, nombres_usados = [], set()
    for doc, score in candidatos:
        if score < UMBRAL_SIMILITUD:
            continue
        nombre = doc.metadata.get("nombre", "")
        if nombre in nombres_usados:
            continue  # ya se incluyó una ficha de este medicamento
        nombres_usados.add(nombre)
        ficha = {campo: doc.metadata.get(campo, "") for campo in CAMPOS_FICHA}
        ficha["cita"] = f"Vademécum · ficha ID {doc.metadata.get('id', '?')}"
        ficha["score"] = round(score, 3)  # visible solo para calibrar el umbral; quitar en producción
        fichas.append(ficha)
        if len(fichas) >= TOP_K:
            break
    return {"encontrado": len(fichas) > 0, "fichas": fichas, "modo": "semantico"}


if __name__ == "__main__":
    pregunta = " ".join(sys.argv[1:]) or "¿para qué sirve la cetirizina?"
    print(f"Pregunta: {pregunta}\n")
    resultado = tool_rag_vademecum(pregunta)
    print(f"Modo: {resultado['modo']} · Umbral actual: {UMBRAL_SIMILITUD}\n")

    if not resultado["encontrado"]:
        print("Sin fichas relevantes — el guardrail debe responder que no tiene información suficiente.")
    else:
        for f in resultado["fichas"]:
            score_txt = f" · score={f['score']}" if "score" in f else ""
            print(f"— {f['nombre']} ({f['nombre_generico']}) · {f['concentracion']} · {f['forma_farmaceutica']}{score_txt}")
            print(f"  Indicaciones: {f['indicaciones']}")
            print(f"  Fuente: {f['cita']}\n")
