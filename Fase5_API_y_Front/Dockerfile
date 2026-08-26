# Dockerfile — Trabajo Final M04 · Asistente de farmacias (tarea 22)
# Contexto de build: la carpeta "trabajo final/" (raíz de las fases):
#   docker build -f Fase5_API_y_Front/Dockerfile -t farmacias-grupo1 .
# Las llaves NUNCA van en la imagen: llegan como secretos del entorno (fly secrets).
FROM python:3.11-slim

WORKDIR /app

COPY Fase5_API_y_Front/fase5_avance/requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Se replica la estructura de fases que los imports relativos esperan.
COPY Fase1_Datos_y_RAG/fase1_entrega/tool_rag_vademecum.py Fase1_Datos_y_RAG/fase1_entrega/
COPY Fase2_Fuente_En_Vivo/fase2_entrega/tool_minsal.py Fase2_Fuente_En_Vivo/fase2_entrega/
COPY Fase3_Orquestacion_LangGraph/fase3_entrega/grafo_farmacias.py Fase3_Orquestacion_LangGraph/fase3_entrega/
COPY Fase3_Orquestacion_LangGraph/fase3_entrega/tool_minsal_stub.py Fase3_Orquestacion_LangGraph/fase3_entrega/
COPY Fase4_Seguridad_Guardrails/fase4_entrega/guardrails.py Fase4_Seguridad_Guardrails/fase4_entrega/
COPY Fase5_API_y_Front/fase5_avance/api_server.py Fase5_API_y_Front/fase5_avance/
COPY Fase5_API_y_Front/fase5_avance/historial.py Fase5_API_y_Front/fase5_avance/
COPY Fase5_API_y_Front/fase5_avance/rate_limiting.py Fase5_API_y_Front/fase5_avance/
COPY Fase5_API_y_Front/fase5_avance/trazabilidad.py Fase5_API_y_Front/fase5_avance/
COPY Fase5_API_y_Front/fase5_avance/front/ Fase5_API_y_Front/fase5_avance/front/

WORKDIR /app/Fase5_API_y_Front/fase5_avance
EXPOSE 8080
CMD ["python", "-m", "uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8080"]
