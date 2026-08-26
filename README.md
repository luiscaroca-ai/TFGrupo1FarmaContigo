# TaraFinalFarmaContigo

Proyecto de asistente para consultar información farmacéutica y farmacias de turno en Chile. Integra recuperación de información (RAG), datos públicos de MINSAL, orquestación con LangGraph, guardrails de seguridad y una API con interfaz web.

## Estructura del proyecto

- `Fase1_Datos_y_RAG`: herramienta RAG para consultar información de medicamentos.
- `Fase2_Fuente_En_Vivo`: consulta y captura de datos de farmacias de turno desde MINSAL.
- `Fase3_Orquestacion_LangGraph`: grafo que coordina las herramientas y el modelo de lenguaje.
- `Fase4_Seguridad_Guardrails`: validaciones y controles de seguridad.
- `Fase5_API_y_Front`: API, trazabilidad, historial, limitación de solicitudes e interfaz web.
- `fly.toml`: configuración de despliegue en Fly.io.

## Requisitos

- Python 3.11 o superior.
- Una clave de API de OpenAI.
- Una instancia de Qdrant con su URL y clave de acceso.

## Configuración

Instala las dependencias:

```bash
pip install -r Fase5_API_y_Front/fase5_avance/requirements-deploy.txt
```

Configura las variables de entorno requeridas:

```bash
export OPENAI_API_KEY="tu_clave_openai"
export QDRANT_URL="https://tu-instancia-qdrant"
export QDRANT_API_KEY="tu_clave_qdrant"
```

Las credenciales no deben guardarse en el repositorio. Los archivos `.env` están excluidos mediante `.gitignore`.

## Ejecución local

Desde la raíz del proyecto:

```bash
uvicorn Fase5_API_y_Front.fase5_avance.api_server:app --reload
```

Luego abre la dirección indicada por Uvicorn, normalmente `http://127.0.0.1:8000`.

## Despliegue

El proyecto incluye un `Dockerfile` y configuración para Fly.io. Antes de desplegar, registra las credenciales como secretos de la plataforma y nunca dentro de la imagen Docker.

## Aviso

La información entregada por el asistente es orientativa y no reemplaza la evaluación de un profesional de la salud.
