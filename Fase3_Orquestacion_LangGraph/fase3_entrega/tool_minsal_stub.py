"""
tool_minsal_stub.py — Contrato de la tool MINSAL + stub con snapshot rotulado.

La Fase 2 (fuente en vivo) todavía no tiene dueño. Este archivo deja definido
el CONTRATO que el grafo de la Fase 3 ya consume, con una implementación de
respaldo que usa la muestra capturada en el enunciado (13 comunas, 2026-07-14).

Contrato que la Fase 2 debe respetar al construir la tool real:

    tool_minsal(comuna: str) -> dict
    {
      "encontrado": bool,
      "comuna": str,                    # comuna consultada, tal como se buscó
      "farmacias": [
        {
          "nombre": str, "direccion": str, "comuna": str,
          "horario_apertura": str,      # "09:00"
          "horario_cierre": str,        # "08:59" puede significar el día siguiente
          "telefono": str,              # puede venir vacío
        },
        ...
      ],
      "fuente": str,                    # OBLIGATORIO: "dato MINSAL en vivo · <hora>"
                                        # o "respaldo (snapshot) · <fecha>" — el
                                        # generador SIEMPRE informa esta etiqueta.
    }

La tool real de Fase 2 además debe aplicar el pipeline de calidad de 5 pasos del
enunciado: recibir (timeout ~5 s + validar esquema) → normalizar (strip, espacios,
teléfonos) → filtrar (fecha vigente y comuna; ojo: fk_region es id interno) →
interpretar turnos nocturnos (cierre < apertura = día siguiente) → responder.
Este stub ya deja los datos normalizados (paso 2) para fijar el formato.

Uso suelto:
    python tool_minsal_stub.py Temuco
"""
import re
import sys
import unicodedata

FECHA_SNAPSHOT = "2026-07-14"

# Muestra capturada del enunciado (assets/trabajo-final.js del deck) — 13 comunas.
# Sirve para desarrollar y probar el grafo; NO representa el estado actual.
_SNAPSHOT = [
    {"nombre": "ISAIAS 2", "comuna": "ARICA", "direccion": "AVENIDA TUCAPEL 2596",
     "apertura": "09:00", "cierre": "21:00", "telefono": ""},
    {"nombre": "FARMACIA CRUZ VERDE", "comuna": "IQUIQUE", "direccion": "TARAPACA 496",
     "apertura": "00:00", "cierre": "23:59", "telefono": "+562425641"},
    {"nombre": "VICTORIA", "comuna": "TOCOPILLA", "direccion": "CALLE 21 DE MAYO 1513",
     "apertura": "09:00", "cierre": "23:00", "telefono": "2813238"},
    {"nombre": "FARMACIA VIDA", "comuna": "HUASCO", "direccion": "PASAJE RIQUELME 147",
     "apertura": "09:00", "cierre": "21:00", "telefono": "512539390"},
    {"nombre": "FARMACIA RIVERA 1", "comuna": "VICUÑA", "direccion": "CALLE SAN MARTIN 291",
     "apertura": "09:00", "cierre": "23:00", "telefono": "2411286"},
    {"nombre": "CRUZ VERDE", "comuna": "LA CALERA", "direccion": "AVENIDA JJ PEREZ 202",
     "apertura": "09:00", "cierre": "08:59", "telefono": "2724714"},
    {"nombre": "CRUZ VERDE", "comuna": "BUIN", "direccion": "AVENIDA JOSE MANUEL BALMACEDA 114",
     "apertura": "08:00", "cierre": "07:59", "telefono": "28221985"},
    {"nombre": "FUTURO", "comuna": "ARAUCO", "direccion": "ESMERALDA Nº 515-B",
     "apertura": "09:00", "cierre": "08:59", "telefono": "+56412552942"},
    {"nombre": "FARMACIA CRUZ VERDE L-367", "comuna": "TEMUCO", "direccion": "AVENIDA ALEMANIA 0780",
     "apertura": "00:00", "cierre": "23:59", "telefono": "2244915"},
    {"nombre": "ALBORADA", "comuna": "FUTRONO", "direccion": "BALMACEDA Nº 500",
     "apertura": "09:00", "cierre": "08:59", "telefono": "+56639481390"},
    {"nombre": "FARMACIAS CRUZ VERDE", "comuna": "PUERTO MONTT", "direccion": "AVDA. PRESIDENTE IBAÑEZ 173 LOCAL 1",
     "apertura": "00:00", "cierre": "23:59", "telefono": "+56993796664"},
    {"nombre": "FARMACIA CRUZ VERDE", "comuna": "CHILE CHICO", "direccion": "CALLE AVENIDA BERNARDO O'HIGGINS 505",
     "apertura": "09:00", "cierre": "20:00", "telefono": "976684836"},
    {"nombre": "ITALIA", "comuna": "BULNES", "direccion": "CALLE CARLOS PALACIOS 394",
     "apertura": "08:00", "cierre": "22:00", "telefono": "2631272"},
]


def _normalizar(texto: str) -> str:
    """minúsculas, sin tildes y con espacios colapsados — para comparar comunas."""
    t = unicodedata.normalize("NFKD", texto.lower().strip())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", t)


def tool_minsal(comuna: str) -> dict:
    """Busca farmacias de turno por comuna en el snapshot rotulado.

    La Fase 2 reemplaza esta implementación por la consulta en vivo a
    getLocalesTurnos.php (con timeout, cache y este mismo snapshot como
    fallback), MANTENIENDO el contrato de salida.
    """
    objetivo = _normalizar(comuna)
    farmacias = [
        {
            "nombre": f["nombre"], "direccion": f["direccion"], "comuna": f["comuna"],
            "horario_apertura": f["apertura"], "horario_cierre": f["cierre"],
            "telefono": f["telefono"],
        }
        for f in _SNAPSHOT
        if _normalizar(f["comuna"]) == objetivo
    ]
    return {
        "encontrado": len(farmacias) > 0,
        "comuna": comuna,
        "farmacias": farmacias,
        "fuente": f"respaldo (snapshot) · {FECHA_SNAPSHOT} — NO es dato en vivo",
    }


if __name__ == "__main__":
    consulta = " ".join(sys.argv[1:]) or "Temuco"
    resultado = tool_minsal(consulta)
    print(f"Comuna: {consulta} · encontrado={resultado['encontrado']} · fuente: {resultado['fuente']}")
    for f in resultado["farmacias"]:
        tel = f" · tel {f['telefono']}" if f["telefono"] else ""
        print(f"— {f['nombre']} · {f['direccion']} · {f['horario_apertura']}-{f['horario_cierre']}{tel}")
