import json
import random
from datetime import datetime, timedelta
import os
import pytz # Importar pytz para manejo de zonas horarias

# Rutas de los archivos
INPUT_JSON = 'datasets/estructura_arduino.json' # Plantilla de entrada
OUTPUT_JSON = 'datasets/estructura_arduino_simulado.json' # Salida de la simulación

try:
    # Leer la plantilla base del JSON de entrada.
    # Se asume que el archivo de entrada es una lista y tomamos el primer elemento como plantilla.
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        base_event = json.load(f)[0]
except FileNotFoundError:
    print(f"Error: El archivo de entrada '{INPUT_JSON}' no fue encontrado.")
    # Crear una plantilla de evento por defecto si el archivo no existe
    base_event = {
        "ID": "0",
        "TIMESTAMP": "2025-06-01 00:00:00",
        "FECHA": "01/06/2025",
        "HORA": "00:00:00",
        "HORA_PC": "00:00:00",
        "EVENTO": "INFO",
        "DESCRIPCION": "Plantilla por defecto",
        "TOTAL": "0",
        "LED1": "OFF",
        "LED2": "OFF",
        "LED3": "OFF"
    }
except json.JSONDecodeError:
    print(f"Error: El archivo '{INPUT_JSON}' no es un JSON válido.")
    base_event = {
        "ID": "0",
        "TIMESTAMP": "2025-06-01 00:00:00",
        "FECHA": "01/06/2025",
        "HORA": "00:00:00",
        "HORA_PC": "00:00:00",
        "EVENTO": "INFO",
        "DESCRIPCION": "Plantilla por defecto (JSON inválido)",
        "TOTAL": "0",
        "LED1": "OFF",
        "LED2": "OFF",
        "LED3": "OFF"
    }

# Configuración de la zona horaria para la simulación
# Asumimos que la base de datos Django está configurada con 'America/Lima' (UTC-5)
# Esto es crucial para replicar el comportamiento observado.
TIME_ZONE_STR = 'America/Lima'
try:
    local_tz = pytz.timezone(TIME_ZONE_STR)
except pytz.UnknownTimeZoneError:
    print(f"Advertencia: Zona horaria '{TIME_ZONE_STR}' desconocida. Usando UTC.")
    local_tz = pytz.utc


# Inicialización de la lista para almacenar los registros simulados
registros_simulados = []
# Establecer la fecha y hora de inicio de la simulación (naive datetime)
fecha_hora_base_naive = datetime.strptime("2025-06-01 07:00:00", "%Y-%m-%d %H:%M:%S")
# Localizar la fecha y hora base a la zona horaria definida
fecha_hora_base_aware = local_tz.localize(fecha_hora_base_naive)

# Inicializar el contador total de estudiantes
total_actual = 0

# Generar 300 registros simulados
for i in range(1000):
    # Crear una copia del evento base para cada nuevo registro
    evento = base_event.copy()
    evento['ID'] = str(i + 1) # Asignar un ID único

    # Calcular la fecha y hora del evento (aware datetime)
    fecha_hora_evento_aware = fecha_hora_base_aware + timedelta(minutes=i * 10)

    # Para HORA y HORA_PC: Generar la hora en UTC (que es 5 horas antes de Lima)
    # Esto replicaría el comportamiento donde HORA es 5 horas antes que el timestamp local.
    hora_utc_str = fecha_hora_evento_aware.astimezone(pytz.utc).strftime("%H:%M:%S")

    # Para FECHA y TIMESTAMP: Generar la fecha y hora en la zona horaria local (Lima)
    fecha_str = fecha_hora_evento_aware.strftime("%d/%m/%Y")
    timestamp_local_str = fecha_hora_evento_aware.strftime("%Y-%m-%d %H:%M:%S")

    evento['FECHA'] = fecha_str
    evento['HORA'] = hora_utc_str # HORA ahora es la hora UTC
    evento['HORA_PC'] = hora_utc_str # HORA_PC también es la hora UTC
    evento['TIMESTAMP'] = timestamp_local_str # TIMESTAMP es la hora local (Lima)

    # Simular eventos de ENTRADA o SALIDA de estudiantes
    # Se elige aleatoriamente entre "ENTRADA" y "SALIDA"
    tipo_evento = random.choice(["ENTRADA", "SALIDA"])
    evento['EVENTO'] = tipo_evento

    if tipo_evento == "ENTRADA":
        # Si es ENTRADA, añadir entre 1 y 5 estudiantes
        cantidad = random.randint(1, 5)
        # Asegurarse de que el total de estudiantes no supere los 20
        total_actual = min(20, total_actual + cantidad)
        evento['DESCRIPCION'] = f"{cantidad} estudiantes entraron al aula"
    else:
        # Si es SALIDA, restar entre 1 y un máximo de 5, o el total actual si es menor
        cantidad = random.randint(1, min(5, total_actual)) if total_actual > 0 else 0
        total_actual -= cantidad
        evento['DESCRIPCION'] = f"{cantidad} estudiantes salieron del aula"

    # Asegurarse de que el total de estudiantes no sea negativo
    total_actual = max(0, total_actual)
    evento['TOTAL'] = str(total_actual) # Guardar TOTAL como cadena

    # Aplicar las reglas de los LEDs basadas en el total_actual
    # LED1 se enciende si hay más de 0 estudiantes
    evento['LED1'] = "ON" if total_actual > 0 else "OFF"
    # LED2 se enciende si hay más de 4 estudiantes
    evento['LED2'] = "ON" if total_actual > 4 else "OFF"
    # LED3 se enciende si hay más de 8 estudiantes
    evento['LED3'] = "ON" if total_actual > 8 else "OFF"

    # Añadir el evento generado a la lista de registros
    registros_simulados.append(evento)

# Asegurarse de que el directorio 'datasets' exista
os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

# Guardar los registros simulados en un nuevo archivo JSON
with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(registros_simulados, f, indent=4, ensure_ascii=False)

print(f"✅ Simulación generada con TIMESTAMP real en: {OUTPUT_JSON}")
