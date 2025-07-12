import json
import random
from datetime import datetime, timedelta

# Rutas
INPUT_JSON = 'datasets/estructura_arduino.json'
OUTPUT_JSON = 'datasets/estructura_arduino_simulado.json'

# Leer plantilla base (acceder al primer objeto si es lista)
with open(INPUT_JSON, 'r') as f:
    base_event = json.load(f)[0]

# Inicialización
registros_simulados = []
fecha_hora_base = datetime.strptime("2025-06-01 07:00:00", "%Y-%m-%d %H:%M:%S")
total_actual = 0

for i in range(300):
    evento = base_event.copy()
    evento['ID'] = str(i + 1)

    # Calcular fecha y hora del evento (incrementos de 10 minutos)
    fecha_hora_evento = fecha_hora_base + timedelta(minutes=i * 10)
    fecha_str = fecha_hora_evento.strftime("%d/%m/%Y")
    hora_str = fecha_hora_evento.strftime("%H:%M:%S")
    timestamp_str = fecha_hora_evento.strftime("%Y-%m-%d %H:%M:%S")

    evento['FECHA'] = fecha_str
    evento['HORA'] = hora_str
    evento['HORA_PC'] = hora_str  # puedes cambiar si deseas simular desfase
    evento['TIMESTAMP'] = timestamp_str  # ahora representa fecha y hora exactas

    # ENTRADA o SALIDA
    tipo_evento = random.choice(["ENTRADA", "SALIDA"])
    evento['EVENTO'] = tipo_evento

    if tipo_evento == "ENTRADA":
        cantidad = random.randint(1, 5)
        total_actual += cantidad
        evento['DESCRIPCION'] = f"{cantidad} estudiantes entraron al aula"
    else:
        cantidad = random.randint(1, min(5, total_actual)) if total_actual > 0 else 0
        total_actual -= cantidad
        evento['DESCRIPCION'] = f"{cantidad} estudiantes salieron del aula"

    evento['TOTAL'] = str(total_actual)

    # LEDs
    evento['LED1'] = "ON" if total_actual > 0 else "OFF"
    evento['LED2'] = "ON" if total_actual > 4 else "OFF"
    evento['LED3'] = "ON" if total_actual > 8 else "OFF"

    registros_simulados.append(evento)

# Guardar nuevo archivo JSON
with open(OUTPUT_JSON, 'w') as f:
    json.dump(registros_simulados, f, indent=4, ensure_ascii=False)

print(f"✅ Simulación generada con TIMESTAMP real en: {OUTPUT_JSON}")
