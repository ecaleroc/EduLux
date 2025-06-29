# ------------------------------------------------------------------------------
# Archivo: api/data_loader.py
# Carga y preprocesamiento inicial de los datos desde estructura_arduino.json.
# ------------------------------------------------------------------------------
import json
import os
import pandas as pd
from datetime import datetime, time, date

# Simula la ruta del archivo JSON, asume que está en la raíz del proyecto Django
# (donde está manage.py). Ajusta según sea necesario.
DATA_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'estructura_arduino.json')

def load_arduino_data():
    """
    Carga los datos desde el archivo estructura_arduino.json.
    """
    if not os.path.exists(DATA_FILE_PATH):
        print(f"Error: El archivo de datos no se encuentra en {DATA_FILE_PATH}")
        return []
    try:
        with open(DATA_FILE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError as e:
        print(f"Error al decodificar JSON: {e}")
        return []
    except Exception as e:
        print(f"Error al cargar el archivo de datos: {e}")
        return []

def preprocess_data(raw_data):
    """
    Preprocesa los datos brutos del Arduino para su uso en Machine Learning.
    - Convierte 'ON'/'OFF' a 1/0 para los LEDs.
    - Convierte FECHA y HORA a formatos adecuados.
    - Crea DataFrame de Pandas.
    """
    processed_data = []
    for record in raw_data:
        try:
            # Convierte estados de LED a numérico
            led1_num = 1 if record.get('LED1') == 'ON' else 0
            led2_num = 1 if record.get('LED2') == 'ON' else 0
            led3_num = 1 if record.get('LED3') == 'ON' else 0

            # Intenta parsear la fecha y hora
            # Asume que la fecha es DD/MM/YYYY
            fecha_obj = datetime.strptime(record.get('FECHA'), '%d/%m/%Y').date()
            # Asume que la hora es HH:MM:SS
            hora_obj = datetime.strptime(record.get('HORA'), '%H:%M:%S').time()
            hora_pc_obj = datetime.strptime(record.get('HORA_PC'), '%H:%M:%S').time()

            processed_data.append({
                'ID': record.get('ID'),
                'TIMESTAMP': int(record.get('TIMESTAMP')),
                'FECHA': fecha_obj,
                'HORA': hora_obj,
                'HORA_PC': hora_pc_obj,
                'EVENTO': record.get('EVENTO'),
                'DESCRIPCION': record.get('DESCRIPCION'),
                'TOTAL': int(record.get('TOTAL')),
                'LED1_NUM': led1_num,
                'LED2_NUM': led2_num,
                'LED3_NUM': led3_num
            })
        except ValueError as ve:
            print(f"Error de formato de fecha/hora en registro {record.get('ID')}: {ve}")
            continue
        except Exception as e:
            print(f"Error inesperado al procesar registro {record.get('ID')}: {e}")
            continue

    df = pd.DataFrame(processed_data)
    return df

# Función para cargar datos y guardarlos en la base de datos de Django
def populate_db_from_json():
    """
    Carga los datos desde el JSON y los guarda en el modelo RegistroArduino.
    Evita duplicados basándose en 'id_registro' y 'timestamp'.
    """
    from api.models import RegistroArduino # Importa aquí para evitar circular imports

    raw_data = load_arduino_data()
    if not raw_data:
        print("No hay datos para cargar en la base de datos.")
        return

    records_to_create = []
    existing_ids = set(RegistroArduino.objects.values_list('id_registro', flat=True))

    for record in raw_data:
        record_id = record.get('ID')
        # Verificar si el ID ya existe para evitar duplicados.
        # Una estrategia más robusta podría ser ID + TIMESTAMP o un hash del registro.
        if record_id in existing_ids:
            # print(f"Saltando registro {record_id} porque ya existe.")
            continue

        try:
            # Asegúrate de que los tipos de datos coincidan con el modelo Django
            records_to_create.append(RegistroArduino(
                id_registro=record.get('ID'),
                timestamp=int(record.get('TIMESTAMP')),
                fecha=datetime.strptime(record.get('FECHA'), '%d/%m/%Y').date(),
                hora=datetime.strptime(record.get('HORA'), '%H:%M:%S').time(),
                hora_pc=datetime.strptime(record.get('HORA_PC'), '%H:%M:%S').time(),
                evento=record.get('EVENTO'),
                descripcion=record.get('DESCRIPCION'),
                total_estudiantes=int(record.get('TOTAL')),
                led1_estado=record.get('LED1'),
                led2_estado=record.get('LED2'),
                led3_estado=record.get('LED3'),
            ))
        except Exception as e:
            print(f"Error al preparar el registro {record.get('ID')} para la base de datos: {e}")
            continue

    if records_to_create:
        RegistroArduino.objects.bulk_create(records_to_create, ignore_conflicts=True)
        print(f"Se insertaron {len(records_to_create)} nuevos registros de Arduino en la base de datos.")
    else:
        print("No se encontraron nuevos registros para insertar.")
