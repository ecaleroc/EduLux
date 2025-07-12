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
        print(f"Error al decodificar JSON en {DATA_FILE_PATH}: {e}")
        return []
    except Exception as e:
        print(f"Error al cargar el archivo de datos {DATA_FILE_PATH}: {e}")
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
        record_id = record.get('ID', 'UNKNOWN') # Default for logging
        try:
            led1_num = 1 if record.get('LED1') == 'ON' else 0
            led2_num = 1 if record.get('LED2') == 'ON' else 0
            led3_num = 1 if record.get('LED3') == 'ON' else 0

            fecha_str = record.get('FECHA')
            hora_str = record.get('HORA')
            hora_pc_str = record.get('HORA_PC')

            # Ensure they are strings before stripping and parsing
            if not all(isinstance(s, str) for s in [fecha_str, hora_str, hora_pc_str]):
                print(f"Skipping record {record_id} in preprocess_data: Date/time values are not strings. Types: Fecha={type(fecha_str)}, Hora={type(hora_str)}, Hora_PC={type(hora_pc_str)}")
                continue

            # Robust conversion for TOTAL
            total_val = None
            try:
                total_val = int(record.get('TOTAL', '0'))
            except (ValueError, TypeError):
                print(f"Skipping record {record_id}: 'TOTAL' is not a valid integer (value: '{record.get('TOTAL')}', type: {type(record.get('TOTAL'))}).")
                continue # Skip this record entirely if TOTAL is invalid


            fecha_obj = datetime.strptime(fecha_str.strip(), '%d/%m/%Y').date()
            hora_obj = datetime.strptime(hora_str.strip(), '%H:%M:%S').time()
            hora_pc_obj = datetime.strptime(hora_pc_str.strip(), '%H:%M:%S').time()


            processed_data.append({
                'ID': record_id,
                # 'TIMESTAMP': timestamp_val, # Removed as it's not used for ML features and causes parsing issues
                'FECHA': fecha_obj,
                'HORA': hora_obj,
                'HORA_PC': hora_pc_obj,
                'EVENTO': record.get('EVENTO', ''),
                'DESCRIPCION': record.get('DESCRIPCION', ''),
                'TOTAL': total_val,
                'LED1_NUM': led1_num,
                'LED2_NUM': led2_num,
                'LED3_NUM': led3_num
            })
        except ValueError as ve:
            print(f"Error de formato de fecha/hora en record {record_id} during preprocess_data: {ve}. Data: Fecha='{fecha_str}', Hora='{hora_str}', Hora_PC='{hora_pc_str}'. Skipping.")
            continue
        except TypeError as te:
            print(f"TypeError in record {record_id} during preprocess_data: {te}. Skipping.")
            continue
        except Exception as e:
            print(f"Unexpected error in record {record_id} during preprocess_data: {e}. Skipping.")
            continue

    df = pd.DataFrame(processed_data)
    return df

# Función para cargar datos y guardarlos en la base de datos de Django
def populate_db_from_json():
    """
    Carga los datos desde el JSON y los guarda en el modelo RegistroArduino.
    Evita duplicados basándose en 'id_registro'.
    """
    from api.models import RegistroArduino # Importa aquí para evitar circular imports

    raw_data = load_arduino_data()
    if not raw_data:
        print("No hay datos para cargar en la base de datos.")
        return

    records_to_create = []
    # Fetch existing IDs once for efficiency
    existing_ids = set(RegistroArduino.objects.values_list('id_registro', flat=True))

    for record in raw_data:
        _id_registro = record.get('ID', 'UNKNOWN') # Default for logging
        if _id_registro in existing_ids:
            continue

        try:
            # Explicitly get and validate types for all fields before use
            _fecha = record.get('FECHA')
            _hora = record.get('HORA')
            _hora_pc = record.get('HORA_PC')
            _evento = record.get('EVENTO')
            _descripcion = record.get('DESCRIPCION')
            _total = record.get('TOTAL')
            _led1 = record.get('LED1')
            _led2 = record.get('LED2')
            _led3 = record.get('LED3')

            # --- Type and Value Validation ---
            if not isinstance(_id_registro, str):
                print(f"Skipping record (ID: {_id_registro}): 'ID' is not a string (type: {type(_id_registro)}).")
                continue
            if not isinstance(_fecha, str):
                print(f"Skipping record {_id_registro}: 'FECHA' is not a string (type: {type(_fecha)}).")
                continue
            if not isinstance(_hora, str):
                print(f"Skipping record {_id_registro}: 'HORA' is not a string (type: {type(_hora)}).")
                continue
            if not isinstance(_hora_pc, str):
                print(f"Skipping record {_id_registro}: 'HORA_PC' is not a string (type: {type(_hora_pc)}).")
                continue
            if not isinstance(_evento, str):
                print(f"Skipping record {_id_registro}: 'EVENTO' is not a string (type: {type(_evento)}).")
                continue
            if not isinstance(_descripcion, str):
                print(f"Skipping record {_id_registro}: 'DESCRIPCION' is not a string (type: {type(_descripcion)}).")
                continue
            
            # Robust conversion for TOTAL
            try:
                total_int = int(_total)
            except (ValueError, TypeError):
                print(f"Skipping record {_id_registro}: 'TOTAL' is not a valid integer (value: '{_total}', type: {type(_total)}).")
                continue

            if not isinstance(_led1, str) or _led1 not in ['ON', 'OFF']:
                print(f"Skipping record {_id_registro}: 'LED1' is invalid (value: {_led1}, type: {type(_led1)}).")
                continue
            if not isinstance(_led2, str) or _led2 not in ['ON', 'OFF']:
                print(f"Skipping record {_id_registro}: 'LED2' is invalid (value: {_led2}, type: {type(_led2)}).")
                continue
            if not isinstance(_led3, str) or _led3 not in ['ON', 'OFF']:
                print(f"Skipping record {_id_registro}: 'LED3' is invalid (value: {_led3}, type: {type(_led3)}).")
                continue

            # --- Conversion after validation, with .strip() for robustness ---
            # Combine FECHA and HORA to create the datetime object for 'timestamp' field
            combined_datetime_str = f"{_fecha.strip()} {_hora.strip()}"
            timestamp_obj = datetime.strptime(combined_datetime_str, '%d/%m/%Y %H:%M:%S')

            fecha_obj = datetime.strptime(_fecha.strip(), '%d/%m/%Y').date()
            hora_obj = datetime.strptime(_hora.strip(), '%H:%M:%S').time()
            hora_pc_obj = datetime.strptime(_hora_pc.strip(), '%H:%M:%S').time()

            records_to_create.append(RegistroArduino(
                id_registro=_id_registro,
                timestamp=timestamp_obj, # Use the combined datetime object
                fecha=fecha_obj,
                hora=hora_obj,
                hora_pc=hora_pc_obj,
                evento=_evento,
                descripcion=_descripcion,
                total_estudiantes=total_int,
                led1_estado=_led1,
                led2_estado=_led2,
                led3_estado=_led3,
            ))
        except ValueError as ve:
            print(f"Error de formato/valor en record {_id_registro} during populate_db_from_json: {ve}. Data: {record}. Skipping.")
            continue
        except TypeError as te:
            print(f"TypeError in record {_id_registro} during populate_db_from_json: {te}. Data: {record}. Skipping.")
            continue
        except Exception as e:
            print(f"Unexpected error in record {_id_registro} during populate_db_from_json: {e}. Data: {record}. Skipping.")
            continue

    if records_to_create:
        try:
            RegistroArduino.objects.bulk_create(records_to_create, ignore_conflicts=True)
            print(f"Se insertaron {len(records_to_create)} nuevos registros de Arduino en la base de datos.")
        except Exception as e:
            print(f"Error crítico durante bulk_create: {e}")
            raise # Re-raise the original exception for the API to catch
    else:
        print("No se encontraron nuevos registros para insertar.")
