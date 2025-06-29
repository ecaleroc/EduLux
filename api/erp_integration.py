
# ------------------------------------------------------------------------------
# Archivo: api/erp_integration.py
# Implementación de la interconexión con el sistema ERP ficticio vía SSH.
# ------------------------------------------------------------------------------
import paramiko
import json
import os
import io # Para manejar claves desde string si es necesario

# --- Configuración SSH ---
# ADVERTENCIA: En un entorno de producción real, estas credenciales
# NO deben estar directamente en el código. Usa variables de entorno
# o un sistema de gestión de secretos seguro.

# Ruta a la clave privada generada con PuTTYgen (o ssh-keygen)
# Asegúrate de que esta ruta sea correcta y que la clave tenga permisos restrictivos (ej. 600)
# Si tu clave .ppk tiene una contraseña, especifícala en 'password'.
PRIVATE_KEY_PATH = '/path/to/your/private_key.ppk' # ¡ACTUALIZA ESTA RUTA!
# Para ejemplo, podrías usar: os.path.expanduser('~/.ssh/id_rsa') si generaste con ssh-keygen

# Credenciales del servidor ERP ficticio
ERP_HOST = 'tu.servidor.erp.com' # ¡CAMBIA ESTO! (Puede ser localhost para pruebas)
ERP_PORT = 22 # Puerto SSH estándar
ERP_USER = 'erp_user' # ¡CAMBIA ESTO!
ERP_REMOTE_PATH = '/home/erp_user/data_from_sensors/' # Ruta remota donde se guardarán los archivos

def authenticate_and_connect_ssh(data_to_send=None):
    """
    Establece una conexión SSH segura con el ERP ficticio, autentica
    con clave privada y simula el envío/consulta de datos.

    Args:
        data_to_send (dict, optional): Datos JSON para enviar al ERP.
                                        Se guardará como 'sensor_data_<timestamp>.json'.
    """
    client = paramiko.SSHClient()
    # Permite añadir hosts desconocidos. Para producción, se recomienda
    # load_system_host_keys() y/o añadir hosts manualmente.
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        # Cargar la clave privada
        # Paramiko puede tener problemas con .ppk directamente.
        # Si no funciona, convierte tu .ppk a formato OpenSSH con PuTTYgen
        # (Conversions -> Export OpenSSH key) y usa esa clave.
        if not os.path.exists(PRIVATE_KEY_PATH):
            print(f"Error: La clave privada no se encontró en {PRIVATE_KEY_PATH}")
            return {"status": "error", "message": "Clave privada no encontrada."}

        try:
            # Intenta cargar como RSA (común para PuTTYgen o ssh-keygen)
            key = paramiko.RSAKey.from_private_key_file(PRIVATE_KEY_PATH, password=None) # Si tiene contraseña, pasala aquí
        except paramiko.SSHException:
            # Si RSA falla, intenta con DSA (menos común hoy en día)
            key = paramiko.dsskey.DSSKey.from_private_key_file(PRIVATE_KEY_PATH, password=None)
        except Exception as e:
            print(f"Error al cargar la clave SSH: {e}")
            return {"status": "error", "message": f"Error al cargar la clave SSH: {e}"}

        print(f"Intentando conectar a {ERP_USER}@{ERP_HOST}:{ERP_PORT}...")
        client.connect(hostname=ERP_HOST, port=ERP_PORT, username=ERP_USER, pkey=key, timeout=10)
        print("Conexión SSH establecida exitosamente.")

        sftp = client.open_sftp()
        print("Canal SFTP abierto.")

        response_messages = []

        # Asegurarse de que el directorio remoto exista
        try:
            sftp.stat(ERP_REMOTE_PATH)
        except FileNotFoundError:
            sftp.mkdir(ERP_REMOTE_PATH)
            response_messages.append(f"Directorio remoto {ERP_REMOTE_PATH} creado.")
            print(f"Directorio remoto {ERP_REMOTE_PATH} creado.")


        if data_to_send:
            timestamp_str = datetime.now().strftime("%Y%m%d%H%M%S")
            local_file_name = f'sensor_data_{timestamp_str}.json'
            remote_file_path_full = os.path.join(ERP_REMOTE_PATH, local_file_name)

            # Crea un archivo temporal local para escribir los datos
            with open(local_file_name, 'w', encoding='utf-8') as f:
                json.dump(data_to_send, f, indent=4)

            sftp.put(local_file_name, remote_file_path_full)
            response_messages.append(f"Datos enviados al ERP en: {remote_file_path_full}")
            print(f"Datos enviados a {remote_file_path_full} en el ERP.")

            # Opcional: limpiar el archivo local temporal
            os.remove(local_file_name)

        # Ejemplo de consulta: listar archivos en el directorio remoto
        stdin, stdout, stderr = client.exec_command(f'ls -l {ERP_REMOTE_PATH}')
        output = stdout.read().decode('utf-8')
        error = stderr.read().decode('utf-8')

        if output:
            response_messages.append(f"Listado de archivos en ERP:\n{output}")
            print(f"Salida del comando remoto:\n{output}")
        if error:
            response_messages.append(f"Error al ejecutar comando remoto: {error}")
            print(f"Error al ejecutar comando remoto: {error}")


        sftp.close()
        client.close()
        print("Conexión SSH cerrada.")
        return {"status": "success", "message": "Conexión ERP exitosa.", "details": response_messages}

    except paramiko.AuthenticationException:
        print("Error de autenticación. Verifique la clave privada, la contraseña o el nombre de usuario.")
        return {"status": "error", "message": "Error de autenticación SSH. Clave o credenciales incorrectas."}
    except paramiko.SSHException as ssh_err:
        print(f"Error SSH: {ssh_err}")
        return {"status": "error", "message": f"Error de conexión SSH: {ssh_err}"}
    except Exception as e:
        print(f"Ocurrió un error inesperado durante la conexión SSH: {e}")
        return {"status": "error", "message": f"Error inesperado: {e}"}

