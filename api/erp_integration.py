import paramiko
import json
import os
import io
import socket
from datetime import datetime
from typing import Optional, Dict, Any

# --- Configuración SSH ---
# NOTA: En producción usa variables de entorno o un gestor de secretos
ERP_HOST = 'localhost'  # Cambiar por la IP/hostname del servidor SFTP
ERP_PORT = 22           # Puerto SSH estándar
ERP_USER = 'User' # Usuario para autenticación
ERP_REMOTE_PATH = '/datasets/'  # Ruta donde se guardarán los archivos
#PRIVATE_KEY_PATH = os.path.expanduser('~/.ssh/id_rsa')  # Ruta a clave privada
PRIVATE_KEY_PATH = os.path.join(os.path.dirname(__file__), '..\ssh_connector\claves', 'id_rsa')
PASSWORD = '1175321'  # Opcional: contraseña si se usa autenticación mixta

def authenticate_and_connect_ssh(data_to_send: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Establece conexión SSH/SFTP segura con autenticación por clave privada y/o contraseña.
    
    Args:
        data_to_send: Diccionario con datos a enviar (se guardará como JSON)
    
    Returns:
        Dict con status, message y detalles de la operación
    """
    # Validación inicial
    if not os.path.exists(PRIVATE_KEY_PATH):
        error_msg = f"Clave privada no encontrada en: {PRIVATE_KEY_PATH}"
        print(error_msg)
        return {"status": "error", "message": error_msg}

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        # 1. Cargar clave privada
        try:
            key = paramiko.RSAKey.from_private_key_file(PRIVATE_KEY_PATH, password=PASSWORD)
        except paramiko.ssh_exception.PasswordRequiredException:
            return {"status": "error", "message": "La clave privada requiere passphrase"}
        except paramiko.ssh_exception.SSHException:
            try:
                key = paramiko.Ed25519Key.from_private_key_file(PRIVATE_KEY_PATH, password=PASSWORD)
            except:
                return {"status": "error", "message": "Formato de clave privada no válido"}

        # 2. Establecer conexión
        print(f"Conectando a {ERP_USER}@{ERP_HOST}:{ERP_PORT}...")
        client.connect(
            hostname=ERP_HOST,
            port=ERP_PORT,
            username=ERP_USER,
            pkey=key,
            password=PASSWORD,  # Solo si se usa autenticación mixta
            timeout=20,
            look_for_keys=False,
            allow_agent=False
        )
        print("Conexión SSH establecida")

        # 3. Operaciones SFTP
        sftp = client.open_sftp()
        response_messages = []
        
        # Crear directorio remoto si no existe
        try:
            sftp.stat(ERP_REMOTE_PATH)
        except FileNotFoundError:
            sftp.mkdir(ERP_REMOTE_PATH)
            msg = f"Directorio remoto creado: {ERP_REMOTE_PATH}"
            response_messages.append(msg)
            print(msg)

        # 4. Enviar datos si se proporcionaron
        if data_to_send:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                remote_filename = f"sensor_data_{timestamp}.json"
                remote_path = f"{ERP_REMOTE_PATH}/{remote_filename}"
                
                # Usar StringIO para evitar crear archivos temporales
                with io.StringIO() as file_obj:
                    json.dump(data_to_send, file_obj, indent=2)
                    file_obj.seek(0)
                    sftp.putfo(file_obj, remote_path)
                
                msg = f"Datos enviados a: {remote_path}"
                response_messages.append(msg)
                print(msg)
            except Exception as e:
                print(f"Error al enviar datos: {str(e)}")
                response_messages.append(f"Error al enviar datos: {str(e)}")

        # 5. Listar archivos (verificación)
        try:
            stdin, stdout, stderr = client.exec_command(f'ls -la {ERP_REMOTE_PATH}')
            output = stdout.read().decode().strip()
            if output:
                print(f"Contenido remoto:\n{output}")
                response_messages.append(f"Archivos remotos:\n{output}")
        except Exception as e:
            print(f"Error al listar archivos: {str(e)}")

        return {
            "status": "success",
            "message": "Operación SFTP completada",
            "details": response_messages
        }

    except paramiko.AuthenticationException:
        error_msg = "Error de autenticación. Verifique credenciales/clave."
        print(error_msg)
        return {"status": "error", "message": error_msg}
    except paramiko.SSHException as e:
        error_msg = f"Error SSH: {str(e)}"
        print(error_msg)
        return {"status": "error", "message": error_msg}
    except socket.timeout:
        error_msg = "Timeout: El servidor no respondió"
        print(error_msg)
        return {"status": "error", "message": error_msg}
    except Exception as e:
        error_msg = f"Error inesperado: {str(e)}"
        print(error_msg)
        return {"status": "error", "message": error_msg}
    finally:
        try:
            if 'sftp' in locals():
                sftp.close()
            client.close()
            print("Conexión cerrada")
        except:
            pass

# Ejemplo de uso
if __name__ == "__main__":
    # Datos de prueba
    test_data = {
        "sensor_id": "temp_001",
        "value": 25.4,
        "timestamp": datetime.now().isoformat()
    }
    
    # Probar conexión sin enviar datos
    print("\nPrueba 1: Conexión básica")
    result = authenticate_and_connect_ssh()
    print(json.dumps(result, indent=2))
    
    # Probar enviando datos
    print("\nPrueba 2: Envío de datos")
    result = authenticate_and_connect_ssh(test_data)
    print(json.dumps(result, indent=2))

def check_ssh_server():
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(10)
    try:
        result = sock.connect_ex((ERP_HOST, ERP_PORT))
        return result == 0  # True si el puerto está accesible
    except Exception as e:
        print(f"Error al verificar servidor SSH: {e}")
        return False
    finally:
        sock.close()