
# ------------------------------------------------------------------------------
# Archivo: api/views.py
# Lógica de las vistas de la API REST.
# ------------------------------------------------------------------------------
from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.decorators import action
from api.models import RegistroArduino, MLModelPerformance
from api.serializers import RegistroArduinoSerializer, MLModelPerformanceSerializer, LEDPredictionSerializer, ERPIntegrationSerializer
from api.data_loader import populate_db_from_json, preprocess_data, load_arduino_data
from api.ml_models import MLTrainer, BEST_MODEL_FILENAME, MODEL_DIR
from api.erp_integration import authenticate_and_connect_ssh, check_ssh_server # Importa la función SSH
import os
import joblib # Para cargar el scaler de ML
import pandas as pd # Importar pandas al inicio del archivo

# Inicializar el entrenador de ML. Se podría hacer en un Singleton o global para persistencia.
# Para este ejemplo, lo inicializamos una vez al cargar el módulo.
# En un entorno de producción real, los modelos se cargarían o reentrenarían
# de forma más robusta, quizás al iniciar el servidor o con tareas programadas.
ml_trainer = MLTrainer()

class RegistroArduinoViewSet(viewsets.ModelViewSet):
    """
    API endpoint que permite ver o editar los registros del Arduino.
    """
    queryset = RegistroArduino.objects.all().order_by('-fecha', '-hora_pc')
    serializer_class = RegistroArduinoSerializer

    @action(detail=False, methods=['post'], url_path='populate-from-json')
    def populate_db(self, request):
        """
        Endpoint para poblar la base de datos con los datos del archivo JSON.
        """
        try:
            populate_db_from_json()
            return Response({"status": "Base de datos poblada/actualizada correctamente."}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": f"Error al poblar la base de datos: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class MLModelPerformanceViewSet(viewsets.ReadOnlyModelViewSet):
    """
    API endpoint que permite ver el rendimiento de los modelos de ML.
    """
    queryset = MLModelPerformance.objects.all().order_by('-training_date')
    serializer_class = MLModelPerformanceSerializer

    @action(detail=False, methods=['post'])
    def train_models(self, request):
        """
        Endpoint para iniciar el entrenamiento y evaluación de los modelos ML.
        Obtiene los datos directamente de la base de datos de RegistroArduino.
        """
        try:
            # Cargar datos directamente desde la base de datos de Django
            # Convertir el QuerySet a una lista de diccionarios y luego a DataFrame
            # Asegúrate de que los nombres de las columnas coincidan con lo que espera ml_models.py
            # 'total_estudiantes', 'led1_estado', 'led2_estado', 'led3_estado'
            arduino_records = RegistroArduino.objects.all().values(
                'total_estudiantes', 'led1_estado', 'led2_estado', 'led3_estado'
            )
            df = pd.DataFrame(list(arduino_records))

            # --- DEBUGGING PRINTS ---
            print("DataFrame columns:", df.columns)
            print("DataFrame head:\n", df.head())
            # --- END DEBUGGING PRINTS ---

            if df.empty:
                return Response({"error": "No hay datos disponibles en la base de datos para entrenar los modelos."}, status=status.HTTP_400_BAD_REQUEST)

            # Validar que las columnas esenciales existan antes de usarlas
            required_columns = ['total_estudiantes', 'led1_estado', 'led2_estado', 'led3_estado']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                return Response(
                    {"error": f"Faltan columnas esenciales en los datos de la base de datos para el entrenamiento: {', '.join(missing_columns)}"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Convertir los estados de LED de 'ON'/'OFF' a 1/0 para el entrenamiento de ML
            df['LED1_NUM'] = df['led1_estado'].apply(lambda x: 1 if x == 'ON' else 0)
            df['LED2_NUM'] = df['led2_estado'].apply(lambda x: 1 if x == 'ON' else 0)
            df['LED3_NUM'] = df['led3_estado'].apply(lambda x: 1 if x == 'ON' else 0)

            ml_trainer.train_and_evaluate_models(df)
            ml_trainer.save_best_model() # Guarda el mejor modelo después de entrenar
            return Response({"status": "Modelos entrenados y evaluados correctamente. Rendimiento actualizado."}, status=status.HTTP_200_OK)
        except Exception as e:
            # Captura cualquier otra excepción no manejada y la devuelve como un error 500
            print(f"Error inesperado al entrenar modelos: {str(e)}")
            return Response({"error": f"Error al entrenar modelos: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class IluminacionInteligenteViewSet(viewsets.ViewSet):
    """
    API para la lógica de Iluminación Inteligente y predicción.
    """
    @action(detail=False, methods=['post'])
    def predict_led_state(self, request):
        """
        Predice el estado de los LEDs basado en el número de estudiantes.
        Requiere 'total_estudiantes' en el cuerpo de la solicitud.
        """
        serializer = LEDPredictionSerializer(data=request.data)
        if serializer.is_valid():
            total_estudiantes = serializer.validated_data['total_estudiantes']
            prediction = ml_trainer.predict_led_states(total_estudiantes)

            if "error" in prediction:
                return Response({"error": prediction["error"]}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            # Combina la entrada con la predicción para la respuesta
            response_data = {
                "total_estudiantes": total_estudiantes,
                "LED1": prediction.get("LED1", "OFF"),
                "LED2": prediction.get("LED2", "OFF"),
                "LED3": prediction.get("LED3", "OFF")
            }
            return Response(response_data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=False, methods=['get'])
    def get_current_lighting_status(self, request):
        """
        Obtiene el último estado de iluminación registrado en el Arduino.
        """
        latest_record = RegistroArduino.objects.order_by('-timestamp').first()
        if latest_record:
            return Response({
                "LED1": latest_record.led1_estado,
                "LED2": latest_record.led2_estado,
                "LED3": latest_record.led3_estado,
                "total_estudiantes": latest_record.total_estudiantes,
                "last_updated": latest_record.hora_pc
            }, status=status.HTTP_200_OK)
        return Response({"message": "No hay registros de iluminación disponibles."}, status=status.HTTP_404_NOT_FOUND)


class ERPIntegrationViewSet(viewsets.ViewSet):
    """
    API para la integración con el sistema ERP ficticio vía SSH.
    """
    @action(detail=False, methods=['post'])
    def send_data_to_erp(self, request):
        """
        Endpoint para enviar datos al ERP a través de SSH.
        Puede recibir 'data_to_send' en el cuerpo de la solicitud.
        Si no se envía data_to_send, intentará enviar el último registro de Arduino.
        """
        serializer = ERPIntegrationSerializer(data=request.data)
        if serializer.is_valid():
            data_to_send = serializer.validated_data.get('data_to_send')

            if not data_to_send:
                # Si no se proporcionan datos, enviar el último registro de Arduino
                latest_record = RegistroArduino.objects.order_by('-timestamp').first()
                if latest_record:
                    data_to_send = RegistroArduinoSerializer(latest_record).data
                    print(f"Enviando el último registro de Arduino al ERP: {data_to_send}")
                else:
                    return Response({"error": "No hay datos para enviar al ERP y no se encontró el último registro de Arduino."}, status=status.HTTP_400_BAD_REQUEST)

            erp_response = authenticate_and_connect_ssh(data_to_send)
            return Response(erp_response, status=status.HTTP_200_OK if erp_response.get("status") == "success" else status.HTTP_500_INTERNAL_SERVER_ERROR)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    #@action(detail=False, methods=['get'])
    #def test_erp_connection(self, request):
    #    """
    #    Endpoint para probar la conexión SSH con el ERP sin enviar datos.
    #    """
    #    erp_response = authenticate_and_connect_ssh(data_to_send=None)
    #    return Response(erp_response, status=status.HTTP_200_OK if erp_response.get("status") == "success" else status.HTTP_500_INTERNAL_SERVER_ERROR)

    @action(detail=False, methods=['get'])
    def test_erp_connection(self, request):
        """
        Endpoint para probar la conexión SSH con el ERP sin enviar datos.
        """
        if not check_ssh_server():
            return Response(
                {"status": "error", "message": "Servidor SSH no accesible en el puerto configurado"},
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )

        try:
            erp_response = authenticate_and_connect_ssh(data_to_send=None)
            return Response(
                erp_response,
                status=status.HTTP_200_OK if erp_response.get("status") == "success" else status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        except Exception as e:
            return Response(
                {"status": "error", "message": f"Error inesperado: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )