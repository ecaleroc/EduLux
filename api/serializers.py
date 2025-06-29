
# ------------------------------------------------------------------------------
# Archivo: api/serializers.py
# Definición de serializadores para convertir datos de modelos a JSON y viceversa.
# ------------------------------------------------------------------------------
from rest_framework import serializers
from api.models import RegistroArduino, MLModelPerformance

class RegistroArduinoSerializer(serializers.ModelSerializer):
    """
    Serializador para el modelo RegistroArduino.
    Convierte instancias del modelo a representaciones JSON y viceversa.
    """
    class Meta:
        model = RegistroArduino
        fields = '__all__' # Incluye todos los campos del modelo

class MLModelPerformanceSerializer(serializers.ModelSerializer):
    """
    Serializador para el modelo MLModelPerformance.
    Muestra las métricas de rendimiento de los modelos ML.
    """
    class Meta:
        model = MLModelPerformance
        fields = '__all__'

class LEDPredictionSerializer(serializers.Serializer):
    """
    Serializador para la entrada y salida de predicciones de LEDs.
    """
    total_estudiantes = serializers.IntegerField(min_value=0, help_text="Número de estudiantes presentes.")
    LED1 = serializers.CharField(max_length=3, read_only=True, help_text="Estado predicho del LED1 (ON/OFF).")
    LED2 = serializers.CharField(max_length=3, read_only=True, help_text="Estado predicho del LED2 (ON/OFF).")
    LED3 = serializers.CharField(max_length=3, read_only=True, help_text="Estado predicho del LED3 (ON/OFF).")

class ERPIntegrationSerializer(serializers.Serializer):
    """
    Serializador para la integración con el ERP.
    """
    data_to_send = serializers.JSONField(required=False, help_text="Datos JSON para enviar al ERP.")
    status = serializers.CharField(read_only=True)
    message = serializers.CharField(read_only=True)
    details = serializers.JSONField(read_only=True)
