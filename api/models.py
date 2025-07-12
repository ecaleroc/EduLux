# Create your models here.

# ------------------------------------------------------------------------------
# Archivo: api/models.py
# Definición de los modelos de base de datos para los registros del Arduino
# y el rendimiento de los modelos ML.
# ------------------------------------------------------------------------------
from django.db import models

class RegistroArduino(models.Model):
    """
    Modelo para almacenar los registros de presencia estudiantil y estado de LEDs
    provenientes del dispositivo Arduino (simulado por el JSON).
    """
    id_registro = models.CharField(max_length=100, unique=True, help_text="ID único del registro en el Arduino.")
    timestamp = models.DateTimeField(help_text="Fecha y hora exacta del evento.")
    fecha = models.DateField(help_text="Fecha del evento.")
    hora = models.TimeField(help_text="Hora del evento (formato Arduino).")
    hora_pc = models.TimeField(help_text="Hora del evento según el PC.")
    evento = models.CharField(max_length=255, help_text="Tipo de evento (INFO, ENTRADA, SALIDA, ERROR).")
    descripcion = models.TextField(help_text="Descripción detallada del evento.")
    total_estudiantes = models.IntegerField(help_text="Número total de estudiantes detectados en el aula.")
    led1_estado = models.CharField(max_length=3, help_text="Estado del LED1 (ON/OFF).")
    led2_estado = models.CharField(max_length=3, help_text="Estado del LED2 (ON/OFF).")
    led3_estado = models.CharField(max_length=3, help_text="Estado del LED3 (ON/OFF).")

    class Meta:
        verbose_name = "Registro de Arduino"
        verbose_name_plural = "Registros de Arduino"
        ordering = ['timestamp'] # Ordenar por la marca de tiempo

    def __str__(self):
        return f"Reg: {self.id_registro} - Total: {self.total_estudiantes} ({self.fecha} {self.hora_pc})"

class MLModelPerformance(models.Model):
    """
    Modelo para almacenar las métricas de rendimiento de los diferentes
    modelos de Machine Learning entrenados.
    """
    model_name = models.CharField(max_length=100, unique=True, help_text="Nombre del modelo de ML.")
    accuracy = models.FloatField(null=True, blank=True, help_text="Precisión (Accuracy) del modelo.")
    f1_score = models.FloatField(null=True, blank=True, help_text="F1-Score del modelo.")
    precision = models.FloatField(null=True, blank=True, help_text="Precisión (Precision) del modelo.")
    recall = models.FloatField(null=True, blank=True, help_text="Exhaustividad (Recall) del modelo.")
    rmse = models.FloatField(null=True, blank=True, help_text="RMSE (si aplica para regresión).")
    silhouette_score = models.FloatField(null=True, blank=True, help_text="Silhouette Score (para clustering).")
    training_date = models.DateTimeField(auto_now=True, help_text="Fecha y hora del último entrenamiento.")
    is_best_model = models.BooleanField(default=False, help_text="Indica si este es el modelo actualmente seleccionado como el mejor.")

    class Meta:
        verbose_name = "Rendimiento del Modelo ML"
        verbose_name_plural = "Rendimiento de los Modelos ML"
        ordering = ['-training_date']

    def __str__(self):
        return f"{self.model_name} - Accuracy: {self.accuracy:.2f}"
