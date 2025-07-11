# Generated by Django 5.2.3 on 2025-07-07 05:27

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='MLModelPerformance',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('model_name', models.CharField(help_text='Nombre del modelo de ML.', max_length=100, unique=True)),
                ('accuracy', models.FloatField(blank=True, help_text='Precisión (Accuracy) del modelo.', null=True)),
                ('f1_score', models.FloatField(blank=True, help_text='F1-Score del modelo.', null=True)),
                ('precision', models.FloatField(blank=True, help_text='Precisión (Precision) del modelo.', null=True)),
                ('recall', models.FloatField(blank=True, help_text='Exhaustividad (Recall) del modelo.', null=True)),
                ('rmse', models.FloatField(blank=True, help_text='RMSE (si aplica para regresión).', null=True)),
                ('silhouette_score', models.FloatField(blank=True, help_text='Silhouette Score (para clustering).', null=True)),
                ('training_date', models.DateTimeField(auto_now=True, help_text='Fecha y hora del último entrenamiento.')),
                ('is_best_model', models.BooleanField(default=False, help_text='Indica si este es el modelo actualmente seleccionado como el mejor.')),
            ],
            options={
                'verbose_name': 'Rendimiento del Modelo ML',
                'verbose_name_plural': 'Rendimiento de los Modelos ML',
                'ordering': ['-training_date'],
            },
        ),
        migrations.CreateModel(
            name='RegistroArduino',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('id_registro', models.CharField(help_text='ID único del registro en el Arduino.', max_length=100, unique=True)),
                ('timestamp', models.DateTimeField(help_text='Fecha y hora exacta del evento.')),
                ('fecha', models.DateField(help_text='Fecha del evento.')),
                ('hora', models.TimeField(help_text='Hora del evento (formato Arduino).')),
                ('hora_pc', models.TimeField(help_text='Hora del evento según el PC.')),
                ('evento', models.CharField(help_text='Tipo de evento (INFO, ENTRADA, SALIDA, ERROR).', max_length=255)),
                ('descripcion', models.TextField(help_text='Descripción detallada del evento.')),
                ('total_estudiantes', models.IntegerField(help_text='Número total de estudiantes detectados en el aula.')),
                ('led1_estado', models.CharField(help_text='Estado del LED1 (ON/OFF).', max_length=3)),
                ('led2_estado', models.CharField(help_text='Estado del LED2 (ON/OFF).', max_length=3)),
                ('led3_estado', models.CharField(help_text='Estado del LED3 (ON/OFF).', max_length=3)),
            ],
            options={
                'verbose_name': 'Registro de Arduino',
                'verbose_name_plural': 'Registros de Arduino',
                'ordering': ['timestamp'],
            },
        ),
    ]
