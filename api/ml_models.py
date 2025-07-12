
# ------------------------------------------------------------------------------
# Archivo: api/ml_models.py
# Implementación, entrenamiento y comparación de los modelos de Machine Learning.
# ------------------------------------------------------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_squared_error
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralCoclustering
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import joblib # Para guardar y cargar modelos
import os

# Ruta para guardar y cargar modelos
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trained_models')
os.makedirs(MODEL_DIR, exist_ok=True)

# Nombre del archivo para guardar el mejor modelo
BEST_MODEL_FILENAME = os.path.join(MODEL_DIR, 'best_led_prediction_model.joblib')

class MLTrainer:
    """
    Clase para manejar el entrenamiento, evaluación y selección
    de los modelos de Machine Learning.
    """
    def __init__(self):
        self.models = {}
        self.performance_metrics = {}
        self.best_model_name = None
        self.best_model_accuracy = -1

    def train_and_evaluate_models(self, df):
        """
        Entrena y evalúa los 10 modelos de ML especificados.
        Para la predicción de LEDs, se enfocará en modelos de clasificación.
        Los modelos de clustering y detección de anomalías se usarán para
        ejemplificar su aplicación en este contexto, aunque no directamente
        para la predicción de LEDs 'ON'/'OFF' en este flujo principal.
        """
        if df.empty:
            print("DataFrame vacío, no se pueden entrenar modelos.")
            return

        # Preprocesamiento de los estados de LED a numérico si no se hizo al cargar
        if 'led1_estado' in df.columns and 'LED1_NUM' not in df.columns:
            df['LED1_NUM'] = df['led1_estado'].apply(lambda x: 1 if x == 'ON' else 0)
            df['LED2_NUM'] = df['led2_estado'].apply(lambda x: 1 if x == 'ON' else 0)
            df['LED3_NUM'] = df['led3_estado'].apply(lambda x: 1 if x == 'ON' else 0)


        # Características (TOTAL de estudiantes)
        X = df[['total_estudiantes']] # Usar 'total_estudiantes' del modelo Django
        # Objetivos (estado de cada LED)
        y_led1 = df['LED1_NUM']
        y_led2 = df['LED2_NUM']
        y_led3 = df['LED3_NUM']

        # Escala las características (importante para algunos modelos como SVM, ANN)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.models['scaler'] = scaler # Guardar el scaler para usarlo en la predicción

        # Dividir datos para entrenamiento y prueba (para modelos supervisados)
        X_train, X_test, y_train_led1, y_test_led1 = train_test_split(X_scaled, y_led1, test_size=0.3, random_state=42)
        _, _, y_train_led2, y_test_led2 = train_test_split(X_scaled, y_led2, test_size=0.3, random_state=42)
        _, _, y_train_led3, y_test_led3 = train_test_split(X_scaled, y_led3, test_size=0.3, random_state=42)


        print("Iniciando entrenamiento y evaluación de modelos...")

        # ======================================================================
        # Modelos de CLASIFICACIÓN (directamente aplicables a LEDs ON/OFF)
        # ======================================================================

        # 1. Support Vector Machine (SVC)
        print("Entrenando SVM para LEDs...")
        self._train_and_evaluate_classifier(SVC(random_state=42), X_train, X_test, y_train_led1, y_test_led1, 'SVM_LED1')
        self._train_and_evaluate_classifier(SVC(random_state=42), X_train, X_test, y_train_led2, y_test_led2, 'SVM_LED2')
        self._train_and_evaluate_classifier(SVC(random_state=42), X_train, X_test, y_train_led3, y_test_led3, 'SVM_LED3')

        # 2. Decision Tree Classifier
        print("Entrenando Decision Tree para LEDs...")
        self._train_and_evaluate_classifier(DecisionTreeClassifier(random_state=42), X_train, X_test, y_train_led1, y_test_led1, 'DecisionTree_LED1')
        self._train_and_evaluate_classifier(DecisionTreeClassifier(random_state=42), X_train, X_test, y_train_led2, y_test_led2, 'DecisionTree_LED2')
        self._train_and_evaluate_classifier(DecisionTreeClassifier(random_state=42), X_train, X_test, y_train_led3, y_test_led3, 'DecisionTree_LED3')

        # 3. Artificial Neural Network (ANN)
        print("Entrenando ANN para LEDs...")
        # Las ANNs requieren un formato específico de entrada y salida
        self._train_and_evaluate_ann(X_train, X_test, y_train_led1, y_test_led1, 'ANN_LED1')
        self._train_and_evaluate_ann(X_train, X_test, y_train_led2, y_test_led2, 'ANN_LED2')
        self._train_and_evaluate_ann(X_train, X_test, y_train_led3, y_test_led3, 'ANN_LED3')

        # 4. Logistic Regression
        print("Entrenando Logistic Regression para LEDs...")
        self._train_and_evaluate_classifier(LogisticRegression(random_state=42), X_train, X_test, y_train_led1, y_test_led1, 'LogisticRegression_LED1')
        self._train_and_evaluate_classifier(LogisticRegression(random_state=42), X_train, X_test, y_train_led2, y_test_led2, 'LogisticRegression_LED2')
        self._train_and_evaluate_classifier(LogisticRegression(random_state=42), X_train, X_test, y_train_led3, y_test_led3, 'LogisticRegression_LED3')

        # 5. Gaussian Naive Bayes
        print("Entrenando Gaussian Naive Bayes para LEDs...")
        self._train_and_evaluate_classifier(GaussianNB(), X_train, X_test, y_train_led1, y_test_led1, 'GaussianNB_LED1')
        self._train_and_evaluate_classifier(GaussianNB(), X_train, X_test, y_train_led2, y_test_led2, 'GaussianNB_LED2')
        self._train_and_evaluate_classifier(GaussianNB(), X_train, X_test, y_train_led3, y_test_led3, 'GaussianNB_LED3')

        # ======================================================================
        # Modelos de CLUSTERING (para análisis de patrones de presencia)
        # No se evaluarán con accuracy/F1 para la predicción de LEDs directamente,
        # sino que se usarán para encontrar grupos en los datos de 'TOTAL'.
        # Se puede usar silhouette_score para calidad del clustering.
        # ======================================================================
        from sklearn.metrics import silhouette_score
        print("\nEntrenando modelos de Clustering (para análisis de datos de TOTAL)...")

        # Se usará el DataFrame completo de X_scaled para clustering
        # 6. KMeans
        try:
            # Asumimos 3-5 clusters para probar, se podría optimizar con el método del codo
            kmeans_model = KMeans(n_clusters=3, random_state=42, n_init=10)
            kmeans_labels = kmeans_model.fit_predict(X_scaled)
            if len(np.unique(kmeans_labels)) > 1: # Asegurarse de que haya más de 1 cluster para silhouette
                kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
            else:
                kmeans_silhouette = None
            self.performance_metrics['KMeans'] = {'Silhouette_Score': kmeans_silhouette}
            print(f"KMeans - Silhouette Score: {kmeans_silhouette:.2f}" if kmeans_silhouette is not None else "KMeans - No Silhouette Score (1 cluster)")
            self.models['KMeans'] = kmeans_model
        except Exception as e:
            print(f"Error al entrenar KMeans: {e}")

        # 7. DBSCAN (requiere ajuste de parámetros epsilon y min_samples)
        # Los valores por defecto pueden no ser adecuados para todos los datasets.
        # Aquí se usan valores de ejemplo.
        try:
            dbscan_model = DBSCAN(eps=0.5, min_samples=5)
            dbscan_labels = dbscan_model.fit_predict(X_scaled)
            if len(np.unique(dbscan_labels)) > 1 and -1 in dbscan_labels and len(np.unique(dbscan_labels)) > 2:
                 # DBSCAN puede crear un cluster -1 para ruido. Si solo hay 2 clusters (ruido y uno más), silhouette no es ideal.
                 # Necesitamos al menos dos clusters válidos (no -1).
                core_samples_mask = np.zeros_like(dbscan_labels, dtype=bool)
                core_samples_mask[dbscan_model.core_sample_indices_] = True
                unique_labels = set(dbscan_labels)
                n_clusters_ = len(unique_labels) - (1 if -1 in unique_labels else 0)
                if n_clusters_ >= 2:
                    dbscan_silhouette = silhouette_score(X_scaled, dbscan_labels)
                else:
                    dbscan_silhouette = None
            else:
                dbscan_silhouette = None
            self.performance_metrics['DBSCAN'] = {'Silhouette_Score': dbscan_silhouette}
            print(f"DBSCAN - Silhouette Score: {dbscan_silhouette:.2f}" if dbscan_silhouette is not None else "DBSCAN - No Silhouette Score (menos de 2 clusters válidos)")
            self.models['DBSCAN'] = dbscan_model
        except Exception as e:
            print(f"Error al entrenar DBSCAN: {e}")


        # 8. Gaussian Mixture Model (GMM)
        try:
            gmm_model = GaussianMixture(n_components=3, random_state=42)
            gmm_labels = gmm_model.fit_predict(X_scaled)
            if len(np.unique(gmm_labels)) > 1:
                gmm_silhouette = silhouette_score(X_scaled, gmm_labels)
            else:
                gmm_silhouette = None
            self.performance_metrics['GMM'] = {'Silhouette_Score': gmm_silhouette}
            print(f"GMM - Silhouette Score: {gmm_silhouette:.2f}" if gmm_silhouette is not None else "GMM - No Silhouette Score (1 cluster)")
            self.models['GMM'] = gmm_model
        except Exception as e:
            print(f"Error al entrenar GMM: {e}")

        # 9. Agglomerative Clustering
        try:
            agg_model = AgglomerativeClustering(n_clusters=3)
            agg_labels = agg_model.fit_predict(X_scaled)
            if len(np.unique(agg_labels)) > 1:
                agg_silhouette = silhouette_score(X_scaled, agg_labels)
            else:
                agg_silhouette = None
            self.performance_metrics['Agglomerative'] = {'Silhouette_Score': agg_silhouette}
            print(f"Agglomerative - Silhouette Score: {agg_silhouette:.2f}" if agg_silhouette is not None else "Agglomerative - No Silhouette Score (1 cluster)")
            self.models['Agglomerative'] = agg_model
        except Exception as e:
            print(f"Error al entrenar Agglomerative Clustering: {e}")

        # 10. Spectral Clustering (para este dataset simple, puede que no sea ideal o falle con n_components=1)
        # En la vida real, requiere una matriz de afinidad o similitud. Para 'TOTAL', es un caso simple.
        try:
            # Spectral Clustering en scikit-learn requiere al menos 2 componentes válidos.
            # Asegúrate de que n_clusters sea <= número de muestras.
            n_clusters_spectral = min(3, len(X_scaled)) # Ajusta si hay pocas muestras
            if n_clusters_spectral > 1:
                spectral_model = SpectralCoclustering(n_clusters=n_clusters_spectral, random_state=42)
                # SpectralCoclustering es para co-clustering de matrices, no solo para clustering de filas
                # Para clustering de filas estándar, usar SpectralClustering
                from sklearn.cluster import SpectralClustering
                spectral_model = SpectralClustering(n_clusters=n_clusters_spectral, random_state=42, assign_labels='kmeans')
                spectral_labels = spectral_model.fit_predict(X_scaled)
                if len(np.unique(spectral_labels)) > 1:
                    spectral_silhouette = silhouette_score(X_scaled, spectral_labels)
                else:
                    spectral_silhouette = None
                self.performance_metrics['Spectral'] = {'Silhouette_Score': spectral_silhouette}
                print(f"Spectral - Silhouette Score: {spectral_silhouette:.2f}" if spectral_silhouette is not None else "Spectral - No Silhouette Score (1 cluster)")
                self.models['Spectral'] = spectral_model
            else:
                print("Spectral Clustering requiere al menos 2 clusters y suficientes muestras.")
                self.performance_metrics['Spectral'] = {'Silhouette_Score': None}
        except Exception as e:
            print(f"Error al entrenar Spectral Clustering: {e}")


        # ======================================================================
        # Modelo de DETECCIÓN DE ANOMALÍAS
        # ======================================================================
        # 11. Isolation Forest (para detectar valores atípicos en 'TOTAL')
        # Isolation Forest es útil para identificar patrones de presencia inusuales.
        print("\nEntrenando Isolation Forest (para detección de anomalías en TOTAL)...")
        try:
            iso_forest_model = IsolationForest(random_state=42)
            iso_forest_model.fit(X_scaled)
            # Predice si una muestra es una anomalía (-1) o no (1)
            # anomaly_predictions = iso_forest_model.predict(X_scaled)
            self.models['IsolationForest'] = iso_forest_model
            self.performance_metrics['IsolationForest'] = {'Description': 'Modelo entrenado para detección de anomalías.'}
            print("Isolation Forest entrenado.")
        except Exception as e:
            print(f"Error al entrenar Isolation Forest: {e}")

        # Guardar métricas en la base de datos
        self._save_performance_to_db()

    def _train_and_evaluate_classifier(self, model, X_train, X_test, y_train, y_test, model_name):
        """
        Función auxiliar para entrenar y evaluar un modelo de clasificación.
        """
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='binary', zero_division=0) # 'binary' para 0/1
            precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
            recall = recall_score(y_test, y_pred, average='binary', zero_division=0)

            self.models[model_name] = model
            self.performance_metrics[model_name] = {
                'Accuracy': accuracy,
                'F1_Score': f1,
                'Precision': precision,
                'Recall': recall
            }
            print(f"{model_name} - Accuracy: {accuracy:.2f}, F1-Score: {f1:.2f}")

            # Identificar el mejor modelo para LED prediction (se hará al final)
            if "LED" in model_name: # Solo para los modelos de predicción de LEDs
                if accuracy > self.best_model_accuracy:
                    self.best_model_accuracy = accuracy
                    self.best_model_name = model_name

        except Exception as e:
            print(f"Error al entrenar/evaluar {model_name}: {e}")
            self.performance_metrics[model_name] = {'Error': str(e)}

    def _train_and_evaluate_ann(self, X_train, X_test, y_train, y_test, model_name):
        """
        Función auxiliar para entrenar y evaluar una Red Neuronal Artificial (ANN).
        """
        try:
            # Definir la arquitectura del modelo
            model = Sequential([
                Dense(10, activation='relu', input_shape=(X_train.shape[1],)),
                Dense(5, activation='relu'),
                Dense(1, activation='sigmoid') # Sigmoid para clasificación binaria
            ])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=0) # verbose=0 para no imprimir cada epoch

            y_pred_proba = model.predict(X_test)
            y_pred = (y_pred_proba > 0.5).astype(int) # Convertir probabilidades a 0 o 1

            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
            precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
            recall = recall_score(y_test, y_pred, average='binary', zero_division=0)

            self.models[model_name] = model
            self.performance_metrics[model_name] = {
                'Accuracy': accuracy,
                'F1_Score': f1,
                'Precision': precision,
                'Recall': recall
            }
            print(f"{model_name} - Accuracy: {accuracy:.2f}, F1-Score: {f1:.2f}")

            if "LED" in model_name:
                if accuracy > self.best_model_accuracy:
                    self.best_model_accuracy = accuracy
                    self.best_model_name = model_name

        except Exception as e:
            print(f"Error al entrenar/evaluar {model_name}: {e}")
            self.performance_metrics[model_name] = {'Error': str(e)}

    def _save_performance_to_db(self):
        """
        Guarda las métricas de rendimiento en la base de datos de Django.
        """
        from api.models import MLModelPerformance # Importa aquí para evitar circular imports
        from django.utils import timezone # Para obtener la hora actual

        # Limpiar la tabla de rendimiento si se desea, o actualizar solo los existentes
        MLModelPerformance.objects.all().update(is_best_model=False) # Resetear el indicador de "mejor modelo"

        for model_name, metrics in self.performance_metrics.items():
            if 'Error' in metrics:
                print(f"Saltando {model_name} debido a un error previo.")
                continue

            # Crear o actualizar el registro de rendimiento
            obj, created = MLModelPerformance.objects.update_or_create(
                model_name=model_name,
                defaults={
                    'accuracy': metrics.get('Accuracy'),
                    'f1_score': metrics.get('F1_Score'),
                    'precision': metrics.get('Precision'),
                    'recall': metrics.get('Recall'),
                    'silhouette_score': metrics.get('Silhouette_Score'),
                    'training_date': timezone.now(),
                    'is_best_model': (model_name == self.best_model_name)
                }
            )
            if created:
                print(f"Guardado nuevo registro de rendimiento para: {model_name}")
            else:
                print(f"Actualizado registro de rendimiento para: {model_name}")

        # Marcar el modelo predilecto (si existe)
        if self.best_model_name:
            MLModelPerformance.objects.filter(model_name=self.best_model_name).update(is_best_model=True)
            print(f"Modelo '{self.best_model_name}' marcado como el mejor.")
        else:
            print("No se pudo determinar un modelo 'mejor' de clasificación.")

    def save_best_model(self):
        """
        Guarda el modelo de predicción de LEDs clasificado como el mejor.
        Este modelo será en realidad un diccionario de 3 modelos (uno por cada LED).
        """
        if self.best_model_name and self.best_model_name.startswith('SVM_LED'): # O DecisionTree, ANN
            # Asumimos que el mejor modelo es el que se usa para los 3 LEDs (ej. SVM para todos)
            # Esto puede ser una simplificación. En un caso real, cada LED podría tener su mejor modelo.
            # Para este ejemplo, tomamos el tipo de modelo (ej. SVM) que dio el mejor resultado global.
            model_type = self.best_model_name.split('_')[0] # Ej. 'SVM'

            best_led_models = {
                'LED1': self.models.get(f'{model_type}_LED1'),
                'LED2': self.models.get(f'{model_type}_LED2'),
                'LED3': self.models.get(f'{model_type}_LED3'),
                'scaler': self.models.get('scaler') # Guardar el scaler también
            }
            if all(best_led_models.values()):
                joblib.dump(best_led_models, BEST_MODEL_FILENAME)
                print(f"Mejor conjunto de modelos ({model_type}) guardado en {BEST_MODEL_FILENAME}")
            else:
                print(f"Advertencia: No se pudo guardar el conjunto de modelos {model_type} completo.")
        elif self.best_model_name and self.best_model_name.startswith('DecisionTree_LED'):
            model_type = self.best_model_name.split('_')[0]
            best_led_models = {
                'LED1': self.models.get(f'{model_type}_LED1'),
                'LED2': self.models.get(f'{model_type}_LED2'),
                'LED3': self.models.get(f'{model_type}_LED3'),
                'scaler': self.models.get('scaler') # Guardar el scaler también
            }
            if all(best_led_models.values()):
                joblib.dump(best_led_models, BEST_MODEL_FILENAME)
                print(f"Mejor conjunto de modelos ({model_type}) guardado en {BEST_MODEL_FILENAME}")
            else:
                print(f"Advertencia: No se pudo guardar el conjunto de modelos {model_type} completo.")
        elif self.best_model_name and self.best_model_name.startswith('ANN_LED'):
             model_type = self.best_model_name.split('_')[0]
             best_led_models = {
                'LED1': self.models.get(f'{model_type}_LED1'),
                'LED2': self.models.get(f'{model_type}_LED2'),
                'LED3': self.models.get(f'{model_type}_LED3'),
                'scaler': self.models.get('scaler') # Guardar el scaler también
            }
             # Keras models need to be saved differently
             if all(best_led_models.values()):
                 # Save Keras models one by one
                 best_led_models['LED1'].save(os.path.join(MODEL_DIR, 'ANN_LED1.keras'))
                 best_led_models['LED2'].save(os.path.join(MODEL_DIR, 'ANN_LED2.keras'))
                 best_led_models['LED3'].save(os.path.join(MODEL_DIR, 'ANN_LED3.keras'))
                 joblib.dump(best_led_models['scaler'], os.path.join(MODEL_DIR, 'ANN_scaler.joblib'))
                 print(f"Mejor conjunto de modelos ({model_type}) guardado en {MODEL_DIR}")
             else:
                 print(f"Advertencia: No se pudo guardar el conjunto de modelos {model_type} completo.")
        else:
            print("No se determinó un modelo 'mejor' de clasificación para guardar.")

    def load_best_model(self):
        """
        Carga el modelo de predicción de LEDs clasificado como el mejor.
        """
        if os.path.exists(BEST_MODEL_FILENAME):
            print(f"Cargando el mejor modelo de predicción desde {BEST_MODEL_FILENAME}")
            # Determinar el tipo de modelo guardado para cargarlo correctamente
            # Esto es una simplificación. Un enfoque más robusto implicaría guardar el nombre del tipo de modelo
            # junto con los modelos mismos.
            loaded_models = joblib.load(BEST_MODEL_FILENAME)
            return loaded_models
        else:
            print(f"No se encontró el archivo del mejor modelo en {BEST_MODEL_FILENAME}.")
            return None

    def predict_led_states(self, total_estudiantes):
        """
        Usa el mejor modelo entrenado para predecir el estado de los LEDs.
        """
        loaded_models = self.load_best_model()
        if not loaded_models:
            print("No hay modelos de predicción cargados. Entrene los modelos primero.")
            return {"LED1": "OFF", "LED2": "OFF", "LED3": "OFF", "error": "Modelos no entrenados."}

        # Extraer el scaler y los modelos de LED individuales
        scaler = loaded_models.get('scaler')
        model_led1 = loaded_models.get('LED1')
        model_led2 = loaded_models.get('LED2')
        model_led3 = loaded_models.get('LED3')

        if not all([scaler, model_led1, model_led2, model_led3]):
            print("Error: No se pudo cargar un modelo completo (scaler o modelos de LED faltantes).")
            return {"LED1": "OFF", "LED2": "OFF", "LED3": "OFF", "error": "Modelos incompletos."}

        # Preparar la entrada para la predicción
        input_data = np.array([[total_estudiantes]])
        input_scaled = scaler.transform(input_data)

        # Predecir el estado de cada LED
        pred_led1 = model_led1.predict(input_scaled)[0]
        pred_led2 = model_led2.predict(input_scaled)[0]
        pred_led3 = model_led3.predict(input_scaled)[0]

        # Convertir predicciones a 'ON'/'OFF'
        status_led1 = 'ON' if (isinstance(pred_led1, np.ndarray) and pred_led1[0] > 0.5) or (isinstance(pred_led1, (int, float)) and pred_led1 > 0.5) else 'OFF'
        status_led2 = 'ON' if (isinstance(pred_led2, np.ndarray) and pred_led2[0] > 0.5) or (isinstance(pred_led2, (int, float)) and pred_led2 > 0.5) else 'OFF'
        status_led3 = 'ON' if (isinstance(pred_led3, np.ndarray) and pred_led3[0] > 0.5) or (isinstance(pred_led3, (int, float)) and pred_led3 > 0.5) else 'OFF'


        return {
            "LED1": status_led1,
            "LED2": status_led2,
            "LED3": status_led3
        }
