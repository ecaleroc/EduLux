# ------------------------------------------------------------------------------
# Archivo: api/urls.py
# Definición de las URLs para la aplicación 'api'.
# ------------------------------------------------------------------------------
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from api.views import RegistroArduinoViewSet, MLModelPerformanceViewSet, IluminacionInteligenteViewSet, ERPIntegrationViewSet

# Crea un router para registrar automáticamente los ViewSets
router = DefaultRouter()
router.register(r'registros', RegistroArduinoViewSet)
router.register(r'ml-performance', MLModelPerformanceViewSet, basename='ml-performance')
router.register(r'iluminacion', IluminacionInteligenteViewSet, basename='iluminacion')
router.register(r'erp-integration', ERPIntegrationViewSet, basename='erp-integration')


urlpatterns = [
    path('', include(router.urls)), # Incluye las URLs generadas por el router
    # Puedes añadir URLs manuales si es necesario
]