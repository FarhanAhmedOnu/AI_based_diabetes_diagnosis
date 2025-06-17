# predictor/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.predict_diabetes, name='predict_diabetes')
]

# diabetes_project/urls.py
from django.urls import path, include

urlpatterns = [
    path('', include('predictor.urls'))
]
