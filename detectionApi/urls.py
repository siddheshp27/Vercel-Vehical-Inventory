from django.urls import path
from . import views


urlpatterns = [
    path('', views.test, name="test"),
    path('getData', views.getData),
    path('putData', views.putData),
    path('vehicalDetection', views.ImageView.as_view(), name='image'),
    path('drowsinessDetection', views.FaceView().as_view(), name='image1'),
    path('hospitalFinder', views.hospital),
]
