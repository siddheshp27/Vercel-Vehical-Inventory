from django.urls import path
from . import views


urlpatterns = [
    path('', views.test, name="test"),
    path('getData', views.getData),
    path('putData', views.putData),
    path('image', views.ImageView.as_view(), name='image'),
]
