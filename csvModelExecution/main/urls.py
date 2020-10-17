from django.conf import settings
from django.conf.urls.static import static
from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('insert', views.insert, name='insert'),
    path('delete', views.delete, name='delete'),
    path('join', views.join, name='join'),
    path('model-create', views.model_create, name='model-create'),
    path('plot', views.get_svg, name='plot'),
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)