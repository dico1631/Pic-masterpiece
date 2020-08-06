from django.contrib import admin
from django.urls import path
# from django.conf.urls import url,include
import font_detect.views as views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('crosscolor/', views.crosscolor, name='crosscolor'),
    path('showchange/', views.showchange, name='showchange'),
]
