"""singleCell_catalog URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.urls import include
from django.views.generic import RedirectView
from django.conf import settings
from django.conf.urls.static import static

from django.contrib.staticfiles.storage import staticfiles_storage

from bokeh_django import autoload
from segmentation import views



urlpatterns = [
    path(r"segmentation/", views.index_test, name="index"),
    path('admin/', admin.site.urls),
    path("segmentation/sea-surface-temp", views.index),
        #path("segmentation", views.sea_surface),

]


urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

bokeh_apps = [
    #autoload("segmentation", views.sea_surface_handler) ,
    autoload("segmentation/sea-surface-temp", views.bokeh_handler),

]