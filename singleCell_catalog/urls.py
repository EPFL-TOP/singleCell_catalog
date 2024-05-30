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

from segmentation import views
from bokeh_django import autoload, directory, document, static_extensions


urlpatterns = [
    path(r"segmentation/", views.index, name="index"),
    path(r"segmentation/bokeh_dashboard", views.bokeh_dashboard, name="bokeh_dashboard"),
    path('admin/', admin.site.urls),
]


urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

bokeh_apps = [
    autoload("segmentation/bokeh_dashboard", views.segmentation_handler),
]

urlpatterns += [
    path('accounts/', include('django.contrib.auth.urls')),
]