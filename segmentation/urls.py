from django.urls import path
from . import views
from django.contrib.staticfiles.storage import staticfiles_storage
from django.views.generic.base import RedirectView
from bokeh_django import autoload



urlpatterns = [
    path('', views.index, name='index'),
    path('favicon.ico', RedirectView.as_view(url=staticfiles_storage.url('images/favicon.ico'))),
    path("embedded-bokeh-app/", views.index),
]


bokeh_apps = [
    autoload("embedded-bokeh-app/", views.bokeh_handler) 
]