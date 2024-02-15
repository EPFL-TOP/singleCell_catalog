from django.urls import path
from . import views
from django.contrib.staticfiles.storage import staticfiles_storage
from django.views.generic.base import RedirectView



urlpatterns = [
    path(r"", views.index, name="index"),
    path(r"", views.bokeh_server, name="bokeh_template"),
    #path('', views.index, name='index'),
    #path('favicon.ico', RedirectView.as_view(url=staticfiles_storage.url('images/favicon.ico'))),
    #path("segmentation", views.index),

]



