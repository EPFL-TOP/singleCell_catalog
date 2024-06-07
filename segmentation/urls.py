from django.urls import path
from . import views
from django.contrib.staticfiles.storage import staticfiles_storage
from django.views.generic.base import RedirectView

from .views import image_view, save_selected_region

urlpatterns = [
    path(r"", views.index, name="index"),
    path(r"bokeh_dashboard", views.bokeh_dashboard, name="bokeh_dashboard"),
    path(r"bokeh_summary_dashboard", views.bokeh_summary_dashboard, name="bokeh_summary_dashboard"),
    #path('download/', views.index, name='download_file'),
    #path("bokeh_template", views.bokeh_server, name="bokeh_template"),
    #path("image_template", views.image_view, name="image_template"),
    #path('', views.index, name='index'),
    #path('favicon.ico', RedirectView.as_view(url=staticfiles_storage.url('images/favicon.ico'))),
    #path("segmentation", views.index),

]



#urlpatterns += [
#    path('', image_view, name='image_view'),
#    path('save-selected-region/', save_selected_region, name='save_selected_region'),
#]
