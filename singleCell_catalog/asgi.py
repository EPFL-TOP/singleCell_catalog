#"""
#ASGI config for singleCell_catalog project.
#
#It exposes the ASGI callable as a module-level variable named ``application``.
#
#For more information on this file, see
#https://docs.djangoproject.com/en/3.2/howto/deployment/asgi/
#"""
#
#import os
#
#from django.core.asgi import get_asgi_application
#
#os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'singleCell_catalog.settings')
#
#application = get_asgi_application()


from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter, URLRouter
from django.apps import apps

bokeh_app_config = apps.get_app_config('bokeh_django')

application = ProtocolTypeRouter({
    'websocket': AuthMiddlewareStack(URLRouter(bokeh_app_config.routes.get_websocket_urlpatterns())),
    'http': AuthMiddlewareStack(URLRouter(bokeh_app_config.routes.get_http_urlpatterns())),
})
