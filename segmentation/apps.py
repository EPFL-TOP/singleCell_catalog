from django.apps import AppConfig


class SegmentationConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'segmentation'


    def ready(self):
        import segmentation.signals #accounts is a name of app