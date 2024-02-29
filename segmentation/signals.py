from django.db.models.signals import post_save,post_delete
#I have used django user model to use post save, post delete.
from django.contrib.auth.models import User
from segmentation.models import Contour

from django.dispatch import receiver

#@receiver(post_save,sender=Contour)
#def post_create_contour(sender,instance,created,**kwargs):
#    if created:
#        #write your logic here
#        print("CONTOUR Created")
        
#@receiver(post_delete,sender=Contour)
#def post_delete_contour(sender,instance,*args,**kwargs):
#    #write your login when user profile is deleted.
#    print("CONTOUR Profile Deleted")
#    if instance.pixels_data_contour:
#        instance.pixels_data_contour.delete()
#        print("pixels_data_contour Deleted")
#
#    if instance.pixels_data_inside:
#        instance.pixels_data_inside.delete()
#        print("pixels_data_inside Deleted")

