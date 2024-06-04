from django.contrib import admin
from .models import Sample, Frame, CellFlag, CellStatus, Contour, Experiment, ExperimentalDataset, CellID, CellROI, ContourSeg
# Register your models here.

class CellFlagAdmin(admin.ModelAdmin):
    search_fields = ["cell_roi__frame__sample__file_name"]

class ContourSegAdmin(admin.ModelAdmin):
    search_fields = ["cell_roi__frame__sample__file_name"]

class ContourAdmin(admin.ModelAdmin):
    search_fields = ["cell_roi__frame__sample__file_name"]

class CellROIAdmin(admin.ModelAdmin):
    search_fields = ["frame__sample__file_name"]

class CellIDAdmin(admin.ModelAdmin):
    search_fields = ["sample__file_name"]

class CellStatusAdmin(admin.ModelAdmin):
    search_fields = ["cellid_cellstatus__sample__file_name"]

class FrameAdmin(admin.ModelAdmin):
    search_fields = ["sample__file_name"]

class SampleAdmin(admin.ModelAdmin):
    search_fields = ["file_name"]

admin.site.register(Experiment)
admin.site.register(ExperimentalDataset)

admin.site.register(Sample, SampleAdmin)
admin.site.register(Frame, FrameAdmin)
admin.site.register(CellFlag, CellFlagAdmin)
admin.site.register(CellStatus, CellStatusAdmin)
admin.site.register(Contour, ContourAdmin)
admin.site.register(ContourSeg, ContourSegAdmin)
admin.site.register(CellID, CellIDAdmin)
admin.site.register(CellROI, CellROIAdmin)
