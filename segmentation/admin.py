from django.contrib import admin
from .models import Sample, Frame, CellFlag, CellStatus, Contour, Experiment, ExperimentalDataset, CellID, CellROI, ContourSeg
# Register your models here.

class CellFlagAdmin(admin.ModelAdmin):
    search_fields = ["cell_roi__frame__sample__file_name"]


class ContourSegAdmin(admin.ModelAdmin):
    search_fields = ["cell_roi__frame__sample__file_name"]

admin.site.register(Sample)
admin.site.register(Frame)
admin.site.register(CellFlag, CellFlagAdmin)
admin.site.register(CellStatus)
admin.site.register(Contour)
admin.site.register(ContourSeg, ContourSegAdmin)
admin.site.register(Experiment)
admin.site.register(ExperimentalDataset)
admin.site.register(CellID)
admin.site.register(CellROI)
