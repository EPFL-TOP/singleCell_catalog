from django.contrib import admin
from .models import Sample, Frame, CellFlag, CellStatus, Contour, Experiment, ExperimentalDataset, Data, Segmentation, SegmentationChannel, CellID, CellROI
# Register your models here.
admin.site.register(Sample)
admin.site.register(Frame)
admin.site.register(CellFlag)
admin.site.register(CellStatus)
admin.site.register(Contour)
admin.site.register(Experiment)
admin.site.register(ExperimentalDataset)
admin.site.register(Data)
admin.site.register(SegmentationChannel)
admin.site.register(Segmentation)
admin.site.register(CellID)
admin.site.register(CellROI)
