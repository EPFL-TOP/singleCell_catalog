from django.contrib import admin
from .models import Sample, Frame, Cell, CellStatus, Contour, Experiment, ExperimentalDataset, Data, Segmentation, SegmentationChannel, CellFrame, CellID
# Register your models here.
admin.site.register(Sample)
admin.site.register(Frame)
admin.site.register(Cell)
admin.site.register(CellStatus)
admin.site.register(Contour)
admin.site.register(Experiment)
admin.site.register(ExperimentalDataset)
admin.site.register(Data)
admin.site.register(SegmentationChannel)
admin.site.register(Segmentation)
admin.site.register(CellFrame)
admin.site.register(CellID)
