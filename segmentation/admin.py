from django.contrib import admin
from .models import Sample, Frame, Cell, CellStatus, Contour, Analysis, Project, Data
# Register your models here.
admin.site.register(Sample)
admin.site.register(Frame)
admin.site.register(Cell)
admin.site.register(CellStatus)
admin.site.register(Contour)
admin.site.register(Analysis)
admin.site.register(Project)
admin.site.register(Data)
