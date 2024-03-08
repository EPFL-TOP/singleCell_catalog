from django.db import models
from django.urls import reverse

# Create your models here.

#___________________________________________________________________________________________
class Experiment(models.Model):
    name         = models.CharField(max_length=200, help_text="name of the experiment (from RawDataset database).")
    date         = models.DateField(null=True, help_text="Date of the experiment (from RawDataset database).")
    description  = models.TextField(blank=True, max_length=2000, help_text="Description of the experiment (from RawDataset database).")

    file_name              = models.CharField(default='',max_length=500, help_text="name of the full unsplitted file (full path)")
    number_of_frames       = models.PositiveSmallIntegerField(default=0, help_text="Number of frames")
    number_of_channels     = models.PositiveSmallIntegerField(default=0, help_text="Number of channels")
    experiment_description = models.CharField(max_length=500, default='', help_text="description of the experiment")
    name_of_channels       = models.CharField(max_length=500, default='', help_text="name of the channels")
    date_of_acquisition    = models.DateTimeField(blank=True, null=True, help_text="Date of the experiment (from nd2 metadata).")

    def __str__(self):
        return '{0}, {1}'.format(self.name, self.date)
    
#___________________________________________________________________________________________
class ExperimentalDataset(models.Model):
    data_type        = models.CharField(default='', max_length=100, help_text='Type of data for this dataset (reflecting the the RCP storage categories)')
    data_name        = models.CharField(default='', max_length=100, help_text="Name of the experimental dataset folder on the RCP storage.")
    experiment       = models.ForeignKey(Experiment, on_delete=models.CASCADE)
    number_of_files  = models.PositiveSmallIntegerField(default=0, help_text="Number of files")
    files            = models.JSONField(null=True)

    def __str__(self):
        return '{0}, {1}'.format(self.data_type, self.data_name)

#    class Meta:
#        verbose_name = 'ExperimentalDataset'
#        verbose_name_plural = 'ExperimentalDatasets'

#___________________________________________________________________________________________
class Sample(models.Model):
    QUALITY = (
        ('High',   'High'),
        ('Medium', 'Medium'),
        ('Low',    'Low'),
    )
    experimental_dataset   = models.ForeignKey(ExperimentalDataset, default='',on_delete=models.CASCADE)
    file_name              = models.CharField(default='', max_length=500, help_text="name of the file (full path)")
#    number_of_frames       = models.PositiveSmallIntegerField(default=0, help_text="Number of frames")
#    number_of_channels     = models.PositiveSmallIntegerField(default=0, help_text="Number of channels")
#    experiment_description = models.CharField(max_length=500, default='', help_text="description of the experiment")
#    name_of_channels       = models.CharField(max_length=500, default='', help_text="name of the channels")
#    date                   = models.DateTimeField(blank=True, null=True)
    #USER specific
    sample_quality     = models.CharField(max_length=200, choices=QUALITY, help_text="")
    keep_sample        = models.BooleanField(help_text="keep this sample flag")

    def __str__(self):
        return 'name={0}, quality={1}'.format(self.file_name, self.sample_quality)

    def get_absolute_url(self):
        return reverse('sample-detail', args=[str(self.id)])



#___________________________________________________________________________________________
class Frame(models.Model):
    sample        = models.ForeignKey(Sample, on_delete=models.CASCADE)
    number        = models.PositiveSmallIntegerField(default=0, help_text="Frame number")
    height        = models.PositiveSmallIntegerField(default=0, help_text="Frame height")
    width         = models.PositiveSmallIntegerField(default=0, help_text="Frame width")
    time          = models.FloatField(default=-9999, help_text="Frame time im milliseconds")
    pos_x         = models.FloatField(default=-9999, help_text="Camera x position in microns")
    pos_y         = models.FloatField(default=-9999, help_text="Camera y position in microns")
    pos_z         = models.FloatField(default=-9999, help_text="Camera z position in microns")
    pixel_microns = models.FloatField(default=-9999, help_text="microns per pixel")
    #USER specific
    keep_sample = models.BooleanField(help_text="keep this sample flag")

    def __str__(self):
        return '{0}, {1}'.format(self.number, self.sample)

#___________________________________________________________________________________________
class Segmentation(models.Model):
    name                 = models.CharField(default='', max_length=200, help_text="name of the segmentation")
    experiment           = models.ForeignKey(Experiment, default='', on_delete=models.SET_DEFAULT)
    algorithm_type       = models.CharField(max_length=200, help_text="type of algorithm used")
    algorithm_version    = models.CharField(max_length=200, help_text="version of algorithm used")
    algorithm_parameters = models.JSONField(help_text="parameters of the algorithm used")

    def __str__(self):
        return '{0}, {1}, {2}, {3}'.format(self.name, self.algorithm_type, self.algorithm_version, self.algorithm_parameters)

#___________________________________________________________________________________________
class SegmentationChannel(models.Model):
    channel_name   = models.CharField(max_length=200, default='', help_text="name of the channel used for the segmentation")
    channel_number = models.PositiveSmallIntegerField(default=-1, help_text="channel number")
    segmentation   = models.ForeignKey(Segmentation, default='', on_delete=models.CASCADE)

    def __str__(self):
        return '{0}, {1}, {2}, {3}'.format(self.channel_name, self.channel_number, self.segmentation.name, self.segmentation.algorithm_type, self.segmentation.algorithm_version, self.segmentation.algorithm_parameters)

#___________________________________________________________________________________________
class Data(models.Model):
    #Only json files because the dimension correspond to the number of channels
    all_pixels    = models.JSONField(default=dict, help_text="Variables calculated over all pixels")
    single_pixels = models.JSONField(default=dict, help_text="Pixels individual coordinates and intensities")







#___________________________________________________________________________________________
class CellStatus(models.Model):

    peaks = models.JSONField(default=dict, help_text="json to store the cell peaks intensities", blank=True)

    time_of_death           = models.FloatField(default=-9999, help_text="Cell time of death in minutes", blank=True)
    start_oscillation       = models.FloatField(default=-9999, help_text="Cell time start of oscillation in minutes", blank=True)
    end_oscillation         = models.FloatField(default=-9999, help_text="Cell time end of oscillation in minutes", blank=True)

    time_of_death_frame     = models.PositiveSmallIntegerField(default=0, help_text="Cell frame of death", blank=True)
    start_oscillation_frame = models.PositiveSmallIntegerField(default=0, help_text="Cell frame start of oscillation", blank=True)
    end_oscillation_frame   = models.PositiveSmallIntegerField(default=0, help_text="Cell frame end of oscillation", blank=True)


    def __str__(self):
        if  hasattr(self, 'cellid_cellstatus'):
            return 'cell={0}, sample={1}'.format(self.cellid_cellstatus.name, self.cellid_cellstatus.sample.file_name)
        else: return 'bad status...'

#___________________________________________________________________________________________
class CellID(models.Model):
    name        = models.CharField(default='', max_length=20, help_text="cell name")
    sample      = models.ForeignKey(Sample, default='', on_delete=models.SET_DEFAULT)
    cell_status = models.OneToOneField(CellStatus, default='', null=True, on_delete=models.CASCADE, related_name="cellid_cellstatus")

    def __str__(self):
        return 'cell={0}, sample={1}'.format(self.name, self.sample.file_name)
    
#___________________________________________________________________________________________
class CellROI(models.Model):
    #Coordinates according to skimage.measure.regionprops.bbox
    #Bounding box (min_row, min_col, max_row, max_col). 

    min_row    = models.PositiveSmallIntegerField(default=0, help_text="skimage.measure.regionprops.bbox min row ROI and bottom in bokeh")
    min_col    = models.PositiveSmallIntegerField(default=0, help_text="skimage.measure.regionprops.bbox min col ROI and left in bokeh")
    max_row    = models.PositiveSmallIntegerField(default=0, help_text="skimage.measure.regionprops.bbox max row ROI and top in bokeh")
    max_col    = models.PositiveSmallIntegerField(default=0, help_text="skimage.measure.regionprops.bbox max col ROI and right in bokeh")
    roi_number = models.PositiveSmallIntegerField(default=-1, help_text="ROI number")

    frame      = models.ForeignKey(Frame, default='',on_delete=models.CASCADE)
    cell_id    = models.ForeignKey(CellID, blank=True, null=True, default='', on_delete=models.SET_NULL)
    #contour     = models.OneToOneField("Contour", default='', null=True,on_delete=models.SET_DEFAULT)

    def __str__(self):
        if self.cell_id != None:
            if  hasattr(self, 'contour_cellroi'):
                return 'file={0}, frame={1}, roi={2}, cell={3}, contour={4}'.format(self.frame.sample.file_name, self.frame.number,self.roi_number, self.cell_id.name, self.contour_cellroi.file_name)
            else:
                return 'file={0}, frame={1}, roi={2}, cell={3}, contour={4}'.format(self.frame.sample.file_name, self.frame.number,self.roi_number, self.cell_id.name, None)

        else:
            if  hasattr(self, 'contour_cellroi'):
                return 'file={0}, frame={1}, roi={2}, cell={3}, contour={4}'.format(self.frame.sample.file_name, self.frame.number,self.roi_number, self.cell_id, self.contour_cellroi.file_name)
            else:
                return 'file={0}, frame={1}, roi={2}, cell={3}, contour={4}'.format(self.frame.sample.file_name, self.frame.number,self.roi_number, self.cell_id, None)


#___________________________________________________________________________________________
class Contour(models.Model):

    MODE = (
        ('manual', 'manual'), 
        ('auto','auto'),
        )
    TYPE = (
        ('cell_ROI', 'cell_ROI'), 
        ('bkg_ROI',  'bkg_ROI'),
        ('cell_contour', 'cell_contour'), 

        )
    center_x_pix     = models.FloatField(default=-9999, help_text="Contour center x position in pixels")
    center_y_pix     = models.FloatField(default=-9999, help_text="Contour center y position in pixels")
    center_z_pix     = models.FloatField(default=-9999, help_text="Contour center z position in pixels")

    center_x_mic     = models.FloatField(default=-9999, help_text="Contour center x position in microns")
    center_y_mic     = models.FloatField(default=-9999, help_text="Contour center y position in microns")
    center_z_mic     = models.FloatField(default=-9999, help_text="Contour center z position in microns")

    intensity_mean   = models.JSONField(help_text="Contour mean intensity")
    intensity_std    = models.JSONField(help_text="Contour std intensity")
    intensity_sum    = models.JSONField(help_text="Contour sum intensity")
    intensity_max    = models.JSONField(help_text="Contour max intensity", default=dict)
    number_of_pixels = models.PositiveSmallIntegerField(default=-9999, help_text="Contour number of pixels")

    file_name        = models.CharField(default='', max_length=1000, help_text="json file name containing all the pixels")
    type             = models.CharField(max_length=200, choices=TYPE, help_text="contour type", default='auto')
    mode             = models.CharField(max_length=200, choices=MODE, help_text="contour type", default='cell_ROI')
    cell_roi         = models.OneToOneField(CellROI, default='', null=True, on_delete=models.CASCADE, related_name="contour_cellroi")

#    pixels_data_contour  = models.OneToOneField(Data, blank=True, null=True, default='', on_delete=models.CASCADE, help_text="pixels data of the contour", related_name="pixels_data_contour")
#    pixels_data_inside   = models.OneToOneField(Data, blank=True, null=True, default='', on_delete=models.CASCADE, help_text="pixels data inside the contour", related_name="pixels_data_inside")
#    segmentation_channel = models.ForeignKey(SegmentationChannel, default='', on_delete=models.CASCADE)

    def __str__(self):
        if self.cell_roi.cell_id != None:
            return 'type={0}, mode={1}, file={2}, frame={3}, roi={4}, cell={5}, contour={6}'.format(self.type, self.mode, self.cell_roi.frame.sample.file_name, self.cell_roi.frame.number, self.cell_roi.roi_number, self.cell_roi.cell_id.name, self.file_name)
        else:
            return 'type={0}, mode={1}, file={2}, frame={3}, roi={4}, cell={5}, contour={6}'.format(self.type, self.mode, self.cell_roi.frame.sample.file_name, self.cell_roi.frame.number, self.cell_roi.roi_number, self.cell_roi.cell_id, self.file_name)


#local status for
#___________________________________________________________________________________________
class CellFlag(models.Model):

    oscillating    = models.BooleanField(help_text="cell oscillation flag", default=True, blank=True)
    alive          = models.BooleanField(help_text="alive cell flag", default=True, blank=True)
    dividing       = models.BooleanField(help_text="dividing cell flag", default=True, blank=True)
    double_nuclei  = models.BooleanField(help_text="double nuclei cell flag", default=True, blank=True)
    multiple_cells = models.BooleanField(help_text="multiple cells flag", default=True, blank=True)
    pair_cell      = models.BooleanField(help_text="pair cell flag", default=True, blank=True)
    maximum        = models.BooleanField(help_text="maximum oscillation signal cell flag", default=True, blank=True)
    minimum        = models.BooleanField(help_text="minimum oscillation signal cell flag", default=True, blank=True)
    rising         = models.BooleanField(help_text="rising oscillation signal cell flag", default=True, blank=True)
    falling        = models.BooleanField(help_text="falling oscillation signal cell flag", default=True, blank=True)

    cell_roi       = models.OneToOneField(CellROI, default='', null=True, on_delete=models.CASCADE, related_name="cellflag_cellroi")

    class Meta:
        verbose_name = 'Cell flags'
        verbose_name_plural = 'Cell flags'
