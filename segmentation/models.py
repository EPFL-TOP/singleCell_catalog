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
    experimental_dataset     = models.ForeignKey(ExperimentalDataset, default='',on_delete=models.CASCADE)
    file_name                = models.CharField(default='', max_length=500, help_text="name of the file (full path)")
    sample_quality           = models.CharField(max_length=200, choices=QUALITY, help_text="", default='High')
    keep_sample              = models.BooleanField(help_text="keep this sample flag", default=True)
    peaks_tod_div_validated  = models.BooleanField(help_text="peaks, tod, division validated flag", default=False)
    bf_features_validated    = models.BooleanField(help_text="other bright field features validated flag", default=False)

    def __str__(self):
        return 'position={0}'.format(self.file_name)

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
class CellStatus(models.Model):

    peaks                   = models.JSONField(default=dict, help_text="json to store the cell peaks intensities", blank=True)
    flags                   = models.JSONField(default=dict, help_text="json to store the cells flags (for each frame the cell exists)", blank=True)
    segmentation            = models.JSONField(default=dict, help_text="json to store the cell segmentation algorithms", blank=True)

    time_of_death           = models.FloatField(default=-9999, help_text="Cell time of death in minutes", blank=True)
    time_of_death_pred      = models.FloatField(default=-9999, help_text="Cell time of death in minutes predicted", blank=True)
    start_oscillation       = models.FloatField(default=-9999, help_text="Cell time start of oscillation in minutes", blank=True)
    end_oscillation         = models.FloatField(default=-9999, help_text="Cell time end of oscillation in minutes", blank=True)

    time_of_death_frame     = models.IntegerField(default=-999, help_text="Cell frame of death", blank=True)
    time_of_death_frame_pred = models.IntegerField(default=-999, help_text="Cell frame of death predicted", blank=True)
    start_oscillation_frame = models.IntegerField(default=-999, help_text="Cell frame start of oscillation", blank=True)
    end_oscillation_frame   = models.IntegerField(default=-999, help_text="Cell frame end of oscillation", blank=True)

    n_oscillations          = models.IntegerField(default=-999, help_text="Cell number of oscillations", blank=True)

    migrating               = models.BooleanField(help_text="migrating cell flag", default=False, blank=True)
    mask                    = models.BooleanField(help_text="mask cell flag", default=False, blank=True)


    def __str__(self):
        if  hasattr(self, 'cellid_cellstatus'):
            return 'cell={0}, sample={1}'.format(self.cellid_cellstatus.name, self.cellid_cellstatus.sample.file_name)
        else: return 'bad status...'

    class Meta:
        verbose_name = 'Cell status'
        verbose_name_plural = 'Cell statuses'

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
    roi_number = models.IntegerField(default=-999, help_text="ROI number")

    frame      = models.ForeignKey(Frame, default='',on_delete=models.CASCADE)
    cell_id    = models.ForeignKey(CellID, blank=True, null=True, default='', on_delete=models.SET_NULL)
    #contour     = models.OneToOneField("Contour", default='', null=True,on_delete=models.SET_DEFAULT)

    def __str__(self):
        if self.cell_id != None:
            if  hasattr(self, 'contour_cellroi'):
                return 'file={0}, frame={1}, roi={2}, cell={3}, contour={4}, id={5}'.format(self.frame.sample.file_name, self.frame.number,self.roi_number, self.cell_id.name, self.contour_cellroi.file_name, self.id)
            else:
                return 'file={0}, frame={1}, roi={2}, cell={3}, contour={4}, id={5}'.format(self.frame.sample.file_name, self.frame.number,self.roi_number, self.cell_id.name, None, self.id)

        else:
            if  hasattr(self, 'contour_cellroi'):
                return 'file={0}, frame={1}, roi={2}, cell={3}, contour={4}, id={5}'.format(self.frame.sample.file_name, self.frame.number,self.roi_number, self.cell_id, self.contour_cellroi.file_name, self.id)
            else:
                return 'file={0}, frame={1}, roi={2}, cell={3}, contour={4}, id={5}'.format(self.frame.sample.file_name, self.frame.number,self.roi_number, self.cell_id, None, self.id)


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
    number_of_pixels = models.PositiveIntegerField(default=0, help_text="Contour number of pixels")

    file_name        = models.CharField(default='', max_length=1000, help_text="json file name containing all the pixels")
    type             = models.CharField(max_length=200, choices=TYPE, help_text="contour type", default='auto')
    mode             = models.CharField(max_length=200, choices=MODE, help_text="contour type", default='cell_ROI')
    cell_roi         = models.OneToOneField(CellROI, default='', null=True, on_delete=models.CASCADE, related_name="contour_cellroi")


    def __str__(self):
        if self.cell_roi.cell_id != None:
            return 'type={0}, mode={1}, file={2}, frame={3}, roi={4}, cell={5}, contour={6}'.format(self.type, self.mode, self.cell_roi.frame.sample.file_name, self.cell_roi.frame.number, self.cell_roi.roi_number, self.cell_roi.cell_id.name, self.file_name)
        else:
            return 'type={0}, mode={1}, file={2}, frame={3}, roi={4}, cell={5}, contour={6}'.format(self.type, self.mode, self.cell_roi.frame.sample.file_name, self.cell_roi.frame.number, self.cell_roi.roi_number, self.cell_roi.cell_id, self.file_name)


#___________________________________________________________________________________________
class ContourSeg(models.Model):

    ALGO = (
        ('localthresholding', 'localthresholding'), 
        
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
    number_of_pixels = models.PositiveIntegerField(default=0, help_text="Contour number of pixels")

    #mask             = models.JSONField(help_text="pixels contour", default=dict)

    file_name        = models.CharField(default='', max_length=1000, help_text="json file containing the mask data")
    algo             = models.CharField(max_length=200, choices=ALGO, help_text="algorithm type", default='localthresholding')
    cell_roi         = models.ForeignKey(CellROI, default='', null=True, on_delete=models.CASCADE, related_name="contourseg_cellroi")

    def __str__(self):
        if self.cell_roi.cell_id != None:
            return 'algo={0}, file={1}, frame={2}, roi={3}, cell={4}, contour={5}, n pixels={6}, intensity={7}'.format(self.algo, self.cell_roi.frame.sample.file_name, self.cell_roi.frame.number, self.cell_roi.roi_number, self.cell_roi.cell_id.name, self.file_name,  self.number_of_pixels, self.intensity_mean)
        else:
            return 'algo={0}, file={1}, frame={2}, roi={3}, cell={4}, contour={5}, n pixels={6}, intensity={7}'.format(self.algo, self.cell_roi.frame.sample.file_name, self.cell_roi.frame.number, self.cell_roi.roi_number, self.cell_roi.cell_id, self.file_name, self.number_of_pixels, self.intensity_mean)


#___________________________________________________________________________________________
class CellFlag(models.Model):

    alive          = models.BooleanField(help_text="alive cell flag", default=True, blank=True)
    oscillating    = models.BooleanField(help_text="cell oscillation flag", default=False, blank=True)
    maximum        = models.BooleanField(help_text="maximum oscillation signal cell flag", default=False, blank=True)
    minimum        = models.BooleanField(help_text="minimum oscillation signal cell flag", default=False, blank=True)
    falling        = models.BooleanField(help_text="falling oscillation signal cell flag", default=False, blank=True)
    rising         = models.BooleanField(help_text="rising oscillation signal cell flag", default=False, blank=True)
    last_osc       = models.BooleanField(help_text="last oscillation signal cell flag", default=False, blank=True)

    mask           = models.BooleanField(help_text="mask cell flag", default=False, blank=True)
    dividing       = models.BooleanField(help_text="dividing cell flag", default=False, blank=True)
    double_nuclei  = models.BooleanField(help_text="double nuclei cell flag", default=False, blank=True)
    multiple_cells = models.BooleanField(help_text="multiple cells flag", default=False, blank=True)
    pair_cell      = models.BooleanField(help_text="pair cell flag", default=False, blank=True)
    flat           = models.BooleanField(help_text="flat cell flag", default=False, blank=True)
    round          = models.BooleanField(help_text="round cell flag", default=False, blank=True)
    elongated      = models.BooleanField(help_text="round cell flag", default=False, blank=True)
    
    cell_roi       = models.OneToOneField(CellROI, default='', null=True, on_delete=models.CASCADE, related_name="cellflag_cellroi")

    class Meta:
        verbose_name = 'Cell flags'
        verbose_name_plural = 'Cell flags'


    def __str__(self):
        if self.cell_roi.cell_id != None:
            return 'file={0}, frame={1}, roi={2}, cell={3}'.format(self.cell_roi.frame.sample.file_name, self.cell_roi.frame.number, self.cell_roi.roi_number, self.cell_roi.cell_id.name)
        else:
            return 'file={0}, frame={1}, roi={2}, cell={3}'.format(self.cell_roi.frame.sample.file_name, self.cell_roi.frame.number, self.cell_roi.roi_number, self.cell_roi.cell_id)
