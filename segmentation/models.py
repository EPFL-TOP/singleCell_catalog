from django.db import models
from django.urls import reverse

# Create your models here.

#___________________________________________________________________________________________
class Experiment(models.Model):
    name         = models.CharField(max_length=200, help_text="name of the experiment.")
    date         = models.DateField(null=True, help_text="Date of the experiment.")
    description  = models.TextField(blank=True, max_length=2000, help_text="Description of the experiment.")

    def __str__(self):
        return '{0}, {1}'.format(self.name, self.date)
    
#___________________________________________________________________________________________
class ExperimentalDataset(models.Model):
    name       = models.CharField(default='',max_length=200, help_text="name of the experimental dataset")
    experiment = models.ForeignKey(Experiment, on_delete=models.CASCADE)

    def __str__(self):
        return self.name

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
    analysis               = models.ForeignKey(ExperimentalDataset, default='',on_delete=models.CASCADE)
    file_name              = models.CharField(max_length=500, help_text="name of the file (full path)")
    number_of_frames       = models.PositiveSmallIntegerField(default=0, help_text="Number of frames")
    number_of_channels     = models.PositiveSmallIntegerField(default=0, help_text="Number of channels")
    experiment_description = models.CharField(max_length=500, default='', help_text="description of the experiment")
    name_of_channels       = models.CharField(max_length=500, default='', help_text="name of the channels")
    date                   = models.DateTimeField(blank=True, null=True)
    #USER specific
    sample_quality     = models.CharField(max_length=200, choices=QUALITY, help_text="")
    keep_sample        = models.BooleanField(help_text="keep this sample flag")

    def __str__(self):
        return self.file_name

    def get_absolute_url(self):
        return reverse('sample-detail', args=[str(self.id)])



#___________________________________________________________________________________________
class Frame(models.Model):
    sample      = models.ForeignKey(Sample, on_delete=models.CASCADE)
    number      = models.PositiveSmallIntegerField(default=0, help_text="Frame number")
    height      = models.PositiveSmallIntegerField(default=0, help_text="Frame height")
    width       = models.PositiveSmallIntegerField(default=0, help_text="Frame width")
    time        = models.FloatField(default=-9999, help_text="Frame time im milliseconds")
    pos_x       = models.FloatField(default=-9999, help_text="Camera x position in microns")
    pos_y       = models.FloatField(default=-9999, help_text="Camera y position in microns")
    pos_z       = models.FloatField(default=-9999, help_text="Camera z position in microns")

    #USER specific
    keep_sample = models.BooleanField(help_text="keep this sample flag")

    def __str__(self):
        return '{0}, {1}'.format(self.number, self.sample)

#___________________________________________________________________________________________
class Segmentation(models.Model):

    frame                = models.ForeignKey(Frame, on_delete=models.CASCADE)
    algorithm_type       = models.CharField(max_length=200, help_text="type of algorithm used")
    algorithm_version    = models.CharField(max_length=200, help_text="version of algorithm used")
    algorithm_parameters = models.JSONField(help_text="parameters of the algorithm used")
    channel              = models.CharField(max_length=200, default='', help_text="name of the channel used for the segmentation")

#___________________________________________________________________________________________
class CellStatus(models.Model):

    STATUS = (
    ('doublenuclei', 'double nuclei'), 
    ('multiplecells','multiple cells'),
    ('pair', 'pair'),
    )

    alive    = models.BooleanField(help_text="alive cell flag")
    dividing = models.BooleanField(help_text="alive cell flag")
    keep     = models.BooleanField(help_text="keep cell flag")
    status   = models.CharField(max_length=200, choices=STATUS, help_text="cell status")


    class Meta:
        verbose_name = 'Cell status'
        verbose_name_plural = 'Cell statuses'

#___________________________________________________________________________________________
class Cell(models.Model):
    name   = models.CharField(max_length=10, help_text="cell name id")
    status = models.OneToOneField(CellStatus, on_delete=models.CASCADE)
    #sample = models.ForeignKey(Sample, default='', on_delete=models.CASCADE)


#___________________________________________________________________________________________
class Data(models.Model):
    #Only json files because the dimension correspond to the number of channels
    all_pixels    = models.JSONField(default=dict, help_text="Variables calculated over all pixels")
    single_pixels = models.JSONField(default=dict, help_text="Pixels individual coordinates and intensities")



#___________________________________________________________________________________________
class Contour(models.Model):


    number_of_pixels     = models.PositiveIntegerField(default=0, help_text="Number of pixels of the contour")
    center               = models.JSONField(help_text="center of the contour")
    pixels_data_contour  = models.OneToOneField(Data, blank=True, null=True, default='', on_delete=models.CASCADE, help_text="pixels data of the contour", related_name="pixels_data_contour")
    pixels_data_inside   = models.OneToOneField(Data, blank=True, null=True, default='', on_delete=models.CASCADE, help_text="pixels data inside the contour", related_name="pixels_data_inside")
    segmentation         = models.ForeignKey(Segmentation, default='', on_delete=models.CASCADE)
    #cell                 = models.ForeignKey(Cell, blank=True, null=True, default='',on_delete=models.CASCADE)
    uid_name             = models.CharField(default='', max_length=1000, help_text="unique name ID used not to create multiple times the same contour. Based on the input file name, frame number, algorithm type, version and parameters")
