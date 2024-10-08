from django.shortcuts import render
from django.db import reset_queries
from django.db import connection
from django.http import HttpRequest, HttpResponse
from django.contrib.auth.decorators import login_required, permission_required
import concurrent.futures

from segmentation.models import Experiment, ExperimentalDataset, Sample, Frame, Contour, CellID, CellROI, CellStatus, CellFlag, ContourSeg

import os, sys, json, glob, gc
import time, datetime
import threading
import subprocess
import imageio
import uuid
import random

from sklearn.cluster import DBSCAN
from skimage import exposure
from skimage.measure import label, regionprops

import numpy as np
from scipy.signal import find_peaks
from scipy import ndimage

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
import io
import urllib, base64

import math

from typing import Any

import nd2
from pathlib import Path
from io import BytesIO
import base64
from PIL import Image

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
import torch.nn.functional as F2
import torch.nn as nn
from torchvision.models import ResNet18_Weights  # Import the appropriate weights enum
from torchvision import models

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor





import bokeh.models
import bokeh.palettes
import bokeh.plotting
import bokeh.embed
import bokeh.layouts


labels_map = {0:'normal', 1:'dead', 2:'flat', 3:'elongated', 4:'dividing'}
labels_map = {0:'normal', 1:'dead'}#, 2:'normal', 3:'normal', 4:'normal'}


class ToTensorNormalize:
    def __call__(self, image):
        if isinstance(image, np.ndarray):
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        else:
            image = F.pil_to_tensor(image).float()
        
        image = (image - image.min()) / (image.max() - image.min())
        return image

class CellClassifier(nn.Module):
    def __init__(self,num_classes):
        super(CellClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # Input is 1 channel (grayscale)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Adjust the input size of the first fully connected layer
        self.fc1 = nn.Linear(128 * 18 * 18, 150)  # Updated size after pooling
        self.fc2 = nn.Linear(150, num_classes)  # 5 classes: "normal", "dead", "flat", "elongated", etc.

        #self.fc1 = nn.Linear(128 * 8 * 8, 150)  # Adjust size according to your input dimensions
        #self.fc2 = nn.Linear(150, 5)  # 4 classes: "normal", "dead", "flat", "elongated"

    def forward(self, x):
        x = self.pool(F2.relu(self.conv1(x)))
        x = self.pool(F2.relu(self.conv2(x)))
        x = self.pool(F2.relu(self.conv3(x)))

        # Flatten the output for the fully connected layer
        x = x.view(-1, 128 * 18 * 18)
        #x = x.view(-1, 128 * 8 * 8)  # Flatten the output for the fully connected layer
        x = F2.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def load_model_detect(model_path, num_classes, device):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='FasterRCNN_ResNet50_FPN_Weights.DEFAULT')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Load the checkpoint
    if device==torch.device('cpu'):
        checkpoint = torch.load(model_path, weights_only=True, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(model_path,weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])

    #when no checkpoint
    #model.load_state_dict(torch.load(model_path, weights_only=True))

    model.to(device)
    model.eval()
    return model



def load_model_label(model_path, num_classes, device):

    # Instantiate the model
    model = CellClassifier(num_classes)
#    Load the saved model parameters
    if device==torch.device('cpu'):
        checkpoint = torch.load(model_path, weights_only=True, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(model_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
# Set the model to evaluation mode
    model.to(device)
    model.eval()
    return model


def load_model_label_resnet(model_path, num_classes, device):
    # Load a pre-trained ResNet model
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    # Modify the first convolutional layer to accept 3-channel grayscale images
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    with torch.no_grad():
        model.conv1.weight = nn.Parameter(model.conv1.weight.mean(dim=1, keepdim=True))

    # Modify the final fully connected layer to output 5 classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    if device==torch.device('cpu'):
        checkpoint = torch.load(model_path, weights_only=True, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(model_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device)
    model.eval()
    
    return model



def preprocess_image_pytorch(image_array):
    transform = ToTensorNormalize()
    image = transform(image_array)
    return image.unsqueeze(0)  # Add batch dimension

def preprocess_image_sam2(image):
    max_value = np.max(image)
    min_value = np.min(image)
    intensity_normalized = (image - min_value)/(max_value-min_value)*255
    intensity_normalized = intensity_normalized.astype(np.uint8)
    image = intensity_normalized
    image = np.expand_dims(image, axis=0)  # Add channel dimension
    image = np.repeat(image, 3, axis=0)
    image = np.transpose(image, (1, 2, 0))
    return image

model_path = 'cell_detection_model.pth'
model_path_labels = 'cell_labels_model.pth'
model_path_labels_resnet = 'cell_labels_model_resnet.pth'
num_classes_detect = 2
num_classes_labels = 2
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model_detect = load_model_detect(model_path, num_classes_detect, device)
model_label = load_model_label(model_path_labels, num_classes_labels, device)
model_label_resnet = load_model_label_resnet(model_path_labels_resnet, num_classes_labels, device)




if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )


sam2_checkpoint = "checkpoints/sam2_hiera_base_plus.pt"
model_cfg = "sam2_hiera_b+.yaml"
sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device)#, apply_postprocessing=False)
predictor_base_plus = SAM2ImagePredictor(sam2)


LOCAL=True
DEBUG=False
DEBUG_TIME=True
RAW_DATA_PATH="raw_data/microscopy/cell_culture"
ANALYSIS_DATA_PATH="analysis_data/singleCell_catalog/contour_data"
NASRCP_MOUNT_POINT=''
MVA_OUTDIR=r'D:'
LOCAL_RAID5=r'D:\single_cells'


#MY macbook
if os.path.isdir('/Users/helsens/Software/github/EPFL-TOP/cellgmenter'):
    sys.path.append('/Users/helsens/Software/github/EPFL-TOP/cellgmenter')

#VMachine
elif os.path.isdir('/home/helsens/Software/segmentationTools/cellgmenter/main'):
    sys.path.append('/home/helsens/Software/segmentationTools/cellgmenter/main')
    LOCAL=False
    import mysql.connector
    import accesskeys

    cnx = mysql.connector.connect(user=accesskeys.RD_DB_RO_user, 
                                  password=accesskeys.RD_DB_RO_password,
                                  host='127.0.0.1',
                                  port=3306,
                                  database=accesskeys.RD_DB_name)
    NASRCP_MOUNT_POINT='/mnt/nas_rcp'

#HIVE
elif os.path.isdir(r'C:\Users\helsens\software\cellgmenter'):
    sys.path.append(r'C:\Users\helsens\software\cellgmenter')
    NASRCP_MOUNT_POINT=r'Y:'
    MVA_OUTDIR=r'D:'
    LOCAL_RAID5=r'D:\single_cells'

    LOCAL=False
    import mysql.connector
    import accesskeys

    cnx = mysql.connector.connect(user=accesskeys.RD_DB_RO_user, 
                                  password=accesskeys.RD_DB_RO_password,
                                  host='127.0.0.1',
                                  port=3336,
                                  database=accesskeys.RD_DB_name)

import reader as read
import segmentationTools as segtools

def change_file_paths():

    count=0
    mylist = Contour.objects.all()
    for X in mylist:
        if 'singleCell_catalog/' in X.file_name:count+=1
        name=X.file_name.replace('singleCell_catalog/','analysis_data/singleCell_catalog/')
        X.file_name=name
        X.save()
        if count%10000==0:
            print(' running = ',count)

    print('changed ',count)

#singleCell_catalog/contour_data/ppf001/ppf001_well1/ppf001_xy006/frame11_ROI0.json

#___________________________________________________________________________________________
def save_categories(cellflags, outname):
    ncells = 300

    for idx, cell in enumerate(cellflags):
        if idx>=ncells:
            break

        val=random.uniform(0,1)
        outdir = os.path.join(MVA_OUTDIR,r'\single_cells\training_cell_detection_categories\train')
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        if val>0.8:
            outdir = os.path.join(MVA_OUTDIR,r'\single_cells\training_cell_detection_categories\valid')
            if not os.path.exists(outdir):
                os.makedirs(outdir)

        cellroi = cell.cell_roi
        frame   = cellroi.frame

        if val<0.8:
            cellrois = CellROI.objects.select_related().filter(frame=frame)
            if len(cellrois)>1:
                continue

        sample_file_name=os.path.join(NASRCP_MOUNT_POINT, frame.sample.file_name)
        print('ttt===',sample_file_name)

        exp_name = frame.sample.experimental_dataset.experiment.name
        well = os.path.split(frame.sample.file_name)[1].replace('.nd2','')
        well = well.split("_")[1]
        out_name = '{}_{}_frame{}_{}'.format(exp_name, well, frame.number, cellroi.cell_id.name)
        outfile_png  = os.path.join(outdir, '{}.png'.format(out_name))
        outfile_json = os.path.join(outdir, '{}.json'.format(out_name))
        outfile_anno = os.path.join(outdir, '{}_annotation.json'.format(out_name))

        if os.path.exists(outfile_png) and os.path.exists(outfile_json) and os.path.exists(outfile_anno):
            continue

        images = nd2.imread(Path(sample_file_name).as_posix())
        images=images.transpose(1,0,2,3)
        image=images[0][frame.number]


        target_size = (150, 150)
        center = (int(cellroi.min_col+(cellroi.max_col-cellroi.min_col)/2.), int(cellroi.min_row+(cellroi.max_row-cellroi.min_row)/2.))
        cropped_image = image[int(center[1]-target_size[1]/2):int(center[1]+target_size[1]/2), int(center[0]-target_size[0]/2):int(center[0]+target_size[0]/2)]

        if cropped_image.shape[0]!=target_size[0] or cropped_image.shape[1]!=target_size[1]:
            continue

        norm_image = (image - image.min()) / (image.max() - image.min())
        plt.imsave(outfile_png, norm_image, cmap='gray')



        outdict={"data":image.tolist(), "data_cropped":cropped_image.tolist()}
        out_file = open(outfile_json, "w") 
        json.dump(outdict, out_file)

        outdict={"bbox":[cellroi.min_col, cellroi.max_col, cellroi.min_row, cellroi.max_row],
                 "image_file":outfile_png,
                 "image_json":outfile_json,
                 "label":outname}
        
        out_file = open(outfile_anno, "w") 
        json.dump(outdict, out_file)

#___________________________________________________________________________________________
def build_mva_detection_categories():
    cellflags_dead      = CellFlag.objects.filter(alive=False).order_by("?")
    cellflags_alive     = CellFlag.objects.filter(cell_roi__frame__sample__peaks_tod_div_validated=True, alive=True, dividing=False, double_nuclei=False, elongated=False, flat=False, multiple_cells=False, pair_cell=False, round=False).order_by("?")
    cellflags_dividing  = CellFlag.objects.filter(alive=True, dividing=True).order_by("?")
    cellflags_elongated = CellFlag.objects.filter(alive=True, elongated=True).order_by("?")
    cellflags_flat      = CellFlag.objects.filter(alive=True, flat=True).order_by("?")

    print('number of dead cells      = ',len(cellflags_dead))
    print('number of alive cells     = ',len(cellflags_alive))
    print('number of dividing cells  = ',len(cellflags_dividing))
    print('number of elongated cells = ',len(cellflags_elongated))
    print('number of flat cells      = ',len(cellflags_flat))
    threads = []
    threads.append(threading.Thread(target = save_categories, args=(cellflags_dead,'dead', )))
    threads.append(threading.Thread(target = save_categories, args=(cellflags_alive,'normal', )))
    threads.append(threading.Thread(target = save_categories, args=(cellflags_dividing,'dividing', )))
    threads.append(threading.Thread(target = save_categories, args=(cellflags_elongated,'elongated', )))
    threads.append(threading.Thread(target = save_categories, args=(cellflags_flat,'flat', )))
    for t in threads: t.start()
    for t in threads: t.join()

#___________________________________________________________________________________________
def get_experiment_details(selected_experiment):
    #GET THE EXPERIMENT DETAILS
    query = (
        "select e.* from experiment_catalog_experiment e"
        " where e.experiment_name = \"{}\"".format(selected_experiment)
        )

    mycursor = cnx.cursor()
    mycursor.execute(query)
    myresult = mycursor.fetchall()

    experiment_dict={}
    if len(myresult)==1:
        experiment_dict['experiment_name']=myresult[0][1]
        experiment_dict['experiment_date']=myresult[0][2]
        experiment_dict['experiment_description']=myresult[0][3]

    return experiment_dict

#___________________________________________________________________________________________
def get_contribution_details(selected_experiment):
    query = (   
        "select contribution.description, person.first_name, person.last_name from experiment_catalog_experiment e"
        " inner join experiment_catalog_experiment_contribution    contrib_exp         on e.id                                = contrib_exp.experiment_id"
        " inner join contribution_catalog_contribution             contribution        on contrib_exp.contribution_id         = contribution.id"
        " inner join contribution_catalog_contribution_contributor contrib_contributor on contrib_contributor.contribution_id = contribution.id"
        " inner join contribution_catalog_contributor              contributor         on contrib_contributor.contributor_id  = contributor.id"
        " inner join contribution_catalog_person                   person              on person.id                      = contributor.person_id"
        " where e.experiment_name = \"{}\"".format(selected_experiment)
        )

    mycursor = cnx.cursor()
    mycursor.execute(query)
    myresult = mycursor.fetchall()

    contribution_dict=[]
    for x in myresult:
        tmpdict={'description':x[0], 'first_name':x[1], 'last_name':x[2]}
        contribution_dict.append(tmpdict)

    return contribution_dict

#___________________________________________________________________________________________
def get_treatment_details(selected_well):
    query = (
        "select treat.name, treat.type, treat.concentration, treat.description, treat.developmental_stage, treat.duration, treat.solvent, treat.temperature from rawdata_catalog_rawdataset rds"
        " inner join experiment_catalog_experimentaldataset              dataset      on rds.id                            = dataset.raw_dataset_id"
        " inner join experimentalcondition_catalog_experimentalcondition expcond      on dataset.experimental_condition_id = expcond.id"
        " inner join experimentalcondition_catalog_experimentalcondition_treatment expcond_treat on expcond.id  = expcond_treat.experimentalcondition_id"
        " inner join experimentalcondition_catalog_treatment treat          on treat.id = expcond_treat.treatment_id"
        " where rds.data_name = \"{}\"".format(selected_well)
        )
    mycursor = cnx.cursor()
    mycursor.execute(query)
    myresult = mycursor.fetchall()

    treatment_dict=[]
    for x in myresult:
        tmpdict={'name':x[0], 'type':x[1], 'concentration':x[2], 'description':x[3], 
                 'developmental_stage':x[4], 'duration':x[5], 'solvent':x[6], 'temperature':x[7]}
        treatment_dict.append(tmpdict)
    return treatment_dict

#___________________________________________________________________________________________
def get_injection_details(selected_well):
    query = (
        "select inj.name, inj.type, inj.concentration, inj.description, inj.developmental_stage, inj.slim_id from rawdata_catalog_rawdataset rds"
        " inner join experiment_catalog_experimentaldataset              dataset      on rds.id                            = dataset.raw_dataset_id"
        " inner join experimentalcondition_catalog_experimentalcondition expcond      on dataset.experimental_condition_id = expcond.id"
        " inner join experimentalcondition_catalog_experimentalcondition_injection expcond_inj on expcond.id  = expcond_inj.experimentalcondition_id"
        " inner join experimentalcondition_catalog_injection inj          on inj.id = expcond_inj.injection_id"
        " where rds.data_name = \"{}\"".format(selected_well)
        )
    mycursor = cnx.cursor()
    mycursor.execute(query)
    myresult = mycursor.fetchall()

    injection_dict=[]
    for x in myresult:
        tmpdict={'name':x[0], 'type':x[1], 'concentration':x[2], 'description':x[3], 
                 'developmental_stage':x[4], 'slim_id':x[5]}
        injection_dict.append(tmpdict)
    return injection_dict

#___________________________________________________________________________________________
def get_instrumental_details(selected_well):
    query = (
        "select inst.name, inst.instrument_name, inst.instrument_name, inst.instrument_name from rawdata_catalog_rawdataset rds"
        " inner join experiment_catalog_experimentaldataset              dataset      on rds.id                            = dataset.raw_dataset_id"
        " inner join experimentalcondition_catalog_experimentalcondition expcond      on dataset.experimental_condition_id = expcond.id"
        " inner join experimentalcondition_catalog_experimentalcondition_instrumeceac expcond_inst on expcond.id  = expcond_inst.experimentalcondition_id"
        " inner join experimentalcondition_catalog_instrumentalcondition inst          on inst.id = expcond_inst.instrumentalcondition_id"
        " where rds.data_name = \"{}\"".format(selected_well)
        )
    mycursor = cnx.cursor()
    mycursor.execute(query)
    myresult = mycursor.fetchall()

    instrumental_dict=[]
    for x in myresult:
        tmpdict={'name':x[0], 'instrument_name':x[1], 'instrument_name':x[2], 'instrument_name':x[3]}
        instrumental_dict.append(tmpdict)
    return instrumental_dict

#___________________________________________________________________________________________
def get_sample_details(selected_well):
    query = (
        "select samp.id, samp.specie, samp.date_of_crossing, samp.developmental_stage, samp.pyrat_crossing_id from rawdata_catalog_rawdataset rds"
        " inner join experiment_catalog_experimentaldataset              dataset      on rds.id                            = dataset.raw_dataset_id"
        " inner join experimentalcondition_catalog_experimentalcondition expcond      on dataset.experimental_condition_id = expcond.id"
        " inner join experimentalcondition_catalog_experimentalcondition_sample expcond_samp on expcond.id  = expcond_samp.experimentalcondition_id"
        " inner join experimentalcondition_catalog_sample samp          on samp.id = expcond_samp.sample_id"
        " where rds.data_name = \"{}\"".format(selected_well)
        )
    mycursor = cnx.cursor()
    mycursor.execute(query)
    sample_res = mycursor.fetchall()

    query = (
        "select samp.id, par.age_at_crossing, par.date_of_birth, par.mutation_grade, par.number_of_female, par.number_of_male, par.number_of_unknown, par.strain_name from rawdata_catalog_rawdataset rds"
        " inner join experiment_catalog_experimentaldataset              dataset      on rds.id                            = dataset.raw_dataset_id"
        " inner join experimentalcondition_catalog_experimentalcondition expcond      on dataset.experimental_condition_id = expcond.id"
        " inner join experimentalcondition_catalog_experimentalcondition_sample expcond_samp on expcond.id  = expcond_samp.experimentalcondition_id"
        " inner join experimentalcondition_catalog_sample samp          on samp.id = expcond_samp.sample_id"
        " inner join experimentalcondition_catalog_sample_parent samp_par on samp_par.sample_id = samp.id"
        " inner join experimentalcondition_catalog_parent par on par.id = samp_par.parent_id"
        " where rds.data_name = \"{}\"".format(selected_well)
        )
    mycursor = cnx.cursor()
    mycursor.execute(query)
    parents_res = mycursor.fetchall()   

    query = (
        "select samp.id, mutname.name, mutgrade.grade from rawdata_catalog_rawdataset rds"
        " inner join experiment_catalog_experimentaldataset              dataset      on rds.id                            = dataset.raw_dataset_id"
        " inner join experimentalcondition_catalog_experimentalcondition expcond      on dataset.experimental_condition_id = expcond.id"
        " inner join experimentalcondition_catalog_experimentalcondition_sample expcond_samp on expcond.id  = expcond_samp.experimentalcondition_id"
        " inner join experimentalcondition_catalog_sample samp          on samp.id = expcond_samp.sample_id"
        " inner join experimentalcondition_catalog_sample_mutation samp_mut on samp_mut.sample_id = samp.id"
        " inner join experimentalcondition_catalog_mutation mut on mut.id = samp_mut.mutation_id"
        " inner join experimentalcondition_catalog_mutationname mutname on mutname.id = mut.name_id"
        " inner join experimentalcondition_catalog_mutationgrade mutgrade on mutgrade.id = mut.grade_id"
        " where rds.data_name = \"{}\"".format(selected_well)
        )
    mycursor = cnx.cursor()
    mycursor.execute(query)
    mutation_res = mycursor.fetchall()


    sample_dict=[]
    for samp in sample_res:
        parents=[]
        for par in parents_res:
            if par[0]==samp[0]:
                parents.append({'age_at_crossing':par[1], 'date_of_birth':par[2], 'mutation_grade':par[3], 'number_of_female':par[4], 
                                'number_of_male':par[5], 'number_of_unknown':par[6], 'strain_name':par[7]})
        mutations=[]
        for mut in mutation_res:
            if mut[0]==samp[0]:
                mutations.append({'name':mut[1], 'grade':mut[2]})
        
        tmpdict={'specie':samp[1], 'date_of_crossing':samp[2], 'developmental_stage':samp[3], 'pyrat_crossing_id':samp[4], 'parents':parents, 'mutations':mutations}
        sample_dict.append(tmpdict)

    return sample_dict

#___________________________________________________________________________________________
def register_rawdataset():
    query = (
        "select e.*, rds.data_type, rds.data_name, rds.number_of_raw_files, rds.raw_files from experiment_catalog_experiment e"
        " inner join experiment_catalog_experiment_experimental_tag ecet on e.id   = ecet.experiment_id"
        " inner join experiment_catalog_experimentaltag tag              on tag.id = ecet.experimentaltag_id"
        " inner join experiment_catalog_experimentaldataset dataset      on e.id   = dataset.experiment_id"
        " inner join rawdata_catalog_rawdataset rds                      on dataset.raw_dataset_id = rds.id"
        " where tag.name = \"SegmentMe\""
        )
    mycursor = cnx.cursor()
    mycursor.execute(query)
    myresult = mycursor.fetchall()


    for x in myresult:
        print('=========================================>',x,'<=========================================')

    experiments = Experiment.objects.values()
    list_experiments = [entry for entry in experiments] 
    list_experiments_uid=[e["name"] for e in list_experiments]

    for x in myresult:
        if x[1] in list_experiments_uid: continue
        unsplit_file = glob.glob(os.path.join(NASRCP_MOUNT_POINT, RAW_DATA_PATH ,x[1],'*.nd2'))
        if DEBUG: print('=========unsplit_file ===',unsplit_file)
        if len(unsplit_file)!=1:
            print('====================== ERROR, unsplit_file not 1, exit ',unsplit_file,'  in ',os.path.join(NASRCP_MOUNT_POINT, RAW_DATA_PATH,x[1],'*.nd2'))
            sys.exit(3)
        metadata = read.nd2reader_getSampleMetadata(unsplit_file[0])
        experiment =  Experiment(name=x[1], 
                                 date=x[2], 
                                 description=x[3],
                                 file_name=RAW_DATA_PATH+'/'+x[1]+'/'+os.path.split(unsplit_file[0])[-1],
                                 number_of_frames=metadata['number_of_frames'], 
                                 number_of_channels=metadata['number_of_channels'], 
                                 name_of_channels=metadata['name_of_channels'].replace(' ',''), 
                                 experiment_description=metadata['experiment_description'],
                                 date_of_acquisition=metadata['date'],
                                 )
        experiment.save()
        list_experiments_uid.append(x[1])
        if DEBUG:print('adding experiment with name:  ',x[1])

    for exp in Experiment.objects.all():
        experimentaldataset = ExperimentalDataset.objects.select_related().filter(experiment = exp)
        list_expds_uid = [os.path.join(entry.data_type, entry.data_name) for entry in experimentaldataset] 

        for x in myresult:
            if x[1]!=exp.name: 
                continue
            if os.path.join(x[4], x[5]) in list_expds_uid: continue
            files_json = json.loads(x[7])
            expds = ExperimentalDataset(data_type=x[4], data_name=x[5], experiment=exp, number_of_files=x[6], files=files_json)
            expds.save()
            if DEBUG:print('    adding experimental dataset with name ',os.path.join(x[4], x[5]))

            for f in files_json["files"]:
                fname=os.path.join(NASRCP_MOUNT_POINT, RAW_DATA_PATH, x[5], "raw_files", f["name"])

                metadata = read.nd2reader_getSampleMetadata(fname)
                sample = Sample(file_name=RAW_DATA_PATH+'/'+ x[5]+"/raw_files/"+f["name"],
                                experimental_dataset=expds,
                                keep_sample=True)
                sample.save()
                if DEBUG:print('        adding sample with name ',fname)

                metadataFrame = read.nd2reader_getFrameMetadata(fname)
                for fr in range(metadata['number_of_frames']):
                    frame = Frame(sample=sample, 
                                  number=fr, 
                                  keep_sample=True,
                                  time=metadataFrame['time'][fr],
                                  pos_x=metadataFrame['x_pos'][fr],
                                  pos_y=metadataFrame['y_pos'][fr],
                                  pos_z=metadataFrame['z_pos'][fr],
                                  height=metadataFrame['height'],
                                  width=metadataFrame['width'],
                                  pixel_microns=metadataFrame['pixel_microns'],
                                  )
                    if DEBUG: print('            adding frame with name ',fr)
                    frame.save()


#___________________________________________________________________________________________
def build_cells_sample(sample, addmode=False):
 
    s=None
    if type(sample) == str:
        s = Sample.objects.get(file_name = sample)
    else:
        s=sample
    print('---- BUILD CELL sample name ',s.file_name)

    cellsid = CellID.objects.select_related().filter(sample = s)
    frames  = Frame.objects.select_related().filter(sample = s)
    nframes = len(frames)
    #addmode false, will build cells as nothing existed on the position
    if addmode==False:
        if len(cellsid)!=0:
            print('can not build cells has cells already exist for this sample')
            return

        cell_dict={}
        for f in range(nframes):
            frame = frames.filter(number=f)
            print('===== frame  ',frame)
            cellrois_frame = CellROI.objects.select_related().filter(frame=frame[0])
            for cellroi_frame in cellrois_frame:
                print('     ===== cell roi=',cellroi_frame.roi_number, ' center=',cellroi_frame.contour_cellroi.center_x_pix,' , ',cellroi_frame.contour_cellroi.center_y_pix)
                if f == 0:
                    cell_dict['cell{}'.format(cellroi_frame.roi_number)]={'frame':[f], 
                                                                          'roi_number':[cellroi_frame.roi_number],
                                                                          'x':[cellroi_frame.contour_cellroi.center_x_pix], 
                                                                          'y':[cellroi_frame.contour_cellroi.center_y_pix]}
                    continue
                minDR=10000000
                maxDR=75
                sel_cell=None
                print('cell_dict beofre loop',cell_dict)

                for cell in cell_dict:
                    print('          ===== cell cell_dict=',cell)
                    if f in cell_dict[cell]['frame']:
                        print('cell already in dic=',cell)
                        continue
                    #print("cell_dict[cell]['frame'][-1]=",cell_dict[cell]['frame'][-1])
                    #print("cell_dict[cell]['x']=",cell_dict[cell]['x'])
                    dR = math.sqrt(math.pow((cell_dict[cell]['x'][len(cell_dict[cell]['frame'])-1] - cellroi_frame.contour_cellroi.center_x_pix),2) +  
                                   math.pow((cell_dict[cell]['y'][len(cell_dict[cell]['frame'])-1] - cellroi_frame.contour_cellroi.center_y_pix),2)) 
                    if dR<minDR and dR<maxDR:
                        minDR=dR
                        sel_cell = cell
                        print('minDR=',minDR,'  cell=',cell)
                if sel_cell!=None:
                    print('cell found')
                    cell_dict[sel_cell]['frame'].append(f)
                    cell_dict[sel_cell]['roi_number'].append(cellroi_frame.roi_number)
                    cell_dict[sel_cell]['x'].append(cellroi_frame.contour_cellroi.center_x_pix)
                    cell_dict[sel_cell]['y'].append(cellroi_frame.contour_cellroi.center_y_pix)
                else:
                    print('cell not found')
                    cell_dict['cell{}'.format(len(cell_dict))] = {'frame':[f], 
                                                                  'roi_number':[cellroi_frame.roi_number],
                                                                  'x':[cellroi_frame.contour_cellroi.center_x_pix], 
                                                                  'y':[cellroi_frame.contour_cellroi.center_y_pix]}



        print('before: ',cell_dict)

        topop=[]
        for cell in cell_dict:
            if len(cell_dict[cell]['frame'])<int(nframes/2.): topop.append(cell)
        for cell in topop:
            cell_dict.pop(cell, None)
        cell_dict_final={}
        count=0
        for cell in cell_dict:
            cell_dict_final['cell{}'.format(count)]=cell_dict[cell]
            count+=1
        print('after: ',cell_dict_final)

        cellid_dict={}
        for cell in cell_dict_final:
            print('---- cell=',cell)
            cellstatus = CellStatus()
            cellstatus.save()
            cellid = CellID(sample=s, name=cell, cell_status=cellstatus)
            cellid.save()

            cellid_dict[cell]=cellid

            for nroi in range(len(cell_dict_final[cell]['roi_number'])):
                print('    ---- nroi=',nroi, '  frame=',cell_dict_final[cell]['frame'][nroi], '   roi=',cell_dict_final[cell]['roi_number'][nroi])
                frame = frames.get(number=cell_dict_final[cell]['frame'][nroi])
                roi   = CellROI.objects.select_related().filter(frame=frame).get(roi_number=cell_dict_final[cell]['roi_number'][nroi])
                roi.cell_id = cellid_dict[cell]
                roi.save()
            predict_time_of_death(cellid_dict[cell])


    if addmode==True:
        cellsid = CellID.objects.select_related().filter(sample = s)
        cellrois_frame = CellROI.objects.select_related().filter(frame__sample = s, cell_id=None)
        for cellroi_frame in cellrois_frame:
            print('==============  ADDMODE=',addmode, '  cellroi_frame=',cellroi_frame)
            minDR=9999999
            mincellid=None
            for cellid in cellsid:
                #find the closest frame to calculate DeltaR
                cellsroi_cell = CellROI.objects.select_related().filter(cell_id=cellid)
                minDelta=999999
                cframe_num = None
                for cellroi_cell in cellsroi_cell:
                    delta=math.fabs(cellroi_frame.frame.number-cellroi_cell.frame.number)
                    if delta<minDelta:
                        minDelta=delta
                        cframe_num = cellroi_cell.frame.number
                if minDelta<1.:
                    continue
                cframe = frames.get(number=cframe_num)
                #get the cell ROI of the closest frame
                cellroi_cell = CellROI.objects.select_related().filter(frame=cframe).get(cell_id=cellid)
                dR = math.sqrt(math.pow((cellroi_cell.contour_cellroi.center_x_pix - cellroi_frame.contour_cellroi.center_x_pix),2) +  
                               math.pow((cellroi_cell.contour_cellroi.center_y_pix - cellroi_frame.contour_cellroi.center_y_pix),2))                    
                if dR<minDR:
                    minDR=dR
                    mincellid=cellid
            if mincellid!=None: 
                cellroi_frame.cell_id = mincellid
                cellroi_frame.save()

#___________________________________________________________________________________________
def removeROIs(sample):
    s=None
    if type(sample) == str:
        s = Sample.objects.get(file_name = sample)
    else:
        s=sample
    frames = Frame.objects.select_related().filter(sample=s)
    for frame in frames:
        cellROIs = CellROI.objects.select_related().filter(frame=frame)
        for cellROI in cellROIs:
            if cellROI.cell_id == None:
                cellROI.delete()
    print('removeROIs sample ',s)



#___________________________________________________________________________________________
def build_segmentation_sam2_single_frame(x_list, y_list, image):
    input_point = np.array([[x, y] for x, y in zip(x_list,y_list)])
    input_label = np.array([1 for i in range(len(x_list))])
    image_prepro = preprocess_image_sam2(image)
    predictor_base_plus.set_image(image_prepro)

    masks, scores, logits = predictor_base_plus.predict(point_coords=input_point,point_labels=input_label,multimask_output=True)
    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    logits = logits[sorted_ind]
    print('scores ',scores)
    return masks[0]

    
#___________________________________________________________________________________________
# Function to prepare the intensity plot
def predict_time_of_death(cellid=None):
    time_of_death_pred=-1000
    time_of_death_frame_pred=-100
    print(' in predict_time_of_death ')
    current_file = cellid.sample.file_name

    current_file=os.path.join(NASRCP_MOUNT_POINT,current_file)
    time_lapse_path = Path(current_file)
    time_lapse = nd2.imread(time_lapse_path.as_posix())
    images=time_lapse.transpose(1,0,2,3)
    BF_images=images[0]
    target_size = (150, 150)
    predictions=[]
    time_list=[]
    for i in range(cellid.sample.experimental_dataset.experiment.number_of_frames):
        predictions.append(None)
        time_list.append(None)
    cellrois = CellROI.objects.select_related().filter(cell_id=cellid)
    for cellroi in cellrois:
        center = (int(cellroi.min_col+(cellroi.max_col-cellroi.min_col)/2.), int(cellroi.min_row+(cellroi.max_row-cellroi.min_row)/2.))
        cropped_image = BF_images[cellroi.frame.number][int(center[1]-target_size[1]/2):int(center[1]+target_size[1]/2), int(center[0]-target_size[0]/2):int(center[0]+target_size[0]/2)]

        if cropped_image.shape[0]!=target_size[0] or cropped_image.shape[1]!=target_size[1]:
            continue
    

        image_cropped = preprocess_image_pytorch(cropped_image).to(device)
        with torch.no_grad():
            labels = model_label(image_cropped)


        probabilities = F2.softmax(labels, dim=1)
        predictions[cellroi.frame.number]=float(probabilities[0].cpu().numpy()[1])
        print('probabilities : ',probabilities)
        pred_label = labels_map[int(torch.argmax(labels, dim=1)[0].cpu().numpy())]
        print('frame ',cellroi.frame.number,'pred_label = ',pred_label)
        time_list[cellroi.frame.number]=cellroi.frame.time/60000.
    #source_intensity_predicted_death.data={'time':[], 'intensity':[]}
    #out_dict={'time':[], 'intensity':[]}
    n=3
    for i in range(len(predictions)-n):
        pred=0
        npred=0

        for j in range(i, i+n):
            if predictions[j]!=None: 
                pred+=predictions[j]
                npred+=1
        if npred>0:
            print(i, '   ',pred/npred)
            if pred/npred>0.8:
                print('frame dead= ',i)
                #source_intensity_predicted_death.data={'time':[source_intensity_ch1.data["time"][i]], 'intensity':[source_intensity_ch1.data["intensity"][i]]}
                #out_dict={'time':[source_intensity_ch1.data["time"][i]], 'intensity':[source_intensity_ch1.data["intensity"][i]]}
                time_of_death_pred=time_list[i]
                time_of_death_frame_pred=i

                break
    cellstatus=cellid.cell_status
    cellstatus.time_of_death_pred=time_of_death_pred
    cellstatus.time_of_death_frame_pred=time_of_death_frame_pred
    cellstatus.save()
    #return out_dict

#___________________________________________________________________________________________
def build_segmentation_sam2(sample=None, force=False):
    s=None
    if type(sample) == str:
        s = Sample.objects.get(file_name = sample)

    else:
        s=sample
    expds = s.experimental_dataset
    exp = expds.experiment

    print('build_segmentation_sam2: ',s.file_name)

    frames = Frame.objects.select_related().filter(sample=s)
    file_name=os.path.join(LOCAL_RAID5,s.file_name)
    if not os.path.exists(file_name):
        file_name=os.path.join(NASRCP_MOUNT_POINT,s.file_name)

    images, channels = read.nd2reader_getFrames(file_name)
    images=images.transpose(1,0,2,3)

    #images are t, c, x, y 
    BF_images=images[0]
    for frame in frames:
        print(frame)

        image_prepro = preprocess_image_sam2(BF_images[frame.number])
        predictor_base_plus.set_image(image_prepro)
        npix=2
        cellROIs = CellROI.objects.select_related().filter(frame=frame)
        for cellroi in cellROIs:


            #This is to fill the new element cell_status.segmentation in case it does not exist yet
            try:
                toto = cellroi.cell_id.cell_status.segmentation['algo']
            except KeyError:
                cell_status = cellroi.cell_id.cell_status
                cell_status.segmentation = {'algo':['roi']}
                cell_status.save()


            if 'SAM2_b+' not in cellroi.cell_id.cell_status.segmentation['algo']:
                cell_status = cellroi.cell_id.cell_status
                cell_status.segmentation['algo'].append('SAM2_b+')
                cell_status.save()

            eflag={'SAM2_b+':False}
            contoursSeg = ContourSeg.objects.select_related().filter(cell_roi=cellroi)
            for contourSeg in contoursSeg:
                eflag[contourSeg.algo] = True

            for flag in eflag:  
                if eflag[flag]: continue

                input_point = np.array([[cellroi.min_col+(cellroi.max_col-cellroi.min_col)/2., cellroi.min_row+(cellroi.max_row-cellroi.min_row)/2.],
                                        [cellroi.min_col+(cellroi.max_col-cellroi.min_col)/2.+npix, cellroi.min_row+(cellroi.max_row-cellroi.min_row)/2.],
                                        [cellroi.min_col+(cellroi.max_col-cellroi.min_col)/2.-npix, cellroi.min_row+(cellroi.max_row-cellroi.min_row)/2.],
                                        [cellroi.min_col+(cellroi.max_col-cellroi.min_col)/2., cellroi.min_row+(cellroi.max_row-cellroi.min_row)/2.+npix],
                                        [cellroi.min_col+(cellroi.max_col-cellroi.min_col)/2., cellroi.min_row+(cellroi.max_row-cellroi.min_row)/2.-npix],
                                        ])
                input_label = np.array([1,1,1,1,1])
                masks, scores, logits = predictor_base_plus.predict(point_coords=input_point,point_labels=input_label,multimask_output=True)
                sorted_ind = np.argsort(scores)[::-1]
                masks = masks[sorted_ind]
                scores = scores[sorted_ind]
                logits = logits[sorted_ind]

                contourseg = ContourSeg(cell_roi=cellroi)

                #build_contours_sam2(contourseg, masks[0], flag, cellroi, images, channels, exp.name, expds.data_name, s.file_name)

                center = ndimage.center_of_mass(masks[0])
                contourseg.center_x_pix = center[0]
                contourseg.center_y_pix = center[1]
                contourseg.center_x_mic = center[0]*cellroi.frame.pixel_microns+cellroi.frame.pos_x
                contourseg.center_y_mic = center[1]*cellroi.frame.pixel_microns+cellroi.frame.pos_y
                contourseg.algo = flag

                intensity_mean={}
                intensity_std={}
                intensity_sum={}
                intensity_max={}
                for ch in range(len(channels)): 
                    segment=masks[0]*images[ch][cellroi.frame.number]
                    sum=float(np.sum(segment))
                    mean=float(sum/masks[0].sum())
                    std=float(np.std(segment))
                    max=float(np.max(segment))
                    ch_name=channels[ch].replace(" ","")
                    intensity_mean[ch_name]=mean
                    intensity_std[ch_name]=std
                    intensity_sum[ch_name]=sum
                    intensity_max[ch_name]=max

                contourseg.intensity_max  = intensity_max
                contourseg.intensity_mean = intensity_mean
                contourseg.intensity_std  = intensity_std
                contourseg.intensity_sum  = intensity_sum
                contourseg.number_of_pixels = masks[0].sum()

                segment_dict = {'mask':masks[0].tolist()}
                out_dir_name  = os.path.join(NASRCP_MOUNT_POINT, ANALYSIS_DATA_PATH,exp.name, expds.data_name, os.path.split(s.file_name)[-1].replace('.nd2',''))
                out_file_name = os.path.join(out_dir_name, "frame{0}_ROI{1}_{2}_mask.json".format(cellroi.frame.number, cellroi.roi_number, flag))

                out_dir_name_DB  = ANALYSIS_DATA_PATH+"/"+exp.name+"/"+ expds.data_name+"/"+os.path.split(s.file_name)[-1].replace('.nd2','')
                out_file_name_DB = out_dir_name_DB+ "/frame{0}_ROI{1}_{2}_mask.json".format(cellroi.frame.number, cellroi.roi_number, flag)

                out_file = open(out_file_name, "w") 
                json.dump(segment_dict, out_file) 
                out_file.close() 
                contourseg.file_name = out_file_name_DB
                contourseg.save()






                #print('scores ',scores)
                #label_im = label(masks[0])
                #region=regionprops(label_im)

                #sel_region = None 
                #if len(region)==1: sel_region=region[0]
                #if len(region)>1:
                #    for r in region:
                #        print('r  =  ',r.bbox,'  ',r.area, '  ',r.centroid)
                #        print('input_point ',input_point)
                #        if r.area<80: continue
                #        if math.sqrt( math.pow((input_point[0][0] - r.centroid[1]),2) +  math.pow((input_point[0][1]- r.centroid[0]),2))>50:continue
                #        sel_region=r
                #        print(' selected r  =  ',r.bbox,'  ',r.area, '  ',r.centroid)
                #if sel_region!=None:
                #    build_contours(sel_region, contourseg, cellroi, BF_images[frame.number].shape, flag, images, channels, exp.name, expds.data_name, s.file_name)


#___________________________________________________________________________________________
def build_contours_sam2(contourseg, mask, segname, cellroi, images, channels, exp_name, expds_data_name, s_file_name):

    #contourseg.mask={'mask':mask.tolist()}
    center = ndimage.center_of_mass(mask)
    contourseg.center_x_pix = center[0]
    contourseg.center_y_pix = center[1]
    contourseg.center_x_mic = center[0]*cellroi.frame.pixel_microns+cellroi.frame.pos_x
    contourseg.center_y_mic = center[1]*cellroi.frame.pixel_microns+cellroi.frame.pos_y
    contourseg.algo = segname

    intensity_mean={}
    intensity_std={}
    intensity_sum={}
    intensity_max={}
    for ch in range(len(channels)): 
        segment=mask*images[ch][cellroi.frame.number]
        sum=float(np.sum(segment))
        mean=float(sum/mask.sum())
        std=float(np.std(segment))
        max=float(np.max(segment))
        ch_name=channels[ch].replace(" ","")
        intensity_mean[ch_name]=mean
        intensity_std[ch_name]=std
        intensity_sum[ch_name]=sum
        intensity_max[ch_name]=max

    contourseg.intensity_max  = intensity_max
    contourseg.intensity_mean = intensity_mean
    contourseg.intensity_std  = intensity_std
    contourseg.intensity_sum  = intensity_sum
    contourseg.number_of_pixels = mask.sum()

    segment_dict = {'mask':mask.tolist()}
    out_dir_name  = os.path.join(NASRCP_MOUNT_POINT, ANALYSIS_DATA_PATH,exp_name, expds_data_name, os.path.split(s_file_name)[-1].replace('.nd2',''))
    out_file_name = os.path.join(out_dir_name, "frame{0}_ROI{1}_{2}_mask.json".format(cellroi.frame.number, cellroi.roi_number, segname))

    out_dir_name_DB  = ANALYSIS_DATA_PATH+"/"+exp_name+"/"+ expds_data_name+"/"+os.path.split(s_file_name)[-1].replace('.nd2','')
    out_file_name_DB = out_dir_name_DB+ "/frame{0}_ROI{1}_{2}_mask.json".format(cellroi.frame.number, cellroi.roi_number, segname)

    out_file = open(out_file_name, "w") 
    json.dump(segment_dict, out_file) 
    out_file.close() 
    contourseg.file_name = out_file_name_DB
    contourseg.save()



#___________________________________________________________________________________________
def build_segmentation_parallel(exp_name=''):

    max_workers=5
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        exp_list = Experiment.objects.all()
        for exp in exp_list:
            if exp_name!='' and exp.name!=exp_name:
                continue
            print('exp ', exp.name)
            experimentaldataset = ExperimentalDataset.objects.select_related().filter(experiment = exp)
            for expds in experimentaldataset:
                print('  expds ', expds.data_name)
                if 'well2' not in expds.data_name:continue
                samples = Sample.objects.select_related().filter(experimental_dataset = expds)
                for s in samples:
                    if 'xy00' not in s.file_name:continue
                    print('build segments sample: ',s.file_name)
                    executor.submit(build_segmentation_sam2, sample=s, force=False)

#___________________________________________________________________________________________
def build_segmentation(exp_name=''):

    exp_list = Experiment.objects.all()
    for exp in exp_list:
        if exp_name!='' and exp.name!=exp_name:
            continue
        print('exp ', exp.name)
        experimentaldataset = ExperimentalDataset.objects.select_related().filter(experiment = exp)
        for expds in experimentaldataset:
            #if 'well3' not in expds.data_name:continue
            #print('  expds ', expds.data_name)

            samples = Sample.objects.select_related().filter(experimental_dataset = expds)
            for s in samples:
                #if 'xy00' in s.file_name:continue
                print('build segments sample: ',s.file_name)
                build_segmentation_sam2( sample=s, force=False)



#___________________________________________________________________________________________
def build_contours(contour, contourseg, cellroi, img_shape, segname, images, channels, exp_name, expds_data_name, s_file_name):
    x_coords=[]
    y_coords=[]
    mask0=np.zeros(img_shape, dtype=bool)

    for coord in contour.coords:
        x_coords.append(coord[0])
        y_coords.append(coord[1])
        mask0[coord[0]][coord[1]]=True

    cs=plt.contour(mask0, [0.5],linewidths=1.,  colors='red')
    contcoords = cs.allsegs[0][0]
    x_cont_coords=[]
    y_cont_coords=[]
    for p in contcoords:
        x_cont_coords.append(p[0])
        y_cont_coords.append(p[1])

    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

    contourseg.pixels={'x':x_cont_coords, 'y':y_cont_coords}
    contourseg.center_x_pix = contour.centroid[0]
    contourseg.center_y_pix = contour.centroid[1]
    contourseg.center_x_mic = contour.centroid[0]*cellroi.frame.pixel_microns+cellroi.frame.pos_x
    contourseg.center_y_mic = contour.centroid[1]*cellroi.frame.pixel_microns+cellroi.frame.pos_y
    contourseg.algo = segname

    intensity_mean={}
    intensity_std={}
    intensity_sum={}
    intensity_max={}
    for ch in range(len(channels)): 
        segment=mask0*images[ch][cellroi.frame.number]
        sum=float(np.sum(segment))
        mean=float(sum/contour.num_pixels)
        std=float(np.std(segment))
        max=float(np.max(segment))
        ch_name=channels[ch].replace(" ","")
        intensity_mean[ch_name]=mean
        intensity_std[ch_name]=std
        intensity_sum[ch_name]=sum
        intensity_max[ch_name]=max

    contourseg.intensity_max  = intensity_max
    contourseg.intensity_mean = intensity_mean
    contourseg.intensity_std  = intensity_std
    contourseg.intensity_sum  = intensity_sum
    contourseg.number_of_pixels = contour.num_pixels

    segment_dict = {}
    out_dir_name  = os.path.join(NASRCP_MOUNT_POINT, ANALYSIS_DATA_PATH,exp_name, expds_data_name, os.path.split(s_file_name)[-1].replace('.nd2',''))
    out_file_name = os.path.join(out_dir_name, "frame{0}_ROI{1}_{2}.json".format(cellroi.frame.number, cellroi.roi_number, segname))

    out_dir_name_DB  = ANALYSIS_DATA_PATH+"/"+exp_name+"/"+ expds_data_name+"/"+os.path.split(s_file_name)[-1].replace('.nd2','')
    out_file_name_DB = out_dir_name_DB+ "/frame{0}_ROI{1}_{2}.json".format(cellroi.frame.number, cellroi.roi_number, segname)

    if not os.path.exists(out_dir_name):
        os.makedirs(out_dir_name)
    segment_dict['npixels']=int(contour.num_pixels)
    segment_dict['type']=segname

    segment_dict['x'] = []
    segment_dict['y'] = []
    for ch in range(len(channels)):
        segment_dict['intensity_{}'.format(channels[ch].replace(" ",""))] = []
    
    for coord in contour.coords:
        segment_dict['x'].append(int(coord[0]))
        segment_dict['y'].append(int(coord[1]))
        for ch in range(len(channels)):
            segment_dict['intensity_{}'.format(channels[ch].replace(" ",""))].append(float(images[ch][cellroi.frame.number][coord[0]][coord[1]]))
    out_file = open(out_file_name, "w") 
    json.dump(segment_dict, out_file) 
    out_file.close() 
    contourseg.file_name = out_file_name_DB
    contourseg.save()


#___________________________________________________________________________________________
def fix_alive_status():
    exp_list = Experiment.objects.all()
    for exp in exp_list:
        experimentaldataset = ExperimentalDataset.objects.select_related().filter(experiment = exp)
        for expds in experimentaldataset:
            samples = Sample.objects.select_related().filter(experimental_dataset = expds)
            for sample in samples:
                cellsid = CellID.objects.select_related().filter(sample=sample)
                for cellid in cellsid:
                    cellstatus = cellid.cell_status
                    if cellstatus.time_of_death<0:
                        cellstatus.time_of_death_frame = -999
                        cellstatus.save()

                    cellrois = CellROI.objects.select_related().filter(cell_id=cellid)
                    for cellroi in cellrois:
                        framenumber = cellroi.frame.number
                        print('frame=',framenumber, '  cellstatus.time_of_death_frame= ',cellstatus.time_of_death_frame)
                        cellflag = cellroi.cellflag_cellroi
                        if framenumber>=cellstatus.time_of_death_frame and cellstatus.time_of_death_frame>=0:
                            cellflag.alive = False
                        else:
                            cellflag.alive = True
                        cellflag.save()
                        print('frame=',framenumber, '  cellflag.alive= ',cellflag.alive )

#___________________________________________________________________________________________
def build_ROIs_loop(exp_name):
    print('build_ROIs_loop exp_name=',exp_name)
    exp_list = Experiment.objects.all()
    for exp in exp_list:
        if exp_name!='' and exp.name!=exp_name:
            continue
        experimentaldataset = ExperimentalDataset.objects.select_related().filter(experiment = exp)
        for expds in experimentaldataset:
            samples = Sample.objects.select_related().filter(experimental_dataset = expds)
            for s in samples:
                build_ROIs(sample=s, force=False)
                

#___________________________________________________________________________________________
def build_ROIs_loop_parallel(exp_name):
    print('build_ROIs_loop exp_name=',exp_name)

    max_workers=25
    # Create a process pool to parallelize build_ROIs
    #with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        exp_list = Experiment.objects.all()
        for exp in exp_list:
            if exp_name!='' and exp.name!=exp_name:
                continue
            experimentaldataset = ExperimentalDataset.objects.select_related().filter(experiment = exp)
            for expds in experimentaldataset:
                samples = Sample.objects.select_related().filter(experimental_dataset = expds)
                for s in samples:
                    executor.submit(build_ROIs, sample=s, force=False)


#___________________________________________________________________________________________
def build_tod_loop_parallel(exp_name):
    print('build_ROIs_loop exp_name=',exp_name)

    max_workers=25
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        exp_list = Experiment.objects.all()
        for exp in exp_list:
            if exp_name!='' and exp.name!=exp_name:
                continue
            experimentaldataset = ExperimentalDataset.objects.select_related().filter(experiment = exp)
            for expds in experimentaldataset:
                samples = Sample.objects.select_related().filter(experimental_dataset = expds)
                for s in samples:
                    cellids = CellID.objects.select_related().filter(sample=s)
                    for cellid in cellids:
                        executor.submit(predict_time_of_death, cellid=cellid)


#___________________________________________________________________________________________
def build_ROIs(sample=None, force=False):
    s=None
    if type(sample) == str:
        s = Sample.objects.get(file_name = sample)
    else:
        s=sample
    cellids = CellID.objects.select_related().filter(sample=s)
    #uncomment to speed up, this will continue if cell is already associated with position
    if force==False and len(cellids)>0:return
    print('build roi sample: ',s.file_name)

    frames = Frame.objects.select_related().filter(sample=s)
    file_name=os.path.join(LOCAL_RAID5,s.file_name)
    if not os.path.exists(file_name):
        file_name=os.path.join(NASRCP_MOUNT_POINT,s.file_name)


    images, channels = read.nd2reader_getFrames(file_name)
    #images are t, c, x, y 
    BF_images=images.transpose(1,0,2,3)
    BF_images=BF_images[0]
    for frame in frames:
        rois_DB = CellROI.objects.select_related().filter(frame = frame)
        #Just for now, should normally check that same ROI don't overlap
        #if len(rois)>0: continue
        #ROIs = segtools.get_ROIs_per_frame(BF_images[frame.number], 2)
        #rois_seg = segtools.triangle_opening(BF_images[frame.number])
        rois_seg=[]
        image_prepro = preprocess_image_pytorch(BF_images[frame.number]).to(device)
        with torch.no_grad():
            predictions = model_detect(image_prepro)
            print(predictions)
            for idx, box in enumerate(predictions[0]['boxes']):
                x_min, y_min, x_max, y_max = box.cpu().numpy()
                if float(predictions[0]['scores'][idx].cpu().numpy())<0.5:continue
                if (x_max-x_min)*(y_max-y_min)<150:continue
                rois_seg.append((int(y_min), int(x_min), int(y_max), int(x_max)))


        roi_number=0
        roi_seg_count=0
        roi_DB_list=[]
        for roi_seg in rois_seg:
            roi=None
            print('----- roi_seg_count=',roi_seg_count)
            roi_seg_count+=1
            x_roi_seg = roi_seg[1]+(roi_seg[3]-roi_seg[1])/2.
            y_roi_seg = roi_seg[0]+(roi_seg[2]-roi_seg[0])/2.
            minDR=100000
            roi_DB_count=0
            for roi_DB in rois_DB:
                if roi_DB_count in roi_DB_list: continue
                x_roi_DB = roi_DB.min_col+(roi_DB.max_col-roi_DB.min_col)/2.
                y_roi_DB = roi_DB.min_row+(roi_DB.max_row-roi_DB.min_row)/2.

                if (math.sqrt(pow(x_roi_seg-x_roi_DB,2) + pow(y_roi_seg-y_roi_DB,2))<50 and 
                    math.sqrt(pow(x_roi_seg-x_roi_DB,2) + pow(y_roi_seg-y_roi_DB,2))<minDR) or \
                    (roi_DB.contour_cellroi.mode == "manual" and roi_DB.contour_cellroi.type == "cell_ROI"):
                    roi = roi_DB
                    roi.roi_number = roi_number
                    minDR = math.sqrt(pow(x_roi_seg-x_roi_DB,2) + pow(y_roi_seg-y_roi_DB,2))
                    print("          take roi in DB ",roi_number, '     ----- roi_DB_count=',roi_DB_count,'  minDR ',minDR)
                roi_DB_count+=1

            if roi==None:
                roi = CellROI(min_row = roi_seg[0], min_col = roi_seg[1],
                                max_row = roi_seg[2], max_col = roi_seg[3], 
                                frame = frame, roi_number=roi_number)
                print("          take new ROI ",roi_number)
                bbox = segtools.validate_roi(BF_images[frame.number], roi.min_row, roi.min_col, roi.max_row, roi.max_col)
                roi.min_row = bbox[0]
                roi.min_col = bbox[1]
                roi.max_row = bbox[2]
                roi.max_col = bbox[3]
                roi.save()

            if hasattr(roi, 'contour_cellroi'):
                if roi.contour_cellroi.mode == "auto" and roi.contour_cellroi.type == "cell_ROI":
                    bbox = segtools.validate_roi(BF_images[frame.number], roi.min_row, roi.min_col, roi.max_row, roi.max_col)
                    roi.min_row = bbox[0]
                    roi.min_col = bbox[1]
                    roi.max_row = bbox[2]
                    roi.max_col = bbox[3]
            roi.save()

            #Bounding box (min_row, min_col, max_row, max_col). 
            cropped_dict = {'shape_original':BF_images[frame.number].shape}
            out_dir_name  = os.path.join(NASRCP_MOUNT_POINT, ANALYSIS_DATA_PATH,s.experimental_dataset.experiment.name, s.experimental_dataset.data_name, os.path.split(s.file_name)[-1].replace('.nd2',''))
            out_file_name = os.path.join(out_dir_name, "frame{0}_ROI{1}.json".format(frame.number, roi_number))
            out_dir_name_DB  = ANALYSIS_DATA_PATH+"/"+s.experimental_dataset.experiment.name+"/"+ s.experimental_dataset.data_name+"/"+os.path.split(s.file_name)[-1].replace('.nd2','')
            out_file_name_DB = out_dir_name_DB+ "/frame{0}_ROI{1}.json".format(frame.number, roi_number)

            if not os.path.exists(out_dir_name):
                os.makedirs(out_dir_name)
            cropped_img = images[frame.number][:, roi.min_row:roi.max_row, roi.min_col:roi.max_col]
            cropped_dict['shape']=[cropped_img.shape[1], cropped_img.shape[2]]
            cropped_dict['npixels']=cropped_img.shape[1]*cropped_img.shape[2]
            cropped_dict['shift']=[roi.min_row, roi.min_col]
            cropped_dict['type']="auto"

            for ch in range(len(channels)):
                cropped_dict['intensity_{}'.format(channels[ch].replace(" ",""))] = cropped_img[ch].tolist()     
            out_file = open(out_file_name, "w") 
            json.dump(cropped_dict, out_file) 
            out_file.close() 

            intensity_mean={}
            intensity_std={}
            intensity_sum={}
            intensity_max={}
            for ch in range(len(channels)): 
                sum=float(np.sum(cropped_img[ch]))
                mean=float(np.mean(cropped_img[ch]))
                std=float(np.std(cropped_img[ch]))
                max=float(np.max(cropped_img[ch]))
                ch_name=channels[ch].replace(" ","")
                intensity_mean[ch_name]=mean
                intensity_std[ch_name]=std
                intensity_sum[ch_name]=sum
                intensity_max[ch_name]=max
            
            contour = None
            if hasattr(roi, 'contour_cellroi'):
                contour = roi.contour_cellroi
            else:
                contour = Contour(cell_roi=roi)

            contour.center_x_pix     = roi.min_col+(roi.max_col-roi.min_col)/2.
            contour.center_y_pix     = roi.min_row+(roi.max_row-roi.min_row)/2.
            contour.center_z_pix     = 0 
            contour.center_x_mic     = (roi.min_col+(roi.max_col-roi.min_col)/2.)*roi.frame.pixel_microns+roi.frame.pos_x
            contour.center_y_mic     = (roi.min_row+(roi.max_row-roi.min_row)/2.)*roi.frame.pixel_microns+roi.frame.pos_y
            contour.center_z_mic     = 0
            contour.intensity_mean   = intensity_mean
            contour.intensity_std    = intensity_std
            contour.intensity_sum    = intensity_sum
            contour.intensity_max    = intensity_max
            contour.number_of_pixels = cropped_img.shape[1]*cropped_img.shape[2]
            contour.file_name        = out_file_name_DB
            contour.type             = "cell_ROI"
            contour.mode             = "auto"
            contour.save()

            if not hasattr(roi, 'cellflag_cellroi'):
                cellflag = CellFlag(cell_roi=roi)
                cellflag.save()

            roi_number+=1
        #check overlapping ROIs
        rois_DB_final = CellROI.objects.select_related().filter(frame = frame)
        if len(rois_DB_final)>1:
            for roi_final_1 in range(len(rois_DB_final)-1):
                print('== roi_final_1=',roi_final_1)
                for roi_final_2 in range(roi_final_1, len(rois_DB_final)):
                    if roi_final_1 == roi_final_2: continue
                    print('    == roi_final_2=',roi_final_2)
                    img_1=np.zeros((frame.height, frame.width))
                    img_2=np.zeros((frame.height, frame.width))
                    img_1[rois_DB_final[roi_final_1].min_row:rois_DB_final[roi_final_1].max_row, rois_DB_final[roi_final_1].min_col:rois_DB_final[roi_final_1].max_col]=True
                    img_2[rois_DB_final[roi_final_2].min_row:rois_DB_final[roi_final_2].max_row, rois_DB_final[roi_final_2].min_col:rois_DB_final[roi_final_2].max_col]=True
                    count_img1=np.count_nonzero(img_1)
                    count_img2=np.count_nonzero(img_2)
                    overlap=img_1*img_2
                    count_overlap=np.count_nonzero(overlap)
                    print('count_overlap/count_img1=',count_overlap/count_img1, '  count_overlap/count_img2=',count_overlap/count_img2)
                    if count_overlap/count_img1 > 0.5 or count_overlap/count_img2>0.5:
                        if count_img1>count_img2:
                            print('1>2: ',rois_DB_final[roi_final_2])
                            if rois_DB_final[roi_final_2].id!=None:rois_DB_final[roi_final_2].delete()
                        else:
                            print('2>1: ',rois_DB_final[roi_final_1])
                            if rois_DB_final[roi_final_1].id!=None:rois_DB_final[roi_final_1].delete()
            #set the roi order after it has been deleted
            rois_DB_final = CellROI.objects.select_related().filter(frame = frame)
            for roi_final in range(len(rois_DB_final)):
                rois_DB_final[roi_final].roi_number = roi_final
                rois_DB_final[roi_final].save()
    print('about to build cells')
    build_cells_sample(s, addmode=False)
    build_cells_sample(s, addmode=True)
    removeROIs(s)


#___________________________________________________________________________________________
async def saveROI(request):
    #roi = sync_to_async(ROI)(min_row=1, max_row=1, roi_number=10000)
    samples = Sample.objects.all()
    print('nsample :',len(samples))
    for s in samples:
        print(s)
    sample = Sample(file_name='totot')
    sample.save()
    roi = CellROI(min_row=1, max_row=1, roi_number=10000, sample=sample)
    roi.asave()
    print('----',roi)


def with_request(f):
    def wrapper(doc):
        return f(doc, doc.session_context.request)
    return wrapper


def print_time(text, prev):
    now=datetime.datetime.now()
    if DEBUG_TIME: print(f'\033[91m {text} deltaT = \033[0m',now-prev)

#___________________________________________________________________________________________
def preprocess_image(image, target_size = (150, 150)):
    # Calculate padding
    image_data = np.array(image, dtype=np.int16)

    if image_data.shape[1]>target_size[1] or image_data.shape[0]>target_size[0]:
        print('-====================== shape',image_data.shape)
        image_data = image_data.resize(target_size)
        print(image_data.shape)
        image_data = image_data / np.max(image_data)
        image_data = np.expand_dims(image_data, axis=-1)
        return image_data

    delta_w = target_size[1] - image_data.shape[1]
    delta_h = target_size[0] - image_data.shape[0]
    pad_width = delta_w // 2
    pad_height = delta_h // 2

    padding = ((pad_height, pad_height), (pad_width, pad_width))

    # Check if the padding difference is odd and distribute padding accordingly
    if delta_w % 2 != 0:
        padding = ((pad_height, pad_height), (pad_width, pad_width+1))

    if delta_h % 2 != 0:
        padding = ((pad_height, pad_height+1), (pad_width, pad_width))

    if delta_h % 2 != 0 and delta_w % 2 != 0:
        padding = ((pad_height, pad_height+1), (pad_width, pad_width+1))
    # Pad the image
    padded_image = np.pad(image, padding, mode='constant', constant_values=0)

    padded_image = padded_image / np.max(padded_image)
    padded_image = np.expand_dims(padded_image, axis=-1)

    return padded_image
#___________________________________________________________________________________________

#___________________________________________________________________________________________
def load_and_preprocess_images(file_list):
    images = []
    filenames = []
    for filename in file_list:
        with open(filename, 'r') as f:
            data = json.load(f)
            for key in data:
                if 'BF' in key.split('_')[-1]:
                    image = data[key]
                    processed_image = preprocess_image(image, (150, 150))
                    images.append(processed_image)
                    filenames.append(filename)
    return np.array(images), filenames
#___________________________________________________________________________________________



#___________________________________________________________________________________________
def segmentation_handler(doc: bokeh.document.Document) -> None:
    start_time=datetime.datetime.now()
    print('****************************  segmentation_handler ****************************')

    #TO BE CHANGED WITH ASYNC?????
    os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"

    experiments=[]
    wells={}
    positions={}
    files={}
    files_id={}
    image_stack_dict={}


    flags_dict = {'mask':0,
                  'dividing':50,
                  'double_nuclei':100,
                  'multiple_cells':150,
                  'pair_cell':200,
                  'flat':250,
                  'round':300,
                  'elongated':350}
    
    for exp in Experiment.objects.all():
        experiments.append(exp.name)
        wells[exp.name] = []
        experimentaldatasets = ExperimentalDataset.objects.select_related().filter(experiment = exp)
        for expds in experimentaldatasets:
            wells[exp.name].append(expds.data_name)
            samples = Sample.objects.select_related().filter(experimental_dataset = expds)
            positions['{0}_{1}'.format(exp.name, expds.data_name)] = []
            files['{0}_{1}'.format(exp.name, expds.data_name)] = []
            files_id['{0}_{1}'.format(exp.name, expds.data_name)] = []
            for samp in samples:
                positions['{0}_{1}'.format(exp.name, expds.data_name)].append(os.path.split(samp.file_name)[1])
                files    ['{0}_{1}'.format(exp.name, expds.data_name)].append(samp.file_name)
                files_id ['{0}_{1}'.format(exp.name, expds.data_name)].append(samp.id)

    experiments=sorted(experiments)
    for i in wells:
        wells[i] = sorted(wells[i])
    for i in positions:
        positions[i] = sorted(positions[i])
        files[i]     = sorted(files[i])

    dropdown_exp  = bokeh.models.Select(value=experiments[0], title='Experiment', options=experiments)
    dropdown_well = bokeh.models.Select(value=wells[experiments[0]][0], title='Well', options=wells[dropdown_exp.value])
    dropdown_pos  = bokeh.models.Select(value=positions['{0}_{1}'.format(experiments[0], wells[experiments[0]][0])][0],title='Position', options=positions['{0}_{1}'.format(dropdown_exp.value, dropdown_well.value)])

    for pos in positions['{0}_{1}'.format(dropdown_exp.value, dropdown_well.value)]:
        image_stack_dict[pos]=None


    initial_position=-9999
    line_position = bokeh.models.Span(location=initial_position, dimension='height', line_color='red', line_width=2)

    source_roi    = bokeh.models.ColumnDataSource(data=dict(left=[], right=[], top=[], bottom=[]))
    source_labels = bokeh.models.ColumnDataSource(data=dict(height=[],weight=[],names=[]))
    source_cells  = bokeh.models.ColumnDataSource(data=dict(height=[],weight=[],names=[]))

    start_oscillation_position = bokeh.models.Span(location=initial_position, dimension='height', line_color='blue', line_width=2)
    end_oscillation_position   = bokeh.models.Span(location=initial_position, dimension='height', line_color='blue', line_width=2)
    time_of_death_position     = bokeh.models.Span(location=initial_position, dimension='height', line_color='black', line_width=2)

    source_varea_death = bokeh.models.ColumnDataSource(data=dict(x=[], y1=[], y2=[]))

    source_varea_rising1  = bokeh.models.ColumnDataSource(data=dict(x=[], y1=[], y2=[]))
    source_varea_rising2  = bokeh.models.ColumnDataSource(data=dict(x=[], y1=[], y2=[]))
    source_varea_rising3  = bokeh.models.ColumnDataSource(data=dict(x=[], y1=[], y2=[]))
    source_varea_rising4  = bokeh.models.ColumnDataSource(data=dict(x=[], y1=[], y2=[]))
    source_varea_rising5  = bokeh.models.ColumnDataSource(data=dict(x=[], y1=[], y2=[]))
    source_varea_rising6  = bokeh.models.ColumnDataSource(data=dict(x=[], y1=[], y2=[]))
    source_varea_rising7  = bokeh.models.ColumnDataSource(data=dict(x=[], y1=[], y2=[]))
    source_varea_rising8  = bokeh.models.ColumnDataSource(data=dict(x=[], y1=[], y2=[]))
    source_varea_rising9  = bokeh.models.ColumnDataSource(data=dict(x=[], y1=[], y2=[]))
    source_varea_rising10 = bokeh.models.ColumnDataSource(data=dict(x=[], y1=[], y2=[]))
    source_varea_rising11 = bokeh.models.ColumnDataSource(data=dict(x=[], y1=[], y2=[]))
    source_varea_rising12 = bokeh.models.ColumnDataSource(data=dict(x=[], y1=[], y2=[]))
    source_varea_rising13 = bokeh.models.ColumnDataSource(data=dict(x=[], y1=[], y2=[]))
    source_varea_rising14 = bokeh.models.ColumnDataSource(data=dict(x=[], y1=[], y2=[]))
    source_varea_rising15 = bokeh.models.ColumnDataSource(data=dict(x=[], y1=[], y2=[]))

    source_varea_falling1  = bokeh.models.ColumnDataSource(data=dict(x=[], y1=[], y2=[]))
    source_varea_falling2  = bokeh.models.ColumnDataSource(data=dict(x=[], y1=[], y2=[]))
    source_varea_falling3  = bokeh.models.ColumnDataSource(data=dict(x=[], y1=[], y2=[]))
    source_varea_falling4  = bokeh.models.ColumnDataSource(data=dict(x=[], y1=[], y2=[]))
    source_varea_falling5  = bokeh.models.ColumnDataSource(data=dict(x=[], y1=[], y2=[]))
    source_varea_falling6  = bokeh.models.ColumnDataSource(data=dict(x=[], y1=[], y2=[]))
    source_varea_falling7  = bokeh.models.ColumnDataSource(data=dict(x=[], y1=[], y2=[]))
    source_varea_falling8  = bokeh.models.ColumnDataSource(data=dict(x=[], y1=[], y2=[]))
    source_varea_falling9  = bokeh.models.ColumnDataSource(data=dict(x=[], y1=[], y2=[]))
    source_varea_falling10 = bokeh.models.ColumnDataSource(data=dict(x=[], y1=[], y2=[]))
    source_varea_falling11 = bokeh.models.ColumnDataSource(data=dict(x=[], y1=[], y2=[]))
    source_varea_falling12 = bokeh.models.ColumnDataSource(data=dict(x=[], y1=[], y2=[]))
    source_varea_falling13 = bokeh.models.ColumnDataSource(data=dict(x=[], y1=[], y2=[]))
    source_varea_falling14 = bokeh.models.ColumnDataSource(data=dict(x=[], y1=[], y2=[]))
    source_varea_falling15 = bokeh.models.ColumnDataSource(data=dict(x=[], y1=[], y2=[]))

    source_nosc      = bokeh.models.ColumnDataSource(data=dict(x=[], top=[]))
    source_nosc_dk   = bokeh.models.ColumnDataSource(data=dict(x=[], top=[]))
    source_nosc_all  = bokeh.models.ColumnDataSource(data=dict(x=[], top=[]))
    source_start_osc = bokeh.models.ColumnDataSource(data=dict(x=[], top=[]))
    source_end_osc   = bokeh.models.ColumnDataSource(data=dict(x=[], top=[]))
    source_tod       = bokeh.models.ColumnDataSource(data=dict(x=[], top=[]))
    source_tod_dk    = bokeh.models.ColumnDataSource(data=dict(x=[], top=[]))
    source_tod_all   = bokeh.models.ColumnDataSource(data=dict(x=[], top=[]))
    source_tod_pred   = bokeh.models.ColumnDataSource(data=dict(x=[], top=[]))

    source_osc_period  = bokeh.models.ColumnDataSource(data=dict(cycle=[], time=[]))
    source_osc_period_err = bokeh.models.ColumnDataSource(data=dict(base=[], upper=[], lower=[]))
    source_osc_period_line = bokeh.models.ColumnDataSource(data=dict(x=[], y=[]))

    source_test_dead = bokeh.models.ColumnDataSource(data=dict(top=[], left=[], right=[]))

    source_segmentation_points = bokeh.models.ColumnDataSource(data=dict(x=[], y=[]))


    ncells_div = bokeh.models.Div(text="<b style='color:black; ; font-size:18px;'> Number of cells=</b>")

    dropdown_filter_position_keep  = bokeh.models.Select(value='all', title='keep', options=['all', 'keep', 'do not keep'])

    print_time('------- PREPARE 1', start_time)



    #___________________________________________________________________________________________
    def filter_position_keep_callback(attr, old, new):

        if dropdown_filter_position_keep.value == 'all':
            dropdown_pos.options = positions['{0}_{1}'.format(dropdown_exp.value, dropdown_well.value)]
            if len(positions['{0}_{1}'.format(dropdown_exp.value, dropdown_well.value)])>0:
                dropdown_pos.value   = positions['{0}_{1}'.format(dropdown_exp.value, dropdown_well.value)][0]

        elif dropdown_filter_position_keep.value == 'keep':
            filtered_list = [k for k in positions['{0}_{1}'.format(dropdown_exp.value, dropdown_well.value)] if 'dk' not in k.split(' - ')[-1]]
            dropdown_pos.options = filtered_list
            if len(filtered_list)>0:
                dropdown_pos.value = filtered_list[0]
        
        else:
            filtered_list = [k for k in positions['{0}_{1}'.format(dropdown_exp.value, dropdown_well.value)] if 'dk' in k.split(' - ')[-1]]
            dropdown_pos.options = filtered_list
            if len(filtered_list)>0:
                dropdown_pos.value = filtered_list[0]            

    dropdown_filter_position_keep.on_change('value', filter_position_keep_callback)
    #___________________________________________________________________________________________


    #___________________________________________________________________________________________
    def next_position_callback():
        if DEBUG: print("****************************  next_position_callback ****************************")
        current_pos = dropdown_pos.options.index(dropdown_pos.value)
        next_pos    = (current_pos + 1) % len(dropdown_pos.options)
        dropdown_pos.value = dropdown_pos.options[next_pos]
    next_position_button = bokeh.models.Button(label="Next pos")
    next_position_button.on_click(next_position_callback)
    #___________________________________________________________________________________________

    #___________________________________________________________________________________________
    def prev_position_callback():
        if DEBUG: print("****************************  prev_position_callback ****************************")
        current_pos = dropdown_pos.options.index(dropdown_pos.value)
        prev_pos    = (current_pos - 1) % len(dropdown_pos.options)
        dropdown_pos.value = dropdown_pos.options[prev_pos]
    prev_position_button = bokeh.models.Button(label="Prev pos")
    prev_position_button.on_click(prev_position_callback)
    #___________________________________________________________________________________________


    #___________________________________________________________________________________________
    def update_position_select(change=True):
        local_pos = []
        local_val = 0
        local_pos_val = dropdown_pos.value

        current_files = files['{0}_{1}'.format(dropdown_exp.value, dropdown_well.value)]
        counter=0
        for pos in dropdown_pos.options:
            mypos=pos.split(' - ')[0]
            current_file = ''
            for f in current_files:
                if mypos in f:
                    current_file = f
                    break
            sample = Sample.objects.get(file_name=current_file)
            if sample.peaks_tod_div_validated and sample.keep_sample and not sample.bf_features_validated:
                local_pos.append('{} - c1'.format(mypos))
            elif sample.peaks_tod_div_validated and not sample.keep_sample and not sample.bf_features_validated:
                local_pos.append('{} - c1,dk'.format(mypos))
            elif not sample.peaks_tod_div_validated and not sample.keep_sample and not sample.bf_features_validated:
                local_pos.append('{} - dk'.format(mypos))
            elif sample.peaks_tod_div_validated and sample.keep_sample and sample.bf_features_validated:
                local_pos.append('{} - c1,c2'.format(mypos))
            elif sample.peaks_tod_div_validated and not sample.keep_sample and sample.bf_features_validated:
                local_pos.append('{} - c1,c2,dk'.format(mypos))
            elif not sample.peaks_tod_div_validated and sample.keep_sample and sample.bf_features_validated:
                local_pos.append('{} - c2'.format(mypos))
            elif not sample.peaks_tod_div_validated and not sample.keep_sample and sample.bf_features_validated:
                local_pos.append('{} - c2,dk'.format(mypos))
            else:
                local_pos.append('{}'.format(mypos))
            if mypos == local_pos_val.split(' - ')[0]:local_val=counter
            counter+=1
        dropdown_pos.options = local_pos
        dropdown_pos.value = local_pos[local_val]
    #___________________________________________________________________________________________

    update_position_select()

    #___________________________________________________________________________________________
    # Function to get the image data stack
    def get_stack_data(current_file, text=''):
        local_time=datetime.datetime.now()
        print('current_filecurrent_filecurrent_filecurrent_filecurrent_filecurrent_filecurrent_filecurrent_file====',current_file)
        current_file_name=os.path.join(LOCAL_RAID5,current_file)
        if not os.path.exists(current_file_name):
            current_file_name=os.path.join(NASRCP_MOUNT_POINT,current_file)
            print('using RCP')
        else:
            print('using local')
        current_file = current_file_name
        print('current_filecurrent_filecurrent_filecurrent_filecurrent_filecurrent_filecurrent_filecurrent_file====',current_file)
        time_lapse_path = Path(current_file)
        print('time_lapse_path = ',time_lapse_path)
        


        print_time(f'------- time_lapse_path get_stack_data {text}', local_time)
        time_lapse = nd2.imread(time_lapse_path.as_posix())
        print_time(f'------- imread get_stack_data {text}', local_time)

        ind_images_list= []
        ind_images_list_norm=[]
        for nch in range(time_lapse.shape[1]):
            time_lapse_tmp = time_lapse[:,nch,:,:] # Assume I(t, c, x, y)
            time_domain = np.asarray(np.linspace(0, time_lapse_tmp.shape[0] - 1, time_lapse_tmp.shape[0]), dtype=np.uint)
            ind_images = [np.flip(time_lapse_tmp[i,:,:],0) for i in time_domain]
            ind_images_norm = []
            for im in ind_images:
                max_value = np.max(im)
                min_value = np.min(im)
                intensity_normalized = (im - min_value)/(max_value-min_value)*255
                intensity_normalized = intensity_normalized.astype(np.uint8)
                ind_images_norm.append(intensity_normalized)
            ind_images_list.append(ind_images)
            ind_images_list_norm.append(ind_images_norm)

        print_time(f'------- end loop get_stack_data {text}', local_time)

        ind_images_array = np.array(ind_images_list)
        ind_images_norm_array = np.array(ind_images_list_norm)
        print_time(f'------- END get_stack_data {text}', local_time)
        return ind_images_array, ind_images_norm_array
    #___________________________________________________________________________________________

    #___________________________________________________________________________________________
    # Function to get the image data stack
    def get_stack_rois_data(current_file, text=''):
        local_time=datetime.datetime.now()
        rois_dict = {'rois':{'left':[], 'right':[], 'top':[], 'bottom':[]}, 
                     'labels':{'weight':[], 'height':[], 'names':[]},
                     'cells':{'weight':[], 'height':[], 'names':[]}
                     }
        sample = Sample.objects.get(file_name=current_file)
        frames = Frame.objects.select_related().filter(sample=sample)
        for frame in frames:
            left_rois=[]
            right_rois=[]
            top_rois=[]
            bottom_rois=[]
            height_labels=[]
            weight_labels=[]
            names_labels=[]
            height_cells=[]
            weight_cells=[]
            names_cells=[]     
            cellrois = CellROI.objects.select_related().filter(frame=frame)
            for roi in cellrois:
                left_rois.append(roi.min_col)
                right_rois.append(roi.max_col)
                top_rois.append(frame.height-roi.min_row)
                bottom_rois.append(frame.height-roi.max_row)
                
                weight_labels.append(roi.min_col)
                height_labels.append(frame.height-roi.min_row)
                names_labels.append('ROI{0} {1}'.format(roi.roi_number,roi.contour_cellroi.mode ))

                weight_cells.append(roi.min_col)
                height_cells.append(frame.height-roi.max_row)
                if roi.cell_id !=None: names_cells.append(roi.cell_id.name)
                else:names_cells.append("none")

            rois_dict['rois']['left'].append(left_rois)
            rois_dict['rois']['right'].append(right_rois)
            rois_dict['rois']['top'].append(top_rois)
            rois_dict['rois']['bottom'].append(bottom_rois)

            rois_dict['labels']['weight'].append(weight_labels)
            rois_dict['labels']['height'].append(height_labels)
            rois_dict['labels']['names'].append(names_labels)

            rois_dict['cells']['weight'].append(weight_cells)
            rois_dict['cells']['height'].append(height_cells)
            rois_dict['cells']['names'].append(names_cells)

        print_time(f'------- END get_stack_rois_data {text}', local_time)
        return rois_dict
    
    #___________________________________________________________________________________________
    def get_masks_data(current_file, tp=-9999, cellname=''):
        local_time=datetime.datetime.now()
        sample = Sample.objects.get(file_name=current_file)
        frames = Frame.objects.select_related().filter(sample=sample)
        cellids = CellID.objects.select_related().filter(sample=sample)
        out_dict={}
        mask1=np.zeros((frames[0].width,frames[0].height), dtype=bool)
        for cellid in cellids:
            out_dict[cellid.name]={}
            try:
                for seg in cellid.cell_status.segmentation['algo']:
                    if seg=='roi':continue
                    out_dict[cellid.name][seg]=[mask1 for i in range(sample.experimental_dataset.experiment.number_of_frames)]
            except KeyError:
                pass

        for frame in frames:
            if tp>-10 and tp!=frame.number:continue
            cellrois = CellROI.objects.select_related().filter(frame=frame)
            for roi in cellrois:
                if cellname!='' and roi.cell_id.name!=cellname:continue
                segmentations = ContourSeg.objects.select_related().filter(cell_roi=roi)
                for seg in segmentations:

                    file_name = os.path.join(LOCAL_RAID5,seg.file_name)
                    if not os.path.exists(file_name):
                        file_name = os.path.join(NASRCP_MOUNT_POINT, seg.file_name)

                    print('file name get_masks_data ', file_name)
                    f = open(file_name)
                    data = json.load(f)
                    out_dict[roi.cell_id.name][seg.algo][frame.number] = np.flip(np.array(data["mask"], dtype=bool),0)

        print_time(f'------- END get_masks_data ', local_time)
        return out_dict

    #___________________________________________________________________________________________
    def get_current_stack():
        if DEBUG: print('****************************  get_current_stack ****************************')
        local_time = datetime.datetime.now()
        current_file = get_current_file(index=0)
        current_pos  = os.path.split(current_file)[1]
        
        if DEBUG_TIME: print_time('------- get_current_stack 1 pos={}, image_stack_dict[current_pos]={}'.format(current_pos, None if image_stack_dict[current_pos]==None else '-->NOT NONE'), local_time)

        if image_stack_dict[current_pos]==None:
            ind_images_list, ind_images_list_norm = get_stack_data(current_file)
            rois_data = get_stack_rois_data(current_file)
            masks_data = get_masks_data(current_file)

            image_stack_dict[current_pos]={'ind_images_list':ind_images_list, 
                                           'ind_images_list_norm':ind_images_list_norm,
                                           'rois':rois_data['rois'],
                                           'labels':rois_data['labels'],
                                           'cells':rois_data['cells'],
                                           'masks':masks_data}

        if DEBUG_TIME: print_time('------- get_current_stack END ', local_time)
        #return image_stack_dict[current_pos]
        return current_pos
    #___________________________________________________________________________________________

    #___________________________________________________________________________________________
    def get_adjacent_stack(number=5):
        current_pos_list=[]
        current_file_list=[]
        for n in range(-number+1, number+2):
            #if n==0:continue
            current_file = get_current_file(index=n)
            current_file_list.append(current_file)
            current_pos_list.append(os.path.split(current_file)[1])

        for k in image_stack_dict:
            if k in current_pos_list:
                if image_stack_dict[k]==None:
                    ind_images_list, ind_images_list_norm = get_stack_data(current_file_list[current_pos_list.index(k)], 'get_adjacent_stack')
                    rois_data = get_stack_rois_data(current_file_list[current_pos_list.index(k)], 'get_adjacent_stack')
                    masks_data = get_masks_data(current_file_list[current_pos_list.index(k)])

                    image_stack_dict[k]={'ind_images_list':ind_images_list, 
                                         'ind_images_list_norm':ind_images_list_norm,
                                         'rois':rois_data['rois'],
                                         'labels':rois_data['labels'],
                                         'cells':rois_data['cells'],
                                         'masks':masks_data}


            else:
                if image_stack_dict[k]!=None:
                    image_stack_dict[k]=None
    #___________________________________________________________________________________________

 
    #___________________________________________________________________________________________
    def get_adjacent_stack_test(number=0):

        current_file = get_current_file(index=number)
        current_pos = os.path.split(current_file)[1]


        if image_stack_dict[current_pos]==None:
            image_stack_dict[current_pos]='Processing'
            ind_images_list, ind_images_list_norm = get_stack_data(current_file, 'get_adjacent_stack_test')
            rois_data = get_stack_rois_data(current_file, 'get_adjacent_stack_test')
            masks_data = get_masks_data(current_file)

            image_stack_dict[current_pos]={'ind_images_list':ind_images_list, 
                                    'ind_images_list_norm':ind_images_list_norm,
                                    'rois':rois_data['rois'],
                                    'labels':rois_data['labels'],
                                    'cells':rois_data['cells'],
                                    'masks':masks_data}

    #___________________________________________________________________________________________

    #___________________________________________________________________________________________
    # Function to get the current file
    def get_current_file(index=0):
        current_files = files['{0}_{1}'.format(dropdown_exp.value, dropdown_well.value)]
        current_file  = ''
        current_file_index = -9999
        indexed_file_index = -9999
        for f in range(len(current_files)):

            if dropdown_pos.value.split(' - ')[0] in current_files[f]:
                current_file_index = f
                break
        if index!=0:
            indexed_file_index = (current_file_index + index) % len(current_files)
            current_file = current_files[indexed_file_index]
        else:
            current_file = current_files[current_file_index]

        return current_file
    #___________________________________________________________________________________________


    #___________________________________________________________________________________________
    # Function to get the current file
    def get_current_file_id(index=0):
        current_files = files['{0}_{1}'.format(dropdown_exp.value, dropdown_well.value)]
        current_files_id = files_id['{0}_{1}'.format(dropdown_exp.value, dropdown_well.value)]
        current_file_id  = -9999
        current_file_index = -9999
        indexed_file_index = -9999
        for f in range(len(current_files)):
            if dropdown_pos.value.split(' - ')[0] in current_files[f]:
                current_file_index = f
                break
        if index!=0:
            indexed_file_index = (current_file_index + index) % len(current_files)
            current_file_id = current_files_id[indexed_file_index]
        else:
            current_file_id = current_files_id[current_file_index]

        return current_file_id
    #___________________________________________________________________________________________

    current_pos_data = get_current_stack()
    current_stack_data = image_stack_dict[current_pos_data]
    ind_images_list = current_stack_data['ind_images_list']
    ind_images_list_norm = current_stack_data['ind_images_list_norm']


    #current images (current index and list of channels)
    data_img_ch={'img':[ind_images_list[ch][0] for ch in range(len(ind_images_list))]}
    source_img_ch = bokeh.models.ColumnDataSource(data=data_img_ch)

    #list of all images for all channels
    data_imgs={'images':ind_images_list}
    source_imgs = bokeh.models.ColumnDataSource(data=data_imgs)

    #list of all images for all channels
    data_imgs_norm={'images':ind_images_list_norm}
    source_imgs_norm = bokeh.models.ColumnDataSource(data=data_imgs_norm)


    #current image to be displayed
    data_img={'img':[data_imgs_norm['images'][0][0]]}
    source_img = bokeh.models.ColumnDataSource(data=data_img)


    source_rois_full   = bokeh.models.ColumnDataSource(data=current_stack_data['rois'])
    source_labels_full = bokeh.models.ColumnDataSource(data=current_stack_data['labels'])
    source_cells_full  = bokeh.models.ColumnDataSource(data=current_stack_data['cells'])

    source_intensity_ch0 = bokeh.models.ColumnDataSource(data={'time':[], 'intensity':[]})
    source_intensity_ch1 = bokeh.models.ColumnDataSource(data={'time':[], 'intensity':[]})
    source_intensity_ch2 = bokeh.models.ColumnDataSource(data={'time':[], 'intensity':[]})
    source_intensity_max = bokeh.models.ColumnDataSource(data={'time':[], 'intensity':[]})
    source_intensity_min = bokeh.models.ColumnDataSource(data={'time':[], 'intensity':[]})

    source_intensity_area = bokeh.models.ColumnDataSource(data={'time':[], 'area':[]})
    source_intensity_predicted_death = bokeh.models.ColumnDataSource(data={'time':[], 'intensity':[]})

    source_segments_cell      = bokeh.models.ColumnDataSource(data={'time':[], 'intensity':[]})
    source_mask_cell          = bokeh.models.ColumnDataSource(data={'time':[], 'intensity_full':[]})
    source_dividing_cell      = bokeh.models.ColumnDataSource(data={'time':[], 'intensity':[], 'intensity_full':[]})
    source_double_nuclei_cell = bokeh.models.ColumnDataSource(data={'time':[], 'intensity':[], 'intensity_full':[]})
    source_multiple_cells     = bokeh.models.ColumnDataSource(data={'time':[], 'intensity':[], 'intensity_full':[]})
    source_pair_cell          = bokeh.models.ColumnDataSource(data={'time':[], 'intensity':[], 'intensity_full':[]})
    source_flat_cell          = bokeh.models.ColumnDataSource(data={'time':[], 'intensity':[], 'intensity_full':[]})
    source_round_cell         = bokeh.models.ColumnDataSource(data={'time':[], 'intensity':[], 'intensity_full':[]})
    source_elongated_cell     = bokeh.models.ColumnDataSource(data={'time':[], 'intensity':[], 'intensity_full':[]})
    

    # Create a Slider widget
    initial_time_point = 0
    slider         = bokeh.models.Slider(start=0, end=len(ind_images_list[0]) - 1, value=initial_time_point, step=1, title="Time Point", width=250)

    x_range = bokeh.models.Range1d(start=0, end=ind_images_list[0][0].shape[0])
    y_range = bokeh.models.Range1d(start=0, end= ind_images_list[0][0].shape[1])
    #plot_image     = bokeh.plotting.figure(x_range=(0, ind_images_list[0][0].shape[0]), y_range=(0, ind_images_list[0][0].shape[1]), tools="box_select,wheel_zoom,box_zoom,reset,undo",width=550, height=550)
    #plot_img_mask  = bokeh.plotting.figure(x_range=(0, ind_images_list[0][0].shape[0]), y_range=(0, ind_images_list[0][0].shape[1]), tools="box_select,wheel_zoom,box_zoom,reset,undo",width=550, height=550)
    plot_image     = bokeh.plotting.figure(x_range=x_range, y_range=y_range, tools="box_select,wheel_zoom,box_zoom,reset,undo",width=550, height=550)
    #plot_img_mask  = bokeh.plotting.figure(x_range=x_range, y_range=y_range, tools="box_select,wheel_zoom,box_zoom,reset,undo",width=550, height=550)

    plot_image.toolbar.active_tap = None

    plot_intensity = bokeh.plotting.figure(title="Intensity vs Time", x_axis_label='Time (minutes)', y_axis_label='Intensity',width=1000, height=500, tools="pan,box_zoom,save,reset")
    plot_tod       = bokeh.plotting.figure(title="Time of death", x_axis_label='Time (30 mins bins)', y_axis_label='Number of positions',width=550, height=350)
    plot_tod_pred  = bokeh.plotting.figure(title="Time of death predicted", x_axis_label='Time (30 mins bins)', y_axis_label='Number of positions',width=550, height=350)
    plot_nosc      = bokeh.plotting.figure(title="Number of oscillations", x_axis_label='Number of oscillations', y_axis_label='Number of positions',width=550, height=350)

    plot_intensity.extra_y_ranges['area'] = bokeh.models.Range1d(start=0, end=100)

    slider_find_peaks  = bokeh.models.Slider(start=0, end=100, value=30, step=1, title="Peak prominence", width=200)

    #plot_intensity.varea(x='x', y1='y1', y2='y2', fill_alpha=0.10, fill_color='black', source=source_varea_death)



    #___________________________________________________________________________________________
    # Function to prepare the intensity plot
    def prepare_intensity():
        local_time = datetime.datetime.now()
        if DEBUG:print('----------------prepare_intensity--------------------dropdown_cell.value=',dropdown_cell.value)
        current_file=get_current_file()
        sample = Sample.objects.get(file_name=current_file)
        allcellids = CellID.objects.select_related().filter(sample=sample)

        ncells_div.text="<b style='color:black; ; font-size:18px;'> Number of cells={}</b>".format(len(allcellids))

        if DEBUG_TIME:print_time('------- prepare_intensity 1 ', local_time)

        if dropdown_cell.value!='':

            cellids = CellID.objects.select_related().filter(sample=sample, name=dropdown_cell.value)
            cellid_cell_status = cellids[0].cell_status
            if DEBUG_TIME:print_time('------- prepare_intensity 2 ', local_time)

            #Set start of oscilation if it exist, -999 else
            if cellid_cell_status.start_oscillation>0: 
                start_oscillation_position.location = source_intensity_ch1.data["time"][cellid_cell_status.start_oscillation_frame]
            else: 
                start_oscillation_position.location = -999

            #Set end of oscilation if it exist, -999 else
            if cellid_cell_status.end_oscillation>0: 
                end_oscillation_position.location   = source_intensity_ch1.data["time"][cellid_cell_status.end_oscillation_frame]
            else: 
                end_oscillation_position.location   = -999

            source_intensity_predicted_death.data={'time':[], 'intensity':[]}

            if cellid_cell_status.time_of_death_pred<-9900:
                predict_time_of_death(cellids[0])
                if cellid_cell_status.time_of_death_frame_pred>-10:
                    source_intensity_predicted_death.data={'time':[cellid_cell_status.time_of_death_pred], 'intensity':[source_intensity_ch1.data["intensity"][cellid_cell_status.time_of_death_frame_pred]]}
            elif cellid_cell_status.time_of_death_pred<-90 and cellid_cell_status.time_of_death_pred>-10000:
                source_intensity_predicted_death.data={'time':[], 'intensity':[]}
            else:
                source_intensity_predicted_death.data={'time':[cellid_cell_status.time_of_death_pred], 'intensity':[source_intensity_ch1.data["intensity"][cellid_cell_status.time_of_death_frame_pred]]}

            #Set time of death and varea if it exist, -999 [] else
            if cellid_cell_status.time_of_death>0:
                time_of_death_position.location = source_intensity_ch1.data["time"][cellid_cell_status.time_of_death_frame]
                source_varea_death.data['x']    = [source_intensity_ch1.data["time"][t] for t in range(cellid_cell_status.time_of_death_frame, len(source_intensity_ch1.data["time"])) ]
                source_varea_death.data['y1']   = [source_intensity_ch1.data["intensity"][t] for t in range(cellid_cell_status.time_of_death_frame, len(source_intensity_ch1.data["intensity"]))]
                source_varea_death.data['y2']   = [0 for i in range(len(source_varea_death.data['y1']))]

            else:
                time_of_death_position.location = -999
                source_varea_death.data['x']    = []
                source_varea_death.data['y1']   = []
                source_varea_death.data['y2']   = []


            #Set maximums and mimimums if exists [] else
            if len(cellids[0].cell_status.peaks)>=6:
                source_intensity_max.data={'time':cellids[0].cell_status.peaks["max_time"], 
                                           #'intensity':cellids[0].cell_status.peaks["max_int"]}
                                           'intensity':[source_intensity_ch1.data["intensity"][t] for t in cellids[0].cell_status.peaks["max_frame"]]}
                source_intensity_min.data={'time':cellids[0].cell_status.peaks["min_time"], 
                                           #'intensity':cellids[0].cell_status.peaks["min_int"]}
                                           'intensity':[source_intensity_ch1.data["intensity"][t] for t in cellids[0].cell_status.peaks["min_frame"]]}

            else:
                source_intensity_max.data={'time':[], 'intensity':[]}
                source_intensity_min.data={'time':[], 'intensity':[]}


            source_segments_cell.data={'time':[], 'intensity':[]}

            cell_flags = cellid_cell_status.flags

            #____________________________________________________
            try: 
                source_mask_cell.data={'time':cell_flags["mask_time"], 
                                       'intensity_full':[source_intensity_ch1.data["intensity"][t] for t in cell_flags["mask_frame"]]}
            except KeyError:
                source_mask_cell.data={'time':[], 'intensity_full':[]}

            #____________________________________________________
            try: 
                source_dividing_cell.data={'time':cell_flags["dividing_time"], 
                                           'intensity_full':[source_intensity_ch1.data["intensity"][t] for t in cell_flags["dividing_frame"]],
                                           'intensity':[flags_dict['dividing'] for t in cell_flags["dividing_frame"]]}
                data_segment = {'time':source_segments_cell.data["time"]+cell_flags["dividing_time"], 
                                'intensity':source_segments_cell.data["intensity"]+[source_intensity_ch1.data["intensity"][t] for t in cell_flags["dividing_frame"]]}
                source_segments_cell.data=data_segment
            except KeyError:
                source_dividing_cell.data={'time':[], 'intensity':[], 'intensity_full':[]}

            #____________________________________________________
            try: 
                source_double_nuclei_cell.data={'time':cell_flags["double_nuclei_time"], 
                                                'intensity_full':[source_intensity_ch1.data["intensity"][t] for t in cell_flags["double_nuclei_frame"]],
                                                'intensity':[flags_dict['double_nuclei'] for t in cell_flags["double_nuclei_frame"]]}
                data_segment = {'time':source_segments_cell.data["time"]+cell_flags["double_nuclei_time"], 
                                'intensity':source_segments_cell.data["intensity"]+[source_intensity_ch1.data["intensity"][t] for t in cell_flags["double_nuclei_frame"]]}
                source_segments_cell.data=data_segment
            except KeyError:
                source_double_nuclei_cell.data={'time':[], 'intensity':[], 'intensity_full':[]}


            #____________________________________________________
            try: 
                source_multiple_cells.data={'time':cell_flags["multiple_cells_time"], 
                                           'intensity_full':[source_intensity_ch1.data["intensity"][t] for t in cell_flags["multiple_cells_frame"]],
                                           'intensity':[flags_dict['multiple_cells'] for t in cell_flags["multiple_cells_frame"]]}
                data_segment = {'time':source_segments_cell.data["time"]+cell_flags["multiple_cells_time"], 
                                'intensity':source_segments_cell.data["intensity"]+[source_intensity_ch1.data["intensity"][t] for t in cell_flags["multiple_cells_frame"]]}
                source_segments_cell.data=data_segment
            except KeyError:
                source_multiple_cells.data={'time':[], 'intensity':[], 'intensity_full':[]}

            #____________________________________________________
            try: 
                source_pair_cell.data={'time':cell_flags["pair_cell_time"], 
                                       'intensity_full':[source_intensity_ch1.data["intensity"][t] for t in cell_flags["pair_cell_frame"]],
                                       'intensity':[flags_dict['pair_cell'] for t in cell_flags["pair_cell_frame"]]}
                data_segment = {'time':source_segments_cell.data["time"]+cell_flags["pair_cell_time"], 
                                'intensity':source_segments_cell.data["intensity"]+[source_intensity_ch1.data["intensity"][t] for t in cell_flags["pair_cell_frame"]]}
                source_segments_cell.data=data_segment
            except KeyError:
                source_pair_cell.data={'time':[], 'intensity':[], 'intensity_full':[]}

            #____________________________________________________
            try: 
                source_flat_cell.data={'time':cell_flags["flat_time"], 
                                       'intensity_full':[source_intensity_ch1.data["intensity"][t] for t in cell_flags["flat_frame"]],
                                       'intensity':[flags_dict['flat'] for t in cell_flags["flat_frame"]]}
                data_segment = {'time':source_segments_cell.data["time"]+cell_flags["flat_time"], 
                                'intensity':source_segments_cell.data["intensity"]+[source_intensity_ch1.data["intensity"][t] for t in cell_flags["flat_frame"]]}
                source_segments_cell.data=data_segment
            except KeyError:
                source_flat_cell.data={'time':[], 'intensity':[], 'intensity_full':[]}

            #____________________________________________________
            try: 
                source_round_cell.data={'time':cell_flags["round_time"], 
                                        'intensity_full':[source_intensity_ch1.data["intensity"][t] for t in cell_flags["round_frame"]],
                                        'intensity':[flags_dict['round'] for t in cell_flags["round_frame"]]}
                data_segment = {'time':source_segments_cell.data["time"]+cell_flags["round_time"], 
                                'intensity':source_segments_cell.data["intensity"]+[source_intensity_ch1.data["intensity"][t] for t in cell_flags["round_frame"]]}
                source_segments_cell.data=data_segment
            except KeyError:
                source_round_cell.data={'time':[], 'intensity':[], 'intensity_full':[]}

            #____________________________________________________
            try: 
                source_elongated_cell.data={'time':cell_flags["elongated_time"], 
                                            'intensity_full':[source_intensity_ch1.data["intensity"][t] for t in cell_flags["elongated_frame"]],
                                            'intensity':[flags_dict['elongated'] for t in cell_flags["elongated_frame"]]}
                data_segment = {'time':source_segments_cell.data["time"]+cell_flags["elongated_time"], 
                                'intensity':source_segments_cell.data["intensity"]+[source_intensity_ch1.data["intensity"][t] for t in cell_flags["elongated_frame"]]}
                source_segments_cell.data=data_segment
            except KeyError:
                source_elongated_cell.data={'time':[], 'intensity':[], 'intensity_full':[]}

            if DEBUG_TIME:print_time('------- prepare_intensity 3 ', local_time)
            set_rising_falling(cellids[0])
            if DEBUG_TIME: print_time('------- prepare_intensity 4 ', local_time)

        else:
            if DEBUG: print('in the else ----------------prepare_intensity-------------------- in the else')

            line_position.location = 0
            if len(source_intensity_ch1.data["time"])!=0:
                line_position.location = source_intensity_ch1.data["time"][0]
            else:
                line_position.location = -999
            start_oscillation_position.location = -999
            end_oscillation_position.location   = -999

            time_of_death_position.location = -999

            source_varea_death.data['x']    = []
            source_varea_death.data['y1']   = []
            source_varea_death.data['y2']   = []

            source_intensity_max.data      = {'time':[], 'intensity':[]}
            source_intensity_min.data      = {'time':[], 'intensity':[]}
    
            source_mask_cell.data          = {'time':[], 'intensity_full':[]}
            source_dividing_cell.data      = {'time':[], 'intensity':[], 'intensity_full':[]}
            source_double_nuclei_cell.data = {'time':[], 'intensity':[], 'intensity_full':[]}
            source_multiple_cells.data     = {'time':[], 'intensity':[], 'intensity_full':[]}
            source_pair_cell.data          = {'time':[], 'intensity':[], 'intensity_full':[]}
            source_flat_cell.data          = {'time':[], 'intensity':[], 'intensity_full':[]}
            source_round_cell.data         = {'time':[], 'intensity':[], 'intensity_full':[]}
            source_elongated_cell.data     = {'time':[], 'intensity':[], 'intensity_full':[]}
            source_segments_cell.data      = {'time':[], 'intensity':[]}

            set_rising_falling(None)

        if DEBUG: print('prepare intensity - - - - - - - source_varea_death.data',  source_varea_death.data)
    #___________________________________________________________________________________________



    #___________________________________________________________________________________________
    def set_rising_falling_local(max_list, min_list):

        if DEBUG:print('-----------------set_rising_falling_local-------------------------     ')
        arrays_r = {}
        for i in range(1,15):
            array_x  = 'xr_{}'.format(i)
            array_y1 = 'yr1_{}'.format(i)
            array_y2 = 'yr2_{}'.format(i)
            arrays_r[array_x]  = []
            arrays_r[array_y1] = []
            arrays_r[array_y2] = []

        source_rising = {}
        source_rising[1]=source_varea_rising1
        source_rising[2]=source_varea_rising2
        source_rising[3]=source_varea_rising3
        source_rising[4]=source_varea_rising4
        source_rising[5]=source_varea_rising5
        source_rising[6]=source_varea_rising6
        source_rising[7]=source_varea_rising7
        source_rising[8]=source_varea_rising8
        source_rising[9]=source_varea_rising9
        source_rising[10]=source_varea_rising10
        source_rising[11]=source_varea_rising11
        source_rising[12]=source_varea_rising12
        source_rising[13]=source_varea_rising13
        source_rising[14]=source_varea_rising14
        source_rising[15]=source_varea_rising15

        arrays_f = {}
        for i in range(1,15):
            array_x  = 'xf_{}'.format(i)
            array_y1 = 'yf1_{}'.format(i)
            array_y2 = 'yf2_{}'.format(i)
            arrays_f[array_x]  = []
            arrays_f[array_y1] = []
            arrays_f[array_y2] = []

        source_falling = {}
        source_falling[1]=source_varea_falling1
        source_falling[2]=source_varea_falling2
        source_falling[3]=source_varea_falling3
        source_falling[4]=source_varea_falling4
        source_falling[5]=source_varea_falling5
        source_falling[6]=source_varea_falling6
        source_falling[7]=source_varea_falling7
        source_falling[8]=source_varea_falling8
        source_falling[9]=source_varea_falling9
        source_falling[10]=source_varea_falling10
        source_falling[11]=source_varea_falling11
        source_falling[12]=source_varea_falling12
        source_falling[13]=source_varea_falling13
        source_falling[14]=source_varea_falling14
        source_falling[15]=source_varea_falling15


        if end_oscillation_position.location  < 0 or start_oscillation_position.location < 0:
            for i in range(1,15):
                source_rising[i].data={'x':arrays_r['xr_{}'.format(i)], 'y1':arrays_r['yr1_{}'.format(i)], 'y2':arrays_r['yr2_{}'.format(i)]}
                source_falling[i].data={'x':arrays_f['xf_{}'.format(i)], 'y1':arrays_f['yf1_{}'.format(i)], 'y2':arrays_f['yf2_{}'.format(i)]}
            return
        

        for m in range(len(max_list)):
            min_val=source_intensity_ch1.data["time"].index(start_oscillation_position.location)
            for n in range(len(min_list)):
                if min_list[n]<max_list[m]: 
                    min_val=min_list[n]
            for t in range(source_intensity_ch1.data["time"].index(start_oscillation_position.location), source_intensity_ch1.data["time"].index(end_oscillation_position.location)):

                if t==min_val and source_intensity_ch1.data["time"].index(start_oscillation_position.location)==min_val and t<max_list[m] and max_list[m]>min_val:
                    arrays_r['xr_{}'.format(m+1)].append(source_intensity_ch1.data["time"][t])
                    arrays_r['yr1_{}'.format(m+1)].append(0)
                    arrays_r['yr2_{}'.format(m+1)].append(source_intensity_ch1.data["intensity"][t])

                if t<max_list[m] and max_list[m]>min_val and t>min_val:
                    arrays_r['xr_{}'.format(m+1)].append(source_intensity_ch1.data["time"][t])
                    arrays_r['yr1_{}'.format(m+1)].append(0)
                    arrays_r['yr2_{}'.format(m+1)].append(source_intensity_ch1.data["intensity"][t])
  
        for i in range(1,15):
            source_rising[i].data={'x':arrays_r['xr_{}'.format(i)], 'y1':arrays_r['yr1_{}'.format(i)], 'y2':arrays_r['yr2_{}'.format(i)]}



        for m in range(len(max_list)):
            min_val=source_intensity_ch1.data["time"].index(end_oscillation_position.location)
            for n in range(len(min_list)):
                if max_list[m]<min_list[n]:
                    min_val=min_list[n]
                    break
            for t in range(source_intensity_ch1.data["time"].index(start_oscillation_position.location), source_intensity_ch1.data["time"].index(end_oscillation_position.location)+1):

                if t>max_list[m] and max_list[m]<min_val and t<min_val:
                    arrays_f['xf_{}'.format(m+1)].append(source_intensity_ch1.data["time"][t])
                    arrays_f['yf1_{}'.format(m+1)].append(0)
                    arrays_f['yf2_{}'.format(m+1)].append(source_intensity_ch1.data["intensity"][t])

                if t==source_intensity_ch1.data["time"].index(end_oscillation_position.location) and min_val==source_intensity_ch1.data["time"].index(end_oscillation_position.location):
                    arrays_f['xf_{}'.format(m+1)].append(source_intensity_ch1.data["time"][t])
                    arrays_f['yf1_{}'.format(m+1)].append(0)
                    arrays_f['yf2_{}'.format(m+1)].append(source_intensity_ch1.data["intensity"][t])

        for i in range(1,15):
            source_falling[i].data={'x':arrays_f['xf_{}'.format(i)], 'y1':arrays_f['yf1_{}'.format(i)], 'y2':arrays_f['yf2_{}'.format(i)]}
    #___________________________________________________________________________________________



    #___________________________________________________________________________________________
    def set_rising_falling(cellid, save_status=False, delete_status=False):
        if DEBUG: print("*************************set_rising_falling*****************************************")
        arrays_r = {}
        for i in range(1,15):
            array_x  = 'xr_{}'.format(i)
            array_y1 = 'yr1_{}'.format(i)
            array_y2 = 'yr2_{}'.format(i)
            arrays_r[array_x]  = []
            arrays_r[array_y1] = []
            arrays_r[array_y2] = []

        source_rising = {}
        source_rising[1]=source_varea_rising1
        source_rising[2]=source_varea_rising2
        source_rising[3]=source_varea_rising3
        source_rising[4]=source_varea_rising4
        source_rising[5]=source_varea_rising5
        source_rising[6]=source_varea_rising6
        source_rising[7]=source_varea_rising7
        source_rising[8]=source_varea_rising8
        source_rising[9]=source_varea_rising9
        source_rising[10]=source_varea_rising10
        source_rising[11]=source_varea_rising11
        source_rising[12]=source_varea_rising12
        source_rising[13]=source_varea_rising13
        source_rising[14]=source_varea_rising14
        source_rising[15]=source_varea_rising15


        arrays_f = {}
        for i in range(1,15):
            array_x  = 'xf_{}'.format(i)
            array_y1 = 'yf1_{}'.format(i)
            array_y2 = 'yf2_{}'.format(i)
            arrays_f[array_x]  = []
            arrays_f[array_y1] = []
            arrays_f[array_y2] = []

        source_falling = {}
        source_falling[1]=source_varea_falling1
        source_falling[2]=source_varea_falling2
        source_falling[3]=source_varea_falling3
        source_falling[4]=source_varea_falling4
        source_falling[5]=source_varea_falling5
        source_falling[6]=source_varea_falling6
        source_falling[7]=source_varea_falling7
        source_falling[8]=source_varea_falling8
        source_falling[9]=source_varea_falling9
        source_falling[10]=source_varea_falling10
        source_falling[11]=source_varea_falling11
        source_falling[12]=source_varea_falling12
        source_falling[13]=source_varea_falling13
        source_falling[14]=source_varea_falling14
        source_falling[15]=source_varea_falling15

        if cellid==None:
            for i in range(1,15):
                source_rising[i].data  = {'x':arrays_r['xr_{}'.format(i)], 'y1':arrays_r['yr1_{}'.format(i)], 'y2':arrays_r['yr2_{}'.format(i)]}
                source_falling[i].data = {'x':arrays_f['xf_{}'.format(i)], 'y1':arrays_f['yf1_{}'.format(i)], 'y2':arrays_f['yf2_{}'.format(i)]}
            return
        
        osc_dict={'rising_frame':[], 'falling_frame':[],
                  'rising_time':[],  'falling_time':[]}
        

        #####BEGIN OF THIS IS TEMPORARY TO RESTORE STATUS IN CELLSTATUS.FLAGS
        cellstatus = cellid.cell_status
        if len(cellstatus.flags)==6 or len(cellstatus.flags)==4:
            print('IN TEMPOREARUYUUUUUUUUUUUUUUUUUUUUU set_rising_falling')
            cellrois = CellROI.objects.select_related().filter(cell_id=cellid)
            osc_dict_final={"mask_frame":[],"mask_time":[],
                            "dividing_frame":[],"dividing_time":[],
                            "double_nuclei_frame":[], "double_nuclei_time":[],
                            "multiple_cells_frame":[],"multiple_cells_time":[],
                            "pair_cell_frame":[],"pair_cell_time":[],
                            "flat_frame":[],"flat_time":[],
                            "round_frame":[],"round_time":[],
                            "elongated_frame":[], "elongated_time":[]}
            for cellroi in cellrois:
                framenumber = cellroi.frame.number
                cellflag = cellroi.cellflag_cellroi

                if cellflag.mask:
                    osc_dict_final["mask_frame"].append(framenumber)
                    osc_dict_final["mask_time"].append(source_intensity_ch1.data["time"][framenumber])
                if cellflag.dividing:
                    osc_dict_final["dividing_frame"].append(framenumber)
                    osc_dict_final["dividing_time"].append(source_intensity_ch1.data["time"][framenumber])
                if cellflag.double_nuclei:
                    osc_dict_final["double_nuclei_frame"].append(framenumber)
                    osc_dict_final["double_nuclei_time"].append(source_intensity_ch1.data["time"][framenumber])
                if cellflag.multiple_cells:
                    osc_dict_final["multiple_cells_frame"].append(framenumber)
                    osc_dict_final["multiple_cells_time"].append(source_intensity_ch1.data["time"][framenumber])
                if cellflag.pair_cell:
                    osc_dict_final["pair_cell_frame"].append(framenumber)
                    osc_dict_final["pair_cell_time"].append(source_intensity_ch1.data["time"][framenumber])
                if cellflag.flat:
                    osc_dict_final["flat_frame"].append(framenumber)
                    osc_dict_final["flat_time"].append(source_intensity_ch1.data["time"][framenumber])
                if cellflag.round:
                    osc_dict_final["round_frame"].append(framenumber)
                    osc_dict_final["round_time"].append(source_intensity_ch1.data["time"][framenumber])
                if cellflag.elongated:
                    osc_dict_final["elongated_frame"].append(framenumber)
                    osc_dict_final["elongated_time"].append(source_intensity_ch1.data["time"][framenumber])

            osc_dict_final.update(cellstatus.flags)
            cellstatus.flags = osc_dict_final
            cellstatus.save()
        #####END OF THIS IS TEMPORARY TO RESTORE STATUS IN CELLSTATUS.FLAGS




        if len(cellid.cell_status.peaks)==6:
            peaks = cellid.cell_status.peaks
            start_frame=cellid.cell_status.start_oscillation_frame
            end_frame=cellid.cell_status.end_oscillation_frame

            for m in range(len(peaks["max_frame"])):
                min_val=start_frame
                for n in range(len(peaks["min_frame"])):
                    if peaks["min_frame"][n]<peaks["max_frame"][m]: 
                        min_val=peaks["min_frame"][n]
                for t in range(start_frame, end_frame):

                    if t==min_val and start_frame==min_val and t<peaks["max_frame"][m] and peaks["max_frame"][m]>min_val:
                        arrays_r['xr_{}'.format(m+1)].append(source_intensity_ch1.data["time"][t])
                        arrays_r['yr1_{}'.format(m+1)].append(0)
                        arrays_r['yr2_{}'.format(m+1)].append(source_intensity_ch1.data["intensity"][t])
                        osc_dict["rising_frame"].append(t)
                        osc_dict["rising_time"].append(source_intensity_ch1.data["time"][t])

                    if t<peaks["max_frame"][m] and peaks["max_frame"][m]>min_val and t>min_val:
                        arrays_r['xr_{}'.format(m+1)].append(source_intensity_ch1.data["time"][t])
                        arrays_r['yr1_{}'.format(m+1)].append(0)
                        arrays_r['yr2_{}'.format(m+1)].append(source_intensity_ch1.data["intensity"][t])
                        osc_dict["rising_frame"].append(t)
                        osc_dict["rising_time"].append(source_intensity_ch1.data["time"][t])


            for m in range(len(peaks["max_frame"])):
                min_val=end_frame
                for n in range(len(peaks["min_frame"])):
                    if peaks["max_frame"][m]<peaks["min_frame"][n]:
                        min_val=peaks["min_frame"][n]
                        break
                for t in range(start_frame, end_frame+1):

                    if t>peaks["max_frame"][m] and peaks["max_frame"][m]<min_val and t<min_val:
                        arrays_f['xf_{}'.format(m+1)].append(source_intensity_ch1.data["time"][t])
                        arrays_f['yf1_{}'.format(m+1)].append(0)
                        arrays_f['yf2_{}'.format(m+1)].append(source_intensity_ch1.data["intensity"][t])
                        osc_dict["falling_frame"].append(t)
                        osc_dict["falling_time"].append(source_intensity_ch1.data["time"][t])

                    if t==cellid.cell_status.end_oscillation_frame and min_val==end_frame:
                        arrays_f['xf_{}'.format(m+1)].append(source_intensity_ch1.data["time"][t])
                        arrays_f['yf1_{}'.format(m+1)].append(0)
                        arrays_f['yf2_{}'.format(m+1)].append(source_intensity_ch1.data["intensity"][t])
                        osc_dict["falling_frame"].append(t)
                        osc_dict["falling_time"].append(source_intensity_ch1.data["time"][t])                        
        for i in range(1,15):
            source_falling[i].data={'x':arrays_f['xf_{}'.format(i)], 'y1':arrays_f['yf1_{}'.format(i)], 'y2':arrays_f['yf2_{}'.format(i)]}
            source_rising[i].data={'x':arrays_r['xr_{}'.format(i)], 'y1':arrays_r['yr1_{}'.format(i)], 'y2':arrays_r['yr2_{}'.format(i)]}

        if save_status and cellid!=None:
            cellrois = CellROI.objects.select_related().filter(cell_id=cellid)
            if DEBUG or True:
                print('cellid=',cellid, ' cellrois=',len(cellrois))
                print('osc_dict=',osc_dict)
                print('cellid.cell_status.start_oscillation_frame ',cellid.cell_status.start_oscillation_frame)
                print('cellid.cell_status.end_oscillation_frame ',cellid.cell_status.end_oscillation_frame)
                print('cellid.cell_status.time_of_death_frame ',cellid.cell_status.time_of_death_frame)
            for cellroi in cellrois:
                framenumber = cellroi.frame.number
                if DEBUG:print('  frame=',framenumber)
                cellflag = cellroi.cellflag_cellroi

                if framenumber in osc_dict['rising_frame']:  cellflag.rising = True
                else: cellflag.rising = False

                if framenumber in osc_dict['falling_frame']: cellflag.falling = True
                else: cellflag.falling = False

                if framenumber in cellid.cell_status.peaks["max_frame"]: cellflag.maximum = True
                else: cellflag.maximum = False

                if framenumber in cellid.cell_status.peaks["min_frame"]: cellflag.minimum = True
                else: cellflag.minimum = False

                if framenumber>=cellid.cell_status.start_oscillation_frame and \
                    framenumber<=cellid.cell_status.end_oscillation_frame:
                    cellflag.oscillating = True
                else: cellflag.oscillating = False

                if framenumber>=cellid.cell_status.time_of_death_frame and cellid.cell_status.time_of_death>0: cellflag.alive = False
                else: cellflag.alive = True

                #print('framenumber = ',framenumber, '   cellflag.alive = ',cellflag.alive)
                cellflag.save()


            osc_dict_final = cellstatus.flags
            osc_dict_final.update(osc_dict)
            cellstatus = cellid.cell_status
            cellstatus.flags = osc_dict_final
            cellstatus.save()

        if delete_status and cellid!=None:
            cellrois = CellROI.objects.select_related().filter(cell_id=cellid)
            for cellroi in cellrois:
                cellflag = cellroi.cellflag_cellroi
                cellflag.alive          = True
                cellflag.rising         = False
                cellflag.falling        = False
                cellflag.maximum        = False
                cellflag.minimum        = False
                cellflag.oscillating    = False
                cellflag.last_osc       = False
                cellflag.mask           = False
                cellflag.dividing       = False
                cellflag.double_nuclei  = False
                cellflag.multiple_cells = False
                cellflag.pair_cell      = False
                cellflag.flat           = False
                cellflag.round          = False
                cellflag.elongated      = False
                cellflag.save()

        #if DEBUG:print('osc_dict=',osc_dict)
        #cellrois = CellROI.objects.select_related().filter(cell_id=cellid)

        #for cellroi in cellrois:
        #    cellflag = cellroi.cellflag_cellroi
    #___________________________________________________________________________________________

 
    #___________________________________________________________________________________________
    # Function to update the well depending on the experiment
    def update_dropdown_well(attr, old, new):
        if DEBUG: print('****************************  update_dropdown_well ****************************')
        dropdown_well.options = wells[dropdown_exp.value]
        if DEBUG:
            print('+++++++++++++++++  dropdown_exp.value ',dropdown_exp.value, '  wells[dropdown_exp.value]  +++++++++++  ',wells[dropdown_exp.value],'   ',)
            print('+++++++++++++++++  positions[{0}_{1}.format(dropdown_exp.value, wells[dropdown_exp.value][0])][0] +++  ',positions['{0}_{1}'.format(dropdown_exp.value, wells[dropdown_exp.value][0])][0])
            print('+++++++++++++++++  {0}_{1}.format(dropdown_exp.value, wells[dropdown_exp.value][0]) +++++++++++++++++  ', '{0}_{1}'.format(dropdown_exp.value, wells[dropdown_exp.value][0]))
        dropdown_well.value   = wells[dropdown_exp.value][0]
        dropdown_pos.options  = positions['{0}_{1}'.format(dropdown_exp.value, wells[dropdown_exp.value][0])]

        update_position_select()

        if slider.value == 0:
            if DEBUG:print('in the if update_dropdown_well')
            left_rois,right_rois,top_rois,bottom_rois,height_labels, weight_labels, names_labels,height_cells, weight_cells, names_cells=update_source_roi_cell_labels()
        
            source_roi.data = {'left': left_rois, 'right': right_rois, 'top': top_rois, 'bottom': bottom_rois}
            source_labels.data = {'height':height_labels, 'weight':weight_labels, 'names':names_labels}
            source_cells.data = {'height':height_cells, 'weight':weight_cells, 'names':names_cells}
        else:
            if DEBUG:print('in the else update_dropdown_well')
            slider.value = 0
        slider.start = 0
        slider.end=len(source_imgs.data['images'][0]) - 1

    dropdown_exp.on_change('value', update_dropdown_well)
    #___________________________________________________________________________________________


    #___________________________________________________________________________________________
    def update_oscillation_cycle():
        current_file = get_current_file()
        sample       = Sample.objects.get(file_name=current_file)
        expds        = ExperimentalDataset.objects.get(id=sample.experimental_dataset.id)
        samples = Sample.objects.select_related().filter(experimental_dataset=expds)
        cycle=[]
        time=[]
        for s in samples:
            if not s.peaks_tod_div_validated: continue
            if not s.keep_sample:continue
            cellIDs = CellID.objects.select_related().filter(sample=s)
            for cellID in cellIDs:
                peaks = cellID.cell_status.peaks
                max_time=[]
                try:
                    max_time = peaks["max_time"]
                except KeyError:
                    continue
                if len(max_time)<2: break
                for i in range(len(max_time)-1):
                    cycle.append(i+1)
                    time.append(max_time[i+1]-max_time[i])
        source_osc_period.data=dict(cycle=cycle, time=time)
        

        classes = list(set(cycle))
        tmp_dict={}
        for cl in classes:
            tmp_dict[cl]=[]
        for i in range(len(cycle)):
            tmp_dict[cycle[i]].append(time[i])
        upper=[]
        lower=[]
        mean=[]
        for c in range(1, len(classes)+1):
            array = np.array(tmp_dict[c])
            upper.append(np.mean(array)+np.std(array)/2)
            lower.append(np.mean(array)-np.std(array)/2)
            mean.append(np.mean(array))
        source_osc_period_err.data=dict(base=classes, upper=upper, lower=lower)
        source_osc_period_line.data=dict(x=classes, y=mean)

    #___________________________________________________________________________________________




    #___________________________________________________________________________________________
    # Function to update the position depending on the experiment and the well
    def update_dropdown_pos(attr, old, new):
        if DEBUG:print('****************************  update_dropdown_pos ****************************')

        image_stack_dict.clear
        for pos in positions['{0}_{1}'.format(dropdown_exp.value, dropdown_well.value)]:
            image_stack_dict[pos] = None

        dropdown_pos.options = positions['{0}_{1}'.format(dropdown_exp.value, dropdown_well.value)]
        dropdown_pos.value   = positions['{0}_{1}'.format(dropdown_exp.value, dropdown_well.value)][0]

        source_osc_period.data=dict(cycle=[], time=[])

        update_oscillation_cycle()

        update_position_select()

        if slider.value == 0:
            if DEBUG:print('in the if update_dropdown_pos')
            left_rois,right_rois,top_rois,bottom_rois,height_labels, weight_labels, names_labels,height_cells, weight_cells, names_cells=update_source_roi_cell_labels()
        
            source_roi.data = {'left': left_rois, 'right': right_rois, 'top': top_rois, 'bottom': bottom_rois}
            source_labels.data = {'height':height_labels, 'weight':weight_labels, 'names':names_labels}
            source_cells.data = {'height':height_cells, 'weight':weight_cells, 'names':names_cells}
        else:
            if DEBUG:print('in the else update_dropdown_pos')
            slider.value = 0
        slider.start = 0
        slider.end=len(source_imgs.data['images'][0]) - 1
        #CLEMENT TEST TIME
        update_source_osc_tod()

        #slider_find_peaks.value = 30
    dropdown_well.on_change('value', update_dropdown_pos)
    #___________________________________________________________________________________________


    #___________________________________________________________________________________________
    def update_dropdown_channel(attr, old, new):
        if DEBUG:print('****************************  update_dropdown_channel ****************************')
        ch_list=[]
        for ch in range(len(source_img_ch.data['img'])):
            ch_list.append(str(ch))
        dropdown_channel.options = ch_list
        dropdown_channel.value   = dropdown_channel.value        
        
        new_image = source_imgs_norm.data['images'][int(dropdown_channel.value)][slider.value]
        source_img.data   = {'img':[new_image]}

        update_source_segment(slider.value)
        #img_min = new_image.min()
        #img_max = new_image.max()
        #color_mapper.low  = img_min
        #color_mapper.high = img_max

        #contrast_slider.value = (img_min, img_max)
        #contrast_slider.start = img_min
        #contrast_slider.end = img_max
        if DEBUG:
            print('update_dropdown_channel options: ',dropdown_channel.options)
            print('update_dropdown_channel value  : ',dropdown_channel.value)

    channel_list=[]
    for ch in range(len(ind_images_list)):channel_list.append(str(ch))
    dropdown_channel  = bokeh.models.Select(value='0', title='Channel', options=channel_list)   
    dropdown_channel.on_change('value', update_dropdown_channel)
    #___________________________________________________________________________________________

    #___________________________________________________________________________________________
    def update_dropdown_cell(attr, old, new):
        if DEBUG:print('****************************  update_dropdown_cell ****************************')
        local_time=datetime.datetime.now()

        source_intensity_ch0.data={'time':[], 'intensity':[]}
        source_intensity_ch1.data={'time':[], 'intensity':[]}
        source_intensity_ch2.data={'time':[], 'intensity':[]}

        current_file_id=get_current_file_id()
        sample = Sample.objects.get(id=current_file_id)
        if sample.peaks_tod_div_validated:
            position_check_div.text = "<b style='color:green; ; font-size:18px;'> Peaks/ToD/Division validated (c1)</b>"
            position_check_button.label = "Peaks/ToD/Division not validated"
        else:
            position_check_div.text = "<b style='color:red; ; font-size:18px;'> Peaks/ToD/Division not validated</b>"
            position_check_button.label = "Peaks/ToD/Division validated"

        if sample.bf_features_validated:
            position_check2_div.text = "<b style='color:green; ; font-size:18px;'> BF features validated (c2)</b>"
            position_check2_button.label = "BF features not validated"
        else:
            position_check2_div.text = "<b style='color:red; ; font-size:18px;'> BF features not validated</b>"
            position_check2_button.label = "BF features validated"

        if sample.keep_sample:
            position_keep_div.text  = "<b style='color:green; ; font-size:18px;'> Keep Position</b>"
            position_keep_button.label = "Don't Keep Position"
        else:
            position_keep_div.text  = "<b style='color:red; ; font-size:18px;'> Do not Keep Position (dk)</b>"
            position_keep_button.label = "Keep Position"

        print_time('------- update_dropdown_cell 1 ', local_time)

        cellIDs = CellID.objects.select_related().filter(sample=sample)

        cell_list=[]
        cell_list_id=[]
        for cid in cellIDs:
            cell_list.append(cid.name)
            cell_list_id.append(cid.id)
            if cid.name!=dropdown_cell.value:continue
            time_list={}
            intensity_list={}
            area_list={}
            ROIs = CellROI.objects.select_related().filter(cell_id=cid)

            #This is to fill the new element cell_status.segmentation in case it does not exist yet
            try:
                dropdown_segmentation_type.options = cid.cell_status.segmentation['algo']
            except KeyError:
                cell_status = cid.cell_status
                cell_status.segmentation = {'algo':['roi']}
                dropdown_segmentation_type.options = ['roi']
                cell_status.save()

            old_seg_val = dropdown_segmentation_type.value
            if old_seg_val not in dropdown_segmentation_type.options:
                dropdown_segmentation_type.value=dropdown_segmentation_type.options[0]

            print_time('------- update_dropdown_cell 1.1 ', local_time)


            for roi in ROIs:
                roi_frame = roi.frame
                roi_contour_cellroi = roi.contour_cellroi
                if dropdown_segmentation_type.value == 'roi':
                    for ch in roi_contour_cellroi.intensity_sum:

                        try:
                            time_list[ch]
                        except KeyError:
                            #CHANGE WITH ssample.number_of_frames when available
                            #nframes=sample.number_of_frames
                            frames=Frame.objects.select_related().filter(sample=sample)
                            nframes=len(frames)
                            intensity_list[ch]=[0 for i in range(nframes)]
                            time_list[ch]=[f.time/60000 for f in frames]
                            area_list[ch]=[0 for i in range(nframes)]

                        if   dropdown_intensity_type.value == 'sum': 
                            intensity_list[ch][roi_frame.number]= getattr(roi_contour_cellroi, 'intensity_sum')[ch]
                        elif dropdown_intensity_type.value == 'avg': 
                            intensity_list[ch][roi_frame.number]= getattr(roi_contour_cellroi, 'intensity_mean')[ch]
                        elif   dropdown_intensity_type.value == 'max': 
                            intensity_list[ch][roi_frame.number]= getattr(roi_contour_cellroi, 'intensity_max')[ch]
                        elif   dropdown_intensity_type.value == 'std': 
                            intensity_list[ch][roi_frame.number]= getattr(roi_contour_cellroi, 'intensity_std')[ch]

                print_time('------- update_dropdown_cell 1.2 ', local_time)

                if dropdown_segmentation_type.value != 'roi':
                    contours = ContourSeg.objects.select_related().filter(cell_roi=roi, algo=dropdown_segmentation_type.value)

                    if len(contours)==0:return
                    contour  = contours[0]
                    for ch in contour.intensity_sum:

                        try:
                            time_list[ch]
                        except KeyError:
                            #CHANGE WITH ssample.number_of_frames when available
                            #nframes=sample.number_of_frames
                            frames=Frame.objects.select_related().filter(sample=sample)
                            nframes=len(frames)
                            intensity_list[ch]=[0 for i in range(nframes)]
                            time_list[ch]=[f.time/60000 for f in frames]
                            area_list[ch]=[0 for i in range(nframes)]
                        area_list[ch][roi.frame.number]=contour.number_of_pixels
                        if   dropdown_intensity_type.value == 'sum': 
                            intensity_list[ch][roi.frame.number]= getattr(contour, 'intensity_sum')[ch]
                        elif dropdown_intensity_type.value == 'avg': 
                            intensity_list[ch][roi.frame.number]= getattr(contour, 'intensity_mean')[ch]
                        elif   dropdown_intensity_type.value == 'max': 
                            intensity_list[ch][roi.frame.number]= getattr(contour, 'intensity_max')[ch]
                        elif   dropdown_intensity_type.value == 'std': 
                            intensity_list[ch][roi.frame.number]= getattr(contour, 'intensity_std')[ch]

            print_time('------- update_dropdown_cell 1.3 ', local_time)

            for index, key in enumerate(time_list):
                if index==0:
                    sorted_lists = sorted(zip(time_list[key], intensity_list[key], area_list[key])) 
                    time_sorted, intensity_sorted, area_sorted = zip(*sorted_lists) 
                    source_intensity_ch0.data={'time':time_sorted, 'intensity':intensity_sorted}
                    source_intensity_area.data={'time':time_sorted, 'area':area_sorted}
                    plot_intensity.extra_y_ranges['area'].start = min(area_sorted)*0.7
                    plot_intensity.extra_y_ranges['area'].end = max(area_sorted)*1.3
                    
                if index==1:
                    sorted_lists = sorted(zip(time_list[key], intensity_list[key])) 
                    time_sorted, intensity_sorted = zip(*sorted_lists) 
                    source_intensity_ch1.data={'time':time_sorted, 'intensity':intensity_sorted}    
                if index==2:
                    sorted_lists = sorted(zip(time_list[key], intensity_list[key])) 
                    time_sorted, intensity_sorted = zip(*sorted_lists) 
                    source_intensity_ch2.data={'time':time_sorted, 'intensity':intensity_sorted}
        dropdown_cell.options=cell_list

        print_time('------- update_dropdown_cell 2 ', local_time)
        print('source_intensity_area  ',source_intensity_area.data)

        #plot_intensity.y_range = bokeh.models.Range1d(max(source_intensity_ch1.data["intensity"])*0.4, max(source_intensity_ch1.data["intensity"])*1.2, bounds="auto")

        if dropdown_cell.value=='':
            if len(dropdown_cell.options)>0:
                dropdown_cell.value = dropdown_cell.options[0]
            else:
                dropdown_cell.value = ''
        if len(dropdown_cell.options)==0:
            dropdown_cell.value = ''
        if dropdown_cell.value not in dropdown_cell.options:
            if len(dropdown_cell.options)>0: 
                dropdown_cell.value=dropdown_cell.options[0]
            else: 
                dropdown_cell.value = ''


        print_time('------- update_dropdown_cell 3 ', local_time)

        #CLEMENT TEST COMMENTED
        #threading.Thread(target = prepare_intensity).start()
        prepare_intensity()
        slider_find_peaks.end   = 30
        if len(source_intensity_ch1.data['intensity'])>0:
            slider_find_peaks.end   = int(np.max(source_intensity_ch1.data['intensity'])-np.min(source_intensity_ch1.data['intensity']))
        slider_find_peaks.value = int(slider_find_peaks.end/3.)
        if slider_find_peaks.end<250:
            slider_find_peaks.step = 1
        elif slider_find_peaks.end>=250 and slider_find_peaks.end<500:
            slider_find_peaks.step = 5
        elif slider_find_peaks.end>=500 and slider_find_peaks.end<1000:
            slider_find_peaks.step = 10
        elif slider_find_peaks.end>=1000 and slider_find_peaks.end<5000:
            slider_find_peaks.step = 20
        elif slider_find_peaks.end>=5000 and slider_find_peaks.end<10000:
            slider_find_peaks.step = 50
        elif slider_find_peaks.end>=10000:
            slider_find_peaks.step = 100

        #CLEMENT TEST COMMENTED
        #prepare_intensity()
        #print_time('------- update_dropdown_cell 5 ', local_time)
        print_time('------- update_dropdown_cell END ', local_time)


        if dropdown_cell.value!='':
            cellID = CellID.objects.get(id=cell_list_id[cell_list.index(dropdown_cell.value)])
            try:
                dropdown_segmentation_type.options = cellID.cell_status.segmentation['algo']
            except KeyError:
                cell_status = cellID.cell_status
                cell_status.segmentation = {'algo':['roi']}
                cell_status.save()
    dropdown_cell  = bokeh.models.Select(value='', title='Cell', options=[])   
    dropdown_cell.on_change('value', update_dropdown_cell)
    #___________________________________________________________________________________________

    #___________________________________________________________________________________________
    def update_dropdown_color(attr, old, new):
        if DEBUG:print('****************************  update_dropdown_channel ****************************')
        palette = dropdown_color.value
        color_mapper.palette = palette


    colormaps = ['Greys256','Inferno256','Viridis256']
    color_mapper = bokeh.models.LinearColorMapper(palette="Greys256", low=source_img.data['img'][0].min(), high=source_img.data['img'][0].max())
    color_bar = bokeh.models.ColorBar(color_mapper=color_mapper, location=(0,0))
    dropdown_color = bokeh.models.Select(title="Color Palette", value="Grey256",options=colormaps)
    dropdown_color.on_change('value',update_dropdown_color)
    #___________________________________________________________________________________________


    #___________________________________________________________________________________________
    # Function to update the position
    def prepare_pos(attr, old, new):
        if DEBUG:print('****************************  prepare_pos ****************************')

        local_time = datetime.datetime.now()
        #current_stack_data = get_current_stack()
        current_pos_data = get_current_stack()
        current_stack_data = image_stack_dict[current_pos_data]
        #threading.Thread(target = get_adjacent_stack).start()

        threads = []
        for iii in range(-2,6):
            threads.append(threading.Thread(target = get_adjacent_stack_test, args=(iii, )))
        for t in threads: t.start()
        #for t in threads: t.join()

        print_time('------- prepare_pos 0 ', local_time)


        images      = current_stack_data['ind_images_list']
        images_norm = current_stack_data['ind_images_list_norm']
        rois_data   = current_stack_data['rois']
        labels_data = current_stack_data['labels']
        cells_data  = current_stack_data['cells']

        source_imgs.data       = {'images':images}
        source_imgs_norm.data  = {'images':images_norm}
        source_img_ch.data     = {'img':[images[ch][0] for ch in range(len(images))]}
        source_img.data        = {'img':[images_norm[int(dropdown_channel.value)][0]]}

        source_segmentation_points.data = dict(x=[], y=[])


        source_rois_full.data['left']   = rois_data['left']
        source_rois_full.data['right']  = rois_data['right']
        source_rois_full.data['top']    = rois_data['top']
        source_rois_full.data['bottom'] = rois_data['bottom']

        source_labels_full.data['weight'] = labels_data['weight']
        source_labels_full.data['height'] = labels_data['height']
        source_labels_full.data['names']  = labels_data['names']

        source_cells_full.data['weight'] = cells_data['weight']
        source_cells_full.data['height'] = cells_data['height']
        source_cells_full.data['names']  = cells_data['names']

        source_roi.data    = {'left'  : source_rois_full.data['left'][0], 
                              'right' : source_rois_full.data['right'][0], 
                              'top'   : source_rois_full.data['top'][0], 
                              'bottom': source_rois_full.data['bottom'][0]}
        
        source_labels.data = {'height':source_labels_full.data['height'][0],
                              'weight':source_labels_full.data['weight'][0], 
                              'names':source_labels_full.data['names'][0]}
        
        source_cells.data  = {'height':source_cells_full.data['height'][0], 
                              'weight':source_cells_full.data['weight'][0], 
                              'names':source_cells_full.data['names'][0]}


        dropdown_channel.value = dropdown_channel.options[0]
        dropdown_color.value   = dropdown_color.options[0]

        print_time('------- prepare_pos 1 ', local_time)

        if slider.value != 0:
            if DEBUG:print('in the else prepare_pos')
            slider.value = 0
            slider.start = 0
            slider.end   = len(source_imgs.data['images'][0]) - 1
    
        print_time('------- prepare_pos 2 ', local_time)

        slider.end=len(source_imgs.data['images'][0]) - 1
        print_time('------- prepare_pos 3 ', local_time)

        reset_tap_tool()
        print_time('------- prepare_pos 4 ', local_time)

        update_dropdown_channel('','','')
        print_time('------- prepare_pos 5 ', local_time)

        intensity_type_callback('','','')
        print_time('------- prepare_pos 6 ', local_time)

        update_source_segment()
        print_time('------- prepare_pos end ', local_time)

        print('prepare pos source_varea_death.data = =  == =  = == =  = ', source_varea_death.data)

    dropdown_pos.on_change('value', prepare_pos)
    #___________________________________________________________________________________________



    #___________________________________________________________________________________________
    # Function to get the current index
    def get_current_index():
        if DEBUG:print('****************************  get_current_index ****************************')
        return slider.value

    refresh_time_list = ["100", "200", "300", "400", "500", "1000"]
    dropdown_refresh_time = bokeh.models.Select(value=refresh_time_list[1], title="time (ms)", options=refresh_time_list)

    # Callback function to handle menu item click
    #___________________________________________________________________________________________
    def refresh_time_callback(attr, old, new):
        if DEBUG:
            print('****************************  refresh_time_callback ****************************')
            print("refresh time : {}".format(dropdown_refresh_time.value))
        play_stop_callback()
        play_stop_callback()
    dropdown_refresh_time.on_change('value',refresh_time_callback)
    #___________________________________________________________________________________________


    #___________________________________________________________________________________________
    def intensity_type_callback(attr, old, new):
        local_time = datetime.datetime.now()
        #threading don't work, document locked
        #threading.Thread(target = update_dropdown_cell, args=('','','',)).start()

        update_dropdown_cell('','','')
        print_time('------- intensity_type_callback 1 ', local_time)

        sample = Sample.objects.get(file_name=get_current_file())
        cellid = CellID.objects.select_related().filter(sample=sample)

        print_time('------- intensity_type_callback 2 ', local_time)

        for cell in cellid:
            if cell.name != dropdown_cell.value:continue
            peaks=cell.cell_status.peaks
            if len(peaks)>=6:
                int_max=[]
                for i in peaks["max_frame"]:
                    int_max.append(source_intensity_ch1.data["intensity"][i])
                source_intensity_max.data={'time':peaks['max_time'], 'intensity':int_max}  
                int_min=[]
                for i in peaks["min_frame"]:
                    int_min.append(source_intensity_ch1.data["intensity"][i])
                source_intensity_min.data={'time':peaks['min_time'], 'intensity':int_min}  
        print_time('------- intensity_type_callback end ', local_time)

    int_type_list = ["avg", "max", "sum",  "std"]
    dropdown_intensity_type = bokeh.models.Select(value=int_type_list[0], title="intensity", options=int_type_list)
    dropdown_intensity_type.on_change('value', intensity_type_callback)
    #___________________________________________________________________________________________



    #___________________________________________________________________________________________
    def segmentation_type_callback(attr, old, new):
        if DEBUG:
            print('segmentation_type_callback value=',dropdown_segmentation_type.value)
        update_dropdown_cell('','','')
        sample = Sample.objects.get(file_name=get_current_file())
        cellid = CellID.objects.select_related().filter(sample=sample)

        for cell in cellid:
            if cell.name != dropdown_cell.value:continue
            peaks=cell.cell_status.peaks
            if len(peaks)>=6:
                int_max=[]
                for i in peaks["max_frame"]:
                    int_max.append(source_intensity_ch1.data["intensity"][i])
                source_intensity_max.data={'time':peaks['max_time'], 'intensity':int_max}  
                int_min=[]
                for i in peaks["min_frame"]:
                    int_min.append(source_intensity_ch1.data["intensity"][i])
                source_intensity_min.data={'time':peaks['min_time'], 'intensity':int_min}  
        update_source_segment(slider.value)

    seg_type_list = ["roi"]#, "SAM2_b+"]
    dropdown_segmentation_type = bokeh.models.Select(value=seg_type_list[0], title="segmentation", options=seg_type_list)
    dropdown_segmentation_type.on_change('value', segmentation_type_callback)
    #___________________________________________________________________________________________

    # update the source_roi
    def update_source_roi_cell_labels():
        if DEBUG:print('****************************  update_source_roi_cell_labels ****************************')
        left_rois=[]
        right_rois=[]
        top_rois=[]
        bottom_rois=[]
        height_labels=[]
        weight_labels=[]
        names_labels=[]
        height_cells=[]
        weight_cells=[]
        names_cells=[]        
        current_file=get_current_file()
        current_index=get_current_index()
        sample = Sample.objects.get(file_name=current_file)
        frame  = Frame.objects.select_related().filter(sample=sample, number=current_index)
        if len(frame)!=1:
            print('NOT ONLY FRAME FOUND< PLEASE CHECKKKKKKKK')
            print('======sample: ',sample)
            for f in frame:
                print('===============frame: ',f)
        rois   = CellROI.objects.select_related().filter(frame=frame[0])

        for roi in rois:
            left_rois.append(roi.min_col)
            right_rois.append(roi.max_col)
            top_rois.append(frame[0].height-roi.min_row)
            bottom_rois.append(frame[0].height-roi.max_row)

            weight_labels.append(roi.min_col)
            height_labels.append(frame[0].height-roi.min_row)
            names_labels.append('ROI{0} {1}'.format(roi.roi_number,roi.contour_cellroi.mode ))

            weight_cells.append(roi.min_col)
            height_cells.append(frame[0].height-roi.max_row)
            if roi.cell_id !=None: names_cells.append(roi.cell_id.name)
            else:names_cells.append("none")
        if DEBUG:print('ppppppp update_source_roi ',left_rois, right_rois, top_rois, bottom_rois)

        return left_rois,right_rois,top_rois,bottom_rois, height_labels, weight_labels, names_labels, height_cells, weight_cells, names_cells


    #___________________________________________________________________________________________
    def run_segmentation_point(x,y):
        current_file = get_current_file()
        current_pos  = os.path.split(current_file)[1]
        image_BF = image_stack_dict[current_pos]['ind_images_list'][0][slider.value]

        sample = Sample.objects.get(file_name=current_file)
        frame = Frame.objects.select_related().filter(sample=sample).get(number=slider.value)
        cellid = CellID.objects.select_related().filter(sample=sample).get(name=dropdown_cell.value)
        cellroi = CellROI.objects.select_related().filter(frame=frame, cell_id=cellid)
        contour = ContourSeg.objects.select_related().filter(cell_roi=cellroi[0]).get(algo='SAM2_b+')
        mask=build_segmentation_sam2_single_frame(x,y,image_BF)#, sample, cell)
        image_stack_dict[current_pos]['masks'][dropdown_cell.value]['SAM2_b+'][slider.value]=mask
        source_img_mask.data = {'img':[mask]}

        area = list(source_intensity_area.data['area'])
        area[slider.value]=mask.sum()
        source_intensity_area.data={'time':source_intensity_area.data['time'], 'area':area}


        channels=sample.experimental_dataset.experiment.name_of_channels.split(',')
        center = ndimage.center_of_mass(mask)
        contour.center_x_pix = center[0]
        contour.center_y_pix = center[1]
        contour.center_x_mic = center[0]*cellroi[0].frame.pixel_microns+cellroi[0].frame.pos_x
        contour.center_y_mic = center[1]*cellroi[0].frame.pixel_microns+cellroi[0].frame.pos_y
        contour.algo = 'SAM2_b+'

        intensity_mean={}
        intensity_std={}
        intensity_sum={}
        intensity_max={}
        for ch in range(len(channels)): 
            segment=mask*image_stack_dict[current_pos]['ind_images_list'][ch][slider.value]
            sum=float(np.sum(segment))
            mean=float(sum/mask.sum())
            std=float(np.std(segment))
            max=float(np.max(segment))
            ch_name=channels[ch].replace(" ","")
            intensity_mean[ch_name]=mean
            intensity_std[ch_name]=std
            intensity_sum[ch_name]=sum
            intensity_max[ch_name]=max

            if ch==0:
                intensity = list(source_intensity_ch0.data['intensity'])
                if dropdown_intensity_type.value=="avg":
                    intensity[slider.value]=mean
                    source_intensity_ch0.data={'time':source_intensity_ch0.data['time'], 'intensity':intensity}
                elif dropdown_intensity_type.value=="max":
                    intensity[slider.value]=max
                    source_intensity_ch0.data={'time':source_intensity_ch0.data['time'], 'intensity':intensity}
                elif dropdown_intensity_type.value=="sum":
                    intensity[slider.value]=sum
                    source_intensity_ch0.data={'time':source_intensity_ch0.data['time'], 'intensity':intensity}
                elif dropdown_intensity_type.value=="std":
                    intensity[slider.value]=std
                    source_intensity_ch0.data={'time':source_intensity_ch0.data['time'], 'intensity':intensity}
            elif ch==1:
                intensity = list(source_intensity_ch1.data['intensity'])
                if dropdown_intensity_type.value=="avg":
                    intensity[slider.value]=mean
                    source_intensity_ch1.data={'time':source_intensity_ch1.data['time'], 'intensity':intensity}
                elif dropdown_intensity_type.value=="max":
                    intensity[slider.value]=max
                    source_intensity_ch1.data={'time':source_intensity_ch1.data['time'], 'intensity':intensity}
                elif dropdown_intensity_type.value=="sum":
                    intensity[slider.value]=sum
                    source_intensity_ch1.data={'time':source_intensity_ch1.data['time'], 'intensity':intensity}
                elif dropdown_intensity_type.value=="std":
                    intensity[slider.value]=std
                    source_intensity_ch1.data={'time':source_intensity_ch1.data['time'], 'intensity':intensity}

            elif ch==2:
                intensity = list(source_intensity_ch2.data['intensity'])
                if dropdown_intensity_type.value=="avg":
                    intensity[slider.value]=mean
                    source_intensity_ch2.data={'time':source_intensity_ch2.data['time'], 'intensity':intensity}
                elif dropdown_intensity_type.value=="max":
                    intensity[slider.value]=max
                    source_intensity_ch2.data={'time':source_intensity_ch2.data['time'], 'intensity':intensity}
                elif dropdown_intensity_type.value=="sum":
                    intensity[slider.value]=sum
                    source_intensity_ch2.data={'time':source_intensity_ch2.data['time'], 'intensity':intensity}
                elif dropdown_intensity_type.value=="std":
                    intensity[slider.value]=std
                    source_intensity_ch2.data={'time':source_intensity_ch2.data['time'], 'intensity':intensity}
        contour.intensity_max  = intensity_max
        contour.intensity_mean = intensity_mean
        contour.intensity_std  = intensity_std
        contour.intensity_sum  = intensity_sum
        contour.number_of_pixels = mask.sum()


        segment_dict = {'mask':np.flip(mask,0).tolist()}
        out_dir_name  = os.path.join(NASRCP_MOUNT_POINT, ANALYSIS_DATA_PATH,sample.experimental_dataset.experiment.name, sample.experimental_dataset.data_name, os.path.split(sample.file_name)[-1].replace('.nd2',''))
        out_file_name = os.path.join(out_dir_name, "frame{0}_ROI{1}_{2}_mask.json".format(cellroi[0].frame.number, cellroi[0].roi_number, 'SAM2_b+'))

        out_dir_name_DB  = ANALYSIS_DATA_PATH+"/"+sample.experimental_dataset.experiment.name+"/"+ sample.experimental_dataset.data_name+"/"+os.path.split(sample.file_name)[-1].replace('.nd2','')
        out_file_name_DB = out_dir_name_DB+ "/frame{0}_ROI{1}_{2}_mask.json".format(cellroi[0].frame.number, cellroi[0].roi_number, 'SAM2_b+')

        out_file = open(out_file_name, "w") 
        json.dump(segment_dict, out_file) 
        out_file.close() 
        contour.file_name = out_file_name_DB
        contour.save()




    #___________________________________________________________________________________________
    def add_segmentation_point_callback(event):
        if isinstance(event, bokeh.events.SelectionGeometry):

            print('add_segmentation_point_callback event ',event)
            print('add_segmentation_point_callback event.geometry ',event.geometry)
            if event.geometry["type"]!="point":
                return
            x, y = event.geometry['x'], event.geometry['y']
            new_point = dict(x=source_segmentation_points.data['x'] + [x], y=source_segmentation_points.data['y'] + [y])
            source_segmentation_points.data = new_point
            run_segmentation_point(source_segmentation_points.data['x'],source_segmentation_points.data['y'])
    plot_image.on_event(bokeh.events.SelectionGeometry, add_segmentation_point_callback)


    #___________________________________________________________________________________________
    def remove_last_segmentation_point_callback(event):
        print('remove_last_segmentation_point_callback event ',event)
        if source_segmentation_points.data['x']:
            new_data = dict(x=source_segmentation_points.data['x'][:-1], y=source_segmentation_points.data['y'][:-1])
            source_segmentation_points.data = new_data
            run_segmentation_point(source_segmentation_points.data['x'],source_segmentation_points.data['y'])
    plot_image.on_event(bokeh.events.DoubleTap, remove_last_segmentation_point_callback)


    #___________________________________________________________________________________________
    def update_source_segment(tp=0):
        current_file = get_current_file(index=0)
        current_pos  = os.path.split(current_file)[1]
        if DEBUG:print('****************************  update_source_segment ****************************')
        try:
            #mask = get_current_stack()['masks'][dropdown_cell.value][dropdown_segmentation_type.value][tp]
            source_img_mask.data = {'img':[image_stack_dict[current_pos]['masks'][dropdown_cell.value][dropdown_segmentation_type.value][tp]]}
        except KeyError:
            source_img_mask.data = {'img':[]}
        return
    #___________________________________________________________________________________________

    #___________________________________________________________________________________________
    def callback_slider(attr: str, old: Any, new: Any) -> None:
        if DEBUG:print('****************************  callback_slider ****************************')
        time_point = slider.value
        source_img.data = {'img':[source_imgs_norm.data['images'][int(dropdown_channel.value)][time_point]]}

        images=source_imgs.data['images']
        source_img_ch.data = {'img':[images[ch][time_point] for ch in range(len(images))]}


        current_file = get_current_file(index=0)
        current_pos  = os.path.split(current_file)[1]

        try:
            source_img_mask.data = {'img':[image_stack_dict[current_pos]['masks'][dropdown_cell.value][dropdown_segmentation_type.value][time_point]]}
        except KeyError:
            source_img_mask.data = {'img':[]}

        source_segmentation_points.data = dict(x=[], y=[])


        source_roi.data    = {'left'  :source_rois_full.data['left'][time_point], 
                              'right' :source_rois_full.data['right'][time_point], 
                              'top'   :source_rois_full.data['top'][time_point], 
                              'bottom':source_rois_full.data['bottom'][time_point]}
        
        source_labels.data = {'height':source_labels_full.data['height'][time_point],
                              'weight':source_labels_full.data['weight'][time_point], 
                              'names' :source_labels_full.data['names'][time_point]}
        
        source_cells.data  = {'height':source_cells_full.data['height'][time_point], 
                              'weight':source_cells_full.data['weight'][time_point], 
                              'names' :source_cells_full.data['names'][time_point]}
        
        if len(source_intensity_ch1.data["time"])==0:
            line_position.location = -999
        else:
            line_position.location = source_intensity_ch1.data["time"][time_point]
    #___________________________________________________________________________________________
    

    callback_slider_test = bokeh.models.CustomJS(args=dict(source_img=source_img, 
                                                           #line_position=line_position, 
                                                            source_imgs_norm=source_imgs_norm, 
                                                           #dropdown_channel=dropdown_channel, source_intensity_ch1=source_intensity_ch1,
    #                                                       image_stack_rois_dict=image_stack_rois_dict, dropdown_pos=dropdown_pos,
    #                                                       image_stack_cells_dict=image_stack_cells_dict, image_stack_labels_dict=image_stack_labels_dict,
                                                           #source_roi=source_roi, source_rois_full=source_rois_full,
                                                           #source_labels=source_labels, source_cells=source_cells)
    ), code="""
        var index     = cb_obj.value;
        //var channel   = parseInt(dropdown_channel.value);
        var new_image = source_imgs_norm.data['images'][channel][index];
        source_img.data['img'][0] = new_image;

        //if (source_intensity_ch1.data['time'].length === 0) {
        //    line_position.location = -999;
        //} else {
        //    line_position.location = source_intensity_ch1.data['time'][index];
        //}

        //console.log('source_rois_full.data',source_rois_full.data)
        //source_roi.data['left']   = source_rois_full.data['left'][index];
        //source_roi.data['right']  = source_rois_full.data['right'][index];
        //source_roi.data['top']    = source_rois_full.data['top'][index];
        //source_roi.data['bottom'] = source_rois_full.data['bottom'][index];

        //source_labels.data['height'] = image_stack_labels_dict[current_file][String(index)]['height']
        //source_labels.data['weight'] = image_stack_labels_dict[current_file][String(index)]['weight']
        //source_labels.data['names']  = image_stack_labels_dict[current_file][String(index)]['names']
        
        //source_cells.data['height'] = image_stack_cells_dict[current_file][String(index)]['height']
        //source_cells.data['weight'] = image_stack_cells_dict[current_file][String(index)]['weight']
        //source_cells.data['names']  = image_stack_cells_dict[current_file][String(index)]['names']

        source_img.change.emit();
        //source_roi.change.emit();
        //source_labels.change.emit();
        //source_cells.change.emit();

        """)
    #slider.js_on_change('value', callback_slider_test)
    #dropdown_pos.js_on_change('value', callback_slider_test)
    
    #PYTHON
    slider.on_change('value', callback_slider)




    #___________________________________________________________________________________________
    # Define a callback function to update the contrast when the slider changes
    def update_contrast(attr, old, new):
        low, high = new 
        color_mapper.low = low
        color_mapper.high = high

    # Create a slider to adjust contrast
    contrast_slider = bokeh.models.RangeSlider(start=0, end=255, value=(0, 255), step=1, title="Contrast", width=150)
    contrast_slider.on_change('value', update_contrast)
    #___________________________________________________________________________________________


    #___________________________________________________________________________________________
    # Define a callback to update the ROI
    def select_roi_callback(event):
        if DEBUG:print('****************************  select_roi_callback ****************************')
        print('select_roi_callback event ',event)
        print('select_roi_callback event ',event.geometry)
        if isinstance(event, bokeh.events.SelectionGeometry):
            if event.geometry["type"]!='rect':return
            data_manual = dict(
                left=source_roi_manual.data['left'] + [event.geometry['x0']],
                right=source_roi_manual.data['right'] + [event.geometry['x1']],
                top=source_roi_manual.data['top'] + [event.geometry['y0']],
                bottom=source_roi_manual.data['bottom'] + [event.geometry['y1']]
                )
            source_roi_manual.data = data_manual

            data = dict(
                left=source_roi.data['left'] + [event.geometry['x0']],
                right=source_roi.data['right'] + [event.geometry['x1']],
                top=source_roi.data['top'] + [event.geometry['y0']],
                bottom=source_roi.data['bottom'] + [event.geometry['y1']]
                )
            source_roi.data = data

            save_roi_callback()
            build_cells_callback()
            prepare_intensity()

            #inspect_cells_callback()
            #print('select_roi_callback x0=left, x1=right, y0=top, y1=bottom',event.geometry['x0'], event.geometry['x1'],nrows-event.geometry['y0'],nrows-event.geometry['y1'])
            print('select_roi_callback x0=left, x1=right, y0=top, y1=bottom',event.geometry['x0'], event.geometry['x1'],event.geometry['y0'],event.geometry['y1'])
    plot_image.on_event(bokeh.events.SelectionGeometry, select_roi_callback)
    #___________________________________________________________________________________________

    #___________________________________________________________________________________________
    # Define a callback to delete the ROI
    def delete_roi_callback():
        if DEBUG:print('****************************  delete_roi_callback ****************************')
        source_roi.data = {'left': [], 'right': [], 'top': [], 'bottom': []}
        source_labels.data = {'height':[], 'weight':[], 'names':[]}
        source_cells.data = {'height':[], 'weight':[], 'names':[]}

        current_file=get_current_file()
        current_index=get_current_index()

        sample = Sample.objects.get(file_name=current_file)
        frame  = Frame.objects.select_related().filter(sample=sample, number=current_index)
        if len(frame)!=1:
            print('NOT ONLY FRAME FOUND< PLEASE CHECKKKKKKKK')
            print('======sample: ',sample)
            for f in frame:
                print('===============frame: ',f)
        rois   = CellROI.objects.select_related().filter(frame=frame[0])


        for roi in rois:
            if os.path.isfile(roi.contour_cellroi.file_name):
                os.remove(roi.contour_cellroi.file_name)
            else:
                print("Error: {} file not found".format(roi.contour_cellroi.file_name) )
            roi.delete()
        update_dropdown_cell('','','')
        #prepare_intensity()

    button_delete_roi = bokeh.models.Button(label="Delete ROI")
    button_delete_roi.on_click(delete_roi_callback)
    #___________________________________________________________________________________________

    #___________________________________________________________________________________________
    # Save ROI
    def save_roi_callback():
        if DEBUG:
            print('****************************  save_roi_callback ****************************')
            print('Saving ROI===================================',source_roi_manual.data)
        current_file  = get_current_file()
        current_index = get_current_index()

        sample   = Sample.objects.get(file_name=current_file)
        expds    = ExperimentalDataset.objects.get(id=sample.experimental_dataset.id)
        exp      = Experiment.objects.get(id=expds.experiment.id)
        frame    = Frame.objects.select_related().filter(sample=sample, number=current_index)
        if len(frame)!=1:
            print('NOT ONLY FRAME FOUND< PLEASE CHECKKKKKKKK')
            print('======sample: ',sample)
            for f in frame:
                print('===============frame: ',f)
        for i in range(len(source_roi_manual.data['left'])):
            cellrois = CellROI.objects.select_related().filter(frame=frame[0])
            roi_exist=False
            #TOBECHANGED CLEMENT really need to save more than1 roi per frame
            roi_number = len(cellrois)
            #roi_number=i+len(source_roi.data['left'])-len(source_roi_manual.data['left'])
            if DEBUG:
                print('roi_number=',roi_number)
                print('source_roi=',len(source_roi.data['left']))
                print('source_roi_manual=',len(source_roi_manual.data['left']))
                print('i=',i)
            for cellroi in cellrois:
                if cellroi.min_col == math.floor(source_roi_manual.data['left'][i]) and \
                    cellroi.min_row == math.floor(frame[0].height-source_roi_manual.data['bottom'][i])  and \
                        cellroi.max_col == math.ceil(source_roi_manual.data['right'][i]) and \
                            cellroi.max_row == math.ceil(frame[0].height-source_roi_manual.data['top'][i]):
                        if DEBUG:print('save_roi_callback already exist ',frame[0])
                        roi_exist=True
            if not roi_exist:
                if DEBUG:print('save_roi_callback saving frame=',frame[0], ' roi_number=',roi_number)
                roi = CellROI(min_col=math.floor(source_roi_manual.data['left'][i]), max_col=math.ceil(source_roi_manual.data['right'][i]), 
                              min_row=math.floor(frame[0].height-source_roi_manual.data['bottom'][i]),  max_row=math.ceil(frame[0].height-source_roi_manual.data['top'][i]),
                              roi_number=roi_number, 
                              frame=frame[0])
                roi.save()

                images=source_img_ch.data['img']
                images=np.array(images)
                for i in range(len(images)):
                    images[i]=np.flip(images[i],0)
                if DEBUG:print('save_roi_callback images shape ', images.shape)
                cropped_dict = {'shape_original':images[0].shape}

                out_dir_name  = os.path.join(NASRCP_MOUNT_POINT, ANALYSIS_DATA_PATH,exp.name, expds.data_name, os.path.split(sample.file_name)[-1].replace('.nd2',''))
                out_file_name = os.path.join(out_dir_name, "frame{0}_ROI{1}.json".format(frame[0].number, roi_number))

                out_dir_name_DB  = ANALYSIS_DATA_PATH+"/"+exp.name+"/"+ expds.data_name+"/"+os.path.split(sample.file_name)[-1].replace('.nd2','')
                out_file_name_DB = out_dir_name_DB+ "/frame{0}_ROI{1}.json".format(frame[0].number, roi_number)


                if not os.path.exists(out_dir_name):
                    os.makedirs(out_dir_name)
                if DEBUG:print('roi.min_row, roi.max_row, roi.min_col,roi.max_col',roi.min_row, roi.max_row, roi.min_col,roi.max_col)
                cropped_img = images[:, roi.min_row:roi.max_row, roi.min_col:roi.max_col]
                
                cropped_dict['shape']=[cropped_img.shape[1],cropped_img.shape[2]]
                cropped_dict['npixels']=cropped_img.shape[1]*cropped_img.shape[2]
                cropped_dict['shift']=[roi.min_row, roi.min_col]
                cropped_dict['type']="manual"

                channels=exp.name_of_channels.split(',')
                for ch in range(len(channels)):
                    cropped_dict['intensity_{}'.format(channels[ch])] = cropped_img[ch].tolist()
                            
                if DEBUG:print('out_file_name=',out_file_name)
                out_file = open(out_file_name, "w") 
                json.dump(cropped_dict, out_file) 
                out_file.close() 

                intensity_mean={}
                intensity_std={}
                intensity_sum={}
                intensity_max={}
                for ch in range(len(channels)): 
                    sum=float(np.sum(cropped_img[ch]))
                    mean=float(np.mean(cropped_img[ch]))
                    std=float(np.std(cropped_img[ch]))
                    max=float(np.max(cropped_img[ch]))
                    ch_name=channels[ch]
                    intensity_mean[ch_name]=mean
                    intensity_std[ch_name]=std
                    intensity_sum[ch_name]=sum
                    intensity_max[ch_name]=max
                

                contour = Contour(center_x_pix=roi.min_col+(roi.max_col-roi.min_col)/2., 
                                  center_y_pix=roi.min_row+(roi.max_row-roi.min_row)/2.,
                                  center_z_pix=0, 
                                  center_x_mic=(roi.min_col+(roi.max_col-roi.min_col)/2.)*frame[0].pixel_microns+frame[0].pos_x,
                                  center_y_mic=(roi.min_row+(roi.max_row-roi.min_row)/2.)*frame[0].pixel_microns+frame[0].pos_y,
                                  center_z_mic=0,
                                  intensity_mean=intensity_mean,
                                  intensity_std=intensity_std,
                                  intensity_sum=intensity_sum,
                                  intensity_max=intensity_max,
                                  number_of_pixels=cropped_img.shape[1]*cropped_img.shape[2],
                                  file_name=out_file_name_DB,
                                  cell_roi=roi,
                                  type="cell_ROI",
                                  mode="manual")
                contour.save()

                cellflag = CellFlag(cell_roi=roi)
                cellflag.save()

        left_rois, right_rois, top_rois, bottom_rois,height_labels, weight_labels, names_labels, height_cells, weight_cells, names_cells= update_source_roi_cell_labels()
        source_labels.data = {'height':height_labels, 'weight':weight_labels, 'names':names_labels}
        source_cells.data = {'height':height_cells, 'weight':weight_cells, 'names':names_cells}


        source_rois_full.data['left'][current_index]   = source_roi.data['left']
        source_rois_full.data['right'][current_index]  = source_roi.data['right']
        source_rois_full.data['top'][current_index]    = source_roi.data['top']
        source_rois_full.data['bottom'][current_index] = source_roi.data['bottom']

        source_labels_full.data['weight'][current_index] = source_labels.data['weight']
        source_labels_full.data['height'][current_index] = source_labels.data['height']
        source_labels_full.data['names'][current_index]  = source_labels.data['names']

        source_cells_full.data['weight'][current_index] = source_cells.data['weight']
        source_cells_full.data['height'][current_index] = source_cells.data['height']
        source_cells_full.data['names'][current_index]  = source_cells.data['names']

        source_roi_manual.data['left']=[]
        source_roi_manual.data['right']=[]
        source_roi_manual.data['top']=[]
        source_roi_manual.data['bottom']=[]

    button_save_roi = bokeh.models.Button(label="Save ROI")
    button_save_roi.on_click(save_roi_callback)
    #___________________________________________________________________________________________

    #___________________________________________________________________________________________
    # Function to update the image displayed
    def update_image(way=1, number=-9999):
        if DEBUG:print('****************************  update_image ****************************')
        current_index=get_current_index()
        current_index = (current_index + 1*way) % len(source_imgs.data["images"][0])
        if current_index>slider.end:current_index=slider.start
        if current_index<slider.start:current_index=slider.end

        if number>=0:
            current_index = number
        slider.value = current_index
        if len(source_intensity_ch1.data["time"])==0:
            line_position.location = -999
        else:
            line_position.location = source_intensity_ch1.data["time"][current_index]
        if DEBUG:print('update_image index=',current_index)

    #___________________________________________________________________________________________


    #___________________________________________________________________________________________
    class PlaybackState:
        def __init__(self):
            self.playing = False
    play_state = PlaybackState()
    #___________________________________________________________________________________________

    #___________________________________________________________________________________________
    class PlaybackID:
        def __init__(self) -> None:
            self.state = None
    play_state_id = PlaybackID()
    #___________________________________________________________________________________________

    #___________________________________________________________________________________________
    # Create play/stop button
    def play_stop_callback():
        if DEBUG:print('****************************  play_stop_callback ****************************')
        refresh_time = int(dropdown_refresh_time.value)
        if  not play_state.playing:
            button_play_stop.label = "Stop"
            timerr = doc.add_periodic_callback(update_image, refresh_time)  # Change the interval as needed
            play_state_id.state = timerr
            play_state.playing = True
        else:
            button_play_stop.label = "Play"
            doc.remove_periodic_callback(play_state_id.state)
            play_state.playing = False
    button_play_stop = bokeh.models.Button(label="Play")
    button_play_stop.on_click(play_stop_callback)
    #___________________________________________________________________________________________

    position_check_div = bokeh.models.Div(text="<b style='color:red; ; font-size:18px;'> Peaks/ToD/Division not validated</b>")
    #___________________________________________________________________________________________
    def position_check_callback():
        current_file=get_current_file()
        sample   = Sample.objects.get(file_name=current_file)
        if sample.peaks_tod_div_validated == False:
            sample.peaks_tod_div_validated = True
            position_check_button.label = "Peaks/ToD/Division not validated"
            position_check_div.text = "<b style='color:green; ; font-size:18px;'> Peaks/ToD/Division validated</b>"
        else:
            sample.peaks_tod_div_validated = False
            position_check_button.label = "Peaks/ToD/Division validated"
            position_check_div.text = "<b style='color:red; ; font-size:18px;'> Peaks/ToD/Division not validated</b>"
        sample.save()
        update_position_select(change=False)
    position_check_button = bokeh.models.Button(label="Peaks/ToD/Division validated")
    position_check_button.on_click(position_check_callback)
    #___________________________________________________________________________________________


    position_check2_div = bokeh.models.Div(text="<b style='color:red; ; font-size:18px;'> BF features not validated</b>")
    #___________________________________________________________________________________________
    def position_check2_callback():
        current_file=get_current_file()
        sample   = Sample.objects.get(file_name=current_file)
        if sample.bf_features_validated == False:
            sample.bf_features_validated = True
            position_check2_button.label = "BF features not validated"
            position_check2_div.text = "<b style='color:green; ; font-size:18px;'> BF features validated</b>"
        else:
            sample.bf_features_validated = False
            position_check2_button.label = "BF features validated"
            position_check2_div.text = "<b style='color:red; ; font-size:18px;'> BF features not validated</b>"
        sample.save()
        update_position_select(change=False)
    position_check2_button = bokeh.models.Button(label="BF features validated")
    position_check2_button.on_click(position_check2_callback)
    #___________________________________________________________________________________________


    position_keep_div = bokeh.models.Div(text="<b style='color:green; ; font-size:18px;'> Keep Position</b>")
    #___________________________________________________________________________________________
    def position_keep_callback():
        current_file=get_current_file()
        sample   = Sample.objects.get(file_name=current_file)
        if sample.keep_sample == False:
            sample.keep_sample = True
            position_keep_button.label = "Don't keep Position"
            position_keep_div.text = "<b style='color:green; ; font-size:18px;'> Keep Position</b>"
        else:
            sample.keep_sample = False
            position_keep_button.label = "Keep Position"
            position_keep_div.text = "<b style='color:red; ; font-size:18px;'> Don't keep Position</b>"

        sample.save()
        update_position_select(change=False)

    position_keep_button = bokeh.models.Button(label="Don't keep Position")
    position_keep_button.on_click(position_keep_callback)
    #___________________________________________________________________________________________


    #___________________________________________________________________________________________
    def remove_roi_callback():
        if DEBUG:print('****************************  remove_roi_callback ****************************')
        removeROIs(get_current_file())

        left_rois,right_rois,top_rois,bottom_rois,height_labels, weight_labels, names_labels,height_cells, weight_cells, names_cells=update_source_roi_cell_labels()
        
        source_roi.data = {'left': left_rois, 'right': right_rois, 'top': top_rois, 'bottom': bottom_rois}
        source_labels.data = {'height':height_labels, 'weight':weight_labels, 'names':names_labels}
        source_cells.data = {'height':height_cells, 'weight':weight_cells, 'names':names_cells}
    button_remove_roi = bokeh.models.Button(label="Remove ROI")
    button_remove_roi.on_click(remove_roi_callback)
    #___________________________________________________________________________________________


    #___________________________________________________________________________________________
    def build_roi_callback():
        if DEBUG:print('****************************  build_roi_callback ****************************')
        build_ROIs(get_current_file(), force=True)
        left_rois,right_rois,top_rois,bottom_rois,height_labels, weight_labels, names_labels,height_cells, weight_cells, names_cells=update_source_roi_cell_labels()
        
        source_roi.data    = {'left': left_rois, 'right': right_rois, 'top': top_rois, 'bottom': bottom_rois}
        source_labels.data = {'height':height_labels, 'weight':weight_labels, 'names':names_labels}
        source_cells.data  = {'height':height_cells, 'weight':weight_cells, 'names':names_cells}
        update_dropdown_cell('','','')
        prepare_intensity()

        s = Sample.objects.get(file_name = get_current_file())
        rois_data = get_stack_rois_data(s.file_name)

        source_rois_full.data['left']   = rois_data['rois']['left']
        source_rois_full.data['right']  = rois_data['rois']['right']
        source_rois_full.data['top']    = rois_data['rois']['top']
        source_rois_full.data['bottom'] = rois_data['rois']['bottom']

        source_labels_full.data['weight'] = rois_data['labels']['weight']
        source_labels_full.data['height'] = rois_data['labels']['height']
        source_labels_full.data['names']  = rois_data['labels']['names']

        source_cells_full.data['weight'] = rois_data['cells']['weight']
        source_cells_full.data['height'] = rois_data['cells']['height']
        source_cells_full.data['names']  = rois_data['cells']['names']


    button_build_roi = bokeh.models.Button(label="Build ROI")
    button_build_roi.on_click(build_roi_callback)
    #___________________________________________________________________________________________

    #___________________________________________________________________________________________
    def build_sam2_callback():
        if DEBUG:print('****************************  build_sam2_callback ****************************')
        build_segmentation_sam2(get_current_file(), force=True)
        if 'SAM2_b+' not in dropdown_segmentation_type.options:
            toto=dropdown_segmentation_type.options
            toto.append('SAM2_b+')
            dropdown_segmentation_type.options = toto
        current_file = get_current_file()
        masks_data=get_masks_data(current_file, cellname=dropdown_cell.value)

        current_pos  = os.path.split(current_file)[1]
        image_stack_dict[current_pos]['masks'][dropdown_cell.value]['SAM2_b+']=masks_data[dropdown_cell.value]['SAM2_b+']
    button_build_sam2 = bokeh.models.Button(label="Build SAM2")
    button_build_sam2.on_click(build_sam2_callback)
    #___________________________________________________________________________________________


    #___________________________________________________________________________________________
    # Create next button
    def next_callback():
        if DEBUG:print('****************************  next_callback ****************************')
        update_image()
    button_next = bokeh.models.Button(label="Next")
    button_next.on_click(next_callback)
    #___________________________________________________________________________________________

    #___________________________________________________________________________________________
    # Create next button
    def prev_callback():
        if DEBUG:print('****************************  prev_callback ****************************')
        update_image(-1)
    button_prev = bokeh.models.Button(label="Prev")
    button_prev.on_click(prev_callback)
    #___________________________________________________________________________________________

    #___________________________________________________________________________________________
    # Go to next frame with possible issue
    def inspect_cells_callback():
        if DEBUG:print('****************************  inspect_cells_callback ****************************')
        current_file=get_current_file()

        sample   = Sample.objects.get(file_name=current_file)
        frames   = Frame.objects.select_related().filter(sample=sample)
        for f in range(len(frames)-1):
            ncells_t1=0
            cellrois_t1 = CellROI.objects.select_related().filter(frame=frames[f])
            for cellroi in cellrois_t1:
                if cellroi.cell_id != None: ncells_t1+=1
            ncells_t2=0
            cellrois_t2 = CellROI.objects.select_related().filter(frame=frames[f+1])
            for cellroi in cellrois_t2:
                if cellroi.cell_id != None: ncells_t2+=1
            if DEBUG: print('inspect_cells_callback: frame ',f, '  ncells_t1 ',ncells_t1,'  ncells_t2 ',ncells_t2)
            if ncells_t1 == 0: 
                update_image(number=f)
                break
            if ncells_t2 == 0: 
                update_image(number=f+1)
                break
            if ncells_t2>ncells_t1:
                update_image(number=f)
                break
            if ncells_t2<ncells_t1:
                update_image(number=f)
                break
    button_inspect = bokeh.models.Button(label="Inspect")
    button_inspect.on_click(inspect_cells_callback)
    #___________________________________________________________________________________________

    #___________________________________________________________________________________________
    def build_cells_callback():
        if DEBUG:print('****************************  build_cells_callback ****************************')
        current_file=get_current_file()
        build_cells_sample(sample=current_file, addmode=True)
        left_rois, right_rois, top_rois, bottom_rois,height_labels, weight_labels, names_labels, height_cells, weight_cells, names_cells= update_source_roi_cell_labels()
        source_cells.data = {'height':height_cells, 'weight':weight_cells, 'names':names_cells}

        source_cells_full.data['weight'][slider.value] = source_cells.data['weight']
        source_cells_full.data['height'][slider.value] = source_cells.data['height']
        source_cells_full.data['names'][slider.value]  = source_cells.data['names']

        update_dropdown_cell('','','')

    button_build_cells = bokeh.models.Button(label="build cells")
    button_build_cells.on_click(build_cells_callback)
    #___________________________________________________________________________________________

    #___________________________________________________________________________________________
    # Set the start of oscillation
    def start_oscillation_callback():
        current_index=get_current_index()
        start_oscillation_position.location = source_intensity_ch1.data["time"][current_index]
        sample = Sample.objects.get(file_name=get_current_file())
        cellsid = CellID.objects.select_related().filter(sample=sample, name=dropdown_cell.value)
        cellstatus = cellsid[0].cell_status
        cellstatus.start_oscillation_frame = current_index
        frame = Frame.objects.select_related().filter(sample=sample, number=current_index)
        cellstatus.start_oscillation = frame[0].time/60000.
        cellstatus.save()
        if start_oscillation_position.location>=0 and end_oscillation_position.location>0:
            find_peaks_slider_callback('','',slider_find_peaks.value)
            cellrois = CellROI.objects.select_related().filter(cell_id=cellsid[0])
            for cellroi in cellrois:
                framenumber = cellroi.frame.number
                cellflag = cellroi.cellflag_cellroi
            if cellstatus.start_oscillation_frame>=0 and framenumber>=cellstatus.start_oscillation_frame and cellstatus.end_oscillation_frame>=1 and frame<=cellstatus.end_oscillation_frame:
                cellflag.oscillating = True
            else:
                cellflag.oscillating = False
            cellflag.save()

    button_start_oscillation = bokeh.models.Button(label="Osc. Start")
    button_start_oscillation.on_click(start_oscillation_callback)
    #___________________________________________________________________________________________

    #___________________________________________________________________________________________
    # Set the end of oscillation
    def end_oscillation_callback():
        current_index=get_current_index()
        end_oscillation_position.location = source_intensity_ch1.data["time"][current_index]
        sample = Sample.objects.get(file_name=get_current_file())
        cellsid = CellID.objects.select_related().filter(sample=sample, name=dropdown_cell.value)
        cellstatus = cellsid[0].cell_status
        cellstatus.end_oscillation_frame = current_index
        frame = Frame.objects.select_related().filter(sample=sample, number=current_index)
        cellstatus.end_oscillation = frame[0].time/60000.
        cellstatus.save()
        if start_oscillation_position.location>=0 and end_oscillation_position.location>0:
            find_peaks_slider_callback('','',slider_find_peaks.value)
            cellrois = CellROI.objects.select_related().filter(cell_id=cellsid[0])
            for cellroi in cellrois:
                framenumber = cellroi.frame.number
                cellflag = cellroi.cellflag_cellroi
            if cellstatus.start_oscillation_frame>=0 and framenumber>=cellstatus.start_oscillation_frame and cellstatus.end_oscillation_frame>=1 and framenumber<=cellstatus.end_oscillation_frame:
                cellflag.oscillating = True
            else:
                cellflag.oscillating = False
            cellflag.save()

    button_end_oscillation = bokeh.models.Button(label="Osc. End")
    button_end_oscillation.on_click(end_oscillation_callback)
    #___________________________________________________________________________________________

    #___________________________________________________________________________________________
    # Set cell time of death
    def time_of_death_callback():
        sample = Sample.objects.get(file_name=get_current_file())
        cellsid = CellID.objects.select_related().filter(sample=sample, name=dropdown_cell.value)
        cellstatus = cellsid[0].cell_status
        if len(source_intensity_ch1.selected.indices)==0:
            cellstatus.time_of_death = -9999
            time_of_death_position.location = -9999
            source_varea_death.data['x']    = []
            source_varea_death.data['y1']   = []
            source_varea_death.data['y2']   = []
            cellstatus.time_of_death_frame = -999

        else:
            current_index=source_intensity_ch1.selected.indices[0]
            time_of_death_position.location = source_intensity_ch1.data["time"][current_index]
            source_varea_death.data['x']  = [source_intensity_ch1.data["time"][t] for t in range(current_index, len(source_intensity_ch1.data["time"])) ]
            source_varea_death.data['y1'] = [source_intensity_ch1.data["intensity"][t] for t in range(current_index, len(source_intensity_ch1.data["intensity"])) ]
            source_varea_death.data['y2'] = [0 for i in range(len(source_varea_death.data['y1']))]

            cellstatus.time_of_death_frame = current_index
            frame = Frame.objects.select_related().filter(sample=sample, number=current_index)
            cellstatus.time_of_death = frame[0].time/60000.
        cellstatus.save()

        cellrois = CellROI.objects.select_related().filter(cell_id=cellsid[0])

        for cellroi in cellrois:
            framenumber = cellroi.frame.number
            cellflag = cellroi.cellflag_cellroi
            if framenumber>=cellstatus.time_of_death_frame and cellstatus.time_of_death_frame>=0:
                cellflag.alive = False
            else:
                cellflag.alive = True
            cellflag.save()
        update_source_osc_tod()
        if DEBUG: print('time_of_death_callback  - - - - - - - source_varea_death.data',  source_varea_death.data)

    button_time_of_death = bokeh.models.Button(label="Dead")
    button_time_of_death.on_click(time_of_death_callback)
    #___________________________________________________________________________________________

    #___________________________________________________________________________________________
    def find_peaks_slider_callback(attr, old, new):
        if DEBUG:print('=======================find_peaks_slider_callback=======================================')
        int_array = np.array(source_intensity_ch1.data["intensity"])
        for int in range(len(int_array)):
            if source_intensity_ch1.data["time"][int]<start_oscillation_position.location:
                int_array[int]=0
            if source_intensity_ch1.data["time"][int]>end_oscillation_position.location:
                int_array[int]=0

        peaksmax, _ = find_peaks(np.array(int_array),  prominence=new)
        peaksmin, _ = find_peaks(-np.array(int_array), prominence=new)

        int_max=[]
        time_max=[]
        for p in peaksmax:
            time_max.append(source_intensity_ch1.data["time"][p])
            int_max.append(source_intensity_ch1.data["intensity"][p])
        source_intensity_max.data={'time':time_max, 'intensity':int_max}  

        int_min=[]
        time_min=[]
        for p in peaksmin:
            time_min.append(source_intensity_ch1.data["time"][p])
            int_min.append(source_intensity_ch1.data["intensity"][p])
        source_intensity_min.data={'time':time_min, 'intensity':int_min}
        if DEBUG:print('peaksmax=',peaksmax,'  peaksmin=',peaksmin)
        set_rising_falling_local(peaksmax, peaksmin)
    slider_find_peaks.on_change('value', find_peaks_slider_callback)
    #___________________________________________________________________________________________

    #___________________________________________________________________________________________
    def save_peaks_callback():
        if DEBUG:print('------------------------------------save_peaks_callback-------------------------------')
        int_array = np.array(source_intensity_ch1.data["intensity"])
        for int in range(len(int_array)):
            if source_intensity_ch1.data["time"][int]<start_oscillation_position.location:
                int_array[int]=0
            if source_intensity_ch1.data["time"][int]>end_oscillation_position.location:
                int_array[int]=0
        peaksmax, _ = find_peaks(np.array(int_array),  prominence=slider_find_peaks.value)
        peaksmin, _ = find_peaks(-np.array(int_array), prominence=slider_find_peaks.value)

        int_max=[]
        time_max=[]
        for p in peaksmax:
            time_max.append(source_intensity_ch1.data["time"][p])
            int_max.append(source_intensity_ch1.data["intensity"][p])
        source_intensity_max.data={'time':time_max, 'intensity':int_max}  

        int_min=[]
        time_min=[]
        for p in peaksmin:
            time_min.append(source_intensity_ch1.data["time"][p])
            int_min.append(source_intensity_ch1.data["intensity"][p])
        source_intensity_min.data={'time':time_min, 'intensity':int_min}

        sample = Sample.objects.get(file_name=get_current_file())
        cellsid = CellID.objects.select_related().filter(sample=sample, name=dropdown_cell.value)
        if DEBUG:print('cellsid ',cellsid, '  ',len(cellsid))
        if len(cellsid)==1:
            cellstatus = cellsid[0].cell_status
            cellstatus.peaks={'min_frame':peaksmin.tolist(), 'max_frame':peaksmax.tolist(), 
                              'min_int':int_min, 'max_int':int_max,
                              'min_time':time_min, 'max_time':time_max}
            cellstatus.n_oscillations = len(peaksmax.tolist())

            cellrois = CellROI.objects.select_related().filter(cell_id=cellsid[0])
            for cellroi in cellrois:
                cellflag = cellroi.cellflag_cellroi

                print('cellflag.last_osc=', cellflag.last_osc)
                print('cellflag.mask    =', cellflag.mask)
                print('cellflag.dividing       =', cellflag.dividing)
                print('cellflag.double_nuclei  =', cellflag.double_nuclei)
                print('cellflag.multiple_cells =', cellflag.multiple_cells)
                print('cellflag.pair_cell      =', cellflag.pair_cell)
                print('cellflag.flat           =', cellflag.flat)
                print('cellflag.round          =', cellflag.round)
                print('cellflag.elongated      =', cellflag.elongated )

            cellstatus.save()


            for cellroi in cellrois:
                cellflag = cellroi.cellflag_cellroi
                print('===========================')
                print('cellflag.last_osc=', cellflag.last_osc)
                print('cellflag.mask    =', cellflag.mask)
                print('cellflag.dividing       =', cellflag.dividing)
                print('cellflag.double_nuclei  =', cellflag.double_nuclei)
                print('cellflag.multiple_cells =', cellflag.multiple_cells)
                print('cellflag.pair_cell      =', cellflag.pair_cell)
                print('cellflag.flat           =', cellflag.flat)
                print('cellflag.round          =', cellflag.round)
                print('cellflag.elongated      =', cellflag.elongated )

            set_rising_falling(cellsid[0], save_status=True)
            update_source_osc_tod()
        if len(cellsid)>1:
            print('more than 1 cellsid=',len(cellsid))

    button_save_peaks = bokeh.models.Button(label="Save Peaks")
    button_save_peaks.on_click(save_peaks_callback)
    #___________________________________________________________________________________________

    #___________________________________________________________________________________________
    def delete_peaks_callback():
        if DEBUG:print('------------------------------------delete_peaks_callback-------------------------------')
        sample = Sample.objects.get(file_name=get_current_file())
        cellsid = CellID.objects.select_related().filter(sample=sample, name=dropdown_cell.value)
        if DEBUG:print('cellsid ',cellsid, '  ',len(cellsid))
        if len(cellsid)==1:
            cellstatus = cellsid[0].cell_status
            cellstatus.peaks={}
            cellstatus.flags = {}
            cellstatus.n_oscillations = -999

            cellstatus.time_of_death_frame     = -999
            cellstatus.start_oscillation_frame = -999
            cellstatus.end_oscillation_frame   = -999

            cellstatus.time_of_death     = -9999
            cellstatus.start_oscillation = -9999
            cellstatus.end_oscillation   = -9999

            cellstatus.save()
            set_rising_falling(cellsid[0], delete_status=True)
            update_source_osc_tod()
            source_intensity_max.data={'time':[], 'intensity':[]}
            source_intensity_min.data={'time':[], 'intensity':[]}
            start_oscillation_position.location = -9999
            end_oscillation_position.location = -9999
            time_of_death_position.location = -9999
            source_varea_death.data['x']    = []
            source_varea_death.data['y1']   = []
            source_varea_death.data['y2']   = []
        if DEBUG: print('delete_peaks_callback  - - - - - - - source_varea_death.data',  source_varea_death.data)

    button_delete_peaks = bokeh.models.Button(label="Delete Peaks")
    button_delete_peaks.on_click(delete_peaks_callback)
    #___________________________________________________________________________________________



    #___________________________________________________________________________________________
    def update_source_osc_tod():#checkbox_status):#tod_checkbox_all=True, tod_checkbox_keep=False, tod_checkbox_dkeep=False):
        if DEBUG:print('------------------------update_source_osc_tod-------------------------')
        well = ExperimentalDataset.objects.get(data_name=dropdown_well.value)
        nframes = well.experiment.number_of_frames
        samples = Sample.objects.select_related().filter(experimental_dataset = well)
        n_osc=[]
        n_osc_dk=[]
        n_osc_all=[]
        tod=[]
        tod_dk=[]
        tod_all=[]
        tod_pred=[]
        start_osc=[]
        end_osc=[]
        for sample in samples:
            cellids = CellID.objects.select_related().filter(sample=sample)
            for cellid in cellids:
                if cellid.sample.keep_sample: 
                    n_osc.append(cellid.cell_status.n_oscillations)
                    n_osc_all.append(cellid.cell_status.n_oscillations)
                else: 
                    n_osc_dk.append(cellid.cell_status.n_oscillations)
                    n_osc_all.append(cellid.cell_status.n_oscillations)
                start_osc.append(cellid.cell_status.start_oscillation)
                if cellid.cell_status.n_oscillations==0:# or cellid.cell_status.start_oscillation_frame==0 or cellid.cell_status.end_oscillation_frame==0:
                    print('---------------  ', cellid.cell_status)
                end_osc.append(cellid.cell_status.end_oscillation)
                if cellid.cell_status.time_of_death>0:
                    
                    if cellid.sample.keep_sample:
                        #if tod_checkbox_keep: tod.append(cellid.cell_status.time_of_death)
                        #if tod_checkbox_all: tod_all.append(cellid.cell_status.time_of_death)
                        tod_all.append(cellid.cell_status.time_of_death)
                    else:
                        #if tod_checkbox_dkeep: tod_dk.append(cellid.cell_status.time_of_death)
                        #if tod_checkbox_all:tod_all.append(cellid.cell_status.time_of_death)
                        tod_all.append(cellid.cell_status.time_of_death)

                if cellid.cell_status.time_of_death_pred>0:
                    tod_pred.append(cellid.cell_status.time_of_death_pred)


        max_osc=0
        if max(n_osc_all, default=-999)>0:
            max_osc=max(n_osc_all, default=0)
            max_osc_dk=max(n_osc_dk, default=0)
            if max_osc_dk>max_osc: max_osc=max_osc_dk
        hist, edges = np.histogram(n_osc, bins=max_osc+2, range=(0, max_osc+2))
        source_nosc.data={'x': edges[:-1], 'top': hist}

        hist, edges = np.histogram(n_osc_dk, bins=max_osc+2, range=(0, max_osc+2))
        source_nosc_dk.data={'x': edges[:-1], 'top': hist}

        hist, edges = np.histogram(n_osc_all, bins=max_osc+2, range=(0, max_osc+2))
        source_nosc_all.data={'x': edges[:-1], 'top': hist}

        max_tod=max(tod_all, default=100.)
        min_tod=min(tod_all, default=0.)
        if max(tod_dk, default=100.)>max_tod:max_tod=max(tod_dk, default=100.)
        if min(tod_dk, default=0.)<min_tod:min_tod=min(tod_dk, default=0.)

        hist, edges = np.histogram(tod, bins=int((max_tod-min_tod)/30.), range=(min_tod, max_tod))
        source_tod.data={'x': edges[:-1], 'top': hist}

        hist, edges = np.histogram(tod_dk, bins=int((max_tod-min_tod)/30.), range=(min_tod, max_tod))
        source_tod_dk.data={'x': edges[:-1], 'top': hist}

        hist, edges = np.histogram(tod_all, bins=int((max_tod-min_tod)/30.), range=(min_tod, max_tod))
        source_tod_all.data={'x': edges[:-1], 'top': hist}



        hist, edges = np.histogram(start_osc, bins=nframes*10, range=(0, nframes*10))
        source_start_osc.data={'x': edges[:-1], 'top': hist}

        hist, edges = np.histogram(end_osc, bins=nframes*10, range=(0, nframes*10))
        source_end_osc.data={'x': edges[:-1], 'top': hist}

        max_tod=max(tod_pred, default=100.)
        min_tod=min(tod_pred, default=0.)
        if min_tod<0: min_tod=0
        if max_tod<0:max_tod=100.
        if min_tod>=max_tod:min_tod=max_tod*0.5
        if int((max_tod-min_tod)/30.) == 0:
            if max_tod<40: max_tod=100
            min_tod=0
        print('max_tod ', max_tod,'  min_tod ',min_tod, '  int((max_tod-min_tod)/30.)  ',int((max_tod-min_tod)/30.))
        hist, edges = np.histogram(tod_pred, bins=int((max_tod-min_tod)/30.), range=(min_tod, max_tod))
        source_tod_pred.data={'x': edges[:-1], 'top': hist}

    #___________________________________________________________________________________________


    tod_checkbox   = bokeh.models.CheckboxGroup(labels=["All", "Keep", "Don't keep"], active=[1,0,0])


    #___________________________________________________________________________________________
    # Select image from click
    def select_tap_callback():
        return """
        const indices = cb_data.source.selected.indices;

        if (indices.length > 0) {
            const index = indices[0];
            other_source.data = {'index': [index]};
            other_source.change.emit();  
        }
        """
    #___________________________________________________________________________________________

    #___________________________________________________________________________________________
    def update_image_tap_callback(attr, old, new):
        if DEBUG:print('****************************  update_image_tap_callback ****************************')
        update_tap_renderers()
        index = new['index'][0]
        if DEBUG:print('index=',index)
        slider.value=index
    #___________________________________________________________________________________________

    #___________________________________________________________________________________________
    def update_tap_renderers():
        # Determine which renderer should be included based on some condition
        # For example, if a certain condition is met, include renderer1, otherwise include renderer2
        tap_tool.renderers = [int_ch1, int_ch2]
        box_select_tool.renderers = [int_ch1, int_ch2]
    #___________________________________________________________________________________________

    #___________________________________________________________________________________________
    # Define a function to reset TapTool's selection
    def reset_tap_tool():
        source_intensity_ch1.selected.indices = []
    #___________________________________________________________________________________________

    #___________________________________________________________________________________________
    def generic_indices_callback(flag):
        if DEBUG:print("==================generic_indices_callback========================")
        sample = Sample.objects.get(file_name=get_current_file())
        cellsid = CellID.objects.select_related().filter(sample=sample, name=dropdown_cell.value)
        cellstatus = cellsid[0].cell_status
        cellflags = cellstatus.flags
        mydict = cellflags

        try:
            mydict['{}_frame'.format(flag)]
            mydict['{}_time'.format(flag)]
        except KeyError:
            mydict['{}_frame'.format(flag)] = []
            mydict['{}_time'.format(flag)]  = []

        if DEBUG:print('cellflags begin=',cellflags)
        if DEBUG:print('mydict begin   =',mydict)

        cellrois = CellROI.objects.select_related().filter(cell_id=cellsid[0])
        data={'time':[], 'intensity':[], 'intensity_full':[]}

        for cellroi in cellrois:
            framenumber = cellroi.frame.number
            cellflag = cellroi.cellflag_cellroi
            if framenumber in source_intensity_ch1.selected.indices:
                setattr(cellflag, flag, True)
                if framenumber not in mydict['{}_frame'.format(flag)]:
                    mydict['{}_frame'.format(flag)].append(framenumber)
                    mydict['{}_time'.format(flag)].append(source_intensity_ch1.data["time"][framenumber])
                    data['time'].append(source_intensity_ch1.data["time"][framenumber])
                    data['intensity'].append(flags_dict[flag])
                    data['intensity_full'].append(source_intensity_ch1.data["intensity"][framenumber])
            elif len(source_intensity_ch1.selected.indices)==0:
                setattr(cellflag, flag, False)
                mydict['{}_frame'.format(flag)] = []
                mydict['{}_time'.format(flag)]  = []

            cellflag.save()

        cellstatus.flags = mydict
        cellstatus.save()

        if DEBUG:
            print('cellflags end=',cellstatus.flags)
            print('mydict end   =',mydict)

        return data
    #___________________________________________________________________________________________

    #___________________________________________________________________________________________
    def mask_cell_callback():
        if DEBUG:print('--------mask_cell_callback-------- ')
        data = generic_indices_callback('mask')
        if len(source_intensity_ch1.selected.indices)==0:
            source_mask_cell.data = {"time":[], "intensity_full":[]}
        else:
            source_mask_cell.data = {"time":source_mask_cell.data["time"]+data["time"], "intensity_full":source_mask_cell.data["intensity_full"]+data["intensity_full"]}
    button_mask_cells = bokeh.models.Button(label="Mask")
    button_mask_cells.on_click(mask_cell_callback)
    #___________________________________________________________________________________________

    #___________________________________________________________________________________________
    def dividing_cell_callback():
        if DEBUG:print('--------dividing_cell_callback-------- ')
        data = generic_indices_callback('dividing')
        if len(source_intensity_ch1.selected.indices)==0:
            try:
                for t1 in range(len(source_dividing_cell.data["time"])):
                   for t2 in range(len(source_segments_cell.data["time"])):
                       if source_dividing_cell.data["time"][t1] == source_segments_cell.data["time"][t2]:
                           source_segments_cell.data["time"].pop(t2)
                           source_segments_cell.data["intensity"].pop(t2)
                           break
            except KeyError:
                pass
            source_dividing_cell.data = {"time":[], "intensity":[]}
            source_segments_cell.data = {"time":source_segments_cell.data["time"], "intensity":source_segments_cell.data["intensity"]}
        else:
            source_dividing_cell.data = {"time":source_dividing_cell.data["time"]+data["time"], "intensity":source_dividing_cell.data["intensity"]+data["intensity"]}
            source_segments_cell.data = {"time":source_segments_cell.data["time"]+data["time"], "intensity":source_segments_cell.data["intensity"]+data["intensity_full"]}
    button_dividing_cells = bokeh.models.Button(label="Dividing")
    button_dividing_cells.on_click(dividing_cell_callback)
    #___________________________________________________________________________________________

    #___________________________________________________________________________________________
    def double_nuclei_cell_callback():
        if DEBUG:print('--------double_nuclei_cell_callback-------- ')
        data = generic_indices_callback('double_nuclei')
        if len(source_intensity_ch1.selected.indices)==0:
            try:
                for t1 in range(len(source_double_nuclei_cell.data["time"])):
                   for t2 in range(len(source_segments_cell.data["time"])):
                       if source_double_nuclei_cell.data["time"][t1] == source_segments_cell.data["time"][t2]:
                           source_segments_cell.data["time"].pop(t2)
                           source_segments_cell.data["intensity"].pop(t2)
                           break
            except KeyError:
                pass
            source_double_nuclei_cell.data = {"time":[], "intensity":[]}
            source_segments_cell.data      = {"time":source_segments_cell.data["time"], "intensity":source_segments_cell.data["intensity"]}
        else:
            source_double_nuclei_cell.data = {"time":source_double_nuclei_cell.data["time"]+data["time"], "intensity":source_double_nuclei_cell.data["intensity"]+data["intensity"]}
            source_segments_cell.data      = {"time":source_segments_cell.data["time"]+data["time"],      "intensity":source_segments_cell.data["intensity"]+data["intensity_full"]}
    button_double_nuclei_cells = bokeh.models.Button(label="Double nuclei")
    button_double_nuclei_cells.on_click(double_nuclei_cell_callback)
    #___________________________________________________________________________________________

    #___________________________________________________________________________________________
    def multiple_cells_callback():
        if DEBUG:print('--------multiple_cells_callback-------- ')
        data = generic_indices_callback('multiple_cells')
        if len(source_intensity_ch1.selected.indices)==0:
            try:
                for t1 in range(len(source_multiple_cells.data["time"])):
                   for t2 in range(len(source_segments_cell.data["time"])):
                       if source_multiple_cells.data["time"][t1] == source_segments_cell.data["time"][t2]:
                           source_segments_cell.data["time"].pop(t2)
                           source_segments_cell.data["intensity"].pop(t2)
                           break
            except KeyError:
                pass
            source_multiple_cells.data = {"time":[], "intensity":[]}
            source_segments_cell.data  = {"time":source_segments_cell.data["time"], "intensity":source_segments_cell.data["intensity"]}
        else:
            source_multiple_cells.data = {"time":source_multiple_cells.data["time"]+data["time"], "intensity":source_multiple_cells.data["intensity"]+data["intensity"]}
            source_segments_cell.data  = {"time":source_segments_cell.data["time"]+data["time"],  "intensity":source_segments_cell.data["intensity"]+data["intensity_full"]}
    button_multiple_cells = bokeh.models.Button(label="Multiple cells")
    button_multiple_cells.on_click(multiple_cells_callback)
    #___________________________________________________________________________________________

    #___________________________________________________________________________________________
    def pair_cell_callback():
        if DEBUG:print('--------pair_cell_callback-------- ')
        data = generic_indices_callback('pair_cell')
        if len(source_intensity_ch1.selected.indices)==0:
            try:
                for t1 in range(len(source_pair_cell.data["time"])):
                   for t2 in range(len(source_segments_cell.data["time"])):
                       if source_pair_cell.data["time"][t1] == source_segments_cell.data["time"][t2]:
                           source_segments_cell.data["time"].pop(t2)
                           source_segments_cell.data["intensity"].pop(t2)
                           break
            except KeyError:
                pass
            source_pair_cell.data     = {"time":[], "intensity":[]}
            source_segments_cell.data = {"time":source_segments_cell.data["time"], "intensity":source_segments_cell.data["intensity"]}
        else:
            source_pair_cell.data     = {"time":source_pair_cell.data["time"]+data["time"], "intensity":source_pair_cell.data["intensity"]+data["intensity"]}
            source_segments_cell.data = {"time":source_segments_cell.data["time"]+data["time"],  "intensity":source_segments_cell.data["intensity"]+data["intensity_full"]}
    button_pair_cell = bokeh.models.Button(label="Pair cell")
    button_pair_cell.on_click(pair_cell_callback)
    #___________________________________________________________________________________________

    #___________________________________________________________________________________________
    def flat_cell_callback():
        if DEBUG:print('--------flat_cell_callback-------- ')
        data = generic_indices_callback('flat')
        if len(source_intensity_ch1.selected.indices)==0:
            try:
                for t1 in range(len(source_flat_cell.data["time"])):
                   for t2 in range(len(source_segments_cell.data["time"])):
                       if source_flat_cell.data["time"][t1] == source_segments_cell.data["time"][t2]:
                           source_segments_cell.data["time"].pop(t2)
                           source_segments_cell.data["intensity"].pop(t2)
                           break
            except KeyError:
                pass
            source_flat_cell.data     = {"time":[], "intensity":[]}
            source_segments_cell.data = {"time":source_segments_cell.data["time"], "intensity":source_segments_cell.data["intensity"]}
        else:
            source_flat_cell.data     = {"time":source_flat_cell.data["time"]+data["time"], "intensity":source_flat_cell.data["intensity"]+data["intensity"]}
            source_segments_cell.data = {"time":source_segments_cell.data["time"]+data["time"],  "intensity":source_segments_cell.data["intensity"]+data["intensity_full"]}
    button_flat_cell = bokeh.models.Button(label="Flat cell")
    button_flat_cell.on_click(flat_cell_callback)
    #___________________________________________________________________________________________

    #___________________________________________________________________________________________
    def round_cell_callback():
        if DEBUG:print('--------round_cell_callback-------- ')
        data = generic_indices_callback('round')
        if len(source_intensity_ch1.selected.indices)==0:
            try:
                for t1 in range(len(source_round_cell.data["time"])):
                   for t2 in range(len(source_segments_cell.data["time"])):
                       if source_round_cell.data["time"][t1] == source_segments_cell.data["time"][t2]:
                           source_segments_cell.data["time"].pop(t2)
                           source_segments_cell.data["intensity"].pop(t2)
                           break
            except KeyError:
                pass
            source_round_cell.data     = {"time":[], "intensity":[]}
            source_segments_cell.data = {"time":source_segments_cell.data["time"], "intensity":source_segments_cell.data["intensity"]}
        else:
            source_round_cell.data     = {"time":source_round_cell.data["time"]+data["time"], "intensity":source_round_cell.data["intensity"]+data["intensity"]}
            source_segments_cell.data = {"time":source_segments_cell.data["time"]+data["time"],  "intensity":source_segments_cell.data["intensity"]+data["intensity_full"]}
    button_round_cell = bokeh.models.Button(label="Round cell")
    button_round_cell.on_click(round_cell_callback)
    #___________________________________________________________________________________________

    #___________________________________________________________________________________________
    def elongated_cell_callback():
        if DEBUG:print('--------elongated_cell_callback-------- ')
        data = generic_indices_callback('elongated')
        if len(source_intensity_ch1.selected.indices)==0:
            try:
                for t1 in range(len(source_elongated_cell.data["time"])):
                   for t2 in range(len(source_segments_cell.data["time"])):
                       if source_elongated_cell.data["time"][t1] == source_segments_cell.data["time"][t2]:
                           source_segments_cell.data["time"].pop(t2)
                           source_segments_cell.data["intensity"].pop(t2)
                           break
            except KeyError:
                pass
            source_elongated_cell.data     = {"time":[], "intensity":[]}
            source_segments_cell.data = {"time":source_segments_cell.data["time"], "intensity":source_segments_cell.data["intensity"]}
        else:
            source_elongated_cell.data     = {"time":source_elongated_cell.data["time"]+data["time"], "intensity":source_elongated_cell.data["intensity"]+data["intensity"]}
            source_segments_cell.data = {"time":source_segments_cell.data["time"]+data["time"],  "intensity":source_segments_cell.data["intensity"]+data["intensity_full"]}
    button_elongated_cell = bokeh.models.Button(label="Elongated cell")
    button_elongated_cell.on_click(elongated_cell_callback)
    #___________________________________________________________________________________________

    #___________________________________________________________________________________________
    def delete_cell_callback():
        sample = Sample.objects.get(file_name=get_current_file())
        cellID = CellID.objects.select_related().filter(sample=sample, name=dropdown_cell.value)
        if len(cellID)==0:return
        cellstatus = cellID[0].cell_status
        cellROIs = CellROI.objects.select_related().filter(cell_id=cellID[0])
        for cellroi in cellROIs:
            cellroi.delete()
        cellstatus.delete()
        update_dropdown_cell('','','')
        left_rois, right_rois, top_rois, bottom_rois,height_labels, weight_labels, names_labels, height_cells, weight_cells, names_cells= update_source_roi_cell_labels()        
        source_roi.data    = {'left': left_rois,      'right': right_rois,    'top': top_rois, 'bottom': bottom_rois}
        source_labels.data = {'height':height_labels, 'weight':weight_labels, 'names':names_labels}
        source_cells.data  = {'height':height_cells,  'weight':weight_cells,  'names':names_cells}

    button_delete_cell = bokeh.models.Button(label="Delete cell")
    button_delete_cell.on_click(delete_cell_callback)
    #___________________________________________________________________________________________

    #___________________________________________________________________________________________
    def box_select_callback(event):
        if DEBUG:print('-----==========-==---=box_select_callback=',source_intensity_ch1.selected.indices)
        if isinstance(event, bokeh.events.SelectionGeometry):
            indices = source_intensity_ch1.selected.indices
            if len(indices)>0:
                slider.start = indices[0]
                slider.end = indices[-1]
                slider.value = indices[0]
            else: 
                slider.start = 0
                slider.end = len(source_intensity_ch0.data["time"])-1
            #slider2.value = len(indices) * 3

    # Attach the Python callback to the figure
    plot_intensity.on_event(bokeh.events.SelectionGeometry, box_select_callback)
    #___________________________________________________________________________________________




    #get_adjacent_stack()
    #threading.Thread(target = get_adjacent_stack).start()

    threads = []
    for iii in range(-2,6):
        threads.append(threading.Thread(target = get_adjacent_stack_test, args=(iii, )))
    for t in threads: t.start()
    for t in threads: t.join()


    # Create a Div widget with some text
    text = bokeh.models.Div(text="<h2>Cell informations</h2>")

    source_roi_manual  = bokeh.models.ColumnDataSource(data=dict(left=[], right=[], top=[], bottom=[]))
    #fill_rois(dropdown_well.value)

    left_rois, right_rois, top_rois, bottom_rois,height_labels, weight_labels, names_labels, height_cells, weight_cells, names_cells= update_source_roi_cell_labels()
    source_roi.data = {'left': left_rois, 'right': right_rois, 'top': top_rois, 'bottom': bottom_rois}
    source_labels.data = {'height':height_labels, 'weight':weight_labels, 'names':names_labels}
    source_cells.data = {'height':height_cells, 'weight':weight_cells, 'names':names_cells}


    labels = bokeh.models.LabelSet(x='weight', y='height', text='names', x_units='data', y_units='data',
                                   x_offset=0, y_offset=0, source=source_labels, text_color='white', text_font_size="10pt")

    plot_image.add_layout(labels)

    labels_cells = bokeh.models.LabelSet(x='weight', y='height', text='names', x_units='data', y_units='data',
                                         x_offset=0, y_offset=-15, source=source_cells, text_color='white', text_font_size="11pt")

    plot_image.add_layout(labels_cells)
    #plot_image.add_layout(color_bar, 'right')

    plot_image.image(image='img', x=0, y=0, dw=ind_images_list[0][0].shape[0], dh=ind_images_list[0][0].shape[1], source=source_img, color_mapper=color_mapper)


    plot_image.star(x='x', y='y', size=10, color='white', source=source_segmentation_points)


    source_segmentation  = bokeh.models.ColumnDataSource(data=dict(x=[], y=[]))
#    plot_image.patch(x='x', y='y', fill_color=None, line_color="red", line_width=3, line_alpha=0.8, source=source_segmentation)
    plot_image.line(x='x', y='y', line_color="red", line_width=3, line_alpha=0.8, source=source_segmentation)


    data_img_mask={'img':[]}
    source_img_mask  = bokeh.models.ColumnDataSource(data=data_img_mask)

    mask_color_mapper = bokeh.models.LinearColorMapper(palette=['#00000000', '#ff000080'], low=0, high=1)
    plot_image.image(image='img', x=0, y=0, dw=ind_images_list[0][0].shape[0], dh=ind_images_list[0][0].shape[1], source=source_img_mask, color_mapper=mask_color_mapper)


    #plot_img_mask     = bokeh.plotting.figure(x_range=(0, ind_images_list[0][0].shape[0]), y_range=(0, ind_images_list[0][0].shape[1]), tools="box_select,wheel_zoom,box_zoom,reset,undo",width=550, height=550)
    #plot_img_mask.image(image='img', x=0, y=0, dw=ind_images_list[0][0].shape[0], dh=ind_images_list[0][0].shape[1], source=source_img_mask, color_mapper=color_mapper)

       # Create a ColumnDataSource to store image data
    #source_url = bokeh.models.ColumnDataSource({'url': [''], 'x': [0], 'y': [0], 'dw': [0], 'dh': [0]})
    #ind_images_list_url = get_current_stack_url()

    #source_url.data = {'url': [f'data:image/png;base64,{ind_images_list_url[0][0]}'], 'x': [0], 'y': [0], 'dw': [ind_images_list[0][0].shape[1]], 'dh': [ind_images_list[0][0].shape[0]]}

    #image_plot = plot_image.image_url(url='url', x='x', y='y', w='dw', h='dh', source=source_url)

    current_file=get_current_file()
    sample = Sample.objects.get(file_name=current_file)

    if sample.peaks_tod_div_validated:
        position_check_div.text = "<b style='color:green; ; font-size:18px;'> Peaks/ToD/Division validated (c1)</b>"
        position_check_button.label = "Peaks/ToD/Division not validated"
    else:
        position_check_div.text = "<b style='color:red; ; font-size:18px;'> Peaks/ToD/Division not validated</b>"
        position_check_button.label = "Peaks/ToD/Division validated"

    if sample.bf_features_validated:
        position_check2_div.text = "<b style='color:green; ; font-size:18px;'> BF features validated (c2)</b>"
        position_check2_button.label = "BF features not validated"
    else:
        position_check2_div.text = "<b style='color:red; ; font-size:18px;'> BF features not validated</b>"
        position_check2_button.label = "BF features validated"

    if sample.keep_sample:
        position_keep_div.text  = "<b style='color:green; ; font-size:18px;'> Keep Position</b>"
        position_keep_button.label = "Don't keep Position"
    else:
        position_keep_div.text  = "<b style='color:red; ; font-size:18px;'> Do not Keep Position (dk)</b>"
        position_keep_button.label = "Keep Position"
    cellIDs = CellID.objects.select_related().filter(sample=sample)
    cell_list=[]
    for cid in cellIDs:
        cell_list.append(cid.name)
    dropdown_cell.options=cell_list
    if len(cell_list)>0:dropdown_cell.value=cell_list[0]
    time_list={}
    intensity_list={}
    if len(cellIDs)>0:

        ROIs = CellROI.objects.select_related().filter(cell_id=cellIDs[0])
        for roi in ROIs:
            for ch in roi.contour_cellroi.intensity_sum:
                try:
                    time_list[ch]
                except KeyError:
                    #CHANGE WITH ssample.number_of_frames when available
                    #nframes=sample.number_of_frames
                    frames=Frame.objects.select_related().filter(sample=sample)
                    nframes=len(frames)
                    intensity_list[ch]=[0 for i in range(nframes)]
                    time_list[ch]=[f.time/60000 for f in frames]
                #time_list[ch].append((roi.frame.time/60000))
                #intensity_list[ch].append(roi.contour_cellroi.intensity_sum[ch]/roi.contour_cellroi.number_of_pixels)
                intensity_list[ch][roi.frame.number]= roi.contour_cellroi.intensity_sum[ch]/roi.contour_cellroi.number_of_pixels


    plot_intensity.line('time', 'intensity', source=source_intensity_ch1, line_color='blue')
    plot_intensity.line('time', 'area', y_range_name="area", source=source_intensity_area, line_color='black')
    plot_intensity.circle('time', 'area', y_range_name="area", source=source_intensity_area, line_color='black')
    plot_intensity.plus('time', 'intensity', source=source_intensity_predicted_death, line_color='black',size=20)
    
    ax2 = bokeh.models.LinearAxis(
        axis_label="area",
        y_range_name="area",
    )
    ax2.axis_label_text_color = bokeh.palettes.Sunset6[5]
    plot_intensity.add_layout(ax2, 'left')


    int_ch1 = plot_intensity.circle('time', 'intensity', source=source_intensity_ch1, fill_color="white", size=10, line_color='blue')
    plot_intensity.line('time', 'intensity', source=source_intensity_ch2, line_color='black')
    int_ch2 = plot_intensity.circle('time', 'intensity', source=source_intensity_ch2, fill_color="white", size=10, line_color='black')
    plot_intensity.circle('time', 'intensity', source=source_intensity_max, fill_color="red", size=10, line_color='red')
    plot_intensity.circle('time', 'intensity', source=source_intensity_min, fill_color="green", size=10, line_color='green')
    plot_intensity.circle('time', 'intensity_full', source=source_mask_cell, fill_color="black", size=10, line_color='black')

    plot_intensity.square_pin('time', 'intensity', source=source_dividing_cell, fill_color=None, size=8, line_color='black')
    plot_intensity.circle_dot('time', 'intensity', source=source_double_nuclei_cell, fill_color=None, size=8, line_color='black')
    plot_intensity.circle_x('time', 'intensity', source=source_multiple_cells, fill_color=None, size=8, line_color='black')
    plot_intensity.circle_y('time', 'intensity', source=source_pair_cell, fill_color=None, size=8, line_color='black')
    plot_intensity.triangle_pin('time', 'intensity', source=source_flat_cell, fill_color=None, size=8, line_color='black')
    plot_intensity.circle_cross('time', 'intensity', source=source_round_cell, fill_color=None, size=8, line_color='black')
    plot_intensity.dash('time', 'intensity', source=source_elongated_cell, fill_color=None, size=8, line_color='black')

    plot_markers = bokeh.plotting.figure(width=200, height=500, title="cell flags", tools="")

    # Sample data
    x = [1, 1, 1, 1, 1]
    y = [1, 1, 1, 1, 1]

    # Available marker types
    markers = ['square_pin','circle_dot', 'circle_x', 'circle_y', 'triangle_pin', 'circle_cross','dash']
    labels  = ['dividing', 'double nuclei', 'multiple cells', 'pair cells', 'flat', 'round', 'elongated']
    # Plot each marker type
    for i in range(len(markers)):
        plot_markers.scatter(x=x, y=y, marker=markers[i], size=8, legend_label=labels[i], fill_color=None)

    # Adjust plot properties
    plot_markers.xaxis.visible = False
    plot_markers.yaxis.visible = False
    plot_markers.legend.title = "Marker Type"
    plot_markers.legend.location = "top_left"
    plot_markers.legend.click_policy = "hide"



    plot_intensity.segment(x0='time', y0=0, x1='time', y1='intensity', line_color='black', line_width=0.5, source=source_segments_cell, line_dash="dotted")

    index_source = bokeh.models.ColumnDataSource(data=dict(index=[]))  # Data source for the image
    tap_tool = bokeh.models.TapTool(callback=bokeh.models.CustomJS(args=dict(other_source=index_source),code=select_tap_callback()))
    tap_tool_seg = bokeh.models.TapTool()
    plot_image.add_tools(tap_tool_seg)

    plot_intensity.add_tools(tap_tool)
    index_source.on_change('data', update_image_tap_callback)

    box_select_tool = bokeh.models.BoxSelectTool(select_every_mousemove=False)
    plot_intensity.add_tools(box_select_tool)

    plot_intensity.y_range.start=0
    plot_intensity.x_range.start=-10

    for index, key in enumerate(time_list):
        if index==0:
            source_intensity_ch0.data={'time':time_list[key], 'intensity':intensity_list[key]}
        if index==1:
            source_intensity_ch1.data={'time':time_list[key], 'intensity':intensity_list[key]}
        if index==2:
            source_intensity_ch2.data={'time':time_list[key], 'intensity':intensity_list[key]}

    initial_position = 0
    if len(source_intensity_ch1.data["time"])!=0:
        initial_position = source_intensity_ch1.data["time"][0]
    plot_intensity.add_layout(line_position)

    plot_intensity.add_layout(start_oscillation_position)
    plot_intensity.add_layout(end_oscillation_position)
    plot_intensity.add_layout(time_of_death_position)


    
    plot_intensity.varea(x='x', y1='y1', y2='y2', fill_alpha=0.20, fill_color='red', source=source_varea_rising1)
    plot_intensity.varea(x='x', y1='y1', y2='y2', fill_alpha=0.20, fill_color='red', source=source_varea_rising2)
    plot_intensity.varea(x='x', y1='y1', y2='y2', fill_alpha=0.20, fill_color='red', source=source_varea_rising3)
    plot_intensity.varea(x='x', y1='y1', y2='y2', fill_alpha=0.20, fill_color='red', source=source_varea_rising4)
    plot_intensity.varea(x='x', y1='y1', y2='y2', fill_alpha=0.20, fill_color='red', source=source_varea_rising5)
    plot_intensity.varea(x='x', y1='y1', y2='y2', fill_alpha=0.20, fill_color='red', source=source_varea_rising6)
    plot_intensity.varea(x='x', y1='y1', y2='y2', fill_alpha=0.20, fill_color='red', source=source_varea_rising7)
    plot_intensity.varea(x='x', y1='y1', y2='y2', fill_alpha=0.20, fill_color='red', source=source_varea_rising8)
    plot_intensity.varea(x='x', y1='y1', y2='y2', fill_alpha=0.20, fill_color='red', source=source_varea_rising9)
    plot_intensity.varea(x='x', y1='y1', y2='y2', fill_alpha=0.20, fill_color='red', source=source_varea_rising10)


    plot_intensity.varea(x='x', y1='y1', y2='y2', fill_alpha=0.20, fill_color='green', source=source_varea_falling1)
    plot_intensity.varea(x='x', y1='y1', y2='y2', fill_alpha=0.20, fill_color='green', source=source_varea_falling2)
    plot_intensity.varea(x='x', y1='y1', y2='y2', fill_alpha=0.20, fill_color='green', source=source_varea_falling3)
    plot_intensity.varea(x='x', y1='y1', y2='y2', fill_alpha=0.20, fill_color='green', source=source_varea_falling4)
    plot_intensity.varea(x='x', y1='y1', y2='y2', fill_alpha=0.20, fill_color='green', source=source_varea_falling5)
    plot_intensity.varea(x='x', y1='y1', y2='y2', fill_alpha=0.20, fill_color='green', source=source_varea_falling6)
    plot_intensity.varea(x='x', y1='y1', y2='y2', fill_alpha=0.20, fill_color='green', source=source_varea_falling7)
    plot_intensity.varea(x='x', y1='y1', y2='y2', fill_alpha=0.20, fill_color='green', source=source_varea_falling8)
    plot_intensity.varea(x='x', y1='y1', y2='y2', fill_alpha=0.20, fill_color='green', source=source_varea_falling9)
    plot_intensity.varea(x='x', y1='y1', y2='y2', fill_alpha=0.20, fill_color='green', source=source_varea_falling10)

    plot_intensity.varea(x='x', y1='y1', y2='y2', fill_alpha=0.10, fill_color='black', source=source_varea_death)


    update_source_osc_tod()
    #plot_osc_tod.vbar(x='x', top='top', width=3, source=source_start_osc, alpha=0.5, color='green', line_color=None)
    #plot_osc_tod.vbar(x='x', top='top', width=3, source=source_end_osc, alpha=0.5, color='red', line_color=None)
    
    #Commment for now on tod keep and don't keep and only display all
    #plot_tod.vbar(x='x', top='top', width=28., source=source_tod, alpha=0.25, color='green', line_color=None, legend_label='keep')
    #plot_tod.vbar(x='x', top='top', width=28., source=source_tod_dk, alpha=0.25, color='red', line_color=None, legend_label='don\'t keep')
    plot_tod.vbar(x='x', top='top', width=28., source=source_tod_all, alpha=0.25, color='black', line_color=None, legend_label='all')
    plot_tod_pred.vbar(x='x', top='top', width=28., source=source_tod_pred, alpha=0.25, color='black', line_color=None, legend_label='pred')

    #Comment for now don't keep and all nosc and only show the ones we keep
    plot_nosc.vbar(x='x', top='top', width=0.9, source=source_nosc, alpha=0.25, color='green', line_color=None, legend_label='keep')
    #plot_nosc.vbar(x='x', top='top', width=0.9, source=source_nosc_dk, alpha=0.25, color='red', line_color=None, legend_label='don\'t keep')
    #plot_nosc.vbar(x='x', top='top', width=0.9, source=source_nosc_all, alpha=0.25, color='black', line_color=None, legend_label='all')

    prepare_intensity() 

    update_source_segment()

    # Add the rectangle glyph after adding the image
    quad = bokeh.models.Quad(left='left', right='right', top='top', bottom='bottom', fill_color=None)#, fill_alpha=0.0, fill_color='#009933')
    plot_image.add_glyph(source_roi, quad, selection_glyph=quad, nonselection_glyph=quad)

    # Remove the axes
    plot_image.axis.visible = False
    plot_image.grid.visible = False
    #plot_img_mask.axis.visible = False
    #plot_img_mask.grid.visible = False

 
    plot_oscillation_cycle  = bokeh.plotting.figure(title="Osc Cycle", x_axis_label='cycle number', y_axis_label='Period [min]',width=500, height=400)
    #whisker = bokeh.models.Whisker(base=bokeh.transform.jitter('base', width=0.25, range=plot_oscillation_cycle.x_range),
    whisker = bokeh.models.Whisker(base='base',
                                   upper='upper', lower='lower', source=source_osc_period_err, level="annotation", line_width=2)
    #whisker.upper_head.size=20
    #whisker.lower_head.size=20
    plot_oscillation_cycle.add_layout(whisker)
    plot_oscillation_cycle.scatter(x=bokeh.transform.jitter('cycle', width=0.25, range=plot_oscillation_cycle.x_range), y='time', source=source_osc_period, size=8)
    plot_oscillation_cycle.line('x', 'y', source=source_osc_period_line, line_color='black')

    update_oscillation_cycle()

    # Sample data
    from bokeh.palettes import Category20c
    from bokeh.transform import cumsum
    from math import pi

    import pandas as pd

    x = {
        'United States': 157,
        'United Kingdom': 93,
        'Japan': 89,
        'China': 63,
        'Germany': 44,
        'India': 42,
        'Italy': 40,
        'Australia': 35,
        'Brazil': 32,
        'France': 31,
        'Taiwan': 31,
        'Spain': 29,
    }

    data = pd.Series(x).reset_index(name='value').rename(columns={'index': 'country'})
    data['angle'] = data['value']/data['value'].sum() * 2*pi
    data['color'] = Category20c[len(x)]

    pie = bokeh.plotting.figure(height=150, title="Pie Chart", toolbar_location=None,
            tools="hover", tooltips="@country: @value", x_range=(-0.5, 1.0))

    pie.wedge(x=0, y=1, radius=0.4,
            start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
            line_color="white", fill_color='color', legend_field='country', source=data)

    exp_color_col = bokeh.layouts.column(bokeh.layouts.row(dropdown_exp),
                                         bokeh.layouts.row(dropdown_well),
                                         bokeh.layouts.row(dropdown_pos), 
                                         bokeh.layouts.row(prev_position_button, next_position_button),
                                         bokeh.layouts.row(dropdown_channel),
                                         bokeh.layouts.row(dropdown_color),
                                         bokeh.layouts.row(bokeh.layouts.Spacer(width=10),contrast_slider),
                                         bokeh.layouts.row(position_check_button),
                                         bokeh.layouts.row(position_check2_button),
                                         bokeh.layouts.row(position_keep_button),
                                         bokeh.layouts.row(button_build_roi),
                                         bokeh.layouts.row(button_remove_roi),
                                         bokeh.layouts.row(button_build_sam2),
                                         )

    right_col = bokeh.layouts.column(bokeh.layouts.row(slider),
                                     bokeh.layouts.row(button_play_stop, button_prev, button_next, dropdown_refresh_time ),
                                     bokeh.layouts.row(button_delete_roi, button_save_roi, dropdown_cell ),
                                     bokeh.layouts.row(button_inspect, button_build_cells, button_delete_cell),
                                     bokeh.layouts.row(button_start_oscillation,button_end_oscillation,button_time_of_death),
                                     bokeh.layouts.row(button_save_peaks, button_delete_peaks) ,
                                     bokeh.layouts.row(dropdown_intensity_type, dropdown_segmentation_type),
                                     bokeh.layouts.row(slider_find_peaks),
                                     bokeh.layouts.row(button_mask_cells, button_dividing_cells, button_double_nuclei_cells),
                                     bokeh.layouts.row(button_multiple_cells, button_pair_cell),
                                     bokeh.layouts.row(button_flat_cell, button_round_cell, button_elongated_cell),
                                     )
    
    intensity_plot_col = bokeh.layouts.column(bokeh.layouts.row(plot_intensity, plot_markers),
                                              bokeh.layouts.row(plot_tod, plot_nosc),#tod_checkbox,
                                              bokeh.layouts.row(plot_tod_pred, plot_oscillation_cycle),)

    cell_osc_plot_col = bokeh.layouts.column(bokeh.layouts.row(plot_image),
                                             #bokeh.layouts.row(plot_nosc),
                                             #bokeh.layouts.row(plot_img_mask),
                                             )

    #cell_osc_plot_col =  bokeh.layouts.column(bokeh.layouts.gridplot([[plot_image], [plot_img_mask]]))

    norm_layout = bokeh.layouts.column(bokeh.layouts.row(dropdown_filter_position_keep),
        bokeh.layouts.row(position_check_div, bokeh.layouts.Spacer(width=10), position_check2_div, 
                                                         bokeh.layouts.Spacer(width=10), position_keep_div, bokeh.layouts.Spacer(width=40), ncells_div),
                                       bokeh.layouts.row(exp_color_col, cell_osc_plot_col, right_col, intensity_plot_col),
                                       #bokeh.layouts.row(pie),
                                       bokeh.layouts.row(text))

    doc.add_root(norm_layout)



#___________________________________________________________________________________________
#___________________________________________________________________________________________
#___________________________________________________________________________________________
def phenocheck_handler(doc: bokeh.document.Document) -> None:
    print('****************************  phenocheck_handler ****************************')
    os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"

    annot_dict_train = {}
    annot_dict_valid = {}
    image_dict_train = {}
    image_dict_valid = {}
    image_cropped_dict_train = {}
    image_cropped_dict_valid = {}

    map_img_pos_train = {}
    map_img_pos_valid = {}

    source_image  = bokeh.models.ColumnDataSource(dict(img=[]))
    source_image_cropped  = bokeh.models.ColumnDataSource(dict(img=[]))
    source_roi = bokeh.models.ColumnDataSource(dict(left=[], right=[], top=[], bottom=[]))
    source_roi_pred = bokeh.models.ColumnDataSource(dict(left=[], right=[], top=[], bottom=[]))

    cell_types = ["normal",  "dead", "elongated", "flat", "dividing"]
    select_cell_label = bokeh.models.Select(title="Cell label", value=cell_types[0], options=cell_types)
    train_set = ["train",  "valid"]
    select_train_set = bokeh.models.Select(title="Image set", value=train_set[0], options=train_set)    

    quad = bokeh.models.Quad(left='left', right='right', top='top', bottom='bottom', fill_color=None, line_color="red", line_width=2)
    quad_pred = bokeh.models.Quad(left='left', right='right', top='top', bottom='bottom', fill_color=None, line_color="white", line_width=3)

    cell_label = bokeh.models.Div(text="")
    n_images = bokeh.models.Div(text="")
    n_images_detect = bokeh.models.Div(text="")
    n_images_label = bokeh.models.Div(text="")

    source_scores_detect = bokeh.models.ColumnDataSource(dict(height=[], width=[], names=[]))
    scores_detect = bokeh.models.LabelSet(x='width', y='height', text='names', x_units='data', y_units='data', x_offset=0, y_offset=0, source=source_scores_detect, text_color='white', text_font_size="10pt")

    source_scores_label = bokeh.models.ColumnDataSource(dict(height=[], width=[], names=[]))
    scores_label = bokeh.models.LabelSet(x='width', y='height', text='names', x_units='data', y_units='data', x_offset=0, y_offset=0, source=source_scores_label, text_color='white', text_font_size="10pt")


    #___________________________________________________________________________________________
    def set_numbers():
        n_valid_detect   = 0
        n_invalid_detect = 0
        n_notano_detect  = 0
        n_valid_label   = 0
        n_invalid_label = 0
        n_notano_label  = 0
        if select_train_set.value == "train":
            n_images.text = "<b style='color:black; ; font-size:18px;'> N images={} </b>".format(len(map_img_pos_train))
            for img in annot_dict_train:
                try:
                    if annot_dict_train[img]['dict']['valid_detect']==True:
                        n_valid_detect+=1
                    else:
                        n_invalid_detect+=1
                except KeyError:
                    n_notano_detect+=1
                try:
                    if annot_dict_train[img]['dict']['valid_label']==True:
                        n_valid_label+=1
                    else:
                        n_invalid_label+=1
                except KeyError:
                    n_notano_label+=1
            n_images_detect.text = "<b style='color:black; ; font-size:18px;'> Detect: valid={} invalid={} missing={}</b>".format(n_valid_detect, n_invalid_detect, n_notano_detect)
            n_images_label.text  = "<b style='color:black; ; font-size:18px;'> Label:  valid={} invalid={} missing={}</b>".format(n_valid_label, n_invalid_label, n_notano_label)

        elif select_train_set.value == "valid":
            n_images.text = "<b style='color:black; ; font-size:18px;'> N images={} </b>".format(len(map_img_pos_valid))
            for img in annot_dict_valid:
                try:
                    if annot_dict_valid[img]['dict']['valid_detect']==True:
                        n_valid_detect+=1
                    else:
                        n_invalid_detect+=1
                except KeyError:
                    n_notano_detect+=1
                try:
                    if annot_dict_valid[img]['dict']['valid_label']==True:
                        n_valid_label+=1
                    else:
                        n_invalid_label+=1
                except KeyError:
                    n_notano_label+=1
            n_images_detect.text = "<b style='color:black; ; font-size:18px;'> Detect: valid={} invalid={} missing={}</b>".format(n_valid_detect, n_invalid_detect, n_notano_detect)
            n_images_label.text  = "<b style='color:black; ; font-size:18px;'> Label:  valid={} invalid={} missing={}</b>".format(n_valid_label, n_invalid_label, n_notano_label)
        

    #___________________________________________________________________________________________
    def build_dict(folder_path):
        print('build_dict  ',folder_path)
        image_paths = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('_annotation.json')])
        titles = [os.path.split(t.replace('_annotation.json',''))[1] for t in image_paths]
        for idx,fname in enumerate(image_paths):
            with open(fname, 'r') as f:
                data = json.load(f)
                if os.path.split(folder_path)[-1]=='train':
                    annot_dict_train[titles[idx]]={'dict':data,'file':fname}
                    image_dict_train[titles[idx]]=None
                    image_cropped_dict_train[titles[idx]]=None
                    map_img_pos_train[idx]=titles[idx]

                elif os.path.split(folder_path)[-1]=='valid':
                    annot_dict_valid[titles[idx]]={'dict':data,'file':fname}
                    image_dict_valid[titles[idx]]=None
                    image_cropped_dict_valid[titles[idx]]=None
                    map_img_pos_valid[idx]=titles[idx]

    #___________________________________________________________________________________________
    def normalise(data):
        image = np.array(data)
        max_value = np.max(image)
        min_value = np.min(image)
        intensity_normalized = (image - min_value)/(max_value-min_value)*255
        intensity_normalized = intensity_normalized.astype(np.uint8)
        return intensity_normalized

    #___________________________________________________________________________________________
    def get_images_thread(input_dict, val):
        for img in input_dict:            
            with open(input_dict[img]["dict"]["image_json"], 'r') as f:
                print(val, '  ===  ',img)
                data = json.load(f)
                image = normalise(data["data"])
                image_cropped = normalise(data["data_cropped"])

                if val == "train":
                    image_dict_train[img]=image
                    image_cropped_dict_train[img]=image_cropped
                elif val == "valid":
                    image_dict_valid[img]=image
                    image_cropped_dict_valid[img]=image_cropped

    #___________________________________________________________________________________________
    def get_images(input_dict, val):
        threads=[]
        thread_dic={}
        target=10
        num=0
        for img in input_dict:
            thread_dic[img]=input_dict[img]
            num+=1
            if num==target:
                threads.append(threading.Thread(target = get_images_thread, args=(thread_dic, val, )))
                num=0
                thread_dic={}
        if num!=0 and num<target:threads.append(threading.Thread(target = get_images_thread, args=(thread_dic, val, )))
        for t in threads: 
            t.start()
            print('start thread ',t)
        for t in threads: t.join()


    #___________________________________________________________________________________________
    def update_image(way=1, number=-9999):
        current_index = slider.value
        nimg = len(annot_dict_train)
        if select_train_set.value == 'valid':
            nimg = len(annot_dict_valid)

        current_index = (current_index + 1*way) % nimg
        if current_index>slider.end:current_index=slider.start
        if current_index<slider.start:current_index=slider.end

        if number>=0:
            current_index = number
        slider.value = current_index


    #___________________________________________________________________________________________
    def fill_color(input_dict, fig_img, key, button):
        try:
            if input_dict[key]==True:
                fig_img.background_fill_color = 'rgba(0, 255, 0, 0.4)'
                fig_img.border_fill_color     = 'rgba(0, 255, 0, 0.4)'
                if key=="valid_detect":
                    button.label     = "Invalid detect"
                elif key=="valid_label":
                    button.label     = "Invalid label"
            elif input_dict[key]==False:
                fig_img.background_fill_color = 'rgba(255, 0, 0, 0.4)'
                fig_img.border_fill_color     = 'rgba(255, 0, 0, 0.4)'
                if key=="valid_detect":
                    button.label     = "Valid detect"
                elif key=="valid_label":
                    button.label     = "Valid label"
        except KeyError:
                fig_img.background_fill_color = 'white'
                fig_img.border_fill_color     = 'white'
                if key=="valid_detect":
                    button.label     = "Valid detect"
                elif key=="valid_label":
                    button.label     = "Valid label"


    #___________________________________________________________________________________________
    def callback_slider(attr: str, old: Any, new: Any) -> None:
        time_point = slider.value
        if select_train_set.value=='train':
            cell_label.text = "<b style='color:black; ; font-size:18px;'> {} </b>".format(map_img_pos_train[time_point])
            source_image.data = {'img':[image_dict_train[map_img_pos_train[time_point]]]}
            source_image_cropped.data = {'img':[image_cropped_dict_train[map_img_pos_train[time_point]]]}
            source_roi.data = {'left':[annot_dict_train[map_img_pos_train[time_point]]['dict']['bbox'][0]], 'right' :[annot_dict_train[map_img_pos_train[time_point]]['dict']['bbox'][1]], 
                               'top' :[annot_dict_train[map_img_pos_train[time_point]]['dict']['bbox'][2]], 'bottom':[annot_dict_train[map_img_pos_train[time_point]]['dict']['bbox'][3]]}
            select_cell_label.value = annot_dict_train[map_img_pos_train[time_point]]['dict']['label']
            fill_color(annot_dict_train[map_img_pos_train[slider.value]]['dict'], fig_img, 'valid_detect', valid_detect_button)
            fill_color(annot_dict_train[map_img_pos_train[slider.value]]['dict'], fig_img_cropped, 'valid_label', valid_label_button)
            
            source_roi_pred.data = {'left':[], 'right' :[], 'top' :[], 'bottom':[]}
            source_scores_detect.data = {'height':[], 'width':[], 'names':[]}
            source_scores_label.data = {'height':[], 'width':[], 'names':[]}

        elif select_train_set.value=='valid':
            cell_label.text = "<b style='color:black; ; font-size:18px;'> {} </b>".format(map_img_pos_valid[time_point])
            source_image.data = {'img':[image_dict_valid[map_img_pos_valid[time_point]]]}
            source_image_cropped.data = {'img':[image_cropped_dict_valid[map_img_pos_valid[time_point]]]}
            source_roi.data = {'left':[annot_dict_valid[map_img_pos_valid[time_point]]['dict']['bbox'][0]], 'right':[annot_dict_valid[map_img_pos_valid[time_point]]['dict']['bbox'][1]], 
                               'top':[annot_dict_valid[map_img_pos_valid[time_point]]['dict']['bbox'][2]], 'bottom':[annot_dict_valid[map_img_pos_valid[time_point]]['dict']['bbox'][3]]}
            select_cell_label.value = annot_dict_valid[map_img_pos_valid[time_point]]['dict']['label']
            fill_color(annot_dict_valid[map_img_pos_valid[slider.value]]['dict'], fig_img, 'valid_detect', valid_detect_button)
            fill_color(annot_dict_valid[map_img_pos_valid[slider.value]]['dict'], fig_img_cropped, 'valid_label', valid_label_button)

            print('=-----------  ',image_dict_valid[map_img_pos_valid[time_point]].shape)
            image = preprocess_image_pytorch(image_dict_valid[map_img_pos_valid[time_point]]).to(device)
            image_cropped = preprocess_image_pytorch(image_cropped_dict_valid[map_img_pos_valid[time_point]]).to(device)
            with torch.no_grad():
                predictions = model_detect(image)
                labels = model_label(image_cropped)
                #labels = model_label_resnet(image_cropped)

                print('labels        : ',labels)
                probabilities = F2.softmax(labels, dim=1)
                print('probabilities : ',probabilities)
                print('predictions   : ',predictions)
                left=[]
                right=[]
                top=[]
                bottom=[]
                height=[]
                width=[]
                names=[]

                for idx, box in enumerate(predictions[0]['boxes']):
                    x_min, y_min, x_max, y_max = box.cpu().numpy()
                    left.append(x_min)
                    right.append(x_max)
                    top.append(y_min)
                    bottom.append(y_max)
                    height.append(y_min)
                    width.append(x_min)
                    names.append(f"{predictions[0]['scores'][idx].cpu().numpy() :.3f}")
                    #plt.text(x_min, y_min-10, f"{predictions[0]['scores'][idx].cpu().numpy() :.3f}", fontsize=10, color='white')
        
                source_roi_pred.data = {'left':left, 'right' :right, 'top' :top, 'bottom':bottom}
                source_scores_detect.data = {'height':height, 'width':width, 'names':names}


                pred_label = labels_map[int(torch.argmax(labels, dim=1)[0].cpu().numpy())]
                print('pred_label = ',pred_label)
                pred_label_proba = probabilities.cpu().numpy()
                print('probabilities.cpu().numpy()  : ',probabilities.cpu().numpy())
                pred_label_proba = pred_label_proba[0][torch.argmax(labels, dim=1)]
                print('pred_label_proba = ',pred_label_proba)

                source_scores_label.data = {'height':[10], 'width':[10], 'names':[f"{pred_label} {pred_label_proba :.3f}"]}


    #___________________________________________________________________________________________
    def callback_label(attr: str, old: Any, new: Any) -> None:
        if select_train_set.value=="train":
            annot_dict_train[map_img_pos_train[slider.value]]['dict']['label']=new
            out_file = open(annot_dict_train[map_img_pos_train[slider.value]]['file'], "w") 
            json.dump(annot_dict_train[map_img_pos_train[slider.value]]['dict'], out_file) 
            out_file.close()
        elif select_train_set.value=="valid":
            annot_dict_valid[map_img_pos_valid[slider.value]]['dict']['label']=new
            out_file = open(annot_dict_valid[map_img_pos_valid[slider.value]]['file'], "w") 
            json.dump(annot_dict_valid[map_img_pos_valid[slider.value]]['dict'], out_file) 
            out_file.close()
    select_cell_label.on_change('value', callback_label)

    #___________________________________________________________________________________________
    def next_callback():
        update_image()
    button_next = bokeh.models.Button(label="Next")
    button_next.on_click(next_callback)

    #___________________________________________________________________________________________
    def prev_callback():
        update_image(-1)
    button_prev = bokeh.models.Button(label="Prev")
    button_prev.on_click(prev_callback)
    
    #___________________________________________________________________________________________
    def valid_detect_callback():

        if select_train_set.value=="train":
            annot_dict_train[map_img_pos_train[slider.value]]['dict']['valid_detect']=True if valid_detect_button.label == "Valid detect" else False
            out_file = open(annot_dict_train[map_img_pos_train[slider.value]]['file'], "w") 
            json.dump(annot_dict_train[map_img_pos_train[slider.value]]['dict'], out_file) 
            out_file.close()
        elif select_train_set.value=="valid":
            annot_dict_valid[map_img_pos_valid[slider.value]]['dict']['valid_detect']=True if valid_detect_button.label == "Valid detect" else False
            out_file = open(annot_dict_valid[map_img_pos_valid[slider.value]]['file'], "w") 
            json.dump(annot_dict_valid[map_img_pos_valid[slider.value]]['dict'], out_file) 
            out_file.close()

        if valid_detect_button.label == "Valid detect":
            fig_img.background_fill_color = 'rgba(0, 255, 0, 0.4)'
            fig_img.border_fill_color     = 'rgba(0, 255, 0, 0.4)'
            valid_detect_button.label = "Invalid detect"
        elif valid_detect_button.label == "Invalid detect":
            fig_img.background_fill_color = 'rgba(255, 0, 0, 0.4)'
            fig_img.border_fill_color     = 'rgba(255, 0, 0, 0.4)'
            valid_detect_button.label = "Valid detect"
        #set_numbers()

    valid_detect_button = bokeh.models.Button(label="")
    valid_detect_button.on_click(valid_detect_callback)
    valid_detect_button.on_click(set_numbers)

    #___________________________________________________________________________________________
    def valid_label_callback():
        if select_train_set.value=="train":
            annot_dict_train[map_img_pos_train[slider.value]]['dict']['valid_label']=True if valid_label_button.label == "Valid label" else False
            out_file = open(annot_dict_train[map_img_pos_train[slider.value]]['file'], "w") 
            json.dump(annot_dict_train[map_img_pos_train[slider.value]]['dict'], out_file) 
            out_file.close()
        elif select_train_set.value=="valid":
            annot_dict_valid[map_img_pos_valid[slider.value]]['dict']['valid_label']=True if valid_label_button.label == "Valid label" else False
            out_file = open(annot_dict_valid[map_img_pos_valid[slider.value]]['file'], "w") 
            json.dump(annot_dict_valid[map_img_pos_valid[slider.value]]['dict'], out_file) 
            out_file.close()

        if valid_label_button.label == "Valid label":
            fig_img_cropped.background_fill_color = 'rgba(0, 255, 0, 0.4)'
            fig_img_cropped.border_fill_color     = 'rgba(0, 255, 0, 0.4)'
            valid_label_button.label = "Invalid label"
        elif valid_label_button.label == "Invalid label":
            fig_img_cropped.background_fill_color = 'rgba(255, 0, 0, 0.4)'
            fig_img_cropped.border_fill_color     = 'rgba(255, 0, 0, 0.4)'
            valid_label_button.label = "Valid label"
        #set_numbers()
    valid_label_button = bokeh.models.Button(label="")
    valid_label_button.on_click(valid_label_callback)
    valid_label_button.on_click(set_numbers)

    #___________________________________________________________________________________________
    def callback_set(attr: str, old: Any, new: Any) -> None:
        if slider.value == 0:
            callback_slider(None, None, None)
        else:
            slider.value = 0

        if select_train_set.value=="train":
            slider.end = len(annot_dict_train) - 1
        else:
            slider.end = len(annot_dict_valid) - 1
        set_numbers()
    select_train_set.on_change('value', callback_set)


    folder_path = os.path.join(MVA_OUTDIR, 'single_cells/training_cell_detection_categories')
    build_dict(os.path.join(MVA_OUTDIR, 'single_cells/training_cell_detection_categories', 'train'))
    build_dict(os.path.join(MVA_OUTDIR, 'single_cells/training_cell_detection_categories', 'valid'))
    get_images(annot_dict_train, "train")
    get_images(annot_dict_valid, "valid")


    initial_time_point = 0
    slider  = bokeh.models.Slider(start=0, end=len(annot_dict_train) - 1, value=initial_time_point, step=1, title="image", width=250)
    slider.on_change('value', callback_slider)


    first_key = list(annot_dict_train.keys())[0]
    source_image.data = {'img':[image_dict_train[first_key]]}
    source_image_cropped.data = {'img':[image_cropped_dict_train[first_key]]}
    source_roi.data = {'left':[annot_dict_train[first_key]['dict']['bbox'][0]], 'right':[annot_dict_train[first_key]['dict']['bbox'][1]], 
                       'top':[annot_dict_train[first_key]['dict']['bbox'][2]], 'bottom':[annot_dict_train[first_key]['dict']['bbox'][3]]}



    color_mapper = bokeh.models.LinearColorMapper(palette="Greys256", low=source_image.data["img"][0].min(), high=source_image.data["img"][0].max())
    x_range = bokeh.models.Range1d(start=0, end=source_image.data["img"][0].shape[0])
    y_range = bokeh.models.Range1d(start=0, end=source_image.data["img"][0].shape[1])
    fig_img = bokeh.plotting.figure(x_range=x_range, y_range=y_range,  width=500, height=500, tools="box_select,wheel_zoom,box_zoom,reset,undo", title='detect ROI valid')
    fig_img.axis.visible = False
    fig_img.grid.visible = False
    fig_img.add_layout(scores_detect)

    fig_img.image(image='img', x=0, y=0, dw=source_image.data["img"][0].shape[0], dh=source_image.data["img"][0].shape[1], color_mapper=color_mapper, source=source_image)
    fig_img.add_glyph(source_roi, quad, selection_glyph=quad, nonselection_glyph=quad)
    fig_img.add_glyph(source_roi_pred, quad_pred, selection_glyph=quad_pred, nonselection_glyph=quad_pred)
    cell_label.text = "<b style='color:black; ; font-size:18px;'> {} </b>".format(first_key)
    select_cell_label.value = annot_dict_train[first_key]['dict']['label']
    fill_color(annot_dict_train[first_key]['dict'], fig_img, 'valid_detect', valid_detect_button)

    x_range_cropped = bokeh.models.Range1d(start=0, end=source_image_cropped.data["img"][0].shape[0])
    y_range_cropped = bokeh.models.Range1d(start=0, end=source_image_cropped.data["img"][0].shape[1])
    fig_img_cropped = bokeh.plotting.figure(x_range=x_range_cropped, y_range=y_range_cropped,  width=500, height=500, tools="box_select,wheel_zoom,box_zoom,reset,undo", title='cell label ID')
    fig_img_cropped.axis.visible = False
    fig_img_cropped.grid.visible = False
    fig_img_cropped.image(image='img', x=0, y=0, dw=source_image_cropped.data["img"][0].shape[0], dh=source_image_cropped.data["img"][0].shape[1], color_mapper=color_mapper, source=source_image_cropped)
    fig_img_cropped.add_layout(scores_label)
    fill_color(annot_dict_train[first_key]['dict'], fig_img_cropped, 'valid_label', valid_label_button)

    set_numbers()

    select_col = bokeh.layouts.column(bokeh.layouts.row(select_train_set), slider, bokeh.layouts.row(button_prev, button_next, select_cell_label), bokeh.layouts.row(valid_detect_button, valid_label_button))
    info_col   = bokeh.layouts.column(cell_label,n_images,n_images_detect, n_images_label)
    layout=bokeh.layouts.column(bokeh.layouts.row(fig_img,fig_img_cropped,select_col, info_col))
    doc.add_root(layout)








#___________________________________________________________________________________________
#___________________________________________________________________________________________
#___________________________________________________________________________________________
def summary_handler(doc: bokeh.document.Document) -> None:
    print('****************************  summary_handler ****************************')
    #TO BE CHANGED WITH ASYNC?????
    start_time=datetime.datetime.now()
    os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"
    experiments=[]
    wells={}
    positions={}
    files={}

    for exp in Experiment.objects.all():
        experiments.append(exp.name)
        wells[exp.name] = []
        experimentaldatasets = ExperimentalDataset.objects.select_related().filter(experiment = exp)
        for expds in experimentaldatasets:
            wells[exp.name].append(expds.data_name)
            samples = Sample.objects.select_related().filter(experimental_dataset = expds)
            positions['{0}_{1}'.format(exp.name, expds.data_name)] = []
            files['{0}_{1}'.format(exp.name, expds.data_name)] = []
            for samp in samples:
                positions['{0}_{1}'.format(exp.name, expds.data_name)].append(samp.file_name.split('/')[-1])
                files    ['{0}_{1}'.format(exp.name, expds.data_name)].append(samp.file_name)

    experiments=sorted(experiments)
    for i in wells:
        wells[i] = sorted(wells[i])
    for i in positions:
        positions[i] = sorted(positions[i])
        files[i]     = sorted(files[i])


    dropdown_exp  = bokeh.models.Select(value=experiments[0], title='Experiment', options=experiments)
    dropdown_well = bokeh.models.Select(value=wells[experiments[0]][0], title='Well', options=wells[dropdown_exp.value])
    dropdown_grid = bokeh.models.Select(value='5', title='Grid', options=['3','4','5','6','7','8'])
    dropdown_intensity_type = bokeh.models.Select(value='mean', title='intensity', options=['mean','max','std','sum'])
    checkbox_yrange = bokeh.models.CheckboxGroup(labels=["Same y-range"], active=[1])
    checkbox_tod    = bokeh.models.CheckboxGroup(labels=["Predict ToD"], active=[1])
    intensity_map = {'max':'intensity_max', 
                     'mean':'intensity_mean',
                     'std':'intensity_std',
                     'sum':'intensity_sum'}
    color_map=['blue', 'red']
    num_plots = len(wells[dropdown_exp.value])
    grid_size = 5

    #___________________________________________________________________________________________
    def create_plots(num):
        plots = []
        for i in range(num):
            p = bokeh.plotting.figure(width=400, height=200, title=f"Plot {i+1}")
            p.circle([1, 2, 3], [1, 4, 9])
            plots.append(p)
        return plots
    #___________________________________________________________________________________________


    #___________________________________________________________________________________________
    def create_grid(plots, grid_size):
        return bokeh.layouts.gridplot([plots[i:i + grid_size] for i in range(0, len(plots), grid_size)])
    #___________________________________________________________________________________________


    plots = create_plots(num_plots)
    grid  = create_grid(plots, grid_size)

    #___________________________________________________________________________________________
    def update_plot(attr, old, new):
        selected_experiment  = Experiment.objects.get(name=dropdown_exp.value)
        experimentaldataset  = ExperimentalDataset.objects.select_related().filter(experiment = selected_experiment).get(data_name = dropdown_well.value)
        samples = Sample.objects.select_related().filter(experimental_dataset = experimentaldataset)
        selected_positons    = positions['{0}_{1}'.format(dropdown_exp.value, dropdown_well.value)]
        selected_num_plots  = len(selected_positons)
        print('selected_num_plots=',selected_num_plots)
        print('selected_experiment=',selected_experiment)
        print('experimentaldataset=',experimentaldataset)
        print('samples=',samples)

        intensity_traces = {}
        for samp in samples:
            sample=samp.file_name.split('/')[-1]
            print('       ---- sample ',sample)
            intensity_traces[sample]={}
            cellsID = CellID.objects.select_related().filter(sample=samp)
            for cellID in cellsID:
                intensity_traces[sample][cellID.name]={}
                intensity_traces[sample][cellID.name]["ROI"]={}
                time           = []
                intensity_max  = []
                intensity_mean = []
                intensity_std  = []
                intensity_sum  = []
                file_name      = []
                frame_number   = []

                cellsROI = CellROI.objects.select_related().filter(cell_id=cellID)
                for cellROI in cellsROI:
                    time.append(cellROI.frame.time)
                    intensity_max.append(cellROI.contour_cellroi.intensity_max)
                    intensity_mean.append(cellROI.contour_cellroi.intensity_mean)
                    intensity_std.append(cellROI.contour_cellroi.intensity_std)
                    intensity_sum.append(cellROI.contour_cellroi.intensity_sum)
                    file_name.append(cellROI.contour_cellroi.file_name)
                    frame_number.append(cellROI.frame.number)

                sorted_lists    = list(zip(time, intensity_max,intensity_mean, intensity_sum, intensity_std, file_name, frame_number)) 
                sorted_combined = sorted(sorted_lists, key=lambda x: x[0])

                time_sorted           = [x[0] for x in sorted_combined]
                intensity_max_sorted  = [x[1] for x in sorted_combined]
                intensity_mean_sorted = [x[2] for x in sorted_combined]
                intensity_sum_sorted  = [x[3] for x in sorted_combined]
                intensity_std_sorted  = [x[4] for x in sorted_combined]
                file_name_sorted      = [x[5] for x in sorted_combined]
                frame_number_sorted   = [x[6] for x in sorted_combined]

                intensity_traces[sample][cellID.name]["ROI"]["time"]           = time_sorted
                intensity_traces[sample][cellID.name]["ROI"]["intensity_max"]  = intensity_max_sorted
                intensity_traces[sample][cellID.name]["ROI"]["intensity_mean"] = intensity_mean_sorted
                intensity_traces[sample][cellID.name]["ROI"]["intensity_std"]  = intensity_std_sorted
                intensity_traces[sample][cellID.name]["ROI"]["intensity_sum"]  = intensity_sum_sorted
                intensity_traces[sample][cellID.name]["ROI"]["file_name"]      = file_name_sorted
                intensity_traces[sample][cellID.name]["ROI"]["frame_number"]   = frame_number_sorted

        new_plots = []

        #get min/max
        y_min=9999999999999
        y_max=0
        for i in range(selected_num_plots):
            for cell in intensity_traces[selected_positons[i]]:
                for t in range(len(intensity_traces[selected_positons[i]][cell]['ROI']['time'])):
                    for ch in intensity_traces[selected_positons[i]][cell]['ROI'][intensity_map[dropdown_intensity_type.value]][t]:
                        if 'BF' in ch:continue
                        val=intensity_traces[selected_positons[i]][cell]['ROI'][intensity_map[dropdown_intensity_type.value]][t][ch]
                        if val<y_min: y_min=val
                        if val>y_max: y_max=val
        shared_y_range = bokeh.models.Range1d(start=y_min, end=y_max)

        for i in range(selected_num_plots):

            p = None
            if 0 in checkbox_yrange.active:
                p = bokeh.plotting.figure(width=400, height=200, y_range=shared_y_range, title=f"{dropdown_exp.value} {selected_positons[i].split('_')[-1].replace('.nd2','')} ncells={len(intensity_traces[selected_positons[i]])}")
            else:
                p = bokeh.plotting.figure(width=400, height=200, title=f"{dropdown_exp.value} {selected_positons[i].split('_')[-1].replace('.nd2','')} ncells={len(intensity_traces[selected_positons[i]])}")
            #print('selected_positons=',selected_positons[i], '  ncells ',len(intensity_traces[selected_positons[i]]))
            
            for cell in intensity_traces[selected_positons[i]]:
                time_list = []
                int_list  = {}
                frame_list = []
                file_list = []

                for ch in intensity_traces[selected_positons[i]][cell]['ROI'][intensity_map[dropdown_intensity_type.value]][0]:
                    int_list[ch]=[]
                for t in range(len(intensity_traces[selected_positons[i]][cell]['ROI']['time'])):
                    time_list.append(intensity_traces[selected_positons[i]][cell]['ROI']['time'][t]/60000.)
                    file_list.append(intensity_traces[selected_positons[i]][cell]['ROI']['file_name'][t])
                    frame_list.append(intensity_traces[selected_positons[i]][cell]['ROI']['frame_number'][t])
                    for ch in intensity_traces[selected_positons[i]][cell]['ROI'][intensity_map[dropdown_intensity_type.value]][t]:
                        int_list[ch].append(intensity_traces[selected_positons[i]][cell]['ROI'][intensity_map[dropdown_intensity_type.value]][t][ch])
                
                ch_num=0
                frame_num=-9999
                added_ch=False
                for ch in int_list:
                    if 'BF' in ch:
                        if 0 in checkbox_tod.active:
                            prediction = get_mva_prediction_alive(file_list)
                        #prediction_osc = get_mva_prediction_oscillating(file_list)
                            if prediction!=None:
                                file_name = prediction.split('/')[-1]
                                frame_num = int(file_name.split('_')[0].replace('frame',''))
                        continue
                    p.line(time_list, int_list[ch], line_color=color_map[ch_num])
                    if frame_num>-1 and added_ch==False and 0 in checkbox_tod.active:
                        added_ch=True
                        x = [time_list[t] for t in range(frame_list.index(frame_num), len(frame_list))]
                        y1 = [int_list[ch][t] for t in range(frame_list.index(frame_num), len(frame_list))]
                        y2 = [0 for t in range(frame_list.index(frame_num), len(frame_list))]
                        p.varea(x=x, y1=y1, y2=y2, fill_alpha=0.10, fill_color='black')
                    ch_num+=1

            new_plots.append(p)

        # Create a new grid with the updated number of plots
        new_grid = create_grid(new_plots, int(dropdown_grid.value))
        
        # Update the layout
        norm_layout.children[1] = new_grid
    dropdown_well.on_change('value', update_plot)
    #___________________________________________________________________________________________


    #___________________________________________________________________________________________
    def predict_tod_button(attr, old, new):
        update_plot('','','')
    checkbox_tod.on_change('active', predict_tod_button)
    #___________________________________________________________________________________________


    #___________________________________________________________________________________________
    def select_grid_size(attr, old, new):
        update_plot('','','')
    dropdown_grid.on_change('value', select_grid_size)
    #___________________________________________________________________________________________

    #___________________________________________________________________________________________
    def select_intensity_type(attr, old, new):
        update_plot('','','')
    dropdown_intensity_type.on_change('value', select_intensity_type)
    #___________________________________________________________________________________________

    #___________________________________________________________________________________________
    def update_dropdown_well(attr, old, new):
        if DEBUG: print('****************************  update_dropdown_well ****************************')
        dropdown_well.options = wells[dropdown_exp.value]
        dropdown_well.value   = wells[dropdown_exp.value][0]
    dropdown_exp.on_change('value', update_dropdown_well)
    #___________________________________________________________________________________________


    #___________________________________________________________________________________________
    def select_y_range(attr, old, new):
        update_plot('','','')
    checkbox_yrange.on_change('active', select_y_range)
    #___________________________________________________________________________________________

    exp_color_col = bokeh.layouts.column(bokeh.layouts.row(dropdown_exp),
                                         bokeh.layouts.row(dropdown_well),
                                         bokeh.layouts.row(dropdown_grid), 
                                         bokeh.layouts.row(dropdown_intensity_type),
                                         bokeh.layouts.row(checkbox_yrange),
                                         bokeh.layouts.row(checkbox_tod))



    norm_layout = bokeh.layouts.row(exp_color_col, grid)

    update_plot('','','')


    print('==========norm norm_layout.children[1]  ',norm_layout.children[1])

    doc.add_root(norm_layout)


selected_dict={
    'experiment':'',
    'well':'',
    'position':'',
    'segmentation':'',
    'segmentation_channel':''
}

#___________________________________________________________________________________________
def create_tarball(input_path, output_tarball):
    # Construct the tar command
    tar_command = ['tar', '-czf', output_tarball, '-C', input_path, '.']
    
    # Start the subprocess
    process = subprocess.Popen(tar_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for the process to complete and get the output and errors
    stdout, stderr = process.communicate()
    
    # Check if the process was successful
    if process.returncode == 0:
        print(f'Tarball created successfully at {output_tarball}')
    else:
        print(f'Error creating tarball: {stderr.decode("utf-8")}')
        

def delete_files_in_directory(directory_path):
   try:
     files = os.listdir(directory_path)
     for file in files:
       file_path = os.path.join(directory_path, file)
       if os.path.isfile(file_path):
         os.remove(file_path)
     print("All files deleted successfully.")
   except OSError:
     print("Error occurred while deleting files.")



#___________________________________________________________________________________________
@login_required
def index(request: HttpRequest) -> HttpResponse:
#def index(request):


    """View function for home page of site."""
    print('The visualisation request method is:', request.method)
    print('The visualisation POST data is:     ', request.POST)
    print('The visualisation GET data is:      ', request.GET)
    cell_dict={}


    #CLEMENT DOES NOT work yet
    #try to reconnect the cnx cursor
    if LOCAL==False:
        try:
            mycursor = cnx.cursor()
        except mysql.connector.Error as err:
            print('err: ',err)
            cnx.reconnect()
    
    #THIS BUILDS THE FRAMES FROM THE RAWDATASET CATALOG WITH TAG "SEGMENTME", CREATES UP TO FRAMES
    if 'register_rawdataset' in request.POST and LOCAL==False:
        register_rawdataset()

    if 'fix_tod' in request.POST:
        fix_alive_status()

    #dictionary to provide possible selection choices
    select_dict={
        'experiment_list':[],
        'well_list':[],
        'position_list':[],
    }

    #dictionary that maps the database
    experiment_dict={        
        'experiments':[]
    }

    #build for front page
    for exp in Experiment.objects.all():
        print(' ---- Experiment name ',exp.name)
        experimentaldataset = ExperimentalDataset.objects.select_related().filter(experiment = exp)
        select_dict['experiment_list'].append(exp.name)
        expds_list=[]
        for expds in experimentaldataset:
            print('    ---- experimental dataset name ',expds.data_name)
            samples = Sample.objects.select_related().filter(experimental_dataset = expds)
            sample_list=[]
            for s in samples:
                sample_list.append(s.file_name)
            expds_list.append({'data_name':expds.data_name, 'data_type':expds.data_type, 'files':sample_list})
            #del samples
        tmp_exp_dict={'name':exp.name, 'wells':expds_list}
        experiment_dict['experiments'].append(tmp_exp_dict)
        #del experimentaldataset



    #dictionary to keep the selected choices, this is for the front end page


    selected_experiment=request.POST.get('select_experiment')
    if selected_experiment!=None:selected_dict['experiment']=selected_experiment
    selected_well=request.POST.get('select_well')
    selected_dict['well']=selected_well
    selected_position=request.POST.get('select_position')
    selected_dict['position']=selected_position
    selected_segmentation=request.POST.get('select_segmentation')
    selected_dict['segmentation']=selected_segmentation
    selected_segmentation_channel=request.POST.get('select_segmentation_channel')
    selected_dict['segmentation_channel']=selected_segmentation_channel

    if 'change_files' in request.POST:
        change_file_paths()


    #THIS BUILDS THE ROIS FOR ALL THE EXISTING SAMPLES
    if 'build_ROIs' in request.POST:
        #build_ROIs_loop(selected_dict['experiment'])
        print('selected_dict[experiment]'  ,selected_dict['experiment'])

        if selected_dict['experiment'] == None or selected_dict['experiment'] == '':
            print('no experiment selected')
        else:
            build_ROIs_loop_parallel(selected_dict['experiment'])

    if 'predict_tod' in request.POST:
        print('selected_dict[experiment]'  ,selected_dict['experiment'])

        if selected_dict['experiment'] == None or selected_dict['experiment'] == '':
            print('no experiment selected')
        else:
            build_tod_loop_parallel(selected_dict['experiment'])


    if 'build_mva_detection_categories' in request.POST:
        build_mva_detection_categories()

    #THIS SEGMENTS ALL THE EXPERIMENTS/POSITIONS IT WILL FIND. CREATES UP TO CONTOUR/DATA
    if 'segment' in request.POST:
        if selected_dict['experiment'] == None or selected_dict['experiment'] == '':
            print('no experiment selected')
        else:
            #build_segmentation_parallel(selected_dict['experiment'])
            build_segmentation(selected_dict['experiment'])


    if selected_experiment!='':
        for e in experiment_dict['experiments']:
            if e['name']!=selected_experiment:continue
            print('experiment selected=',e['name'])
            for d in e['wells']:
                select_dict['well_list'].append(d['data_name'])
            if selected_well!='':
                for s in e['wells']:
                    if s['data_name']!=selected_well:continue
                    print('well selected=',s['data_name'])
                    for f in s['files']:
                        #select_dict['file_list'].append(os.path.split(f)[-1])
                        select_dict['position_list'].append(f)

    select_dict['experiment_list']=sorted(select_dict['experiment_list'])
    select_dict['well_list']=sorted(select_dict['well_list'])
    select_dict['position_list']=sorted(select_dict['position_list'])
    #print('experiment_dict     =  ', experiment_dict)
    #print('selected_project =  ', selected_project)
    print('selected_dict    =  ', selected_dict)
    #print('select_dict      =  ', select_dict)

    # Render the HTML template index.html with the data in the context variable


    experiment_dict={}
    contribution_dict=[]
    ##CLEMENT UNCOMMENT WHEN DB CONNECTION UNDERSTOOD
    if selected_experiment != None:
        ##GET THE LIST OF EXPERIMENTS
        experiment_dict=get_experiment_details(selected_experiment)
        ##GET THE CONTRIBUTION TO THE EXPERIMENT 
        contribution_dict=get_contribution_details(selected_experiment)

    treatment_dict=[]
    injection_dict=[]
    instrumental_dict=[]
    sample_dict=[]
    ##CLEMENT UNCOMMENT WHEN DB CONNECTION UNDERSTOOD
    if selected_well != None:
        ##GET THE EXPERIMENTAL DATASET DETAILS: TREATMENT
        treatment_dict = get_treatment_details(selected_well)
        ##GET THE EXPERIMENTAL DATASET DETAILS: INSTRUMENTAL CONDITIONS
        injection_dict = get_injection_details(selected_well)
        ##GET THE EXPERIMENTAL DATASET DETAILS: INJECTION
        instrumental_dict = get_instrumental_details(selected_well)
        ##GET THE EXPERIMENTAL DATASET DETAILS: SAMPLE
        sample_dict = get_sample_details(selected_well)
    


    delete_files_in_directory('/data/tmp/')
    if 'prepare_data_files' in request.POST:
        print('in prepare data')
        print('selected_experiment=',selected_experiment)
        print('selected_well=',selected_well)
        for experiment in Experiment.objects.all():
            exp=experiment.name
            if (selected_experiment!='') and selected_experiment!=exp: continue

            tardir=os.path.join('/data/singleCell_catalog/contour_data', exp)
            output_tarball = '/data/tmp/data_{}_contours.tar.gz'.format(exp)  # The output tarball file name
            create_tarball(tardir, output_tarball)
            if os.path.exists(output_tarball):
               with open(output_tarball, 'rb') as f:
                response = HttpResponse(f.read(), content_type='application/gzip')
                response['Content-Disposition'] = 'attachment; filename="{}"'.format(os.path.basename(output_tarball))
                return response
            else:
                return HttpResponse("File not found.", status=404)



    #build the output json
    download_dict = {}

    if 'prepare_data' in request.POST:
        print('in prepare data')
        print('selected_experiment=',selected_experiment)
        print('selected_well=',selected_well)
        for experiment in Experiment.objects.all():
            exp=experiment.name

            if (selected_experiment!='') and selected_experiment!=exp: continue
            download_dict[exp]={}
            print(' ---- Experiment name ',exp)
            experimentaldataset = ExperimentalDataset.objects.select_related().filter(experiment = experiment)
            for exerimentalpds in experimentaldataset:
                expds=exerimentalpds.data_name
                #if (selected_well!='') and selected_well!=expds: continue
                download_dict[exp][expds]={}
                print('    ---- experimental dataset name ',expds)
                samples = Sample.objects.select_related().filter(experimental_dataset = exerimentalpds)
                for samp in samples:
                    sample=samp.file_name.split('/')[-1]
                    print('       ---- sample ',sample)
                    download_dict[exp][expds][sample]={}
                    download_dict[exp][expds][sample]['keep_sample']=samp.keep_sample
                    download_dict[exp][expds][sample]['sample_quality']=samp.sample_quality
                    download_dict[exp][expds][sample]['peaks_tod_div_validated']=samp.peaks_tod_div_validated
                    download_dict[exp][expds][sample]['bf_features_validated']=samp.bf_features_validated
                    cellsID = CellID.objects.select_related().filter(sample=samp)
                    download_dict[exp][expds][sample]['cells']=[]
                    for cellID in cellsID:
                        download_dict[exp][expds][sample]['cells'].append(cellID.name)
                        download_dict[exp][expds][sample][cellID.name]={}
                        download_dict[exp][expds][sample][cellID.name]["start_oscillation_time"]=cellID.cell_status.start_oscillation
                        download_dict[exp][expds][sample][cellID.name]["end_oscillation_time"]=cellID.cell_status.end_oscillation
                        download_dict[exp][expds][sample][cellID.name]["start_oscillation_frame"]=cellID.cell_status.start_oscillation_frame
                        download_dict[exp][expds][sample][cellID.name]["end_oscillation_frame"]=cellID.cell_status.end_oscillation_frame
                        download_dict[exp][expds][sample][cellID.name]["mask"]=cellID.cell_status.mask
                        download_dict[exp][expds][sample][cellID.name]["migrating"]=cellID.cell_status.migrating
                        download_dict[exp][expds][sample][cellID.name]["n_oscillations"]=cellID.cell_status.n_oscillations
                        download_dict[exp][expds][sample][cellID.name]["time_of_death"]=cellID.cell_status.time_of_death
                        download_dict[exp][expds][sample][cellID.name]["time_of_death_frame"]=cellID.cell_status.time_of_death_frame
                        download_dict[exp][expds][sample][cellID.name]["peaks"]=cellID.cell_status.peaks
                        download_dict[exp][expds][sample][cellID.name]["flags"]=cellID.cell_status.flags
                        download_dict[exp][expds][sample][cellID.name]["ROI"]={}

                        download_dict[exp][expds][sample][cellID.name]["segmentation"]={}

                        for seg in cellID.cell_status.segmentation['algo']:
                            if seg=='roi':continue
                            download_dict[exp][expds][sample][cellID.name]["segmentation"][seg]={}

                        alive          = []
                        oscillating    = []
                        maximum        = []
                        minimum        = []
                        falling        = []
                        rising         = []
                        last_osc       = []
                        mask           = []
                        dividing       = []
                        double_nuclei  = []
                        multiple_cells = []
                        pair_cell      = []
                        flat           = []
                        round          = []
                        elongated      = []
                        time           = []

                        center_x_mic     = []
                        center_y_mic     = []
                        center_x_pix     = []
                        center_y_pix     = []
                        file_name        = []
                        intensity_max    = []
                        intensity_mean   = []
                        intensity_std    = []
                        intensity_sum    = []
                        mode             = []
                        number_of_pixels = []
                        type             = []

                        cellsROI = CellROI.objects.select_related().filter(cell_id=cellID)
                        for cellROI in cellsROI:
                            time.append(cellROI.frame.time)
                            alive.append(cellROI.cellflag_cellroi.alive)
                            oscillating.append(cellROI.cellflag_cellroi.oscillating)
                            maximum.append(cellROI.cellflag_cellroi.maximum)
                            minimum.append(cellROI.cellflag_cellroi.minimum)
                            falling.append(cellROI.cellflag_cellroi.falling)
                            rising.append(cellROI.cellflag_cellroi.rising)
                            last_osc.append(cellROI.cellflag_cellroi.last_osc)
                            mask.append(cellROI.cellflag_cellroi.mask)
                            dividing.append(cellROI.cellflag_cellroi.dividing)
                            double_nuclei.append(cellROI.cellflag_cellroi.double_nuclei)
                            multiple_cells.append(cellROI.cellflag_cellroi.multiple_cells)
                            pair_cell.append(cellROI.cellflag_cellroi.pair_cell)
                            flat.append(cellROI.cellflag_cellroi.flat)
                            round.append(cellROI.cellflag_cellroi.round)
                            elongated.append(cellROI.cellflag_cellroi.elongated)

                            center_x_mic.append(cellROI.contour_cellroi.center_x_mic)
                            center_y_mic.append(cellROI.contour_cellroi.center_y_mic)
                            center_x_pix.append(cellROI.contour_cellroi.center_x_pix)
                            center_y_pix.append(cellROI.contour_cellroi.center_y_pix)
                            file_name.append(cellROI.contour_cellroi.file_name)
                            intensity_max.append(cellROI.contour_cellroi.intensity_max)
                            intensity_mean.append(cellROI.contour_cellroi.intensity_mean)
                            intensity_std.append(cellROI.contour_cellroi.intensity_std)
                            intensity_sum.append(cellROI.contour_cellroi.intensity_sum)
                            mode.append(cellROI.contour_cellroi.mode)
                            number_of_pixels.append(cellROI.contour_cellroi.number_of_pixels)
                            type.append(cellROI.contour_cellroi.type)

                        sorted_lists = sorted(zip(time, alive, oscillating, maximum, minimum, falling, rising, last_osc, mask, dividing, double_nuclei, multiple_cells, pair_cell, flat, round, elongated)) 
                        time_sorted, alive_sorted, oscillating_sorted, maximum_sorted, minimum_sorted, falling_sorted, rising_sorted, last_osc_sorted, mask_sorted, dividing_sorted, double_nuclei_sorted, multiple_cells_sorted, pair_cell_sorted, flat_sorted, round_sorted, elongated_sorted = zip(*sorted_lists) 
                        download_dict[exp][expds][sample][cellID.name]["time"]           = time_sorted
                        download_dict[exp][expds][sample][cellID.name]["alive"]          = alive_sorted
                        download_dict[exp][expds][sample][cellID.name]["oscillating"]    = oscillating_sorted
                        download_dict[exp][expds][sample][cellID.name]["maximum"]        = maximum_sorted
                        download_dict[exp][expds][sample][cellID.name]["minimum"]        = minimum_sorted
                        download_dict[exp][expds][sample][cellID.name]["falling"]        = falling_sorted
                        download_dict[exp][expds][sample][cellID.name]["rising"]         = rising_sorted
                        download_dict[exp][expds][sample][cellID.name]["last_osc"]       = last_osc_sorted
                        download_dict[exp][expds][sample][cellID.name]["mask"]           = mask_sorted
                        download_dict[exp][expds][sample][cellID.name]["dividing"]       = dividing_sorted
                        download_dict[exp][expds][sample][cellID.name]["double_nuclei"]  = double_nuclei_sorted
                        download_dict[exp][expds][sample][cellID.name]["multiple_cells"] = multiple_cells_sorted
                        download_dict[exp][expds][sample][cellID.name]["pair_cell"]      = pair_cell_sorted
                        download_dict[exp][expds][sample][cellID.name]["flat"]           = flat_sorted
                        download_dict[exp][expds][sample][cellID.name]["round"]          = round_sorted
                        download_dict[exp][expds][sample][cellID.name]["elongated"]      = elongated_sorted
                        
                        sorted_lists2 = sorted(zip(time, center_x_mic, center_y_mic, center_x_pix, center_y_pix, file_name, intensity_max, intensity_mean, intensity_std, intensity_sum, mode, number_of_pixels, type))
                        time_sorted, center_x_mic_sorted, center_y_mic_sorted, center_x_pix_sorted, center_y_pix_sorted, file_name_sorted, intensity_max_sorted, intensity_mean_sorted, intensity_std_sorted, intensity_sum_sorted, mode_sorted, number_of_pixels_sorted, type_sorted = zip(*sorted_lists2)
                        download_dict[exp][expds][sample][cellID.name]["ROI"]["center_x_mic"]     = center_x_mic_sorted
                        download_dict[exp][expds][sample][cellID.name]["ROI"]["center_y_mic"]     = center_y_mic_sorted
                        download_dict[exp][expds][sample][cellID.name]["ROI"]["center_x_pix"]     = center_x_pix_sorted
                        download_dict[exp][expds][sample][cellID.name]["ROI"]["center_y_pix"]     = center_y_pix_sorted
                        download_dict[exp][expds][sample][cellID.name]["ROI"]["file_name"]        = file_name_sorted
                        download_dict[exp][expds][sample][cellID.name]["ROI"]["intensity_max"]    = intensity_max_sorted
                        download_dict[exp][expds][sample][cellID.name]["ROI"]["intensity_mean"]   = intensity_mean_sorted
                        download_dict[exp][expds][sample][cellID.name]["ROI"]["intensity_std"]    = intensity_std_sorted
                        download_dict[exp][expds][sample][cellID.name]["ROI"]["intensity_sum"]    = intensity_sum_sorted
                        download_dict[exp][expds][sample][cellID.name]["ROI"]["mode"]             = mode_sorted
                        download_dict[exp][expds][sample][cellID.name]["ROI"]["number_of_pixels"] = number_of_pixels_sorted
                        download_dict[exp][expds][sample][cellID.name]["ROI"]["type"]             = type_sorted

        json_content = json.dumps(download_dict)#, indent=4)
        #print('download_dict=',download_dict)
        response = HttpResponse(json_content, content_type='application/json')
        

        outfilename_download='data_AllExp.json'
        if selected_well!='': outfilename_download='data_{}.json'.format(selected_well)
        if selected_well=='' and selected_experiment!='': outfilename_download='data_{}.json'.format(selected_experiment)
        # Set the Content-Disposition header to specify the filename
        response['Content-Disposition'] = 'attachment; filename="{}"'.format(outfilename_download)

        return response

                    
    #build the output json
    download_dict_laurel = {}
    if 'prepare_data_laurel' in request.POST:
        print('in prepare data laurel')
        print('selected_experiment=',selected_experiment)
        print('selected_well=',selected_well)
        for exp in Experiment.objects.all():
            if 'wscepfl' not in exp.name: continue
            download_dict_laurel[exp.name]={}
            print(' ---- Experiment name ',exp.name)
            experimentaldataset = ExperimentalDataset.objects.select_related().filter(experiment = exp)
            for expds in experimentaldataset:
                download_dict_laurel[exp.name][expds.data_name]={}
                print('    ---- experimental dataset name ',expds.data_name)
                samples = Sample.objects.select_related().filter(experimental_dataset = expds)
                for samp in samples:
                    sample=samp.file_name.split('/')[-1]
                    print('       ---- sample ',sample)
                    download_dict_laurel[exp.name][expds.data_name][sample]={}            

                    if samp.keep_sample==False: continue
                    if samp.peaks_tod_div_validated==False: continue
                    
                    cellsID = CellID.objects.select_related().filter(sample=samp)
                    for cellID in cellsID:
                        print('           ---- cellid ',cellID.name)
                        download_dict_laurel[exp.name][expds.data_name][sample][cellID.name]={}
                        cellsROI = CellROI.objects.select_related().filter(cell_id=cellID)
                        time             = []
                        intensity_max    = []
                        intensity_mean   = []
                        intensity_std    = []
                        intensity_sum    = []
                        number_of_pixels = []

                        for cellROI in cellsROI:
                            if cellROI.cellflag_cellroi.alive==False: break
                            time.append(cellROI.frame.time)
                            intensity_max.append(cellROI.contour_cellroi.intensity_max)
                            intensity_mean.append(cellROI.contour_cellroi.intensity_mean)
                            intensity_std.append(cellROI.contour_cellroi.intensity_std)
                            intensity_sum.append(cellROI.contour_cellroi.intensity_sum)
                            number_of_pixels.append(cellROI.contour_cellroi.number_of_pixels)
                        if len(time)>0:
                            
                            sorted_lists = sorted(zip(time, intensity_max, intensity_mean, intensity_std, intensity_sum, number_of_pixels))
                            time_sorted, intensity_max_sorted, intensity_mean_sorted, intensity_std_sorted, intensity_sum_sorted, number_of_pixels_sorted = zip(*sorted_lists)

                            download_dict_laurel[exp.name][expds.data_name][sample][cellID.name]["time"]             = time_sorted
                            download_dict_laurel[exp.name][expds.data_name][sample][cellID.name]["intensity_max"]    = intensity_max_sorted
                            download_dict_laurel[exp.name][expds.data_name][sample][cellID.name]["intensity_mean"]   = intensity_mean_sorted
                            download_dict_laurel[exp.name][expds.data_name][sample][cellID.name]["intensity_std"]    = intensity_std_sorted
                            download_dict_laurel[exp.name][expds.data_name][sample][cellID.name]["intensity_sum"]    = intensity_sum_sorted
                            download_dict_laurel[exp.name][expds.data_name][sample][cellID.name]["number_of_pixels"] = number_of_pixels_sorted
                    
        import csv
        from django.http import FileResponse
        header=['experiment', 'well', 'position', 'cell', 'time', 'channel', 'number_of_pixels', 'intensity_max', 'intensity_mean', 'intensity_std', 'intensity_sum']
        with open('laurel.csv', 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for exp in download_dict_laurel:
                for expds in download_dict_laurel[exp]:
                    for sample in download_dict_laurel[exp][expds]:

                        for cell in download_dict_laurel[exp][expds][sample]:
                            if len(download_dict_laurel[exp][expds][sample][cell])==0:continue
                            for ch in download_dict_laurel[exp][expds][sample][cell]["intensity_max"][0]:
                                for timef in range(len(download_dict_laurel[exp][expds][sample][cell]["time"])):
                                    towrite=[exp, 
                                             expds, 
                                             sample, 
                                             cell, 
                                             download_dict_laurel[exp][expds][sample][cell]["time"][timef], 
                                             ch,
                                             download_dict_laurel[exp][expds][sample][cell]["number_of_pixels"][timef], 
                                             download_dict_laurel[exp][expds][sample][cell]["intensity_max"][timef][ch], 
                                             download_dict_laurel[exp][expds][sample][cell]["intensity_mean"][timef][ch], 
                                             download_dict_laurel[exp][expds][sample][cell]["intensity_std"][timef][ch], 
                                             download_dict_laurel[exp][expds][sample][cell]["intensity_sum"][timef][ch], 
                                            ]

                                    writer.writerow(towrite)


        file_path = '/home/helsens/Software/singleCell_catalog/laurel.csv'  # Update with your actual file path
        # Check if the file exists
        if os.path.exists(file_path):
            # Open the file
            file = open(file_path, 'rb')
            response = FileResponse(file)
            # Set the content type for the response
            response['Content-Type'] = 'text/csv'
            # Set the Content-Disposition header to specify the filename
            response['Content-Disposition'] = 'attachment; filename="%s"' % os.path.basename(file_path)
            return response

    context = {
        #'num_samples': num_samples,
        'select_dict':select_dict,
        'selected_dict':selected_dict,
        'experiment_dict':experiment_dict,
        'contribution_dict':contribution_dict,
        'treatment_dict':treatment_dict,
        'injection_dict':injection_dict,
        'instrumental_dict':instrumental_dict,
        'sample_dict':sample_dict,
    }

    if 'download' in request.GET:
    
        json_content = json.dumps(download_dict, indent=4)
        response = HttpResponse(json_content, content_type='application/json')
    
        # Set the Content-Disposition header to specify the filename
        response['Content-Disposition'] = 'attachment; filename="data.json"'

        return response

    else:
        return render(request, 'segmentation/index.html', context=context)

    

#___________________________________________________________________________________________
@login_required
def bokeh_dashboard(request: HttpRequest) -> HttpResponse:

    script = bokeh.embed.server_document(request.build_absolute_uri())
    print("request.build_absolute_uri() ",request.build_absolute_uri())
    context = {'script': script}

    return render(request, 'segmentation/bokeh_dashboard.html', context=context)


#___________________________________________________________________________________________
@login_required
def bokeh_summary_dashboard(request: HttpRequest) -> HttpResponse:

    script = bokeh.embed.server_document(request.build_absolute_uri())
    print("request.build_absolute_uri() ",request.build_absolute_uri())
    context = {'script': script}

    return render(request, 'segmentation/bokeh_summary_dashboard.html', context=context)



#___________________________________________________________________________________________
@login_required
def bokeh_phenocheck_dashboard(request: HttpRequest) -> HttpResponse:

    script = bokeh.embed.server_document(request.build_absolute_uri())
    print("request.build_absolute_uri() ",request.build_absolute_uri())
    context = {'script': script}

    return render(request, 'segmentation/bokeh_phenocheck_dashboard.html', context=context)