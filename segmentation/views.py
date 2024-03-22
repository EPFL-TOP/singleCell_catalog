from django.shortcuts import render
from django.db import reset_queries
from django.db import connection
from django.http import HttpRequest, HttpResponse
from django.contrib.auth.decorators import login_required, permission_required

from segmentation.models import Experiment, ExperimentalDataset, Sample, Frame, Contour, CellID, CellROI, CellStatus, CellFlag, ContourSeg

import os, sys, json, glob, gc
import time

from memory_profiler import profile
from sklearn.cluster import DBSCAN
from skimage import exposure
import numpy as np
from scipy.signal import find_peaks

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

LOCAL=True
DEBUG=False
BASEPATH="/mnt/nas_rcp/raw_data"

#MY macbook
if os.path.isdir('/Users/helsens/Software/github/EPFL-TOP/cellgmenter'):
    sys.path.append('/Users/helsens/Software/github/EPFL-TOP/cellgmenter')
#VMachine
if os.path.isdir('/home/helsens/Software/segmentationTools/cellgmenter/main'):
    sys.path.append('/home/helsens/Software/segmentationTools/cellgmenter/main')
    LOCAL=False
    import mysql.connector
    import accesskeys

    cnx = mysql.connector.connect(user=accesskeys.RD_DB_RO_user, 
                                  password=accesskeys.RD_DB_RO_password,
                                  host='127.0.0.1',
                                  port=3306,
                                  database=accesskeys.RD_DB_name)

import reader as read
import segmentationTools as segtools


import bokeh.models
import bokeh.palettes
import bokeh.plotting
import bokeh.embed
import bokeh.layouts



#___________________________________________________________________________________________
def deltaR(c1, c2):
    return math.sqrt( math.pow((c1['x'] - c2['x']),2) +  math.pow((c1['y'] - c2['y']),2) + math.pow((c1['z'] - c2['z']),2))

#___________________________________________________________________________________________
def get_experiement_details(selected_experiment):
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
        print(x)

    experiments = Experiment.objects.values()
    list_experiments = [entry for entry in experiments] 
    list_experiments_uid=[e["name"] for e in list_experiments]

    for x in myresult:
        if x[1] in list_experiments_uid: continue
        unsplit_file = glob.glob(os.path.join('/mnt/nas_rcp/raw_data/microscopy/cell_culture/',x[1],'*.nd2'))
        if len(unsplit_file)!=1:
            print('====================== ERROR, unsplit_file not 1, exit ',unsplit_file,'  in ',os.path.join('/mnt/nas_rcp/raw_data/microscopy/cell_culture/',x[1],'*.nd2'))
            sys.exit(3)
        metadata = read.nd2reader_getSampleMetadata(unsplit_file[0])
        experiment =  Experiment(name=x[1], 
                                 date=x[2], 
                                 description=x[3],
                                 file_name=unsplit_file[0],
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
                fname=os.path.join(BASEPATH, x[4], x[5], "raw_files", f["name"])
                metadata = read.nd2reader_getSampleMetadata(fname)
                sample = Sample(file_name=fname, 
                                experimental_dataset=expds,
#                                number_of_frames=metadata['number_of_frames'], 
#                                number_of_channels=metadata['number_of_channels'], 
#                                name_of_channels=metadata['name_of_channels'], 
#                                experiment_description=metadata['experiment_description'], 
#                                date=metadata['date'],
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
def segment_localThresholding(sample):

    s=None
    if type(sample) == str:
        s = Sample.objects.get(file_name = sample)
    else:
        s=sample

    segmentation = segtools.customLocalThresholding_Segmentation(threshold=2., delta=2, npix_min=200, npix_max=5000)

    frames = Frame.objects.select_for_related().filter(sample=s)
    for frame in frames:
        cellROIs = CellROI.objects.select_related().filter(frame=frame)
        for cellROI in cellROIs:
            eflag=False
            if cellROI.cell_id == None: continue
            contoursSeg = ContourSeg.objects.select_related().filter(cell_roi=cellROI)
            for contourSeg in contoursSeg:
                if contourSeg.algo == 'localthresholding': 
                    eflag=True
            if eflag: continue


            contour_list = segmentation.segmentation()

#    #loop over all experiments
#    exp_list = Experiment.objects.all()
#
#    for exp in exp_list:
#        print(' ---- SEGMENTATION exp name ',exp.name)
#        print(' ---- SEGMENTATION channels ',exp.name_of_channels.split(','),' number ',exp.number_of_channels, ' full file name ', exp.file_name)
#        segExist=False
#        #build default segmentation class, to be replaced by calls from django app
#        default_segmentation = segtools.customLocalThresholding_Segmentation(threshold=2., delta=2, npix_min=400, npix_max=4000)
#        default_segmentation.channels = exp.name_of_channels.split(',')
#        default_segmentation.channel = 0
#
#        default_segmentation_2 = segtools.customLocalThresholding_Segmentation(threshold=2., delta=1, npix_min=400, npix_max=4000)
#        default_segmentation_2.channels = exp.name_of_channels.split(',')
#        default_segmentation_2.channel = 0
#
#        #check existing segmentation if already registered
#        segmentations = Segmentation.objects.select_related().filter(experiment = exp)
#        for seg in segmentations:
#            if default_segmentation.get_param() == seg.algorithm_parameters and \
#                default_segmentation.get_type() == seg.algorithm_type and \
#                    default_segmentation.get_version() == seg.algorithm_version:
#                print('SEGMENTATION EXISTTTTTTTTT')
#                #check if the segmentation channel exists
#                segmentation_channels = SegmentationChannel.objects.select_related().filter(segmentation = seg)
#                for seg_ch in segmentation_channels:
#                    if seg_ch.channel_number == default_segmentation.channel and \
#                        seg_ch.channel_name == default_segmentation.channels[default_segmentation.channel]:
#                        segExist=True
#        if segExist: continue
#
#        #create segmentation and segmentation channel if it does not exist
#        segmentation = Segmentation(name="default segmentation", 
#                                    experiment=exp,
#                                    algorithm_type=default_segmentation.get_type(),
#                                    algorithm_version=default_segmentation.get_version(),
#                                    algorithm_parameters=default_segmentation.get_param())
#        segmentation.save()
#        segmentation_channel = SegmentationChannel(segmentation=segmentation,
#                                                channel_name=exp.name_of_channels.split(',')[0],
#                                                channel_number=0)
#        segmentation_channel.save()
#
#        print(' ---- SEGMENTATION exp name ',exp.name)
#        experimentaldataset = ExperimentalDataset.objects.select_related().filter(experiment = exp)
#
#        for expds in experimentaldataset:
#            print('    ---- SEGMENTATION experimentaldataset name ',expds.data_name, expds.data_type)
#            samples = Sample.objects.select_related().filter(experimental_dataset = expds)
#
#            counter_samp=0
#            for s in samples:
#                if counter_samp==2: 
#                    print('===========================================')
#                    break
#                counter_samp+=1
#                print('         ---- SEGMENTATION sample name ',s.file_name)
#                frames = Frame.objects.select_related().filter(sample = s)
#                print('getting the images')
#                images, channels = read.nd2reader_getFrames(s.file_name)
#                print ('          ---- SEGMENTATION will loop over ',len(frames),' frames')
#
#                for f in frames:
#                    print( 'getting contour for frame ',f.number)
#                    contour_list = default_segmentation.segmentation(images[f.number])
#
#                    print(' got ',len(contour_list),' contours')
#                    for cont in contour_list:
#                        #pixels_data_contour  = Data(all_pixels=cont['all_pixels_contour'], single_pixels=cont['single_pixels_contour'])
#                        #pixels_data_contour.save()
#                        #pixels_data_inside   = Data(all_pixels=cont['all_pixels_inside'],  single_pixels=cont['single_pixels_inside'])
#                        #pixels_data_inside.save()
#                        print(cont['center'])
#                        contour = Contour(frame=f,
#                        #                  pixels_data_contour=pixels_data_contour,
#                        #                  pixels_data_inside=pixels_data_inside,
#                                          segmentation_channel=segmentation_channel,
#                                          center=cont['center'])
#                        contour.save()
#
#                        del contour
#                    del contour_list
#                    #print('gc collect 1: ',gc.collect())
#
#                print('gc collect 1: ',gc.collect())

#___________________________________________________________________________________________
def build_cells_all_exp(sample=None):
    #loop over all experiments
    exp_list = Experiment.objects.all()
    for exp in exp_list:
        print('---- BUILD CELLS experiment name ',exp.name)
        experimentaldataset = ExperimentalDataset.objects.select_related().filter(experiment = exp)
        for expds in experimentaldataset:
            print('    ---- BUILD CELLS experimentaldataset name ',expds.data_name, expds.data_type)
            samples = Sample.objects.select_related().filter(experimental_dataset = expds)
            for s in samples:
                if sample!=None and sample!=s.file_name:continue
                print('        ---- BUILD CELL sample name ',s.file_name)
                cellsid = CellID.objects.select_related().filter(sample = s)
                #delete the existing cellID
                cellsid.delete()

                frames = Frame.objects.select_related().filter(sample = s)
                cell_roi_list=[]
                cell_roi_coord=[]

                for f in frames:
                    cellrois = CellROI.objects.select_related().filter(frame=f)
                    for cellroi in cellrois:
                        cell_roi_list.append(cellroi)
                        cell_roi_coord.append([cellroi.min_col+(cellroi.max_col-cellroi.min_col)/2., 
                                               cellroi.min_row+(cellroi.max_row-cellroi.min_row)/2.])
                print('number of cell frames=',len(cell_roi_list))
                if len(cell_roi_list)==0:continue
                X = np.array(cell_roi_coord)
                eps= ((cellroi.max_col-cellroi.min_col)/2. + (cellroi.max_row-cellroi.min_row)/2.)/1.
                clustering = DBSCAN(eps=eps, min_samples=25).fit(X)
                print(clustering.labels_)

                #Create the cells ID according to existing clusters (one per cluster >=0)
                #Connect the cellFrames to cellID
                createdcells=[]
                cellid_dict={}
                for cid in range(len(clustering.labels_)):
                    if clustering.labels_[cid] not in createdcells and clustering.labels_[cid]!=-1:
                        cellstatus = CellStatus()
                        cellstatus.save()
                        cellid = CellID(sample=s, name='cell{}'.format(clustering.labels_[cid]), cell_status=cellstatus)
                        cellid.save()
                        createdcells.append(clustering.labels_[cid])
                        cellid_dict['cell{}'.format(clustering.labels_[cid])]=cellid
                    if clustering.labels_[cid]!=-1:
                        cell_roi_list[cid].cell_id = cellid_dict['cell{}'.format(clustering.labels_[cid])]
                        cell_roi_list[cid].save()


#___________________________________________________________________________________________
def build_cells_sample(sample):
 
    s=None
    if type(sample) == str:
        s = Sample.objects.get(file_name = sample)
    else:
        s=sample
    print('build_cells_sample ',s)
    print('        ---- BUILD CELL sample name ',s.file_name)
    cellsid = CellID.objects.select_related().filter(sample = s)
    ##delete the existing cellID
    #cellsid.delete()

    frames = Frame.objects.select_related().filter(sample = s)
    cell_roi_id_list=[]
    for cellid in cellsid:
        cellrois_cellid = CellROI.objects.select_related().filter(cell_id=cellid)
        for cellroi_cellid in cellrois_cellid:
            cell_roi_id_list.append(cellroi_cellid.id)
    cell_roi_list=[]
    cell_roi_coord=[]

    #do a first clustering using all ROIs
    if len(cell_roi_id_list)==0:
        eps=0
        for f in frames:
            cellrois_frame = CellROI.objects.select_related().filter(frame=f)
            for cellroi_frame in cellrois_frame:

                cell_roi_list.append(cellroi_frame)
                cell_roi_coord.append([cellroi_frame.min_col+(cellroi_frame.max_col-cellroi_frame.min_col)/2., 
                                        cellroi_frame.min_row+(cellroi_frame.max_row-cellroi_frame.min_row)/2.])
                eps+= ((cellroi_frame.max_col-cellroi_frame.min_col)/2. + (cellroi_frame.max_row-cellroi_frame.min_row)/2.)/2.

        eps=eps/len(frames)
        print('number of cell frames=',len(cell_roi_list))
        if len(cell_roi_list)==0:return
        X = np.array(cell_roi_coord)
        clustering = DBSCAN(eps=eps, min_samples=20).fit(X)
        print(clustering.labels_)

        #Create the cells ID according to existing clusters (one per cluster >=0)
        #Connect the cellFrames to cellID
        createdcells=[]
        cellid_dict={}
        for cid in range(len(clustering.labels_)):
            if clustering.labels_[cid] not in createdcells and clustering.labels_[cid]!=-1:
                cellstatus = CellStatus()
                cellstatus.save()
                cellid = CellID(sample=s, name='cell{}'.format(clustering.labels_[cid]), cell_status=cellstatus)
                cellid.save()
                createdcells.append(clustering.labels_[cid])
                cellid_dict['cell{}'.format(clustering.labels_[cid])]=cellid
            if clustering.labels_[cid]!=-1:
                cell_roi_list[cid].cell_id = cellid_dict['cell{}'.format(clustering.labels_[cid])]
                cell_roi_list[cid].save()
                cell_roi_id_list.append(cell_roi_list[cid].id)
    if DEBUG: print('cell_roi_id_list afgter DBSCAN ',cell_roi_id_list)

    #Then cluster remaining or new cells ROIs
    cell_pos_dict={}
    cell_id_dict={}
    cellsid = CellID.objects.select_related().filter(sample = s)
    for cellid in cellsid:
        cell_pos_dict[cellid.name]=[]
        cell_id_dict[cellid.name]=cellid
        cellrois = CellROI.objects.select_related().filter(cell_id=cellid)
        for cellroi in cellrois:
            cell_pos_dict[cellid.name].append([cellroi.min_col+(cellroi.max_col-cellroi.min_col)/2., 
                                               cellroi.min_row+(cellroi.max_row-cellroi.min_row)/2.])


    for f in frames:
        cellrois_frame = CellROI.objects.select_related().filter(frame=f)
        for cellroi_frame in cellrois_frame:
            if cellroi_frame.id in cell_roi_id_list: continue


            for cell in cell_pos_dict:
                min_dr_name=''
                min_dr_val=10000000000.
                max_dr_val=((cellroi_frame.max_col-cellroi_frame.min_col)/2. + (cellroi_frame.max_row-cellroi_frame.min_row)/2.)/2.
                tmp_val=0
                for pos in cell_pos_dict[cell]:
                    tmp_val+=              math.sqrt(math.pow(pos[0]-(cellroi_frame.min_col+(cellroi_frame.max_col-cellroi_frame.min_col)/2.),2) + 
                                                     math.pow(pos[1]-(cellroi_frame.min_row+(cellroi_frame.max_row-cellroi_frame.min_row)/2.),2))
                    if DEBUG: print('dr= ',math.sqrt(math.pow(pos[0]-(cellroi_frame.min_col+(cellroi_frame.max_col-cellroi_frame.min_col)/2.),2) + 
                                                     math.pow(pos[1]-(cellroi_frame.min_row+(cellroi_frame.max_row-cellroi_frame.min_row)/2.),2)),
                                           ' n pos=',len(cell_pos_dict[cell]))

                if tmp_val/len(cell_pos_dict[cell])<min_dr_val and tmp_val/len(cell_pos_dict[cell])<max_dr_val:
                    min_dr_val=tmp_val
                    min_dr_name=cell
                    cellroi_frame.cell_id=cell_id_dict[min_dr_name]
                    cellroi_frame.save()
                if DEBUG: print('frame=',f, '   cellroi_frame=',cellroi_frame,'  min_dr_val=',min_dr_val, '  min_dr_name=cellname=',min_dr_name, '  max_dr_val=',max_dr_val,'  tmp_val/len(cell_pos_dict[cell])=',tmp_val/len(cell_pos_dict[cell]))
            #if min_dr_name!='':
            #    cellroi_frame.cell_id=cell_id_dict[min_dr_name]
            #    cellroi_frame.save()
                

#___________________________________________________________________________________________
def removeROIs(sample):
    s=None
    if type(sample) == str:
        s = Sample.objects.get(file_name = sample)
    else:
        s=sample
    frames = Frame.objects.select_for_related().filter(sample=s)
    for frame in frames:
        cellROIs = CellROI.objects.select_related().filter(frame=frame)
        for cellROI in cellROIs:
            if cellROI.cell_id == None:
                cellROI.delete()
    print('removeROIs sample ',s)



#___________________________________________________________________________________________
def checkROI(ROIs):

    for r in range(len(ROIs)):
        if ROIs[r][2]-ROIs[r][0]>50 and ROIs[r][3]-ROIs[r][1]>50:
            return True
    return False


#___________________________________________________________________________________________
def build_ROIs():
    exp_list = Experiment.objects.all()
    for exp in exp_list:
        experimentaldataset = ExperimentalDataset.objects.select_related().filter(experiment = exp)
        for expds in experimentaldataset:
            samples = Sample.objects.select_related().filter(experimental_dataset = expds)
            #counter_samp=0
            for s in samples:
                cellids = CellID.objects.select_related().filter(sample=s)
                if len(cellids)>0:continue
                print('build roi sample: ',s.file_name)
                if 'wscepfl00' not in s.file_name :continue

                #if counter_samp==10: 
                #    print('===================BREAK ROIS========================')
                #    break
                #counter_samp+=1
                frames = Frame.objects.select_related().filter(sample=s)
                images, channels = read.nd2reader_getFrames(s.file_name)
                #images are t, c, x, y 
                BF_images=images.transpose(1,0,2,3)
                BF_images=BF_images[0]
                for frame in frames:
                    rois = CellROI.objects.select_related().filter(frame = frame)
                    #Just for now, should normally check that same ROI don't overlap
                    if len(rois)>0: continue
                    ROIs = segtools.get_ROIs_per_frame(BF_images[frame.number])
                    if len(ROIs)==0 or checkROI(ROIs)==False:
                        ROIs = segtools.get_ROIs_per_frame(BF_images[frame.number], 2.8)
                    if len(ROIs)==0 or checkROI(ROIs)==False:
                        ROIs = segtools.get_ROIs_per_frame(BF_images[frame.number], 2.)
                    if len(ROIs)==0 or checkROI(ROIs)==False:
                        ROIs = segtools.get_ROIs_per_frame(BF_images[frame.number], 1.5)
                    npixmin=10
                    for r in range(len(ROIs)):
                        if ROIs[r][0]<npixmin or ROIs[r][1]<npixmin or ROIs[r][2]>frame.height-npixmin or ROIs[r][3]>frame.width-npixmin:
                            continue
                        roi = CellROI(min_row = ROIs[r][0], min_col = ROIs[r][1],
                                      max_row = ROIs[r][2], max_col = ROIs[r][3], 
                                      frame = frame, roi_number=r)
                        roi.save()
            #Bounding box (min_row, min_col, max_row, max_col). 

                        cropped_dict = {'shape_original':BF_images[frame.number].shape}
                        out_dir_name  = os.path.join(os.sep, "data","singleCell_catalog","contour_data",exp.name, expds.data_name, os.path.split(s.file_name)[-1].replace('.nd2',''))
                        out_file_name = os.path.join(out_dir_name, "frame{0}_ROI{1}.json".format(frame.number, r))
                        if not os.path.exists(out_dir_name):
                            os.makedirs(out_dir_name)
                        cropped_img = images[frame.number][:, ROIs[r][0]:ROIs[r][2], ROIs[r][1]:ROIs[r][3]]
                        cropped_dict['shape']=[cropped_img.shape[1], cropped_img.shape[2]]
                        cropped_dict['npixels']=cropped_img.shape[1]*cropped_img.shape[2]
                        cropped_dict['shift']=[ROIs[r][0], ROIs[r][1]]
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
                        
                        contour = Contour(center_x_pix=ROIs[r][1]+(ROIs[r][3]-ROIs[r][1])/2., 
                                          center_y_pix=ROIs[r][0]+(ROIs[r][2]-ROIs[r][0])/2.,
                                          center_z_pix=0, 
                                          center_x_mic=(ROIs[r][1]+(ROIs[r][3]-ROIs[r][1])/2.)*roi.frame.pixel_microns+roi.frame.pos_x,
                                          center_y_mic=(ROIs[r][0]+(ROIs[r][2]-ROIs[r][0])/2.)*roi.frame.pixel_microns+roi.frame.pos_y,
                                          center_z_mic=0,
                                          intensity_mean=intensity_mean,
                                          intensity_std=intensity_std,
                                          intensity_sum=intensity_sum,
                                          intensity_max=intensity_max,
                                          number_of_pixels=cropped_img.shape[1]*cropped_img.shape[2],
                                          file_name=out_file_name,
                                          cell_roi=roi,
                                          type="cell_ROI",
                                          mode="auto")
                        contour.save()

                        cellflag = CellFlag(cell_roi=roi)
                        cellflag.save()

                build_cells_sample(s)





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


#___________________________________________________________________________________________
def segmentation_handler(doc: bokeh.document.Document) -> None:
    print('****************************  segmentation_handler ****************************')
    print('****************************  segmentation_handler ****************************')
    print('****************************  segmentation_handler ****************************')
    print('****************************  segmentation_handler ****************************')
    #TO BE CHANGED WITH ASYNC?????
    os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"
    experiments=[]
    wells={}
    positions={}
    files={}

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
        #experiments.append(f'<span style="color:red">{exp.name}</span>')
    #formatted_options = [f'<span style="color: {conditions[opt]}">{opt}</span>' for opt in options]
        wells[exp.name] = []
        experimentaldatasets = ExperimentalDataset.objects.select_related().filter(experiment = exp)
        for expds in experimentaldatasets:
            wells[exp.name].append(expds.data_name)
            samples = Sample.objects.select_related().filter(experimental_dataset = expds)
            positions['{0}_{1}'.format(exp.name, expds.data_name)] = []
            files['{0}_{1}'.format(exp.name, expds.data_name)] = []
            for samp in samples:
                positions['{0}_{1}'.format(exp.name, expds.data_name)].append(samp.file_name.split('/')[-1])
                files['{0}_{1}'.format(exp.name, expds.data_name)].append(samp.file_name)

    experiments=sorted(experiments)
    for i in wells:
        wells[i] = sorted(wells[i])
    for i in positions:
        positions[i] = sorted(positions[i])

    dropdown_exp  = bokeh.models.Select(value=experiments[0], title='Experiment', options=experiments)
    dropdown_well = bokeh.models.Select(value=wells[experiments[0]][0], title='Well', options=wells[dropdown_exp.value])
    dropdown_pos  = bokeh.models.Select(value=positions['{0}_{1}'.format(experiments[0], wells[experiments[0]][0])][0],title='Position', options=positions['{0}_{1}'.format(dropdown_exp.value, dropdown_well.value)])


    #___________________________________________________________________________________________
    def update_position_select():
        local_pos = []
        current_files = files['{0}_{1}'.format(dropdown_exp.value, dropdown_well.value)]
        for pos in dropdown_pos.options:
            mypos=pos.split(' - ')[0]
            current_file = ''
            for f in current_files:
                if mypos in f:
                    current_file = f
                    break
            sample = Sample.objects.get(file_name=current_file)
            if sample.check_sample and sample.keep_sample:
                local_pos.append('{} - c'.format(mypos))
            elif sample.check_sample and not sample.keep_sample:
                local_pos.append('{} - c,dk'.format(mypos))
            elif not sample.check_sample and not sample.keep_sample:
                local_pos.append('{} - dk'.format(mypos))
            else:
                local_pos.append('{}'.format(mypos))

        dropdown_pos.options = local_pos
    #___________________________________________________________________________________________

    update_position_select()

    #___________________________________________________________________________________________
    # Function to get the image stack
    def get_current_stack():
        if DEBUG: print('****************************  get_current_stack ****************************')
        current_file=get_current_file()
        time_lapse_path = Path(current_file)
        time_lapse = nd2.imread(time_lapse_path.as_posix())
        ind_images_list=[]
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
            
        return ind_images_list, ind_images_list_norm
    #___________________________________________________________________________________________

    #___________________________________________________________________________________________
    # Function to get the image stack
    def get_current_stack_url():
        current_file=get_current_file()
        time_lapse_path = Path(current_file)
        time_lapse = nd2.imread(time_lapse_path.as_posix())
        ind_images_list=[]
        for nch in range(time_lapse.shape[1]):
            time_lapse_tmp = time_lapse[:,nch,:,:] # Assume I(t, c, x, y)
            time_domain = np.asarray(np.linspace(0, time_lapse_tmp.shape[0] - 1, time_lapse_tmp.shape[0]), dtype=np.uint)
            ind_images = [np.flip(time_lapse_tmp[i,:,:],0) for i in time_domain]
            ind_images_url = []
            for im in ind_images:
                image_pil = Image.fromarray(im)
                buffer = BytesIO()
                image_pil.save(buffer, format='PNG')
                image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                ind_images_url.append(image_base64)
            ind_images_list.append(ind_images_url)
        return ind_images_list
    #___________________________________________________________________________________________  
 

    #___________________________________________________________________________________________
    # Function to get the current file
    def get_current_file():
        if DEBUG:
            print('****************************  get_current_file ****************************')
            print('--------------- get_current_file() dropdown_exp.value=', dropdown_exp.value, '   dropdown_well.value',dropdown_well.value, '  dropdown_pos.value',dropdown_pos.value)

        current_files = files['{0}_{1}'.format(dropdown_exp.value, dropdown_well.value)]
        current_file = ''
        for f in current_files:
            if dropdown_pos.value.split(' - ')[0] in f:
                current_file = f
        if DEBUG: print('--------------- get_current_file() current file  ',current_file)
        return current_file
    #___________________________________________________________________________________________


    ind_images_list,  ind_images_list_norm = get_current_stack()

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

    source_intensity_ch0 = bokeh.models.ColumnDataSource(data={'time':[], 'intensity':[]})
    source_intensity_ch1 = bokeh.models.ColumnDataSource(data={'time':[], 'intensity':[]})
    source_intensity_ch2 = bokeh.models.ColumnDataSource(data={'time':[], 'intensity':[]})
    source_intensity_max = bokeh.models.ColumnDataSource(data={'time':[], 'intensity':[]})
    source_intensity_min = bokeh.models.ColumnDataSource(data={'time':[], 'intensity':[]})

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
    plot_image     = bokeh.plotting.figure(x_range=(0, ind_images_list[0][0].shape[0]), y_range=(0, ind_images_list[0][0].shape[1]), tools="box_select,wheel_zoom,box_zoom,reset,undo",width=550, height=550)
    plot_intensity = bokeh.plotting.figure(title="Intensity vs Time", x_axis_label='Time (minutes)', y_axis_label='Intensity',width=1000, height=500)
    plot_osc_tod   = bokeh.plotting.figure(title="Start/End of Oscilation and Time of death", x_axis_label='Time (minutes)', y_axis_label='Number of positions',width=1000, height=250)
    plot_nosc      = bokeh.plotting.figure(title="Number of oscillations", x_axis_label='Number of oscillations', y_axis_label='Number of positions',width=550, height=250)

    slider_find_peaks  = bokeh.models.Slider(start=0, end=100, value=30, step=1, title="Peak prominence", width=200)

    #___________________________________________________________________________________________
    # Function to prepare the intensity plot
    def prepare_intensity():
        if DEBUG:print('----------------prepare_intensity--------------------dropdown_cell.value=',dropdown_cell.value)
        current_file=get_current_file()
        if dropdown_cell.value!='':
            if DEBUG: print('----------------prepare_intensity-------------------- in the if')

            sample = Sample.objects.get(file_name=current_file)
            cellids = CellID.objects.select_related().filter(sample=sample, name=dropdown_cell.value)

            #Set start of oscilation if it exist, -999 else
            if cellids[0].cell_status.start_oscillation>0: 
                start_oscillation_position.location = source_intensity_ch1.data["time"][cellids[0].cell_status.start_oscillation_frame]
            else: 
                start_oscillation_position.location = -999

            #Set end of oscilation if it exist, -999 else
            if cellids[0].cell_status.end_oscillation>0: 
                end_oscillation_position.location   = source_intensity_ch1.data["time"][cellids[0].cell_status.end_oscillation_frame]
            else: 
                end_oscillation_position.location   = -999

            #Set time of death and varea if it exist, -999 [] else
            if cellids[0].cell_status.time_of_death>0:
                time_of_death_position.location = source_intensity_ch1.data["time"][cellids[0].cell_status.time_of_death_frame]
                source_varea_death.data['x']    = [source_intensity_ch1.data["time"][t] for t in range(cellids[0].cell_status.time_of_death_frame, len(source_intensity_ch1.data["time"])) ]
                source_varea_death.data['y1']   = [source_intensity_ch1.data["intensity"][t] for t in range(cellids[0].cell_status.time_of_death_frame, len(source_intensity_ch1.data["intensity"]))]
                source_varea_death.data['y2']   = [0 for i in range(len(source_varea_death.data['y1']))]

            else:
                time_of_death_position.location = -999
                source_varea_death.data['x']    = []
                source_varea_death.data['y1']   = []
                source_varea_death.data['y2']   = []


            #Set maximums and mimimums if exists [] else
            if len(cellids[0].cell_status.peaks)>=6:
                source_intensity_max.data={'time':cellids[0].cell_status.peaks["max_time"], 'intensity':cellids[0].cell_status.peaks["max_int"]}
                source_intensity_min.data={'time':cellids[0].cell_status.peaks["min_time"], 'intensity':cellids[0].cell_status.peaks["min_int"]}
            else:
                source_intensity_max.data={'time':[], 'intensity':[]}
                source_intensity_min.data={'time':[], 'intensity':[]}


            source_segments_cell.data={'time':[], 'intensity':[]}
            #____________________________________________________
            try: 
                source_mask_cell.data={'time':cellids[0].cell_status.flags["mask_time"], 
                                       'intensity_full':[source_intensity_ch1.data["intensity"][t] for t in cellids[0].cell_status.flags["mask_frame"]]}
            except KeyError:
                source_mask_cell.data={'time':[], 'intensity_full':[]}

            #____________________________________________________
            try: 
                source_dividing_cell.data={'time':cellids[0].cell_status.flags["dividing_time"], 
                                           'intensity_full':[source_intensity_ch1.data["intensity"][t] for t in cellids[0].cell_status.flags["dividing_frame"]],
                                           'intensity':[flags_dict['dividing'] for t in cellids[0].cell_status.flags["dividing_frame"]]}
                data_segment = {'time':source_segments_cell.data["time"]+cellids[0].cell_status.flags["dividing_time"], 
                                'intensity':source_segments_cell.data["intensity"]+[source_intensity_ch1.data["intensity"][t] for t in cellids[0].cell_status.flags["dividing_frame"]]}
                source_segments_cell.data=data_segment
            except KeyError:
                source_dividing_cell.data={'time':[], 'intensity':[], 'intensity_full':[]}

            #____________________________________________________
            try: 
                source_double_nuclei_cell.data={'time':cellids[0].cell_status.flags["double_nuclei_time"], 
                                                'intensity_full':[source_intensity_ch1.data["intensity"][t] for t in cellids[0].cell_status.flags["double_nuclei_frame"]],
                                                'intensity':[flags_dict['double_nuclei'] for t in cellids[0].cell_status.flags["double_nuclei_frame"]]}
                data_segment = {'time':source_segments_cell.data["time"]+cellids[0].cell_status.flags["double_nuclei_time"], 
                                'intensity':source_segments_cell.data["intensity"]+[source_intensity_ch1.data["intensity"][t] for t in cellids[0].cell_status.flags["double_nuclei_frame"]]}
                source_segments_cell.data=data_segment
            except KeyError:
                source_double_nuclei_cell.data={'time':[], 'intensity':[], 'intensity_full':[]}


            #____________________________________________________
            try: 
                source_multiple_cells.data={'time':cellids[0].cell_status.flags["multiple_cells_time"], 
                                           'intensity_full':[source_intensity_ch1.data["intensity"][t] for t in cellids[0].cell_status.flags["multiple_cells_frame"]],
                                           'intensity':[flags_dict['multiple_cells'] for t in cellids[0].cell_status.flags["multiple_cells_frame"]]}
                data_segment = {'time':source_segments_cell.data["time"]+cellids[0].cell_status.flags["multiple_cells_time"], 
                                'intensity':source_segments_cell.data["intensity"]+[source_intensity_ch1.data["intensity"][t] for t in cellids[0].cell_status.flags["multiple_cells_frame"]]}
                source_segments_cell.data=data_segment
            except KeyError:
                source_multiple_cells.data={'time':[], 'intensity':[], 'intensity_full':[]}

            #____________________________________________________
            try: 
                source_pair_cell.data={'time':cellids[0].cell_status.flags["pair_cell_time"], 
                                       'intensity_full':[source_intensity_ch1.data["intensity"][t] for t in cellids[0].cell_status.flags["pair_cell_frame"]],
                                       'intensity':[flags_dict['pair_cell'] for t in cellids[0].cell_status.flags["pair_cell_frame"]]}
                data_segment = {'time':source_segments_cell.data["time"]+cellids[0].cell_status.flags["pair_cell_time"], 
                                'intensity':source_segments_cell.data["intensity"]+[source_intensity_ch1.data["intensity"][t] for t in cellids[0].cell_status.flags["pair_cell_frame"]]}
                source_segments_cell.data=data_segment
            except KeyError:
                source_pair_cell.data={'time':[], 'intensity':[], 'intensity_full':[]}

            #____________________________________________________
            try: 
                source_flat_cell.data={'time':cellids[0].cell_status.flags["flat_time"], 
                                       'intensity_full':[source_intensity_ch1.data["intensity"][t] for t in cellids[0].cell_status.flags["flat_frame"]],
                                       'intensity':[flags_dict['flat'] for t in cellids[0].cell_status.flags["flat_frame"]]}
                data_segment = {'time':source_segments_cell.data["time"]+cellids[0].cell_status.flags["flat_time"], 
                                'intensity':source_segments_cell.data["intensity"]+[source_intensity_ch1.data["intensity"][t] for t in cellids[0].cell_status.flags["flat_frame"]]}
                source_segments_cell.data=data_segment
            except KeyError:
                source_flat_cell.data={'time':[], 'intensity':[], 'intensity_full':[]}

            #____________________________________________________
            try: 
                source_round_cell.data={'time':cellids[0].cell_status.flags["round_time"], 
                                        'intensity_full':[source_intensity_ch1.data["intensity"][t] for t in cellids[0].cell_status.flags["round_frame"]],
                                        'intensity':[flags_dict['round'] for t in cellids[0].cell_status.flags["round_frame"]]}
                data_segment = {'time':source_segments_cell.data["time"]+cellids[0].cell_status.flags["round_time"], 
                                'intensity':source_segments_cell.data["intensity"]+[source_intensity_ch1.data["intensity"][t] for t in cellids[0].cell_status.flags["round_frame"]]}
                source_segments_cell.data=data_segment
            except KeyError:
                source_round_cell.data={'time':[], 'intensity':[], 'intensity_full':[]}

            #____________________________________________________
            try: 
                source_elongated_cell.data={'time':cellids[0].cell_status.flags["elongated_time"], 
                                            'intensity_full':[source_intensity_ch1.data["intensity"][t] for t in cellids[0].cell_status.flags["elongated_frame"]],
                                            'intensity':[flags_dict['elongated'] for t in cellids[0].cell_status.flags["elongated_frame"]]}
                data_segment = {'time':source_segments_cell.data["time"]+cellids[0].cell_status.flags["elongated_time"], 
                                'intensity':source_segments_cell.data["intensity"]+[source_intensity_ch1.data["intensity"][t] for t in cellids[0].cell_status.flags["elongated_frame"]]}
                source_segments_cell.data=data_segment
            except KeyError:
                source_elongated_cell.data={'time':[], 'intensity':[], 'intensity_full':[]}

            set_rising_falling(cellids[0])

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


        #if len(source_intensity_ch1.data["intensity"])>0:
        #    plot_intensity.y_range = bokeh.models.Range1d(max(source_intensity_ch1.data["intensity"])*0.4, max(source_intensity_ch1.data["intensity"])*1.2, bounds="auto")
        ##plot_intensity.y_range.start=0
        ##if len(source_intensity_ch1.data["intensity"])>0:
        ##    plot_intensity.y_range.end=max(source_intensity_ch1.data["intensity"])*1.2
        ##    plot_intensity.y_range.start=min(source_intensity_ch1.data["intensity"])*0.4
        ###plot_intensity.y_range.trigger('start', 'end')
        #plot_intensity.y_range.update(start=plot_intensity.y_range.start, end=plot_intensity.y_range.end)
        if DEBUG: print('===============---------------plot_intensity.y_range.start=',plot_intensity.y_range.start,'  plot_intensity.y_range.end=',plot_intensity.y_range.end)

    #___________________________________________________________________________________________



    #___________________________________________________________________________________________
    def set_rising_falling_local(max_list, min_list):

        if DEBUG:print('-----------------set_rising_falling_local-------------------------     ')
        arrays_r = {}
        for i in range(1,11):
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

        arrays_f = {}
        for i in range(1,11):
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


        if end_oscillation_position.location  < 0 or start_oscillation_position.location < 0:
            for i in range(1,11):
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
  
        for i in range(1,11):
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

        for i in range(1,11):
            source_falling[i].data={'x':arrays_f['xf_{}'.format(i)], 'y1':arrays_f['yf1_{}'.format(i)], 'y2':arrays_f['yf2_{}'.format(i)]}
    #___________________________________________________________________________________________



    #___________________________________________________________________________________________
    def set_rising_falling(cellid, save_status=False, delete_status=False):
        if DEBUG: print("*************************set_rising_falling*****************************************")
        arrays_r = {}
        for i in range(1,11):
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


        arrays_f = {}
        for i in range(1,11):
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
        if cellid==None:
            for i in range(1,11):
                source_rising[i].data  = {'x':arrays_r['xr_{}'.format(i)], 'y1':arrays_r['yr1_{}'.format(i)], 'y2':arrays_r['yr2_{}'.format(i)]}
                source_falling[i].data = {'x':arrays_f['xf_{}'.format(i)], 'y1':arrays_f['yf1_{}'.format(i)], 'y2':arrays_f['yf2_{}'.format(i)]}
            return
        
        osc_dict={'rising_frame':[], 'falling_frame':[], 'mask_frame':[],
                  'rising_time':[],  'falling_time':[],  'mask_time':[]}
        
        if len(cellid.cell_status.peaks)==6:

            for m in range(len(cellid.cell_status.peaks["max_frame"])):
                min_val=cellid.cell_status.start_oscillation_frame
                for n in range(len(cellid.cell_status.peaks["min_frame"])):
                    if cellid.cell_status.peaks["min_frame"][n]<cellid.cell_status.peaks["max_frame"][m]: 
                        min_val=cellid.cell_status.peaks["min_frame"][n]
                for t in range(cellid.cell_status.start_oscillation_frame, cellid.cell_status.end_oscillation_frame):

                    if t==min_val and cellid.cell_status.start_oscillation_frame==min_val and t<cellid.cell_status.peaks["max_frame"][m] and cellid.cell_status.peaks["max_frame"][m]>min_val:
                        arrays_r['xr_{}'.format(m+1)].append(source_intensity_ch1.data["time"][t])
                        arrays_r['yr1_{}'.format(m+1)].append(0)
                        arrays_r['yr2_{}'.format(m+1)].append(source_intensity_ch1.data["intensity"][t])
                        osc_dict["rising_frame"].append(t)
                        osc_dict["rising_time"].append(source_intensity_ch1.data["time"][t])

                    if t<cellid.cell_status.peaks["max_frame"][m] and cellid.cell_status.peaks["max_frame"][m]>min_val and t>min_val:
                        arrays_r['xr_{}'.format(m+1)].append(source_intensity_ch1.data["time"][t])
                        arrays_r['yr1_{}'.format(m+1)].append(0)
                        arrays_r['yr2_{}'.format(m+1)].append(source_intensity_ch1.data["intensity"][t])
                        osc_dict["rising_frame"].append(t)
                        osc_dict["rising_time"].append(source_intensity_ch1.data["time"][t])

        for i in range(1,11):
            source_rising[i].data={'x':arrays_r['xr_{}'.format(i)], 'y1':arrays_r['yr1_{}'.format(i)], 'y2':arrays_r['yr2_{}'.format(i)]}

        if len(cellid.cell_status.peaks)==6:

            for m in range(len(cellid.cell_status.peaks["max_frame"])):
                min_val=cellid.cell_status.end_oscillation_frame
                for n in range(len(cellid.cell_status.peaks["min_frame"])):
                    if cellid.cell_status.peaks["max_frame"][m]<cellid.cell_status.peaks["min_frame"][n]:
                        min_val=cellid.cell_status.peaks["min_frame"][n]
                        break
                for t in range(cellid.cell_status.start_oscillation_frame, cellid.cell_status.end_oscillation_frame+1):

                    if t>cellid.cell_status.peaks["max_frame"][m] and cellid.cell_status.peaks["max_frame"][m]<min_val and t<min_val:
                        arrays_f['xf_{}'.format(m+1)].append(source_intensity_ch1.data["time"][t])
                        arrays_f['yf1_{}'.format(m+1)].append(0)
                        arrays_f['yf2_{}'.format(m+1)].append(source_intensity_ch1.data["intensity"][t])
                        osc_dict["falling_frame"].append(t)
                        osc_dict["falling_time"].append(source_intensity_ch1.data["time"][t])

                    if t==cellid.cell_status.end_oscillation_frame and min_val==cellid.cell_status.end_oscillation_frame:
                        arrays_f['xf_{}'.format(m+1)].append(source_intensity_ch1.data["time"][t])
                        arrays_f['yf1_{}'.format(m+1)].append(0)
                        arrays_f['yf2_{}'.format(m+1)].append(source_intensity_ch1.data["intensity"][t])
                        osc_dict["falling_frame"].append(t)
                        osc_dict["falling_time"].append(source_intensity_ch1.data["time"][t])                        
        for i in range(1,11):
            source_falling[i].data={'x':arrays_f['xf_{}'.format(i)], 'y1':arrays_f['yf1_{}'.format(i)], 'y2':arrays_f['yf2_{}'.format(i)]}

        if save_status and cellid!=None:
            cellrois = CellROI.objects.select_related().filter(cell_id=cellid)
            if DEBUG:
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

                if framenumber>=cellid.cell_status.time_of_death_frame: cellflag.alive = False
                else: cellflag.alive = True
                cellflag.save()


            cellstatus = cellid.cell_status
            cellstatus.flags = osc_dict
            cellstatus.save()

        if delete_status and cellid!=None:
            cellrois = CellROI.objects.select_related().filter(cell_id=cellid)
            for cellroi in cellrois:
                cellflag = cellroi.cellflag_cellroi
                cellflag.rising = False
                cellflag.falling = False
                cellflag.maximum = False
                cellflag.minimum = False
                cellflag.oscillating = False
                cellflag.alive = True
                cellstatus.flags = {}
                cellstatus.save()
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

    print('----------------------------   ',dropdown_pos.css_classes)

    #___________________________________________________________________________________________
    # Function to update the position depending on the experiment and the well
    def update_dropdown_pos(attr, old, new):
        if DEBUG:print('****************************  update_dropdown_pos ****************************')
        dropdown_pos.options = positions['{0}_{1}'.format(dropdown_exp.value, dropdown_well.value)]
        dropdown_pos.value = positions['{0}_{1}'.format(dropdown_exp.value, dropdown_well.value)][0]

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
        #update_source_osc_tod()

        slider_find_peaks.value = 30
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
        source_intensity_ch0.data={'time':[], 'intensity':[]}
        source_intensity_ch1.data={'time':[], 'intensity':[]}
        source_intensity_ch2.data={'time':[], 'intensity':[]}

        current_file=get_current_file()
        sample = Sample.objects.get(file_name=current_file)
        if sample.check_sample:
            position_check_div.text = "<b style='color:green; ; font-size:18px;'> Position validated</b>"
            position_check_button.label = "Position not validated"
        else:
            position_check_div.text = "<b style='color:red; ; font-size:18px;'> Position not validated</b>"
            position_check_button.label = "Position validated"

        if sample.keep_sample:
            position_keep_div.text  = "<b style='color:green; ; font-size:18px;'> Keep Position</b>"
            position_keep_button.label = "Don't Keep Position"
        else:
            position_keep_div.text  = "<b style='color:red; ; font-size:18px;'> Do not Keep Position</b>"
            position_keep_button.label = "Keep Position"


        cellIDs = CellID.objects.select_related().filter(sample=sample)


        if DEBUG:print('dropdown_cell.value = ',dropdown_cell.value)
        cell_list=[]
        for cid in cellIDs:
            cell_list.append(cid.name)
            if cid.name!=dropdown_cell.value:continue
            time_list={}
            intensity_list={}
            ROIs = CellROI.objects.select_related().filter(cell_id=cid)
            for roi in ROIs:
                for ch in roi.contour_cellroi.intensity_sum:
                    try:
                        time_list[ch]
                    except KeyError:
                        #time_list[ch]=[]
                        #intensity_list[ch]=[]
                        #CHANGE WITH ssample.number_of_frames when available
                        #nframes=sample.number_of_frames
                        frames=Frame.objects.select_related().filter(sample=sample)
                        nframes=len(frames)
                        intensity_list[ch]=[0 for i in range(nframes)]
                        time_list[ch]=[f.time/60000 for f in frames]
                        #time_list[ch].append((roi.frame.time/60000))
                        #intensity_list[ch].append(roi.contour_cellroi.intensity_sum[ch]/roi.contour_cellroi.number_of_pixels)

                    if   dropdown_intensity_type.value == 'sum': 
                        intensity_list[ch][roi.frame.number]= getattr(roi.contour_cellroi, 'intensity_sum')[ch]
                    elif dropdown_intensity_type.value == 'avg': 
                        intensity_list[ch][roi.frame.number]= getattr(roi.contour_cellroi, 'intensity_sum')[ch]/roi.contour_cellroi.number_of_pixels
                    elif   dropdown_intensity_type.value == 'max': 
                        intensity_list[ch][roi.frame.number]= getattr(roi.contour_cellroi, 'intensity_max')[ch]
                    elif   dropdown_intensity_type.value == 'std': 
                        intensity_list[ch][roi.frame.number]= getattr(roi.contour_cellroi, 'intensity_std')[ch]



            for index, key in enumerate(time_list):
                if index==0:
                    sorted_lists = sorted(zip(time_list[key], intensity_list[key])) 
                    time_sorted, intensity_sorted = zip(*sorted_lists) 
                    source_intensity_ch0.data={'time':time_sorted, 'intensity':intensity_sorted}
                if index==1:
                    sorted_lists = sorted(zip(time_list[key], intensity_list[key])) 
                    time_sorted, intensity_sorted = zip(*sorted_lists) 
                    source_intensity_ch1.data={'time':time_sorted, 'intensity':intensity_sorted}    
                if index==2:
                    sorted_lists = sorted(zip(time_list[key], intensity_list[key])) 
                    time_sorted, intensity_sorted = zip(*sorted_lists) 
                    source_intensity_ch2.data={'time':time_sorted, 'intensity':intensity_sorted}
        dropdown_cell.options=cell_list

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
        if DEBUG:
            print('dropdown_cell.value = ',dropdown_cell.value)
            print('dropdown_cell.options = ',dropdown_cell.options)
        prepare_intensity()

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
        images, images_norm = get_current_stack()
        source_imgs.data      = {'images':images}
        source_imgs_norm.data = {'images':images_norm}
        source_img_ch.data    = {'img':[images[ch][0] for ch in range(len(images))]}

        new_image = images_norm[int(dropdown_channel.value)][0]
        source_img.data = {'img':[new_image]}
        dropdown_channel.value = dropdown_channel.options[0]
        dropdown_color.value = dropdown_color.options[0]

        if DEBUG:
            print('prepare_pos dropdown_channel.value ',dropdown_channel.value)
            print('prepare_pos dropdown_channel.options ',dropdown_channel.options)
            print('prepare_pos before slider')
        if slider.value == 0:
            if DEBUG:print('in the if prepare_pos')
            left_rois,right_rois,top_rois,bottom_rois,height_labels, weight_labels, names_labels,height_cells, weight_cells, names_cells=update_source_roi_cell_labels()
            source_roi.data = {'left': left_rois, 'right': right_rois, 'top': top_rois, 'bottom': bottom_rois}
            source_labels.data = {'height':height_labels, 'weight':weight_labels, 'names':names_labels}
            source_cells.data = {'height':height_cells, 'weight':weight_cells, 'names':names_cells}
        else:
            if DEBUG:print('in the else prepare_pos')
            slider.value = 0
            slider.start = 0
            slider.end   = len(source_imgs.data['images'][0]) - 1

        if DEBUG:print('prepare_pos after slider')
        update_dropdown_cell('','','')
        slider.end=len(source_imgs.data['images'][0]) - 1
        #prepare_intensity()
        reset_tap_tool()
        update_source_osc_tod()
        update_dropdown_channel('','','')
        intensity_type_callback('','','')

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
        if DEBUG:
            print('dropdown_intensity_type value=',dropdown_intensity_type.value)
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

    int_type_list = ["avg", "max", "sum",  "std"]
    dropdown_intensity_type = bokeh.models.Select(value=int_type_list[0], title="intensity", options=int_type_list)
    dropdown_intensity_type.on_change('value', intensity_type_callback)
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





    #___________________________________________________________________________________________
    # Define a callback to update bf_display with slider
    def callback_slider(attr: str, old: Any, new: Any) -> None:
        if DEBUG:print('****************************  callback_slider ****************************')
        time_point = slider.value
        images_norm=source_imgs_norm.data['images']
        new_image = images_norm[int(dropdown_channel.value)][time_point]
        source_img.data = {'img':[new_image]}

        images=source_imgs.data['images']
        source_img_ch.data = {'img':[images[ch][time_point] for ch in range(len(images))]}

        left_rois,right_rois,top_rois,bottom_rois,height_labels, weight_labels, names_labels,height_cells, weight_cells, names_cells=update_source_roi_cell_labels()

        source_roi.data    = {'left': left_rois, 'right': right_rois, 'top': top_rois, 'bottom': bottom_rois}
        source_labels.data = {'height':height_labels, 'weight':weight_labels, 'names':names_labels}
        source_cells.data  = {'height':height_cells, 'weight':weight_cells, 'names':names_cells}
        if len(source_intensity_ch1.data["time"])==0:
            line_position.location = -999
        else:
            line_position.location = source_intensity_ch1.data["time"][time_point]

    slider.on_change('value', callback_slider)
    #___________________________________________________________________________________________
    

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
        if isinstance(event, bokeh.events.SelectionGeometry):
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

            inspect_cells_callback()
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
        current_file=get_current_file()
        current_index=get_current_index()

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
                out_dir_name  = os.path.join(os.sep, "data","singleCell_catalog","contour_data",exp.name, expds.data_name, os.path.split(sample.file_name)[-1].replace('.nd2',''))
                out_file_name = os.path.join(out_dir_name, "frame{0}_ROI{1}.json".format(frame[0].number, roi_number))
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
                                  file_name=out_file_name,
                                  cell_roi=roi,
                                  type="cell_ROI",
                                  mode="manual")
                contour.save()

                cellflag = CellFlag(cell_roi=roi)
                cellflag.save()

        left_rois, right_rois, top_rois, bottom_rois,height_labels, weight_labels, names_labels, height_cells, weight_cells, names_cells= update_source_roi_cell_labels()
        source_labels.data = {'height':height_labels, 'weight':weight_labels, 'names':names_labels}
        source_cells.data = {'height':height_cells, 'weight':weight_cells, 'names':names_cells}
        
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
        images=source_imgs.data["images"]
        current_index = (current_index + 1*way) % len(images[0])
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

    position_check_div = bokeh.models.Div(text="<b style='color:red; ; font-size:18px;'> Position not validated</b>")
    #___________________________________________________________________________________________
    def position_check_callback():
        current_file=get_current_file()
        sample   = Sample.objects.get(file_name=current_file)
        if sample.check_sample == False:
            sample.check_sample = True
            position_check_button.label = "Position not validated"
            position_check_div.text = "<b style='color:green; ; font-size:18px;'> Position validated</b>"
        else:
            sample.check_sample = False
            position_check_button.label = "Position Validated"
            position_check_div.text = "<b style='color:red; ; font-size:18px;'> Position not validated</b>"
        sample.save()
        update_position_select()
    position_check_button = bokeh.models.Button(label="Validate Position")
    position_check_button.on_click(position_check_callback)
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
        update_position_select()

    position_keep_button = bokeh.models.Button(label="Don't keep Position")
    position_keep_button.on_click(position_keep_callback)
    #___________________________________________________________________________________________

    #___________________________________________________________________________________________
    def remove_roi_callback():
        if DEBUG:print('****************************  remove_roi_callback ****************************')
        removeROIs(get_current_file())
    button_remove_roi = bokeh.models.Button(label="Remove ROI")
    button_remove_roi.on_click(remove_roi_callback)
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
    # Go to next frame with possible issue
    def build_cells_callback():
        if DEBUG:print('****************************  build_cells_callback ****************************')
        current_file=get_current_file()
        build_cells_sample(sample=current_file)
        left_rois, right_rois, top_rois, bottom_rois,height_labels, weight_labels, names_labels, height_cells, weight_cells, names_cells= update_source_roi_cell_labels()
        source_cells.data = {'height':height_cells, 'weight':weight_cells, 'names':names_cells}
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
            find_peaks_slider_callback('','',30)
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
            find_peaks_slider_callback('','',30)
    button_end_oscillation = bokeh.models.Button(label="Osc. End")
    button_end_oscillation.on_click(end_oscillation_callback)
    #___________________________________________________________________________________________

    #___________________________________________________________________________________________
    # Set cell time of death
    def time_of_death_callback():
        current_index=get_current_index()
        time_of_death_position.location = source_intensity_ch1.data["time"][current_index]
        source_varea_death.data['x']  = [source_intensity_ch1.data["time"][t] for t in range(current_index, len(source_intensity_ch1.data["time"])) ]
        source_varea_death.data['y1'] = [source_intensity_ch1.data["intensity"][t] for t in range(current_index, len(source_intensity_ch1.data["intensity"])) ]
        source_varea_death.data['y2'] = [0 for i in range(len(source_varea_death.data['y1']))]
        sample = Sample.objects.get(file_name=get_current_file())
        cellsid = CellID.objects.select_related().filter(sample=sample, name=dropdown_cell.value)
        cellstatus = cellsid[0].cell_status
        cellstatus.time_of_death_frame = current_index
        frame = Frame.objects.select_related().filter(sample=sample, number=current_index)
        cellstatus.time_of_death = frame[0].time/60000.
        cellstatus.save()
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
        if DEBUG:
            print('source_intensity_ch1 ',source_intensity_ch1.data["intensity"])
            print('int_array            ',int_array)
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

            cellstatus.save()

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
            cellstatus.n_oscillations = 0

            cellstatus.time_of_death_frame     = -999
            cellstatus.start_oscillation_frame = -999
            cellstatus.end_oscillation_frame   = -999

            cellstatus.time_of_death     = -9999
            cellstatus.start_oscillation = -9999
            cellstatus.end_oscillation   = -9999

            cellstatus.save()
            set_rising_falling(cellsid[0], delete_status=True)
            update_source_osc_tod()
    button_delete_peaks = bokeh.models.Button(label="Delete Peaks")
    button_delete_peaks.on_click(delete_peaks_callback)
    #___________________________________________________________________________________________

    #___________________________________________________________________________________________
    def update_source_osc_tod():
        if DEBUG:print('------------------------update_source_osc_tod-------------------------')
        well = ExperimentalDataset.objects.get(data_name=dropdown_well.value)
        nframes = well.experiment.number_of_frames
        samples = Sample.objects.select_related().filter(experimental_dataset = well)
        n_osc=[]
        tod=[]
        start_osc=[]
        end_osc=[]
        for sample in samples:
            cellids = CellID.objects.select_related().filter(sample=sample)
            for cellid in cellids:
                n_osc.append(cellid.cell_status.n_oscillations)
                start_osc.append(cellid.cell_status.start_oscillation)
                end_osc.append(cellid.cell_status.end_oscillation)
                tod.append(cellid.cell_status.time_of_death)

        max_osc=0
        if max(n_osc, default=0)>0:
            max_osc=max(n_osc, default=0)
        hist, edges = np.histogram(n_osc, bins=max_osc+2, range=(0, max_osc+2))
        source_nosc.data={'x': edges[:-1], 'top': hist}

        hist, edges = np.histogram(tod, bins=nframes*10, range=(0, nframes*10))
        source_tod.data={'x': edges[:-1], 'top': hist}

        hist, edges = np.histogram(start_osc, bins=nframes*10, range=(0, nframes*10))
        source_start_osc.data={'x': edges[:-1], 'top': hist}

        hist, edges = np.histogram(end_osc, bins=nframes*10, range=(0, nframes*10))
        source_end_osc.data={'x': edges[:-1], 'top': hist}

        if DEBUG:
            print('source_nosc = ',source_nosc.data)
            print('===============---------------plot_intensity.y_range.start=',plot_intensity.y_range.start,'  plot_intensity.y_range.end=',plot_intensity.y_range.end)

    #___________________________________________________________________________________________



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



    #elongated      = models.BooleanField(help_text="round cell flag", default=False, blank=True)


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


    # Create a Div widget with some text
    text = bokeh.models.Div(text="<h2>Cell informations</h2>")

    left_rois, right_rois, top_rois, bottom_rois,height_labels, weight_labels, names_labels, height_cells, weight_cells, names_cells= update_source_roi_cell_labels()
    source_roi  = bokeh.models.ColumnDataSource(data=dict(left=left_rois, right=right_rois, top=top_rois, bottom=bottom_rois))
    source_roi_manual  = bokeh.models.ColumnDataSource(data=dict(left=[], right=[], top=[], bottom=[]))

    source_labels = bokeh.models.ColumnDataSource(data=dict(height=height_labels,weight=weight_labels,names=names_labels))
    labels = bokeh.models.LabelSet(x='weight', y='height', text='names', x_units='data', y_units='data',
                                   x_offset=0, y_offset=0, source=source_labels, text_color='white', text_font_size="10pt")

    plot_image.add_layout(labels)

    source_cells = bokeh.models.ColumnDataSource(data=dict(height=height_cells,weight=weight_cells,names=names_cells))
    labels_cells = bokeh.models.LabelSet(x='weight', y='height', text='names', x_units='data', y_units='data',
                                         x_offset=0, y_offset=-15, source=source_cells, text_color='white', text_font_size="11pt")

    plot_image.add_layout(labels_cells)
    #plot_image.add_layout(color_bar, 'right')

    plot_image.image(image='img', x=0, y=0, dw=ind_images_list[0][0].shape[0], dh=ind_images_list[0][0].shape[1], source=source_img, color_mapper=color_mapper)

       # Create a ColumnDataSource to store image data
    #source_url = bokeh.models.ColumnDataSource({'url': [''], 'x': [0], 'y': [0], 'dw': [0], 'dh': [0]})
    #ind_images_list_url = get_current_stack_url()

    #source_url.data = {'url': [f'data:image/png;base64,{ind_images_list_url[0][0]}'], 'x': [0], 'y': [0], 'dw': [ind_images_list[0][0].shape[1]], 'dh': [ind_images_list[0][0].shape[0]]}

    #image_plot = plot_image.image_url(url='url', x='x', y='y', w='dw', h='dh', source=source_url)

    current_file=get_current_file()
    sample = Sample.objects.get(file_name=current_file)

    if sample.check_sample:
        position_check_div.text = "<b style='color:green; ; font-size:18px;'> Position validated</b>"
        position_check_button.label = "Position not validated"
    else:
        position_check_div.text = "<b style='color:red; ; font-size:18px;'> Position not validated</b>"
        position_check_button.label = "Position validated"

    if sample.keep_sample:
        position_keep_div.text  = "<b style='color:green; ; font-size:18px;'> Keep Position</b>"
        position_keep_button.label = "Don't keep Position"
    else:
        position_keep_div.text  = "<b style='color:red; ; font-size:18px;'> Do not Keep Position</b>"
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
        print('------------------------- time lisrt',time_list)


    plot_intensity.line('time', 'intensity', source=source_intensity_ch1, line_color='blue')
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

    plot_intensity.segment(x0='time', y0=0, x1='time', y1='intensity', line_color='black', line_width=0.5, source=source_segments_cell, line_dash="dotted")

    index_source = bokeh.models.ColumnDataSource(data=dict(index=[]))  # Data source for the image
    tap_tool = bokeh.models.TapTool(callback=bokeh.models.CustomJS(args=dict(other_source=index_source),code=select_tap_callback()))

    plot_intensity.add_tools(tap_tool)
    index_source.on_change('data', update_image_tap_callback)

    box_select_tool = bokeh.models.BoxSelectTool(select_every_mousemove=False)
    plot_intensity.add_tools(box_select_tool)

    plot_intensity.y_range.start=0
    plot_intensity.x_range.start=-10
    #for ch in range(len(time_list)):
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
    line_position = bokeh.models.Span(location=initial_position, dimension='height', line_color='red', line_width=2)
    plot_intensity.add_layout(line_position)

    initial_position=-9999
    start_oscillation_position = bokeh.models.Span(location=initial_position, dimension='height', line_color='blue', line_width=2)
    plot_intensity.add_layout(start_oscillation_position)
    end_oscillation_position = bokeh.models.Span(location=initial_position, dimension='height', line_color='blue', line_width=2)
    plot_intensity.add_layout(end_oscillation_position)
    time_of_death_position = bokeh.models.Span(location=initial_position, dimension='height', line_color='black', line_width=2)
    plot_intensity.add_layout(time_of_death_position)

    source_varea_death = bokeh.models.ColumnDataSource(data=dict(x=[], y1=[], y2=[]))
    plot_intensity.varea(x='x', y1='y1', y2='y2', fill_alpha=0.10, fill_color='black', source=source_varea_death)


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


    source_nosc = bokeh.models.ColumnDataSource(data=dict(x=[], top=[]))
    source_start_osc = bokeh.models.ColumnDataSource(data=dict(x=[], top=[]))
    source_end_osc = bokeh.models.ColumnDataSource(data=dict(x=[], top=[]))
    source_tod = bokeh.models.ColumnDataSource(data=dict(x=[], top=[]))
    update_source_osc_tod()
    plot_osc_tod.vbar(x='x', top='top', width=3, source=source_start_osc, alpha=0.5, color='green', line_color=None)
    plot_osc_tod.vbar(x='x', top='top', width=3, source=source_end_osc, alpha=0.5, color='red', line_color=None)
    plot_osc_tod.vbar(x='x', top='top', width=3, source=source_tod, alpha=0.5, color='black', line_color=None)
    plot_nosc.vbar(x='x', top='top', width=0.5, source=source_nosc, alpha=0.5, color='black', line_color=None)

    prepare_intensity() 

    # Add the rectangle glyph after adding the image
    quad = bokeh.models.Quad(left='left', right='right', top='top', bottom='bottom', fill_color=None)#, fill_alpha=0.0, fill_color='#009933')
    plot_image.add_glyph(source_roi, quad, selection_glyph=quad, nonselection_glyph=quad)

    # Remove the axes
    plot_image.axis.visible = False
    plot_image.grid.visible = False


    # Sample data
    data_pie = {'categories': ['A', 'B', 'C', 'D'], 'values': [20, 30, 25, 25]}

    # Calculate the angles for the sectors
    angles = [data_pie['values'][0]/sum(data_pie['values']) * 2*math.pi]
    for i in range(1, len(data_pie['values'])):
        angles.append(angles[-1] + data_pie['values'][i]/sum(data_pie['values']) * 2*math.pi)

    # Create a figure
    fig_pie = bokeh.plotting.figure(height=250, title="Pie Chart", toolbar_location=None, tools="hover", tooltips="@categories: @values")

    # Draw the sectors of the pie chart
    fig_pie.wedge(x=0, y=1, radius=0.4,
                  start_angle=angles[:-1], end_angle=angles[1:],
                  color=["red", "green", "blue", "orange"],
                  legend_field="categories", source=data_pie)

    # Hide the axes
    fig_pie.axis.visible = False


    exp_color_col = bokeh.layouts.column(bokeh.layouts.row(dropdown_exp),
                                         bokeh.layouts.row(dropdown_well),
                                         bokeh.layouts.row(dropdown_pos), 
                                         bokeh.layouts.row(dropdown_channel),
                                         bokeh.layouts.row(dropdown_color),
                                         bokeh.layouts.row(bokeh.layouts.Spacer(width=10),contrast_slider),
                                         bokeh.layouts.row(position_check_button),
                                         bokeh.layouts.row(position_keep_button),
                                         bokeh.layouts.row(button_remove_roi),
                                         )

    right_col = bokeh.layouts.column(bokeh.layouts.row(slider),
                                     bokeh.layouts.row(button_play_stop, button_prev, button_next, dropdown_refresh_time ),
                                     bokeh.layouts.row(button_delete_roi, button_save_roi, dropdown_cell ),
                                     bokeh.layouts.row(button_inspect, button_build_cells),
                                     bokeh.layouts.row(button_start_oscillation,button_end_oscillation,button_time_of_death),
                                     bokeh.layouts.row(button_save_peaks, button_delete_peaks, dropdown_intensity_type),
                                     bokeh.layouts.row(slider_find_peaks),
                                     bokeh.layouts.row(button_mask_cells, button_dividing_cells, button_double_nuclei_cells),
                                     bokeh.layouts.row(button_multiple_cells, button_pair_cell),
                                     bokeh.layouts.row(button_flat_cell, button_round_cell, button_elongated_cell),
                                     )
    
    intensity_plot_col = bokeh.layouts.column(bokeh.layouts.row(plot_intensity),
                                              bokeh.layouts.row(plot_osc_tod))

    cell_osc_plot_col = bokeh.layouts.column(bokeh.layouts.row(plot_image),
                                             bokeh.layouts.row(plot_nosc))

    norm_layout = bokeh.layouts.column(bokeh.layouts.row(position_check_div, bokeh.layouts.Spacer(width=10),position_keep_div),
                                       bokeh.layouts.row(exp_color_col, cell_osc_plot_col, right_col, intensity_plot_col),
                                       bokeh.layouts.row(fig_pie),
                                       bokeh.layouts.row(text))

    doc.add_root(norm_layout)



#___________________________________________________________________________________________
#@login_required
def index(request: HttpRequest) -> HttpResponse:
#def index(request):


    """View function for home page of site."""
    print('The visualisation request method is:', request.method)
    print('The visualisation POST data is:     ', request.POST)
    cell_dict={}

    #THIS BUILDS THE FRAMES FROM THE RAWDATASET CATALOG WITH TAG "SEGMENTME", CREATES UP TO FRAMES
    if 'register_rawdataset' in request.POST and LOCAL==False:
        register_rawdataset()

    #THIS BUILDS THE ROIS FOR ALL THE EXISTING SAMPLES
    if 'build_ROIs' in request.POST:
        build_ROIs()

    #THIS SEGMENTS ALL THE EXPERIMENTS/POSITIONS IT WILL FIND. CREATES UP TO CONTOUR/DATA
    if 'segment' in request.POST:
        segment()

    if 'build_cells' in request.POST:
        build_cells_all_exp()


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
    selected_dict={
        'experiment':'',
        'well':'',
        'position':'',
        'segmentation':'',
        'segmentation_channel':''
    }

    selected_experiment=request.POST.get('select_experiment')
    selected_dict['experiment']=selected_experiment
    selected_well=request.POST.get('select_well')
    selected_dict['well']=selected_well
    selected_position=request.POST.get('select_position')
    selected_dict['position']=selected_position
    selected_segmentation=request.POST.get('select_segmentation')
    selected_dict['segmentation']=selected_segmentation
    selected_segmentation_channel=request.POST.get('select_segmentation_channel')
    selected_dict['segmentation_channel']=selected_segmentation_channel

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

    print('===========================================',cell_dict)
    uri=None
    if cell_dict != None:
        fig = plt.figure(figsize=(15,5))    
        for cell in cell_dict: 
            channels=[]
            for ch in cell_dict[cell]:
                if ch in channels:continue
                if 'intensity_' not in ch: continue
                channels.append(ch)
            print('channels  ',channels)
            for ch in channels:
                normint=[]
                for p in range(len(cell_dict[cell][ch])):
                    normint.append(cell_dict[cell][ch][p]/cell_dict[cell]['npixels'][p])
                if '_BF' in ch:continue
                plt.plot(cell_dict[cell]['time'], normint)
                print('channel ================== ',ch)
                print(normint)
        fig.tight_layout()
        fig = plt.gcf()
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        string = base64.b64encode(buf.read())
        uri = urllib.parse.quote(string)

    experiment_dict={}
    contribution_dict=[]
    if selected_experiment != None:
        #GET THE LIST OF EXPERIMENTS
        experiment_dict=get_experiement_details(selected_experiment)
        #GET THE CONTRIBUTION TO THE EXPERIMENT 
        contribution_dict=get_contribution_details(selected_experiment)

    treatment_dict=[]
    injection_dict=[]
    instrumental_dict=[]
    sample_dict=[]
    if selected_well != None:
        #GET THE EXPERIMENTAL DATASET DETAILS: TREATMENT
        treatment_dict = get_treatment_details(selected_well)
        #GET THE EXPERIMENTAL DATASET DETAILS: INSTRUMENTAL CONDITIONS
        injection_dict = get_injection_details(selected_well)
        #GET THE EXPERIMENTAL DATASET DETAILS: INJECTION
        instrumental_dict = get_instrumental_details(selected_well)
        #GET THE EXPERIMENTAL DATASET DETAILS: SAMPLE
        sample_dict = get_sample_details(selected_well)
    

    script = None

    script = bokeh.embed.server_document(request.build_absolute_uri())
    print("request.build_absolute_uri() ",request.build_absolute_uri())

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
        #'script': script
    }

    return render(request, 'segmentation/index.html', context=context)

    

#___________________________________________________________________________________________
#@login_required
def bokeh_dashboard(request: HttpRequest) -> HttpResponse:

    script = bokeh.embed.server_document(request.build_absolute_uri())
    print("request.build_absolute_uri() ",request.build_absolute_uri())
    context = {'script': script}

    return render(request, 'segmentation/bokeh_dashboard.html', context=context)
