from django.shortcuts import render
from django.db import reset_queries
from django.db import connection
from django.http import HttpRequest, HttpResponse

from segmentation.models import Experiment, ExperimentalDataset, Sample, Frame, Contour, Data, Segmentation, SegmentationChannel, CellID, CellROI

import os, sys, json, glob, gc
import time

from memory_profiler import profile
from sklearn.cluster import DBSCAN
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
import io
import urllib, base64

import math

from typing import Any

import nd2
from pathlib import Path

LOCAL=True
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
        print('adding experiment with name:  ',x[1])


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
            print('    adding experimental dataset with name ',os.path.join(x[4], x[5]))

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
                print('        adding sample with name ',fname)

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
                    print('            adding frame with name ',fr)
                    frame.save()

#___________________________________________________________________________________________
def segment():
    #loop over all experiments
    exp_list = Experiment.objects.all()

    for exp in exp_list:
        print(' ---- SEGMENTATION exp name ',exp.name)
        print(' ---- SEGMENTATION channels ',exp.name_of_channels.split(','),' number ',exp.number_of_channels, ' full file name ', exp.file_name)
        segExist=False
        #build default segmentation class, to be replaced by calls from django app
        default_segmentation = segtools.customLocalThresholding_Segmentation(threshold=2., delta=2, npix_min=400, npix_max=4000)
        default_segmentation.channels = exp.name_of_channels.split(',')
        default_segmentation.channel = 0

        default_segmentation_2 = segtools.customLocalThresholding_Segmentation(threshold=2., delta=1, npix_min=400, npix_max=4000)
        default_segmentation_2.channels = exp.name_of_channels.split(',')
        default_segmentation_2.channel = 0

        #check existing segmentation if already registered
        segmentations = Segmentation.objects.select_related().filter(experiment = exp)
        for seg in segmentations:
            if default_segmentation.get_param() == seg.algorithm_parameters and \
                default_segmentation.get_type() == seg.algorithm_type and \
                    default_segmentation.get_version() == seg.algorithm_version:
                print('SEGMENTATION EXISTTTTTTTTT')
                #check if the segmentation channel exists
                segmentation_channels = SegmentationChannel.objects.select_related().filter(segmentation = seg)
                for seg_ch in segmentation_channels:
                    if seg_ch.channel_number == default_segmentation.channel and \
                        seg_ch.channel_name == default_segmentation.channels[default_segmentation.channel]:
                        segExist=True
        if segExist: continue

        print('============= default_segmentation.get_param()   = ',default_segmentation.get_param())
        print('============= default_segmentation.get_type()    = ',default_segmentation.get_type())
        print('============= default_segmentation.get_version() = ',default_segmentation.get_version())
        print('============= default_segmentation.channels      = ',default_segmentation.channels)
        print('============= default_segmentation.channel       = ',default_segmentation.channel)

        #create segmentation and segmentation channel if it does not exist
        segmentation = Segmentation(name="default segmentation", 
                                    experiment=exp,
                                    algorithm_type=default_segmentation.get_type(),
                                    algorithm_version=default_segmentation.get_version(),
                                    algorithm_parameters=default_segmentation.get_param())
        segmentation.save()
        segmentation_channel = SegmentationChannel(segmentation=segmentation,
                                                channel_name=exp.name_of_channels.split(',')[0],
                                                channel_number=0)
        segmentation_channel.save()

        print(' ---- SEGMENTATION exp name ',exp.name)
        experimentaldataset = ExperimentalDataset.objects.select_related().filter(experiment = exp)

        for expds in experimentaldataset:
            print('    ---- SEGMENTATION experimentaldataset name ',expds.data_name, expds.data_type)
            samples = Sample.objects.select_related().filter(experimental_dataset = expds)

            counter_samp=0
            for s in samples:
                if counter_samp==4: 
                    print('===========================================')
                    break
                counter_samp+=1
                print('         ---- SEGMENTATION sample name ',s.file_name)
                frames = Frame.objects.select_related().filter(sample = s)
                print('getting the images')
                images, channels = read.nd2reader_getFrames(s.file_name)
                print ('          ---- SEGMENTATION will loop over ',len(frames),' frames')

                for f in frames:
                    print( 'getting contour for frame ',f.number)
                    contour_list = default_segmentation.segmentation(images[f.number])

                    print(' got ',len(contour_list),' contours')
                    for cont in contour_list:
                        pixels_data_contour  = Data(all_pixels=cont['all_pixels_contour'], single_pixels=cont['single_pixels_contour'])
                        pixels_data_contour.save()
                        pixels_data_inside   = Data(all_pixels=cont['all_pixels_inside'],  single_pixels=cont['single_pixels_inside'])
                        pixels_data_inside.save()
                        print(cont['center'])
                        contour = Contour(frame=f,
                                          pixels_data_contour=pixels_data_contour,
                                          pixels_data_inside=pixels_data_inside,
                                          segmentation_channel=segmentation_channel,
                                          center=cont['center'])
                        contour.save()

                        del contour
                    del contour_list
                    #print('gc collect 1: ',gc.collect())

                print('gc collect 1: ',gc.collect())

#___________________________________________________________________________________________
def build_cells(sample=None):
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
                eps= ((cellroi.max_col-cellroi.min_col)/2. + (cellroi.max_row-cellroi.min_row)/2.)/4.
                clustering = DBSCAN(eps=eps, min_samples=20).fit(X)
                print(clustering.labels_)

                #Create the cells ID according to existing clusters (one per cluster >=0)
                #Connect the cellFrames to cellID
                createdcells=[]
                cellid_dict={}
                for cid in range(len(clustering.labels_)):
                    if clustering.labels_[cid] not in createdcells and clustering.labels_[cid]!=-1:
                        cellid = CellID(sample=s, name='cell{}'.format(clustering.labels_[cid]))
                        cellid.save()
                        createdcells.append(clustering.labels_[cid])
                        cellid_dict['cell{}'.format(clustering.labels_[cid])]=cellid
                    if clustering.labels_[cid]!=-1:
                        cell_roi_list[cid].cell_id = cellid_dict['cell{}'.format(clustering.labels_[cid])]
                        cell_roi_list[cid].save()



#___________________________________________________________________________________________
def intensity(experiment='', well='', position=''):
    exp_list = Experiment.objects.all()
    for exp in exp_list:
        if experiment!='' and experiment!=exp.name: continue
        print('---- INTENSITY experiment name ',exp.name)
        experimentaldataset = ExperimentalDataset.objects.select_related().filter(experiment = exp)
        for expds in experimentaldataset:
            if well!='' and well!=expds.data_name: continue
            print('    ---- INTENSITY experimentaldataset name ',expds.data_name, expds.data_type)
            samples = Sample.objects.select_related().filter(experimental_dataset = expds)
            for s in samples:
                if position!='' and position!=s.file_name: continue
                print('        ---- INTENSITY sample name ',s.file_name)
                cellsid = CellID.objects.select_related().filter(sample = s)
                cell_dict={}
                for cid in cellsid:
                    print('            ---- INTENSITY cellid name ',cid.name)
                    cell_dict[cid.name]={'time':[],'npixels':[]}
                    cell_frames = CellFrame.objects.select_related().filter(cell_id=cid)
                    for cf in cell_frames:
                        contours = Contour.objects.select_related().filter(cell_frame=cf)
                        #NEED TO SELECT contours for a given segmentation properly
                        for cont in contours:
                            cell_dict[cid.name]['time'].append(cont.frame.time)
                            cell_dict[cid.name]['npixels'].append(cont.pixels_data_inside.all_pixels['npixels'])

                            for ch in cont.pixels_data_inside.all_pixels['sum_intensity']:
                                try:
                                    cell_dict[cid.name]['intensity_{}'.format(ch.replace(' ',''))]
                                except KeyError:
                                    cell_dict[cid.name]['intensity_{}'.format(ch.replace(' ',''))]=[]
                                cell_dict[cid.name]['intensity_{}'.format(ch.replace(' ',''))].append(cont.pixels_data_inside.all_pixels['sum_intensity'][ch])
                            #print(cont.pixels_data_inside.all_pixels)
                            #print('contour ID=',cont.id,'  center=',cont.center, ' seg channel ',cont.segmentation_channel.channel_name,', ',cont.segmentation_channel.channel_number\
                            #      ,' seg name=',cont.segmentation_channel.segmentation.name,' seg type=',cont.segmentation_channel.segmentation.algorithm_type)
                    print('   n cell_frames=',len(cell_frames))
                print(cell_dict)
                return cell_dict

#___________________________________________________________________________________________
def build_ROIs():
    exp_list = Experiment.objects.all()
    for exp in exp_list:
        experimentaldataset = ExperimentalDataset.objects.select_related().filter(experiment = exp)
        for expds in experimentaldataset:
            samples = Sample.objects.select_related().filter(experimental_dataset = expds)
            counter_samp=0
            for s in samples:
                if counter_samp==1: 
                    print('===================BREAK ROIS========================')
                    break
                counter_samp+=1
                frames = Frame.objects.select_related().filter(sample=s)
                images, channels = read.nd2reader_getFrames(s.file_name)
                #images are t, c, x, y 
                BF_images=images.transpose(1,0,2,3)
                BF_images=BF_images[0]
                counter_frame=0
                for frame in frames:
                    #if counter_frame==2: 
                    #    print('===================BREAK Frame========================')
                    #    break
                    #counter_frame+=1
                    rois = CellROI.objects.select_related().filter(frame = frame)
                    #Just for now, should normally check that same ROI don't overlap
                    if len(rois)>0: continue
                    ROIs = segtools.get_ROIs_per_frame(BF_images[frame.number])

                    for r in range(len(ROIs)):
                        roi = CellROI(min_row = ROIs[r][0], min_col = ROIs[r][1],
                                      max_row = ROIs[r][2], max_col = ROIs[r][3], 
                                      frame = frame, roi_number=r)
                        roi.save()


                        cropped_dict = {'shape_original':BF_images[frame.number].shape}
                        out_dir_name  = os.path.join(os.sep, "data","singleCell_catalog","contour_data",exp.name, expds.data_name, os.path.split(s.file_name)[-1].replace('.nd2',''))
                        out_file_name = os.path.join(out_dir_name, "frame{0}_ROI{1}_all.json".format(frame.number, r))
                        if not os.path.exists(out_dir_name):
                            os.makedirs(out_dir_name)
                        cropped_img = images[frame.number][:, ROIs[r][0]:ROIs[r][2], ROIs[r][1]:ROIs[r][3]]
                        cropped_dict['shape']=[cropped_img.shape[1], cropped_img.shape[2]]
                        cropped_dict['npixels']=cropped_img.shape[1]*cropped_img.shape[2]
                        for ch in range(len(channels)):
                            cropped_dict['intensity_{}'.format(channels[ch].replace(" ",""))] = cropped_img[ch].tolist()     
                        out_file = open(out_file_name, "w") 
                        json.dump(cropped_dict, out_file) 
                        out_file.close() 


                        out_file_name = os.path.join(out_dir_name, "frame{0}_ROI{1}_summary.json".format(frame.number, r))
                        cropped_dict = {'npixels':cropped_img.shape[1]*cropped_img.shape[2], 
                                        'shape_original':BF_images[frame.number].shape, 
                                        'shape':[cropped_img.shape[1], cropped_img.shape[2]],
                                        'shift':[ROIs[r][0], ROIs[r][1]]}
                        for ch in range(len(channels)): 
                            cropped_dict['intensity_{}'.format(channels[ch].replace(" ",""))]={'intensity_sum':float(np.sum(cropped_img[ch])),
                                                                                               'intensity_mean':float(np.mean(cropped_img[ch])), 
                                                                                               'intensity_std':float(np.std(cropped_img[ch])) }
                        out_file = open(out_file_name, "w") 
                        json.dump(cropped_dict, out_file) 
                        out_file.close() 

                        contour = Contour(center_x_pix=ROIs[r][1]+(ROIs[r][3]-ROIs[r][1])/2., 
                                          center_y_pix=ROIs[r][0]+(ROIs[r][2]-ROIs[r][0])/2.,
                                          center_z_pix=0, 
                                          center_x_mic=(ROIs[r][1]+(ROIs[r][3]-ROIs[r][1])/2.)*roi.frame.pixel_microns+roi.frame.pos_x,
                                          center_y_mic=(ROIs[r][0]+(ROIs[r][2]-ROIs[r][0])/2.)*roi.frame.pixel_microns+roi.frame.pos_y,
                                          center_z_mic=0,
                                          intensity_mean=np.mean(cropped_img[ch]),
                                          intensity_std=np.std(cropped_img[ch]),
                                          intensity_sum=np.sum(cropped_img[ch]),
                                          number_of_pixels=cropped_img.shape[1]*cropped_img.shape[2],
                                          file_name=out_file_name,
                                          cell_roi=roi,
                                          type="cell_ROI",
                                          mode="auto")
                        contour.save()


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
                files['{0}_{1}'.format(exp.name, expds.data_name)].append(samp.file_name)

    experiments=sorted(experiments)
    for i in wells:
        wells[i] = sorted(wells[i])
    for i in positions:
        positions[i] = sorted(positions[i])

    dropdown_exp  = bokeh.models.Select(value=experiments[0], title='Experiment', options=experiments)
    dropdown_well = bokeh.models.Select(value=wells[experiments[0]][0], title='Well', options=wells[dropdown_exp.value])    
    dropdown_pos  = bokeh.models.Select(value=positions['{0}_{1}'.format(experiments[0], wells[experiments[0]][0])][0],title='Position', options=positions['{0}_{1}'.format(dropdown_exp.value, dropdown_well.value)])


    # Function to get the current file
    #___________________________________________________________________________________________
    def get_current_file():
        print('****************************  get_current_file ****************************')
        print('--------------- get_current_file() dropdown_exp.value=', dropdown_exp.value, '   dropdown_well.value',dropdown_well.value, '  dropdown_pos.value',dropdown_pos.value)

        current_files = files['{0}_{1}'.format(dropdown_exp.value, dropdown_well.value)]
        current_file = ''
        for f in current_files:
            if dropdown_pos.value in f:
                current_file = f
        print('--------------- get_current_file() current file  ',current_file)
        return current_file


    # Function to get the image stack
    #___________________________________________________________________________________________
    def get_current_stack():
        print('****************************  get_current_stack ****************************')
        current_file=get_current_file()
        time_lapse_path = Path(current_file)
        time_lapse = nd2.imread(time_lapse_path.as_posix())
        ind_images_list=[]
        for nch in range(time_lapse.shape[1]):
            time_lapse_tmp = time_lapse[:,nch,:,:] # Assume I(t, c, x, y)
            time_domain = np.asarray(np.linspace(0, time_lapse_tmp.shape[0] - 1, time_lapse_tmp.shape[0]), dtype=np.uint)
            ind_images = [np.flip(time_lapse_tmp[i,:,:],0) for i in time_domain]
            ind_images_list.append(ind_images)
        return ind_images_list
    ind_images_list = get_current_stack()


    print ('in segmentation_handler ind_images_list (channel)=',len(ind_images_list))
    print ('in segmentation_handler ind_images_list (timepoints)=',len(ind_images_list[0]))
    #current images (current index and list of channels)
    data_img_ch={'img':[ind_images_list[ch][0] for ch in range(len(ind_images_list))]}
    source_img_ch = bokeh.models.ColumnDataSource(data=data_img_ch)

    #current image to be displayed
    norm_img = (data_img_ch['img'][0]-np.min(data_img_ch['img'][0]))/(np.max(data_img_ch['img'][0])-np.min(data_img_ch['img'][0]))
    data_img={'img':[norm_img]}
    source_img = bokeh.models.ColumnDataSource(data=data_img)

    #list of all images for all channels
    data_imgs={'images':ind_images_list}
    source_imgs = bokeh.models.ColumnDataSource(data=data_imgs)

    data_intensity={'time':[1,2,3,4,5,6,7,8,9,10], 'intensity':[1,5,10,5,1,10,5,1,10,5]}
    source_intensity = bokeh.models.ColumnDataSource(data=data_intensity)

    # Create a Slider widget
    initial_time_point = 0
    slider         = bokeh.models.Slider(start=0, end=len(ind_images_list[0]) - 1, value=initial_time_point, step=1, title="Time Point")
    plot_image     = bokeh.plotting.figure(x_range=(0, ind_images_list[0][0].shape[0]), y_range=(0, ind_images_list[0][0].shape[1]), tools="box_select,wheel_zoom,box_zoom,reset,undo")
    plot_intensity = bokeh.plotting.figure(title="Intensity vs Time", x_axis_label='Time', y_axis_label='Intensity')


    # Function to update the well depending on the experiment
    #___________________________________________________________________________________________
    def update_dropdown_well(attr, old, new):
        print('****************************  update_dropdown_well ****************************')
        dropdown_well.options = wells[dropdown_exp.value]
        print('+++++++++++++++++  dropdown_exp.value ',dropdown_exp.value, '  wells[dropdown_exp.value]  +++++++++++  ',wells[dropdown_exp.value],'   ',)
        print('+++++++++++++++++  positions[{0}_{1}.format(dropdown_exp.value, wells[dropdown_exp.value][0])][0] +++  ',positions['{0}_{1}'.format(dropdown_exp.value, wells[dropdown_exp.value][0])][0])
        print('+++++++++++++++++  {0}_{1}.format(dropdown_exp.value, wells[dropdown_exp.value][0]) +++++++++++++++++  ', '{0}_{1}'.format(dropdown_exp.value, wells[dropdown_exp.value][0]))
        dropdown_well.value   = wells[dropdown_exp.value][0]
        dropdown_pos.options  = positions['{0}_{1}'.format(dropdown_exp.value, wells[dropdown_exp.value][0])]
        #dropdown_pos.value    = positions['{0}_{1}'.format(dropdown_exp.value, wells[dropdown_exp.value][0])][0]
        if slider.value == 0:
            print('in the if update_dropdown_well')
            left_rois,right_rois,top_rois,bottom_rois=update_source_roi()
            height_labels, weight_labels, names_labels = update_source_labels_roi()
            height_cells, weight_cells, names_cells = update_source_labels_cells()
        
            source_roi.data = {'left': left_rois, 'right': right_rois, 'top': top_rois, 'bottom': bottom_rois}
            source_labels.data = {'height':height_labels, 'weight':weight_labels, 'names':names_labels}
            source_cells.data = {'height':height_cells, 'weight':weight_cells, 'names':names_cells}
        else:
            print('in the else update_dropdown_well')
            slider.value = 0
        slider.end=len(source_imgs.data['images'][0]) - 1

    dropdown_exp.on_change('value', update_dropdown_well)

    # Function to update the position depending on the experiment and the well
    #___________________________________________________________________________________________
    def update_dropdown_pos(attr, old, new):
        print('****************************  update_dropdown_pos ****************************')
        dropdown_pos.options = positions['{0}_{1}'.format(dropdown_exp.value, dropdown_well.value)]
        dropdown_pos.value = positions['{0}_{1}'.format(dropdown_exp.value, dropdown_well.value)][0]

        if slider.value == 0:
            print('in the if update_dropdown_pos')
            left_rois,right_rois,top_rois,bottom_rois=update_source_roi()
            height_labels, weight_labels, names_labels = update_source_labels_roi()
            height_cells, weight_cells, names_cells = update_source_labels_cells()
        
            source_roi.data = {'left': left_rois, 'right': right_rois, 'top': top_rois, 'bottom': bottom_rois}
            source_labels.data = {'height':height_labels, 'weight':weight_labels, 'names':names_labels}
            source_cells.data = {'height':height_cells, 'weight':weight_cells, 'names':names_cells}
        else:
            print('in the else update_dropdown_pos')
            slider.value = 0
    dropdown_well.on_change('value', update_dropdown_pos)


    channel_list=[]
    for ch in range(len(ind_images_list)):channel_list.append(str(ch))
    dropdown_channel  = bokeh.models.Select(value='0', title='Channel', options=channel_list)
    #___________________________________________________________________________________________
    def update_dropdown_channel(attr, old, new):
        print('****************************  update_dropdown_channel ****************************')
        ch_list=[]
        for ch in range(len(source_img_ch.data['img'])):
            ch_list.append(str(ch))
        dropdown_channel.options = ch_list
        dropdown_channel.value = new
        
        new_image = source_img_ch.data['img'][int(new)]
        x_norm = (new_image-np.min(new_image))/(np.max(new_image)-np.min(new_image))

        source_img.data   = {'img':[x_norm]}
        print('update_dropdown_channel options: ',dropdown_channel.options)
        print('update_dropdown_channel value : ',dropdown_channel.value)
    dropdown_channel.on_change('value', update_dropdown_channel)

    colormaps = ['Greys256','Inferno256','Viridis256']
    color_mapper = bokeh.models.LinearColorMapper(palette="Greys256", low=0, high=1)
    color_bar = bokeh.models.ColorBar(color_mapper=color_mapper, location=(0,0))

    dropdown_color = bokeh.models.Select(title="Color Palette", value="Grey256",options=colormaps)

    #___________________________________________________________________________________________
    def update_dropdown_color(attr, old, new):
        print('****************************  update_dropdown_channel ****************************')
        palette = dropdown_color.value
        color_mapper.palette = palette
        #color_mapper.low=source_img.data['img'][0].min()
        #color_mapper.high=source_img.data['img'][0].max()
    dropdown_color.on_change('value',update_dropdown_color)

    # Function to update the position
    #___________________________________________________________________________________________
    def prepare_pos(attr, old, new):
        print('****************************  prepare_pos ****************************')
        images = get_current_stack()
        source_imgs.data = {'images':images}
        source_img_ch.data = {'img':[images[ch][0] for ch in range(len(images))]}

        new_image = images[int(dropdown_channel.value)][0]
        x_norm = (new_image-np.min(new_image))/(np.max(new_image)-np.min(new_image))

        source_img.data = {'img':[x_norm]}
        dropdown_channel.value = dropdown_channel.options[0]
        dropdown_color.value = dropdown_color.options[0]
        print('prepare_pos dropdown_channel.value ',dropdown_channel.value)
        print('prepare_pos dropdown_channel.options ',dropdown_channel.options)
        print('prepare_pos before slider')
        if slider.value == 0:
            print('in the if prepare_pos')
            left_rois,right_rois,top_rois,bottom_rois=update_source_roi()
            height_labels, weight_labels, names_labels = update_source_labels_roi()
            height_cells, weight_cells, names_cells = update_source_labels_cells()
            source_roi.data = {'left': left_rois, 'right': right_rois, 'top': top_rois, 'bottom': bottom_rois}
            source_labels.data = {'height':height_labels, 'weight':weight_labels, 'names':names_labels}
            source_cells.data = {'height':height_cells, 'weight':weight_cells, 'names':names_cells}
        else:
            print('in the else prepare_pos')
            slider.value = 0
        print('prepare_pos after slider')
    dropdown_pos.on_change('value', prepare_pos)

    # Function to get the current index
    #___________________________________________________________________________________________
    def get_current_index():
        print('****************************  get_current_index ****************************')
        return slider.value

    refresh_time_list = ["100", "200", "300", "400", "500", "750", "1000", "2000"]
    dropdown_refresh_time = bokeh.models.Select(value=refresh_time_list[3], title="time (ms)", options=refresh_time_list)
    # Callback function to handle menu item click
    #___________________________________________________________________________________________
    def refresh_time_callback(attr, old, new):
        print('****************************  refresh_time_callback ****************************')
        print("refresh time : {}".format(dropdown_refresh_time.value))
        play_stop_callback()
        play_stop_callback()
    dropdown_refresh_time.on_change('value',refresh_time_callback)



    left_rois=[]
    right_rois=[]
    top_rois=[]
    bottom_rois=[]
    #___________________________________________________________________________________________
    # update the source_roi
    def update_source_roi():
        print('****************************  update_source_roi ****************************')
        left_rois.clear()
        right_rois.clear()
        top_rois.clear()
        bottom_rois.clear()
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
            #convert databse ROIs im bokeh coordinates
            #top_rois.append(frame[0].height-roi.max_row)
            #bottom_rois.append(frame[0].height-roi.min_row)
            top_rois.append(frame[0].height-roi.min_row)
            bottom_rois.append(frame[0].height-roi.max_row)
        print('ppppppp update_source_roi ',left_rois, right_rois, top_rois, bottom_rois)

        return left_rois,right_rois,top_rois,bottom_rois


    #___________________________________________________________________________________________
    # update the source_labels
    def update_source_labels_roi():
        print('****************************  update_source_labels_roi ****************************')
        height_labels=[]
        weight_labels=[]
        names_labels=[]
        height_labels.clear()
        weight_labels.clear()
        names_labels.clear()
        current_file=get_current_file()
        sample = Sample.objects.get(file_name=current_file)
        current_index=get_current_index()
        frame  = Frame.objects.select_related().filter(sample=sample, number=current_index)
        if len(frame)!=1:
            print('NOT ONLY FRAME FOUND< PLEASE CHECKKKKKKKK')
            print('======sample: ',sample)
            for f in frame:
                print('===============frame: ',f)
        rois   = CellROI.objects.select_related().filter(frame=frame[0])
        for roi in rois:
            weight_labels.append(roi.min_col)
            #back in bokeh coordinates for display
            height_labels.append(frame[0].height-roi.max_row)
            names_labels.append('ROI{0}'.format(roi.roi_number))
        print('ppppppp ',height_labels, weight_labels, names_labels)
        return height_labels, weight_labels, names_labels

    #___________________________________________________________________________________________
    # update the source_labels
    def update_source_labels_cells():
        print('****************************  update_source_labels_cells ****************************')
        height_cells=[]
        weight_cells=[]
        names_cells=[]
        height_cells.clear()
        weight_cells.clear()
        names_cells.clear()
        current_file=get_current_file()
        current_index=get_current_index()
        sample = Sample.objects.get(file_name=current_file)
        frame  = Frame.objects.select_related().filter(sample=sample, number=current_index)
        if len(frame)!=1:
            print('NOT ONLY FRAME FOUND< PLEASE CHECKKKKKKKK')
            print('======sample: ',sample)
            for f in frame:
                print('===============frame: ',f)
        rois = CellROI.objects.select_related().filter(frame=frame[0])
        for roi in rois:
            if roi.cell_id == None: continue
            weight_cells.append(roi.min_col)
            #back in bokeh coordinates for display
            height_cells.append(frame[0].height-roi.min_row)
            names_cells.append(roi.cell_id.name)
        print('ppppppp cells',height_cells, weight_cells, names_cells)
        return height_cells, weight_cells, names_cells

    #___________________________________________________________________________________________
    # Define a callback to update bf_display with slider
    def callback_slider(attr: str, old: Any, new: Any) -> None:
        print('****************************  callback_slider ****************************')
        time_point = slider.value
        images=source_imgs.data['images']
        new_image = images[int(dropdown_channel.value)][time_point]
        norm = (new_image-np.min(new_image))/(np.max(new_image)-np.min(new_image))
        source_img.data = {'img':[norm]}
        left_rois,right_rois,top_rois,bottom_rois=update_source_roi()
        height_labels, weight_labels, names_labels = update_source_labels_roi()
        height_cells, weight_cells, names_cells = update_source_labels_cells()
        
        source_roi.data = {'left': left_rois, 'right': right_rois, 'top': top_rois, 'bottom': bottom_rois}
        source_labels.data = {'height':height_labels, 'weight':weight_labels, 'names':names_labels}
        source_cells.data = {'height':height_cells, 'weight':weight_cells, 'names':names_cells}
    slider.on_change('value', callback_slider)
    
    #___________________________________________________________________________________________
    # Define a callback to update the ROI
    def select_roi_callback(event):
        print('****************************  select_roi_callback ****************************')
        if isinstance(event, bokeh.events.SelectionGeometry):
            data = dict(
                left=source_roi.data['left'] + [event.geometry['x0']],
                right=source_roi.data['right'] + [event.geometry['x1']],
                top=source_roi.data['top'] + [event.geometry['y0']],
                bottom=source_roi.data['bottom'] + [event.geometry['y1']]
                )
            source_roi.data = data
    plot_image.on_event(bokeh.events.SelectionGeometry, select_roi_callback)

    #___________________________________________________________________________________________
    # Define a callback to delete the ROI
    def delete_roi_callback():
        print('****************************  delete_roi_callback ****************************')
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
            roi.delete()
    button_delete_roi = bokeh.models.Button(label="Delete ROI")
    button_delete_roi.on_click(delete_roi_callback)

    #___________________________________________________________________________________________
    # Save ROI
    def save_roi_callback():
        print('****************************  save_roi_callback ****************************')
        print('Saving ROI===================================',source_roi.data)
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
        for i in range(len(source_roi.data['left'])):
            cellrois = CellROI.objects.select_related().filter(frame=frame[0])
            roi_exist=False
            for cellroi in cellrois:
                if cellroi.min_col == math.floor(source_roi.data['left'][i]) and \
                    cellroi.min_row == math.floor(frame[0].height-source_roi.data['top'][i])  and \
                        cellroi.max_col == math.ceil(source_roi.data['right'][i]) and \
                            cellroi.max_row == math.ceil(frame[0].height-source_roi.data['bottom'][i]):
                        print('save_roi_callback already exist ',frame[0])
                        roi_exist=True
            if not roi_exist:
                print('save_roi_callback saving ',frame[0])
                roi = CellROI(min_col=math.floor(source_roi.data['left'][i]), max_col=math.ceil(source_roi.data['right'][i]), 
                              min_row=math.floor(frame[0].height-source_roi.data['top'][i]),  max_row=math.ceil(frame[0].height-source_roi.data['bottom'][i]),
                              roi_number=i, frame=frame[0])
                roi.save()

                images=source_img_ch.data['img']
                images=np.array(images)
                print('save_roi_callback images shape ', images.shape)
                cropped_dict = {'shape_original':images[0].shape}
                out_dir_name  = os.path.join(os.sep, "data","singleCell_catalog","contour_data",exp.name, expds.data_name, os.path.split(sample.file_name)[-1].replace('.nd2',''))
                out_file_name = os.path.join(out_dir_name, "frame{0}_ROI{1}.json".format(frame[0].number, i))
                if not os.path.exists(out_dir_name):
                    os.makedirs(out_dir_name)
                print('roi.min_row, roi.max_row, roi.min_col,roi.max_col',roi.min_row, roi.max_row, roi.min_col,roi.max_col)
                cropped_img = images[:, 512-roi.min_row:512-roi.max_row, roi.min_col:roi.max_col]
                print('cropped_img ',cropped_img)
                print('images ',images)
                print('images shape ',images.shape)
                print('images type ',type(images))
                
                cropped_dict['shape']=[cropped_img.shape[1],cropped_img.shape[2]]
                cropped_dict['npixels']=cropped_img.shape[1]*cropped_img.shape[2]

                channels=exp.name_of_channels.split(',')
                for ch in range(len(channels)):
                    cropped_dict['intensity_{}'.format(channels[ch])] = np.flip(cropped_img[ch],0).tolist()
                            
                print('out_file_name=',out_file_name)
                out_file = open(out_file_name, "w") 
                json.dump(cropped_dict, out_file) 
                out_file.close() 


                out_file_name = os.path.join(out_dir_name, "frame{0}_ROI{1}_summary.json".format(frame[0].number, i))
                cropped_dict = {'npixels':cropped_img.shape[1]*cropped_img.shape[2], 
                                'shape_original':images[0].shape, 
                                'shape':[cropped_img.shape[1], cropped_img.shape[2]]}
                for ch in range(len(channels)): 
                    cropped_dict['intensity_{}'.format(channels[ch].replace(" ",""))]={'intensity_sum':np.sum(cropped_img[ch]),
                                                                                        'intensity_mean':np.mean(cropped_img[ch]), 
                                                                                        'intensity_std':np.std(cropped_img[ch]) }
                out_file = open(out_file_name, "w") 
                json.dump(cropped_dict, out_file) 
                out_file.close()    

                contour = Contour(center_x_pix=roi.min_col+(roi.max_col-roi.min_col)/2., 
                                  center_y_pix=roi.min_row+(roi.max_row-roi.min_row)/2.,
                                  center_z_pix=0, 
                                  center_x_mic=(roi.min_col+(roi.max_col-roi.min_col)/2.)*frame[0].pixel_microns+frame[0].pos_x,
                                  center_y_mic=(roi.min_row+(roi.max_row-roi.min_row)/2.)*frame[0].pixel_microns+frame[0].pos_y,
                                  center_z_mic=0,
                                  file_name=out_file_name,
                                  cell_roi=roi,
                                  type="cell_ROI",
                                  mode="manual")
                contour.save()







        height_labels, weight_labels, names_labels = update_source_labels_roi()
        source_labels.data = {'height':height_labels, 'weight':weight_labels, 'names':names_labels}
        height_cells, weight_cells, names_cells = update_source_labels_cells()
        source_cells.data = {'height':height_cells, 'weight':weight_cells, 'names':names_cells}
        
    button_save_roi = bokeh.models.Button(label="Save ROI")
    button_save_roi.on_click(save_roi_callback)

    #___________________________________________________________________________________________
    # Function to update the image displayed
    def update_image(way=1, number=-9999):
        print('****************************  update_image ****************************')
        current_index=get_current_index()
        images=source_imgs.data["images"]
        source_img_ch.data = {'img':[images[ch][current_index] for ch in range(len(images))]}

        new_image = images[int(dropdown_channel.value)][current_index]
        x_norm = (new_image-np.min(new_image))/(np.max(new_image)-np.min(new_image))

        source_img.data = {'img':[x_norm]}
        current_index = (current_index + 1*way) % len(images[0])
        if number>=0:
            current_index = number
        slider.value = current_index

        print('update_image index=',current_index)


    class PlaybackState:
        def __init__(self):
            self.playing = False
    play_state = PlaybackState()

    class PlaybackID:
        def __init__(self) -> None:
            self.state = None
    play_state_id = PlaybackID()

    #___________________________________________________________________________________________
    # Create play/stop button
    def play_stop_callback():
        print('****************************  play_stop_callback ****************************')
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
    # Create next button
    def next_callback():
        print('****************************  next_callback ****************************')
        update_image()
    button_next = bokeh.models.Button(label="Next")
    button_next.on_click(next_callback)

    #___________________________________________________________________________________________
    # Create next button
    def prev_callback():
        print('****************************  prev_callback ****************************')
        update_image(-1)
    button_prev = bokeh.models.Button(label="Prev")
    button_prev.on_click(prev_callback)

    #___________________________________________________________________________________________
    # Go to next frame with possible issue
    def inspect_cells_callback():
        print('****************************  inspect_cells_callback ****************************')
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
            print('frame ',f, '  ncells_t1 ',ncells_t1,'  ncells_t2 ',ncells_t2)
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
    # Go to next frame with possible issue
    def build_cells_callback():
        print('****************************  build_cells_callback ****************************')
        current_file=get_current_file()
        build_cells(sample=current_file)
        height_cells, weight_cells, names_cells = update_source_labels_cells()
        source_cells.data = {'height':height_cells, 'weight':weight_cells, 'names':names_cells}
    button_build_cells = bokeh.models.Button(label="build cells")
    button_build_cells.on_click(build_cells_callback)

    #___________________________________________________________________________________________
    # Go to next frame with possible issue
    def set_cell_callback():
        height_cells, weight_cells, names_cells = update_source_labels_cells()
        source_cells.data = {'height':height_cells, 'weight':weight_cells, 'names':names_cells}
    button_build_cells = bokeh.models.Button(label="build cells")
    button_build_cells.on_click(build_cells_callback)



    # Create a Div widget with some text
    text = bokeh.models.Div(text="<h2>Cell informations</h2>")

    left_rois, right_rois, top_rois, bottom_rois = update_source_roi()
    source_roi  = bokeh.models.ColumnDataSource(data=dict(left=left_rois, right=right_rois, top=top_rois, bottom=bottom_rois))

    height_labels, weight_labels, names_labels = update_source_labels_roi()
    source_labels = bokeh.models.ColumnDataSource(data=dict(height=height_labels,weight=weight_labels,names=names_labels))
    labels = bokeh.models.LabelSet(x='weight', y='height', text='names', x_units='data', y_units='data',
                                   x_offset=0, y_offset=0, source=source_labels, text_color='white', text_font_size="10pt")

    plot_image.add_layout(labels)

    height_cells, weight_cells, names_cells = update_source_labels_cells()
    source_cells = bokeh.models.ColumnDataSource(data=dict(height=height_cells,weight=weight_cells,names=names_cells))
    labels_cells = bokeh.models.LabelSet(x='weight', y='height', text='names', x_units='data', y_units='data',
                  x_offset=0, y_offset=-15, source=source_cells, text_color='white', text_font_size="11pt")

    plot_image.add_layout(labels_cells)
    #plot_image.add_layout(color_bar, 'right')

    #im = plot_image.image(image='img', x=0, y=0, dw=ind_images_list[0][0].shape[0], dh=ind_images_list[0][0].shape[1], source=source_img, palette='Greys256')
    im = plot_image.image(image='img', x=0, y=0, dw=ind_images_list[0][0].shape[0], dh=ind_images_list[0][0].shape[1], source=source_img, color_mapper=color_mapper)
    int = plot_intensity.line('time', 'intensity', source=source_intensity)
    # Add the rectangle glyph after adding the image
    quad = bokeh.models.Quad(left='left', right='right', top='top', bottom='bottom', fill_alpha=0.3, fill_color='#009933')
    plot_image.add_glyph(source_roi, quad, selection_glyph=quad, nonselection_glyph=quad)

    # Remove the axes
    plot_image.axis.visible = False
    plot_image.grid.visible = False

    left_col = bokeh.layouts.column(bokeh.layouts.row(dropdown_exp),
                                    bokeh.layouts.row(dropdown_well),
                                    bokeh.layouts.row(dropdown_pos), 
                                    bokeh.layouts.row(dropdown_channel),
                                    bokeh.layouts.row(dropdown_color))

    right_col = bokeh.layouts.column(bokeh.layouts.row(slider),
                                     bokeh.layouts.row(button_play_stop, button_prev, button_next, dropdown_refresh_time ),
                                     bokeh.layouts.row(button_delete_roi, button_save_roi ),
                                     bokeh.layouts.row(button_inspect, button_build_cells),
                                     text)
    
    norm_layout = bokeh.layouts.row(left_col, plot_image, right_col, plot_intensity)
    doc.add_root(norm_layout)



#___________________________________________________________________________________________
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
        build_cells()

    if 'intensity' in request.POST:
        intensity()
    #if 'select_experiment' in request.POST and 'select_well' in request.POST and 'select_position' in request.POST:
    #    cell_dict = intensity(experiment=request.POST.get('select_experiment'), 
    #                          well=request.POST.get('select_well'), 
    #                          position=request.POST.get('select_position'))

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
        'script': script
    }

    return render(request, 'segmentation/index.html', context=context)

    

