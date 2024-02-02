from django.shortcuts import render
from django.db import reset_queries
from django.db import connection
from segmentation.models import Experiment, ExperimentalDataset, Sample, Frame, Contour, Data, Segmentation, SegmentationChannel, CellID, CellFrame
import os, sys, json, glob, gc
import time

from memory_profiler import profile
from sklearn.cluster import DBSCAN
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')

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
import math

#___________________________________________________________________________________________
def deltaR(c1, c2):
    return math.sqrt( math.pow((c1['x'] - c2['x']),2) +  math.pow((c1['y'] - c2['y']),2) + math.pow((c1['z'] - c2['z']),2))

#___________________________________________________________________________________________
def build_frames():
    test_arch={
        'projects':[
#            {
#                'name':'testproj1',
#                'analyses':[
#                    {
#                    'name':'testproj1_testana1',
#                    'files':[
#                        '/Users/helsens/data/singleCell/wsc_epfl-wscl_060/wsc_epfl-wscl_060_xy01.nd2',
#                        '/Users/helsens/data/singleCell/wsc_epfl-wscl_060/wsc_epfl-wscl_060_xy02.nd2',
#                        '/Users/helsens/data/singleCell/wsc_epfl-wscl_060/wsc_epfl-wscl_060_xy03.nd2',
#                        '/Users/helsens/data/singleCell/wsc_epfl-wscl_060/wsc_epfl-wscl_060_xy04.nd2'
#                        ]
#                    },

#                    {
#                    'name':'testproj1_testana2',
#                    'files':[
#                        '/Users/helsens/data/singleCell/wsc_epfl-wscl_060/wsc_epfl-wscl_060_xy05.nd2',
#                        '/Users/helsens/data/singleCell/wsc_epfl-wscl_060/wsc_epfl-wscl_060_xy06.nd2',
#                        '/Users/helsens/data/singleCell/wsc_epfl-wscl_060/wsc_epfl-wscl_060_xy07.nd2',
#                        '/Users/helsens/data/singleCell/wsc_epfl-wscl_060/wsc_epfl-wscl_060_xy08.nd2'
#                        ]
#                    }
#                ]
#            },
            {
                'name':'testproj2',
                'analyses':[
                    {
                    'name':'testproj2_testana1',
                    'files':[
                        '/Users/helsens/data/singleCell/wsc_epfl-wscl_060/wsc_epfl-wscl_060_xy01.nd2',
 #                       '/Users/helsens/data/singleCell/wsc_epfl-wscl_060/wsc_epfl-wscl_060_xy02.nd2',
 #                       '/Users/helsens/data/singleCell/wsc_epfl-wscl_060/wsc_epfl-wscl_060_xy03.nd2',
 #                       '/Users/helsens/data/singleCell/wsc_epfl-wscl_060/wsc_epfl-wscl_060_xy04.nd2',
                        '/Users/helsens/data/singleCell/wsc_epfl-wscl_060/wsc_epfl-wscl_060_xy05.nd2',
 #                       '/Users/helsens/data/singleCell/wsc_epfl-wscl_060/wsc_epfl-wscl_060_xy06.nd2',
 #                       '/Users/helsens/data/singleCell/wsc_epfl-wscl_060/wsc_epfl-wscl_060_xy07.nd2',
 #                       '/Users/helsens/data/singleCell/wsc_epfl-wscl_060/wsc_epfl-wscl_060_xy08.nd2'
                        ]
                    }
                ]
            }
        ]
    }


    projects = Project.objects.values()
    list_projects = [entry for entry in projects] 
    list_projects_uid=[e["name"] for e in list_projects]

    for proj in test_arch["projects"]:
        if proj['name'] in list_projects_uid: continue
        project =  Project(name=proj['name'])
        project.save()
        list_projects_uid.append(proj['name'])
        print('adding project with name:  ',proj['name'])

    for p in Project.objects.all():
        analyses = Analysis.objects.select_related().filter(project = p)
        list_analyses_uid = [entry.name for entry in analyses] 

        for proj in test_arch["projects"]:
            if proj['name']!=p.name: 
                continue
            for ana in proj['analyses']:
                if ana['name'] in list_analyses_uid : continue
                analysis = Analysis(name=ana['name'], project=p)
                analysis.save()
                print('    adding analysis with name ',ana['name'])

                for file in ana['files']:
                    metadata = read.nd2reader_getSampleMetadata(file)
                    sample = Sample(file_name=file, 
                                    analysis=analysis,
                                    number_of_frames=metadata['number_of_frames'], 
                                    number_of_channels=metadata['number_of_channels'], 
                                    name_of_channels=metadata['name_of_channels'], 
                                    experiment_description=metadata['experiment_description'], 
                                    date=metadata['date'],
                                    keep_sample=True)
                    sample.save()
                    print('        adding sample with name ',file)

                    metadataFrame = read.nd2reader_getFrameMetadata(file)
                    for f in range(metadata['number_of_frames']):
                        frame = Frame(sample=sample, 
                                      number=f, 
                                      keep_sample=True,
                                      time=metadataFrame['time'][f],
                                      pos_x=metadataFrame['x_pos'][f],
                                      pos_y=metadataFrame['y_pos'][f],
                                      pos_z=metadataFrame['z_pos'][f],
                                      height=metadataFrame['height'],
                                      width=metadataFrame['width'],
                                      )
                        print('            adding frame with name ',f)
                        frame.save()

#___________________________________________________________________________________________
def build_frames_rds():
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
            print('====================== ERROR, unsplit_file not 1, exit ',unsplit_file)
            sys.exit(3)
        metadata = read.nd2reader_getSampleMetadata(unsplit_file[0])
        experiment =  Experiment(name=x[1], 
                                 date=x[2], 
                                 description=x[3],
                                 file_name=unsplit_file[0],
                                 number_of_frames=metadata['number_of_frames'], 
                                 number_of_channels=metadata['number_of_channels'], 
                                 name_of_channels=metadata['name_of_channels'], 
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

            for s in samples:

                if 'xy06' in s.file_name or 'xy74' in s.file_name: 
                    print('===========================================')
                    break
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
def build_cells():
    #loop over all experiments
    exp_list = Experiment.objects.all()
    for exp in exp_list:
        print('---- BUILD CELLS experiment name ',exp.name)
        experimentaldataset = ExperimentalDataset.objects.select_related().filter(experiment = exp)
        for expds in experimentaldataset:
            print('    ---- BUILD CELLS experimentaldataset name ',expds.data_name, expds.data_type)
            samples = Sample.objects.select_related().filter(experimental_dataset = expds)
            for s in samples:
                print('        ---- BUILD CELL sample name ',s.file_name)
                cellsid = CellID.objects.select_related().filter(sample = s)
                #temporary, continue is already cellsid connected to sample
                if len(cellsid)>0:
                    continue
                frames = Frame.objects.select_related().filter(sample = s)
                cell_frame_list=[]
                cell_frame_coord=[]

                for f in frames:
                    cellframes = CellFrame.objects.select_related().filter(frame=f)
                    for cellf in cellframes:
                        cell_frame_list.append(cellf)
                        cell_frame_coord.append([cellf.pos_x, cellf.pos_y, cellf.pos_z])
                print('number of cell frames=',len(cell_frame_list))
                if len(cell_frame_list)==0:continue
                X = np.array(cell_frame_coord)
                clustering = DBSCAN(eps=25, min_samples=10).fit(X)
                print(clustering.labels_)

                #Create the cells ID according to existing clusters (one per cluster >=0)
                #Connect the cellFrames to cellID
                createdcells=[]
                cellid_list=[]
                cellid_dict={}
                for cid in range(len(clustering.labels_)):
                    if clustering.labels_[cid] not in createdcells and clustering.labels_[cid]!=-1:
                        cellid = CellID(sample=s, name='cell{}'.format(clustering.labels_[cid]))
                        cellid.save()
                        createdcells.append(clustering.labels_[cid])
                        #cellid_list.append(cellid)
                        cellid_dict['cell{}'.format(clustering.labels_[cid])]=cellid
                    if clustering.labels_[cid]!=-1:
                        #cell_frame_list[cid].cell_id = cellid_list[clustering.labels_[cid]]
                        cell_frame_list[cid].cell_id = cellid_dict['cell{}'.format(clustering.labels_[cid])]
                        cell_frame_list[cid].save()



#___________________________________________________________________________________________
def build_cell_frames():
    #For now build cells from contours
    #loop over all experiments
    exp_list = Experiment.objects.all()
    for exp in exp_list:
        print('---- BUILD CELL FRAMES experiment name ',exp.name)
        experimentaldataset = ExperimentalDataset.objects.select_related().filter(experiment = exp)
        for expds in experimentaldataset:
            print('    ---- BUILD CELL FRAMES experimentaldataset name ',expds.data_name, expds.data_type)
            samples = Sample.objects.select_related().filter(experimental_dataset = expds)
            for s in samples:
                if 'xy06' in s.file_name or 'xy74' in s.file_name: 
                    print('===========================================')
                    break
                print('        ---- BUILD CELL FRAMES sample name ',s.file_name)
                frames = Frame.objects.select_related().filter(sample = s)
                for f in frames:
                    print('            ---- BUILD CELL FRAMES frame number ',f.number,' ',f.time)
                    contours = Contour.objects.select_related().filter(frame = f)
                    cellframe = CellFrame.objects.select_related().filter(frame = f)
                    if len(contours) == len(cellframe):
                        print('cell frames already exist')

                        continue
                    elif len(contours) > len(cellframe) and len(cellframe)!=0:
                        print('more contours than cell frames, investigate...')
                        continue
                    elif len(cellframe)!=0:
                        print('already cell frames')
                        continue
                    for cont in contours:
                        print('                ---- BUILD CELL FRAMES contour ',cont.center)
                        cellf = CellFrame(frame=f,
                                          time=f.time,
                                          pos_x=cont.center['x'],
                                          pos_y=cont.center['y'],
                                          pos_z=cont.center['z'],
                                          sig_x=10,
                                          sig_y=10,
                                          sig_z=0)
                        cellf.save()
                        cont.cell_frame = cellf
                        cont.save()

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
                            for ch in cont.pixels_data_inside.all_pixels['sum_intensity']:
                                try:
                                    cell_dict[cid.name]['intensity_{}'.format(ch.replace(' ',''))]
                                except KeyError:
                                    cell_dict[cid.name]['intensity_{}'.format(ch.replace(' ',''))]=[]
                                cell_dict[cid.name]['intensity_{}'.format(ch.replace(' ',''))].append(cont.pixels_data_inside.all_pixels['sum_intensity'][ch])
                                cell_dict[cid.name]['npixels'].append(cont.pixels_data_inside.all_pixels['npixels'])
                            #print(cont.pixels_data_inside.all_pixels)
                            #print('contour ID=',cont.id,'  center=',cont.center, ' seg channel ',cont.segmentation_channel.channel_name,', ',cont.segmentation_channel.channel_number\
                            #      ,' seg name=',cont.segmentation_channel.segmentation.name,' seg type=',cont.segmentation_channel.segmentation.algorithm_type)
                    print('   n cell_frames=',len(cell_frames))
                print(cell_dict)
                return cell_dict



#___________________________________________________________________________________________
def index(request):
    """View function for home page of site."""
    print('The visualisation request method is:', request.method)
    print('The visualisation POST data is:     ', request.POST)
    cell_dict={}
    if 'build_frames' in request.POST and LOCAL:
        build_frames()
    if 'build_frames' in request.POST and LOCAL==False:
        build_frames_rds()
    if 'build_cell_frames' in request.POST:
        build_cell_frames()
    if 'build_cells' in request.POST:
        build_cells()
    if 'segment' in request.POST:
        segment()

    if 'intensity' in request.POST:
        intensity()
    if 'select_experiment' in request.POST and 'select_well' in request.POST and 'select_position' in request.POST:
        cell_dict = intensity(experiment=request.POST.get('select_experiment'), 
                              well=request.POST.get('select_well'), 
                              position=request.POST.get('select_position'))


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
        'position':''
    }

    selected_experiment=request.POST.get('select_experiment')
    selected_dict['experiment']=selected_experiment
    selected_well=request.POST.get('select_well')
    selected_dict['well']=selected_well
    selected_position=request.POST.get('select_position')
    selected_dict['position']=selected_position

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

    #print('experiment_dict     =  ', experiment_dict)
    #print('selected_project =  ', selected_project)
    print('selected_dict    =  ', selected_dict)
    #print('select_dict      =  ', select_dict)

    context = {
        #'num_samples': num_samples,
        'select_dict':select_dict,
        'selected_dict':selected_dict
    }

    # Render the HTML template index.html with the data in the context variable


    channels=[]
    for cell in cell_dict: 
        for ch in cell_dict[cell]:
            if ch in channels:continue
            if 'intensity_' not in ch: continue
            channels.append(ch)
        print('channels  ',channels)
        plt.plot()



    return render(request, 'segmentation/index.html', context=context)
