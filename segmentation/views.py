from django.shortcuts import render
from segmentation.models import Experiment, ExperimentalDataset, Sample, Frame, Contour, Data, Cell
import os
import sys
import json
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

    cnx = mysql.connector.connect(user=accesskeys.RD_DB_user, 
                                  password=accesskeys.RD_DB_password,
                                  host='127.0.0.1',
                                  port=3306,
                                  database=accesskeys.RD_DB_name)

import reader as read
import segmentationTools as seg
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
        " where tag.name = \"PSM cell dissociation\""
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
        experiment =  Experiment(name=x[1], date=x[2], description=x[3])
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
                fname=os.path.join(BASEPATH, x[4], x[5], f["name"])
                metadata = read.nd2reader_getSampleMetadata(fname)
                sample = Sample(file_name=f, 
                                experimental_dataset=expds,
                                number_of_frames=metadata['number_of_frames'], 
                                number_of_channels=metadata['number_of_channels'], 
                                name_of_channels=metadata['name_of_channels'], 
                                experiment_description=metadata['experiment_description'], 
                                date=metadata['date'],
                                keep_sample=True)
                sample.save()
                print('        adding sample with name ',fname)
#                            for file in ana['files']:
#                    metadata = read.nd2reader_getSampleMetadata(file)
#                    sample = Sample(file_name=file, 
#                                    analysis=analysis,
#                                    number_of_frames=metadata['number_of_frames'], 
#                                    number_of_channels=metadata['number_of_channels'], 
#                                    name_of_channels=metadata['name_of_channels'], 
#                                    experiment_description=metadata['experiment_description'], 
#                                    date=metadata['date'],
#                                    keep_sample=True)
#                    sample.save()
#                    print('        adding sample with name ',file)



#___________________________________________________________________________________________
def segment():
    #All this seems to be a preprocessing of all existing files
    for p in Project.objects.all():
        print(' ---- project name ',p.name)
        analyses = Analysis.objects.select_related().filter(project = p)
        for ana in analyses:
            print('    ---- analysis name ',ana.name)
            samples = Sample.objects.select_related().filter(analysis = ana)
            for s in samples:
                print('         ---- file name ',s.file_name)
                frames = Frame.objects.select_related().filter(sample = s)
                images, channels = read.nd2reader_getFrames(s.file_name)

                default_segmentation_1 = seg.customLocalThresholding_Segmentation(segchannel=0, channels=channels, threshold=2., delta=2, npix_min=400, npix_max=4000)
                default_segmentation_1_ID = default_segmentation_1.get_AlgoID()

                existing_contours = Contour.objects.values()
                list_contours = [entry for entry in existing_contours] 
                list_contours_uid=[e["uid_name"] for e in list_contours]

                for f in frames:
                    print('              ---- frame number ',f.number)
                    uid_name_segmentation_1="file={0}__frame={1}__{2}".format(s.file_name, f.number, default_segmentation_1_ID)
                    contour_exist = False
                    for c_uid in list_contours_uid:
                        if uid_name_segmentation_1 in c_uid:
                            print("contour already exist=",uid_name_segmentation_1)
                            contour_exist=True
                    if contour_exist: continue
                    contour_list = default_segmentation_1.segmentation(images[f.number-1])
                    for cont in contour_list:
                        print('contour does not exist ',contour_exist)
                        pixels_data_contour  = Data(all_pixels=cont['all_pixels_contour'], single_pixels=cont['single_pixels_contour'])
                        pixels_data_contour.save()
                        pixels_data_inside   = Data(all_pixels=cont['all_pixels_inside'],  single_pixels=cont['single_pixels_inside'])
                        pixels_data_inside.save()
                        print(cont['center'])
                        center="x="+str(int(cont['center']['x']))+"y="+str(int(cont['center']['y']))+"z="+str(int(cont['center']['z']))
                        contour = Contour(algorithm_type=cont['algorithm_type'],
                                          algorithm_version=cont['algorithm_version'],
                                          algorithm_parameters=cont['algorithm_parameters'],
                                          channel=channels[0],
                                          number_of_pixels=cont['number_of_pixels'],
                                          center=cont['center'],
                                          frame=f,
                                          pixels_data_contour=pixels_data_contour,
                                          pixels_data_inside=pixels_data_inside,
                                          uid_name=uid_name_segmentation_1+"__"+center
                                          #"Based on the input file, frame number, algorithm type, version and parameters")
                                          )
                        contour.save()


#___________________________________________________________________________________________
def tracking():
    contours = Contour.objects.all()
    print('number of contours ', len(contours))
    samples = Sample.objects.all()
    for s in samples:
        print("----------------  ",s.file_name)
        frames = Frame.objects.select_related().filter(sample = s)
        contour_list=[]
        for f in frames:
            contours = Contour.objects.select_related().filter(frame = f)
            for c in contours:
                contour_list.append(c)
        
        celldict={}
        cellindex=1
        for cont in contour_list:
            print('=================== contour in full contour list ',cont.uid_name)
            cont_added=False
            for cell in celldict:
                cellcont=celldict[cell][-1]
                dR=deltaR(cellcont.center, cont.center)
                print('=================== contour in cell dict DR= ',dR,cellcont.uid_name)

                if dR<20:
                    print('------- dR ', dR)
                    celldict[cell].append(cont)
                    cont_added=True
                    break

            if not cont_added:
                cellname="cell{}".format(cellindex)
                if cellname in celldict: cellindex+=1
                print("cont not added cellindex=",cellindex,"  cont id=",cont.id,"  cellname=","cell{}".format(cellindex),"  cont=",cont.uid_name)
                celldict["cell{}".format(cellindex)]=[cont]
                print("cell dict=",celldict)

#        print('n contours: ',len(contour_list))
#        for c in celldict:
#            cell = Cell(name=c, sample=s)
#            print('cell=',c, '  n=', len(celldict[c]))
#            for cont in celldict[c]:
#                print('      ',cont.uid_name)


#___________________________________________________________________________________________
def index(request):
    """View function for home page of site."""
    print('The visualisation request method is:', request.method)
    print('The visualisation POST data is:     ', request.POST)

    if 'build_frames' in request.POST and LOCAL:
        build_frames()
    if 'build_frames' in request.POST and LOCAL==False:
        build_frames_rds()
    if 'segment' in request.POST:
        segment()
    if 'tracking' in request.POST:
        tracking()

    #dictionary to provide possible selection choices
    select_dict={
        'experiment_list':[],
        'dataset_list':[],
        'file_list':[],
    }

    #dictionary that maps the database
    project_dict={        
        'projects':[]
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
        tmp_proj_dict={'name':exp.name, 'datasets':expds_list}
        project_dict['projects'].append(tmp_proj_dict)

    #dictionary to keep the selected choices, this is for the front end page
    selected_dict={
        'project':'',
        'analysis':'',
        'file':''
    }

    selected_project=request.POST.get('select_project')
    selected_dict['project']=selected_project
    selected_analysis=request.POST.get('select_analysis')
    selected_dict['analysis']=selected_analysis
    selected_file=request.POST.get('select_file')
    selected_dict['file']=selected_file

    if selected_project!='':
        for p in project_dict['projects']:
            if p['name']!=selected_project:continue
            print('project selected=',p['name'])
            for a in p['analyses']:
                select_dict['analysis_list'].append(a['name'])
            if selected_analysis!='':
                for a in p['analyses']:
                    if a['name']!=selected_analysis:continue
                    print('analysis selected=',a['name'])
                    for f in a['files']:
                        #select_dict['file_list'].append(os.path.split(f)[-1])
                        select_dict['file_list'].append(f)

    print('project_dict     =  ', project_dict)
    print('selected_project =  ', selected_project)
    print('selected_dict    =  ', selected_dict)
    print('select_dict      =  ', select_dict)

    context = {
        #'num_samples': num_samples,
        'select_dict':select_dict,
        'selected_dict':selected_dict
    }

    # Render the HTML template index.html with the data in the context variable
    return render(request, 'segmentation/index.html', context=context)
