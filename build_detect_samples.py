import json, os
import numpy as np
import uuid
import shutil

def make_inputs(files, outdir):
    outdict={
    "info": {
        "year": "2024",
        "version": "1",
        "description": "Exported from UPOATES single cell DB",
        "contributor": "C. HELSENS",
        "url": "",
        "date_created": "2024-06-25T08:05:45+00:00"
    },
    "licenses": [
        {
            "id": 1,
            "url": "https://choosealicense.com/licenses/mit/",
            "name": "MIT"
        }
    ],
    "categories": [
        {
            "id": 0,
            "name": "cells",
            "supercategory": "none"
        },
        {
            "id": 1,
            "name": "cell_roi",
            "supercategory": "cells"
        }
    ],
    "images": [],
    "annotations": []
    }


    
    for f_idx, f in enumerate(files):
        uid = uuid.uuid4().hex
        outfile_name = '{}_{}.jpg'.format(f.replace('.jpg',''), uid)
        shutil.copy(f, os.path.join(outdir,outfile_name))
        data = None
        with open(f.replace('.jpg','.json'), 'r') as f:
            data = json.load(f)

        image = {"id":f_idx,
                 "licence":1,
                 "file_name":outfile_name,
                 "height": data["image"]["height"],
                 "width": data["image"]["width"]
                 }

        outdict["images"].append(image)
        for a_idx, a in enumerate(data["annotations"]):
            annotation = {"id":a_idx,
                          "image_id":f_idx,
                          "category":1,
                          "bbox":a["bbox"],
                          "segmentation": [],
                          "iscrowd": 0}
            outdict["annotations"].append(annotation)

#        {"id": 0,
#            "image_id": 0,
#            "category_id": 1,
#            "bbox": [289,327,20,29],
#            "area": 580,
#            "segmentation": [],
#            "iscrowd": 0}

#        {"id": 0,
#            "license": 1,
#            "file_name": "BloodImage_00322_jpg.rf.eedd60c6eefc381592560aeee7116b58.jpg",
#            "height": 416,
#            "width": 416,
#            "date_captured": "2021-02-24T08:05:45+00:00"},

inputdir = '/data/singleCell_training_images/'
outdir   = '/data/singleCell_training_images_tf/'
outname  = 'testDetectDS'

outdir_train = os.path.join(outdir, outname, 'train')
outdir_valid = os.path.join(outdir, outname, 'valid')
fraction_valid=0.2  

if not os.path.exists(outdir_train):
    os.makedirs(outdir_train)
if not os.path.exists(outdir_valid):
    os.makedirs(outdir_valid)

train_list = []
valid_list = []

for expname in os.listdir(inputdir):
    for wellname in os.listdir(os.path.join(inputdir, expname)):
        for posname in os.listdir(os.path.join(inputdir, expname, wellname)):
            for filename in  os.listdir(os.path.join(inputdir, expname, wellname, posname)):
                if '.jpeg' in filename: 
                    if np.random.uniform(low=0.0, high=1.0)>fraction_valid:
                        train_list.append(os.path.join(inputdir, expname, wellname, posname, filename))
                    else:
                        valid_list.append(os.path.join(inputdir, expname, wellname, posname, filename))


print(len(valid_list),'  ',len(train_list))
make_inputs(valid_list, outdir_valid)