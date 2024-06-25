import json, os
import numpy as np
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
            if np.random.uniform(low=0.0, high=1.0)>fraction_valid:
                if '.jpeg' in posname: 
                    train_list.append(os.path.join(inputdir, expname, wellname, posname))
                else:
                    valid_list.append(os.path.join(inputdir, expname, wellname, posname))




print(valid_list)
print(train_list)
print(len(valid_list),'  ',len(train_list))