# Generate the training sets:
#  Each original 5000*5000*3 satellite image is evenly cropped into 400 250*250*3 images,
#  which are further resized to 128*128*3. Amplitude in the images is rescaled to be in [0, 1] per image.
#  # Rescaling steps are put off until the training process
#  Gt images are cropped like above, resulting in shape 128*128, with all entries either 0 or 1,
#  indicating the possibility that pixel belong to a 'building'. 0 - "not building", 1 - "is building"
#
# Results are saved in "train_small" subfolder under mypath.

import matplotlib.image as mpimg
from PIL import Image
import numpy as np
import os
import shutil

mypath = 'E:\\AerialImageDataset\\'

train_gt = mypath + 'train\\gt\\'
train_img = mypath + 'train\\images\\'
trains_gt = mypath + 'train_small\\gt\\'
trains_img = mypath + 'train_small\\images\\'

try:
    shutil.rmtree(mypath + 'train_small')
except FileNotFoundError:
    pass

os.mkdir(mypath + 'train_small')
os.mkdir(trains_gt)
os.mkdir(trains_img)

img_names = os.listdir(train_gt)

s = 0
for name in img_names:
    s += 1
    print("%d/180"%s)

    # gt = mpimg.imread(train_gt + name)
    # img = mpimg.imread(train_img + name)
    gt = Image.open(train_gt + name)
    img = Image.open(train_img + name)

    count = 0
    for i in range(20):
        for j in range(20):
            count += 1
            crop_gt = gt.crop((i*250, j*250, (i+1)*250, (j+1)*250))
            crop_gt = crop_gt.resize((128,128))
            
            crop_gt.save(trains_gt + '%s_%d.tif' % (os.path.splitext(name)[0], count))

            crop_img = img.crop((i*250, j*250, (i+1)*250, (j+1)*250))
            crop_img = crop_img.resize((128,128))
            
            crop_img.save(trains_img + '%s_%d.tif' % (os.path.splitext(name)[0], count))



