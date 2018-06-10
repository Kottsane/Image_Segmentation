# Generate the training sets:
#  Each original 5000*5000*3 satellite image is evenly cropped into 25 1000*1000*3 images,
#  which are further resized to 25 256*256*3 images. Images without buildings are discarded.
#  Gt images are cropped like above, resulting in shape 256*256, with all entries either 0 or 1,
#  indicating the possibility that pixel belong to a 'building'. 0 - "not building", 1 - "is building"
#
# Results are saved in "train_small" subfolder under mypath.

import matplotlib.image as mpimg
from scipy import misc
import numpy as np
import os
import shutil

mypath = '/Users/zhangjunwei/Downloads/AerialImageDataset/'

train_gt = mypath + '/train/gt/'
train_img = mypath + '/train/images/'
trains_gt = mypath + '/train_small/gt/'
trains_img = mypath + '/train_small/images/'

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

    gt = mpimg.imread(train_gt + name)
    img = mpimg.imread(train_img + name)
    count = 0
    for i in range(5):
        for j in range(5):
            count += 1
            crop_gt = gt[i*1000:(i+1)*1000, j*1000:(j+1)*1000]
            crop_gt = misc.imresize(crop_gt, (256,256))
            crop_gt = np.uint8(np.where(crop_gt > 128, 1, 0))
            if np.sum(crop_gt) == 0:
                continue
            misc.imsave(trains_gt + '%s_%d.tif' % (os.path.splitext(name)[0], count), crop_gt)

            crop_img = img[i*1000:(i+1)*1000, j*1000:(j+1)*1000]
            crop_img = misc.imresize(crop_img, (256,256))

            misc.imsave(trains_img + '%s_%d.tif' % (os.path.splitext(name)[0], count), crop_img)



