import matplotlib.image as mpimg
from PIL import Image
import numpy as np
import os
import shutil

NUM = 5

mypath = '/Users/zhangjunwei/Downloads/AerialImageDataset/'

test_img = mypath + 'train/images/'
#test_img = mypath + 'test/images/'
tests_img = mypath + 'test_small/images/'
tests_img_eg = mypath + 'test_small_eg/images_eg_%d/' % NUM
tests_pred_eg = mypath + 'test_small_eg/predict_eg_%d/' % NUM
tests_pred = mypath + 'test_small/predict/'
'''
try:
    shutil.rmtree(mypath + 'test_small')
except FileNotFoundError:
    pass

os.mkdir(mypath + 'test_small')
os.mkdir(tests_pred)
os.mkdir(tests_img)
'''

os.mkdir(tests_img_eg)
os.mkdir(tests_pred_eg)
#img_names = os.listdir(test_img)
img_names = ['austin1.tif']

s = 0
for name in img_names:
    s += 1
    print("%d/180"%s)
    
    img = Image.open(test_img + name)
    count = 0
    for i in range(20):
        for j in range(20):
            count += 1

            crop_img = img.crop((i*250, j*250, (i+1)*250, (j+1)*250))
            crop_img = crop_img.resize((128,128))
            
            crop_img.save(tests_img_eg + '%s_%d.tif' % (os.path.splitext(name)[0], count))



