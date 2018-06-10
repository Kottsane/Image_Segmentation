
import matplotlib.image as mpimg
from scipy import misc
import os
import shutil

mypath = '/Users/zhangjunwei/Downloads/AerialImageDataset/'

test_img = mypath + '/test/images/'
tests_pred = mypath + '/test_small/predict/'
tests_img = mypath + '/test_small/images/'

try:
    shutil.rmtree(mypath + 'test_small')
except FileNotFoundError:
    pass

os.mkdir(mypath + 'test_small')
os.mkdir(tests_img)
os.mkdir(tests_pred)

img_names = os.listdir(test_img)

s = 0
for name in img_names:
    s += 1
    print("%d/180"%s)

    img = mpimg.imread(test_img + name)
    count = 0
    for i in range(5):
        for j in range(5):
            count += 1

            crop_img = img[i*1000:(i+1)*1000, j*1000:(j+1)*1000]
            crop_img = misc.imresize(crop_img, (256,256))

            misc.imsave(tests_img + '%s_%d.tif' % (os.path.splitext(name)[0], count), crop_img)



