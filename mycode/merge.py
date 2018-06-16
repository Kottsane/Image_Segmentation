# Before running this file, "img_src" folder should contain many 128*128*3 satellite images,
# and "pred_src" folder should contain the 128*128 predicted map with respect to every image in "img_src".

# This file puts every small image in "img_src" folder together and forms the original big image.
# It does the same with "pred_src" folder.
# The way images are put together should comply with the way the original image is divided into smaller
# images in the first place.


import matplotlib.image as mpimg
import numpy as np
from PIL import Image

NUM = 5
NAME = 'austin1'

mypath = '/Users/zhangjunwei/Downloads/AerialImageDataset/test_small_eg/'
big = mypath + 'big/'
img_src = mypath + 'images_eg_%d/' % NUM
pred_src = mypath + 'predict_eg_%d/' % NUM

def main():
    img = np.zeros([2560,2560,3], dtype=np.uint8)
    count = 0
    for j in range(20):
        for i in range(20):
            count += 1
            src = mpimg.imread(img_src + "%s_%d.tif" % (NAME, count))
            img[i*128:(i+1)*128, j*128:(j+1)*128] = src
    img = Image.fromarray(img)
    img.save(big + "%s.tif" % NAME)

    pred = np.zeros([2560,2560], dtype=np.uint8)
    count = 0
    for j in range(20):
        for i in range(20):
            count += 1
            src = mpimg.imread(pred_src + "%s_%d.tif" % (NAME, count))
            pred[i*128:(i+1)*128, j*128:(j+1)*128] = src
    pred = Image.fromarray(pred)
    pred.save(big + "%s_pred.tif" % NAME)
    
if __name__ == '__main__':
    main()





