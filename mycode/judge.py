import matplotlib.image as mpimg
import os
import numpy as np
from PIL import Image

mypath = '/Users/zhangjunwei/Downloads/AerialImageDataset/result_old/'


def judge(img, thres):
    # input an image with entries ranging from 0 to 255, and output judged image with respect to threshold
    img = np.uint8(np.where(img > thres, 255, 0))
    return img

def main():
    img_names = os.listdir(mypath)
    for name in img_names:
        if name[0] == '.':
            continue
        img = mpimg.imread(mypath + name)
        img = judge(img, 127)
        img = Image.fromarray(img)
        img.save(mypath + name)

if __name__ == '__main__':
    main()
