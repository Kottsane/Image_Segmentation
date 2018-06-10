import tensorflow as tf
import matplotlib.image as mpimg
import os
import numpy as np
from scipy import misc

DIM = 256

mypath = '/Users/zhangjunwei/Downloads/AerialImageDataset/'
tests_pred = mypath + '/test_small/predict/'
#tests_img = mypath + '/test_small/images/'
tests_img = mypath + '/train_small/images/'

def conv_layer(inputs, kernel_size, strides, padding, activation, name):
    kernel = tf.get_variable(name + "_k", initializer=tf.truncated_normal(shape=kernel_size, stddev=0.1))
    bias = tf.get_variable(name + "_b", initializer=tf.constant(0.1, shape=[kernel_size[-1]]))
    outputs = activation(tf.nn.conv2d(inputs, kernel, strides=strides, padding=padding) + bias)
    return kernel, bias, outputs

def deconv_layer(inputs, kernel_size, output_shape, strides, padding, name):
    kernel = tf.get_variable(name + "_k", initializer=tf.truncated_normal(shape=kernel_size, stddev=0.1))
    outputs = tf.nn.conv2d_transpose(inputs, kernel, output_shape, strides, padding)
    return kernel, outputs

def identity(x):
    return x


img = tf.placeholder(dtype=tf.float32, shape=[1,DIM,DIM,3], name="inputs") # Batch * 256 * 256 * 3

k1, b1, h1 = conv_layer(img, [3,3,3,32], [1,1,1,1], 'SAME', tf.nn.relu, 'conv1') # Batch * 256 * 256 * 32
k2, b2, h2 = conv_layer(h1, [3,3,32,32], [1,1,1,1], 'SAME', tf.nn.relu, 'conv2') # Batch * 256 * 256 * 32
p1 = tf.nn.max_pool(h2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name="pool1") # Batch * 128 * 128 * 32

k3, b3, h3 = conv_layer(p1, [3,3,32,64], [1,1,1,1], 'SAME', tf.nn.relu, 'conv3') # Batch * 128 * 128 * 64
k4, b4, h4 = conv_layer(h3, [3,3,64,64], [1,1,1,1], 'SAME', tf.nn.relu, 'conv4') # Batch * 128 * 128 * 64
p2 = tf.nn.max_pool(h4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name="pool2") # Batch * 64 * 64 * 64

k5, b5, h5 = conv_layer(p2, [3,3,64,128], [1,1,1,1], 'SAME', tf.nn.relu, 'conv5') # Batch * 64 * 64 * 128
k6, b6, h6 = conv_layer(h5, [3,3,128,128], [1,1,1,1], 'SAME', tf.nn.relu, 'conv6') # Batch * 64 * 64 * 128
p3 = tf.nn.max_pool(h6, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name="pool3") # Batch * 32 * 32 * 128

k7, b7, h7 = conv_layer(p3, [5,5,128,1024], [1,1,1,1], 'SAME', tf.nn.relu, 'conv7') # Batch * 32 * 32 * 1024

k8, b8, h8 = conv_layer(h7, [1,1,1024,128], [1,1,1,1], 'SAME', identity, 'conv8') # Batch * 32 * 32 * 128

k9, h9 = deconv_layer(h8, [16,16,1,128], [1,256,256,1], [1,8,8,1], 'SAME', 'deconv2') # Batch * 256 * 256 * 1

h9 = tf.reshape(h9, [256,256])

z = tf.nn.sigmoid(h9)

def main():
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, mypath + 'save2')
        names = os.listdir(tests_img)
        np.random.shuffle(names)
        test_x = np.zeros([1,256,256,3])
        for name in names:
            image = mpimg.imread(tests_img + 'kitsap11_21.tif')
            test_x[0, :, :, :] = image / 255
            pred = sess.run(z, feed_dict={img: test_x})
            pred *= 255
            misc.imsave(tests_pred + 'kitsap11_21.tif', pred)
            break


if __name__ == '__main__':
    main()