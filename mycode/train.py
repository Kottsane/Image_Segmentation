import tensorflow as tf
import matplotlib.image as mpimg
import os
import numpy as np

BATCH_SIZE = 200
DIM = 128
TRAINING_EPOCHS = 100
LEARNING_RATE = 0.001

mypath = 'E:\\AerialImageDataset\\'
para = mypath + 'para\\'

class DataGenerator:
    def __init__(self, path):
        self.img_path = path + 'train_small\\images\\'
        self.gt_path = path + 'train_small\\gt\\'
        self.names = os.listdir(self.gt_path)
        self.num_images = len(self.names)
        np.random.shuffle(self.names)
        self.cur_idx = 0

    def make_batch(self, batch_size):
        train_to = min(self.cur_idx + batch_size, self.num_images)
        real_batch_size = train_to - self.cur_idx
        train_x = np.zeros((real_batch_size, DIM, DIM, 3), dtype=np.float32)
        train_y = np.zeros((real_batch_size, DIM, DIM, 1), dtype=np.float32)

        for i in range(real_batch_size):
            img = mpimg.imread(self.img_path + self.names[self.cur_idx])
            gt = mpimg.imread(self.gt_path + self.names[self.cur_idx])

            # DO NOT forget to rescale the amplitude to be in [0,1]
            train_x[i, :, :, :] = img / 255
            train_y[i, :, :, 0] = gt / 255
            self.cur_idx += 1

        return train_x, train_y

    def has_data(self):
        return self.cur_idx < self.num_images

    def reset(self):
        np.random.shuffle(self.names)
        self.cur_idx = 0


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

img = tf.placeholder(dtype=tf.float32, shape=[None,DIM,DIM,3], name="inputs") # Batch * 128 * 128 * 3
y_true = tf.placeholder(dtype=tf.float32, shape=[None,DIM,DIM,1], name="labels")

k1, b1, h1 = conv_layer(img, [3,3,3,32], [1,1,1,1], 'SAME', tf.nn.relu, 'conv1') # Batch * 128 * 128 * 32
k2, b2, h2 = conv_layer(h1, [3,3,32,32], [1,1,1,1], 'SAME', tf.nn.relu, 'conv2') # Batch * 128 * 128 * 32
p1 = tf.nn.max_pool(h2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name="pool1") # Batch * 64 * 64 * 32

k3, b3, h3 = conv_layer(p1, [3,3,32,64], [1,1,1,1], 'SAME', tf.nn.relu, 'conv3') # Batch * 64 * 64 * 64
k4, b4, h4 = conv_layer(h3, [3,3,64,64], [1,1,1,1], 'SAME', tf.nn.relu, 'conv4') # Batch * 64 * 64 * 64
p2 = tf.nn.max_pool(h4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name="pool2") # Batch * 32 * 32 * 64

k5, b5, h5 = conv_layer(p2, [3,3,64,128], [1,1,1,1], 'SAME', tf.nn.relu, 'conv5') # Batch * 32 * 32 * 128
k6, b6, h6 = conv_layer(h5, [3,3,128,128], [1,1,1,1], 'SAME', tf.nn.relu, 'conv6') # Batch * 32 * 32 * 128
p3 = tf.nn.max_pool(h6, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name="pool3") # Batch * 16 * 16 * 128

k7, b7, h7 = conv_layer(p3, [5,5,128,1024], [1,1,1,1], 'SAME', tf.nn.relu, 'conv7') # Batch * 16 * 16 * 1024

k8, b8, h8 = conv_layer(h7, [1,1,1024,128], [1,1,1,1], 'SAME', identity, 'conv8') # Batch * 16 * 16 * 128

k9, h9 = deconv_layer(h8, [16,16,1,128], tf.shape(y_true), [1,8,8,1], 'SAME', 'deconv') # Batch * 128 * 128 * 1
# k9, h9 = deconv_layer(h8, [16,16,1,128], [BATCH_SIZE,128,128,1], [1,8,8,1], 'SAME', 'deconv2') # Batch * 128 * 128 * 1

cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=h9)
loss = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
train_op = optimizer.minimize(loss)

def main():

    dg = DataGenerator(mypath)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(TRAINING_EPOCHS):
            batch_num = 0
            while dg.has_data():
                batch_num += 1
                train_x, train_y = dg.make_batch(BATCH_SIZE)
                fd = {img: train_x, y_true: train_y}
                _, loss_ = sess.run([train_op, loss], feed_dict=fd)
                print('Epoch %d/%d: %d batch(s) (%d/%d training images) have been trained, loss is %f'\
                        % (epoch+1, TRAINING_EPOCHS, batch_num, dg.cur_idx, dg.num_images, loss_))
            dg.reset()
            saver.save(sess, para + 'save1')

if __name__ == '__main__':
    main()





