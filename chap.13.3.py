import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.datasets import load_sample_images

images = load_sample_images()['images']
flower = images[0]
china = images[1]
dataset = np.array(images, np.float32)

batch_size, height, width, channels = dataset.shape

filters = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)

filters[:, 3, :, 0] = 1 # x = 3の部分だけ白線
filters[3, :, :, 1] = 1 # y = 3の部分だけ白線

x = tf.placeholder(tf.float32, shape=(None, height, width, channels))
conv = tf.nn.conv2d(x, filters, strides=(1, 2, 2, 1), padding='SAME')
max_pool = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
with tf.Session() as sess:
    output = sess.run(max_pool, feed_dict={x: dataset})


plt.imshow(output[0, :, :, 0], cmap='gray')
plt.show()

plt.imshow(output[1, :, :, 0], cmap='gray')
plt.show()