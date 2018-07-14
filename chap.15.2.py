import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from my_util import  get_mnist_data
import seaborn
seaborn.set()

def create_random_data(m=200):
    w1, w2 = 0.1, 0.3
    noise = 0.1

    angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
    data = np.empty((m, 3))
    data[:, 0] = np.cos(angles) + np.sin(angles) / 2 + noise * np.random.randn(m) / 2
    data[:, 1] = np.sin(angles) * 0.7 + noise * np.random.randn(m) / 2
    data[:, 2] = data[:, 0] * w1 + data[:, 1] * w2 + noise * np.random.randn(m)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(data[:100])
    x_test = scaler.fit_transform(data[:100])

    return x_train, x_test


n_inputs = 3
n_hidden = 2
n_outputs = n_inputs

learning_rate = 0.01

x = tf.placeholder(tf.float32, shape=[None, n_inputs])  # flatten
hidden = tf.layers.dense(x, n_hidden)
outputs = tf.layers.dense(hidden, n_outputs)

reconstruction_loss = tf.reduce_mean(tf.square(outputs - x))  # MSE

optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(reconstruction_loss)

init = tf.global_variables_initializer()

x_train, x_test = create_random_data()
n_iterations = 1000
codings = hidden

with tf.Session() as sess:
    init.run()

    for iteration in range(n_iterations):
        training_op.run(feed_dict={x: x_train})

    codings_val = codings.eval(feed_dict={x: x_test})
    plt.scatter(codings_val[:, 0], codings_val[:, 1])
    plt.show()
