import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from my_util import get_mnist_data, get_logdir, show_multi_image

n_inputs = 28 * 28  # MNIST
n_hidden1 = 300
n_hidden2 = 150
n_hidden3 = n_hidden1
n_outputs = n_inputs

learning_rate = 0.01
l2_reg = 0.0001
mnist = get_mnist_data()

he_init = tf.contrib.layers.variance_scaling_initializer()
l2_regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
activation = tf.nn.elu

x = tf.placeholder(tf.float32, shape=[None, n_inputs])

weights1_init = he_init([n_inputs, n_hidden1])
weights2_init = he_init([n_hidden1, n_hidden2])

weights1 = tf.Variable(weights1_init, dtype=tf.float32, name="weights1")
weights2 = tf.Variable(weights2_init, dtype=tf.float32, name="weights2")
weights3 = tf.transpose(weights2, name='weights3')
weights4 = tf.transpose(weights1, name='weights4')

biases1 = tf.Variable(tf.zeros(n_hidden1), name='biases1')
biases2 = tf.Variable(tf.zeros(n_hidden2), name='biases2')
biases3 = tf.Variable(tf.zeros(n_hidden3), name='biases3')
biases4 = tf.Variable(tf.zeros(n_outputs), name='biases4')

with tf.name_scope('network'):
    hidden1 = activation(tf.matmul(x, weights1) + biases1)
    hidden2 = activation(tf.matmul(hidden1, weights2) + biases2)
    hidden3 = activation(tf.matmul(hidden2, weights3) + biases3)
    outputs = tf.matmul(hidden3, weights4) + biases4

reconstruction_loss = tf.reduce_mean(tf.square(outputs - x))  # MSE
reg_lossess = l2_regularizer(weights1) + l2_regularizer(weights2)

loss = reconstruction_loss + reg_lossess
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)


def train(n_epoches=5,
          batch_size=150,
          save_model_name='models/chap_15_3_2.ckpt'):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(logdir=get_logdir(), graph=tf.get_default_graph())
    loss_summary = tf.summary.scalar('loss', loss)
    n_batches = mnist.train.num_examples // batch_size

    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epoches):
            for iteration in range(n_batches):
                x_batch, _ = mnist.train.next_batch(batch_size)
                sess.run(training_op, feed_dict={x: x_batch})

                if iteration % 100 == 0:
                    loss_val = loss_summary.eval(feed_dict={x: x_batch})
                    writer.add_summary(loss_val, epoch * n_batches + iteration)
                    print("step: ", epoch * n_batches + iteration,
                          "loss : ", loss.eval(feed_dict={x: mnist.test.next_batch(batch_size)[0]}))

        saver.save(sess, save_model_name)
        writer.close()


def predict(images,
            load_model_name='models/chap_15_3_2.ckpt'):
    restore_saver = tf.train.Saver()
    with tf.Session() as sess:
        restore_saver.restore(sess, load_model_name)
        reconstructed_images = outputs.eval(feed_dict={x: images})

    # show
    concated_images = np.concatenate([images, reconstructed_images], axis=0).reshape(images.shape[0] * 2, 28, 28)
    show_multi_image(2, images.shape[0], concated_images)


if __name__ == '__main__':
    train(n_epoches=10)
    test_images = mnist.test.next_batch(10)[0]
    predict(test_images)
