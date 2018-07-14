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

my_dense_layer = partial(tf.layers.dense,
                         activation=tf.nn.relu,
                         kernel_initializer=he_init,
                         kernel_regularizer=l2_regularizer)

x = tf.placeholder(tf.float32, shape=[None, n_inputs])


hidden1 = my_dense_layer(x, n_hidden1)
hidden2 = my_dense_layer(hidden1, n_hidden2)
hidden3 = my_dense_layer(hidden2, n_hidden3)
outputs = my_dense_layer(hidden3, n_outputs, activation=None)

reconstruction_loss = tf.reduce_mean(tf.square(outputs - x))  # MSE
reg_lossess = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

loss = tf.add_n([reconstruction_loss] + reg_lossess)
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)




def train(n_epoches=5,
          batch_size=150,
          save_model_name='models/chap_15_3.ckpt'):
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
            load_model_name='models/chap_15_3.ckpt'):
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
