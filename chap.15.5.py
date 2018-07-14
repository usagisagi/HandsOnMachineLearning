"""auto encoder"""
import sys
from functools import partial

import tensorflow as tf
import numpy as np

from my_util import get_logdir, get_mnist_data, show_multi_image

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 150
n_hidden3 = n_hidden1
n_output = n_inputs

learning_rate = 0.01
noise_level = 1.0
dropout_rate = 0.3
seed = 42
mnist = get_mnist_data()


class Noise(object):
    NONE = 1
    GAUSSIAN = 2
    DROP = 3


he_init = tf.contrib.layers.variance_scaling_initializer(seed=seed)
activation = tf.nn.elu

my_dense_layer = partial(tf.layers.dense, activation=activation, kernel_initializer=he_init)


def construct_graph(noise=Noise.NONE):
    x = tf.placeholder(tf.float32, shape=[None, n_inputs], name='x')
    training = tf.placeholder_with_default(False, (), 'training')

    with tf.variable_scope('network'):
        if noise == Noise.GAUSSIAN:
            x = x + noise_level * tf.random_normal(tf.shape(x), seed=seed)
        elif noise == Noise.DROP:
            x = tf.layers.dropout(x, dropout_rate, seed=seed, training=training)

        hidden1 = my_dense_layer(x, n_hidden1, name='hidden1')
        hidden2 = my_dense_layer(hidden1, n_hidden2, name='hidden2')
        hidden3 = my_dense_layer(hidden2, n_hidden3, name='hidden3')
        outputs = my_dense_layer(hidden3, n_output, activation=None, name='output')

    with tf.variable_scope('optimize'):
        reconstruction_loss = tf.reduce_mean(tf.square(outputs - x))
        optimizer = tf.train.AdamOptimizer(learning_rate)
        training_op = optimizer.minimize(reconstruction_loss)
        return x, training, reconstruction_loss, training_op


def train(x, training, reconstruction_loss, training_op,
          n_epochs=10, batch_size=150, check_interval=100,
          model_name='chap_15_5.ckpt'):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(logdir=get_logdir(), graph=tf.get_default_graph())
    loss_summary = tf.summary.scalar('loss', reconstruction_loss)
    validation_dict = {x: mnist.validation.images}

    with tf.Session() as sess:
        init.run()
        n_batches = mnist.train.num_examples // batch_size

        for epoch in range(n_epochs):
            for batch_index in range(n_batches):
                x_batch, _ = mnist.train.next_batch(batch_size)
                sess.run(training_op, feed_dict={x: x_batch, training: True})

                if batch_index % check_interval == 0:
                    sys.stdout.flush()
                    print("loss :", reconstruction_loss.eval(feed_dict=validation_dict))
                    writer.add_summary(loss_summary.eval(feed_dict=validation_dict),
                                       epoch * n_batches + batch_index)

        saver.save(sess, model_name)
        writer.close()


def generate_reconstructed_images(images,
                                  load_model_name='models/chap_15_5.ckpt'):
    restore_saver = tf.train.import_meta_graph(load_model_name + '.meta')
    with tf.Session() as sess:
        restore_saver.restore(sess, load_model_name)
        x = tf.get_default_graph().get_tensor_by_name('x:0')
        output = tf.get_default_graph().get_tensor_by_name('network/output/BiasAdd:0')
        reconstructed_images = output.eval(feed_dict={x: images})
        return reconstructed_images


def show_reconstructed_images(images_for_view):
    # show
    n_rows = len(images_for_view)
    n_cols = images_for_view[0].shape[0]
    concated_images = np.concatenate(images_for_view, axis=0).reshape(n_rows * n_cols, 28, 28)
    show_multi_image(n_rows, n_cols, concated_images)


if __name__ == '__main__':

    model_names = [f'models/chap_15_5_{m}.ckpt' for m in ['naive', 'gaussian', 'drop']]
    images = mnist.test.images[:10]
    images_for_view = [images]

    for noise, model_name in zip([Noise.NONE, Noise.GAUSSIAN, Noise.DROP],
                                 model_names):
        x, training, loss, op = construct_graph(Noise.NONE)
        train(x, training, loss, op, model_name=model_name)
        tf.reset_default_graph()
        images_for_view.append(
            generate_reconstructed_images(images, load_model_name=model_name))
        tf.reset_default_graph()

    show_reconstructed_images(images_for_view)
