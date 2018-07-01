"""学習シケジュール"""

from datetime import datetime
from functools import partial

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow.contrib.learn.python.learn.datasets.base import Datasets

seed = 42
now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
root_logdir = 'tf_logs'
logdir = f'{root_logdir}/run-{now}/'


def net(mnist: Datasets,
        is_train=True,
        n_epochs=100,
        batch_size=50,
        n_inputs=28 * 28,  # MNIST
        n_hidden1=300,
        n_hidden2=100,
        n_hidden3=50,
        n_hidden4=50,
        n_hidden5=50,
        n_outputs=10,
        initial_learning_rate=0.01,
        threshold=1.0,
        scale=0.001,
        dropout_rate=0.20,
        save_path='./models/chap11_3_model_final.ckpt'):
    x = tf.placeholder(tf.float32, shape=(None, n_inputs), name="x")
    y = tf.placeholder(tf.int32, shape=(None), name="y")
    training = tf.placeholder_with_default(False, (), name='training')

    he_init = tf.contrib.layers.variance_scaling_initializer()
    my_dense_layer = partial(tf.layers.dense,
                             kernel_initializer=he_init,
                             activation=tf.nn.elu)
    my_drop_layer = partial(tf.layers.dropout,
                            rate=dropout_rate,
                            training=training)

    hidden1 = my_dense_layer(x, n_hidden1, name="hidden1")
    hidden1_drop = my_drop_layer(hidden1)

    hidden2 = my_dense_layer(hidden1_drop, n_hidden2, name="hidden2")
    hidden2_drop = my_drop_layer(hidden2)

    hidden3 = my_dense_layer(hidden2_drop, n_hidden3, name="hidden3")
    hidden3_drop = my_drop_layer(hidden3)

    hidden4 = my_dense_layer(hidden3_drop, n_hidden4, name="hidden4")
    hidden4_drop = my_drop_layer(hidden4)

    hidden5 = my_dense_layer(hidden4_drop, n_hidden5, name="hidden5")
    hidden5_drop = my_drop_layer(hidden5)

    logits = tf.layers.dense(hidden5_drop, n_outputs, name="outputs")

    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name="base_loss")

    optimizer = tf.train.MomentumOptimizer(initial_learning_rate, momentum=0.9, use_nesterov=True)
    training_op = optimizer.minimize(loss)

    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

    saver = tf.train.Saver()

    if is_train:
        init = tf.global_variables_initializer()
        writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
        summary_loss = tf.summary.scalar('loss', loss)

        with tf.Session() as sess:
            init.run()
            validation_feed_dict = {x: mnist.validation.images, y: mnist.validation.labels}
            n_batches = mnist.train.num_examples // batch_size

            for epoch in range(n_epochs):
                for i in range(n_batches):
                    x_batch, y_batch = mnist.train.next_batch(batch_size)
                    sess.run(training_op, feed_dict={training: True, x: x_batch, y: y_batch})

                print(epoch, ' accuracy:', accuracy.eval(feed_dict=validation_feed_dict))
                writer.add_summary(summary_loss.eval(feed_dict=validation_feed_dict), epoch)

            saver.save(sess, save_path)
        writer.close()
    else:
        # predict
        with tf.Session() as sess:
            saver.restore(sess, save_path)
            x_new_scaled, y = mnist.test.next_batch(mnist.test.num_examples)
            z = logits.eval(feed_dict={x: x_new_scaled})
            y_pred = np.argmax(z, axis=1)
            print(accuracy_score(y, y_pred))


if __name__ == '__main__':
    from tensorflow.examples.tutorials.mnist import input_data

    mnist = input_data.read_data_sets('/tmp/data/')
    net(mnist, is_train=True)
