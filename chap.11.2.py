from datetime import datetime

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow.contrib.learn.python.learn.datasets.base import Datasets

seed = 42
now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
root_logdir = 'tf_logs'
logdir = f'{root_logdir}/run-{now}/'
save_path = './models/chap11_2_model_final.ckpt'


def net(mnist: Datasets,
        is_train=True,
        n_epochs=30,
        batch_size=50,
        n_inputs=28 * 28,  # MNIST
        n_hidden1=300,
        n_hidden2=50,
        n_hidden3=50,
        n_hidden4=50,
        n_hidden5=50,
        n_outputs=10,
        learning_rate=0.01,
        threshold=1.0,
        save_path='./models/chap11_2_model_final.ckpt'):

    x = tf.placeholder(tf.float32, shape=(None, n_inputs), name="x")
    y = tf.placeholder(tf.int32, shape=(None), name="y")

    with tf.name_scope("dnn"):
        hidden1 = tf.layers.dense(x, n_hidden1, activation=tf.nn.relu, name="hidden1")
        hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu, name="hidden2")
        hidden3 = tf.layers.dense(hidden2, n_hidden3, activation=tf.nn.relu, name="hidden3")
        hidden4 = tf.layers.dense(hidden3, n_hidden4, activation=tf.nn.relu, name="hidden4")
        hidden5 = tf.layers.dense(hidden4, n_hidden5, activation=tf.nn.relu, name="hidden5")
        logits = tf.layers.dense(hidden5, n_outputs, name="outputs")

    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")

    # 勾配クリッピング
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    grads_and_vars = optimizer.compute_gradients(loss)
    capped_gvs = [(tf.clip_by_value(grad, -threshold, threshold), var) for grad, var in grads_and_vars]
    training_op = optimizer.apply_gradients(capped_gvs)

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
                    sess.run(training_op, feed_dict={x: x_batch, y: y_batch})

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
