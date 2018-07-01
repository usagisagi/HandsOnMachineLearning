import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets.base import Datasets
from datetime import datetime
from sklearn.metrics import accuracy_score

seed = 42
now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
root_logdir = 'tf_logs'
logdir = f'{root_logdir}/run-{now}/'


def neuron_layer(x, n_neurons, name, activation=None):
    """n_neurons -> output"""
    with tf.name_scope(name):
        n_inputs = int(x.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs + n_neurons)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev, seed=seed)  # 初期化、±2σの切断正規分布
        W = tf.Variable(init, name='kernel')
        b = tf.Variable(tf.zeros([n_neurons]), name='bias')
        z = tf.matmul(x, W) + b
        if activation is not None:
            return activation(z)
        else:
            return z


def mnist_dnn(
        mnist: Datasets,
        is_train=True,
        n_epochs=50,
        batch_size=50,
        n_inputs=28 * 28,  # MNIST
        n_hidden1=28 * 28,
        n_hidden2=28 * 28,
        n_hidden3=28 * 28,
        n_outputs=10,
        learning_rate=0.01,
        save_path='./models/my_model_final.ckpt'):
    x = tf.placeholder(tf.float32, shape=(None, n_inputs), name='x')
    y = tf.placeholder(tf.int64, shape=(None), name='y')

    with tf.name_scope('dnn'):
        hidden1 = neuron_layer(x, n_hidden1, name='hidden1', activation=tf.nn.elu)
        hidden2 = neuron_layer(hidden1, n_hidden2, name='hidden2', activation=tf.nn.elu)
        hidden3 = neuron_layer(hidden2, n_hidden3, name='hidden3', activation=tf.nn.elu)
        logits = neuron_layer(hidden3, n_outputs, name='outputs')  # 各出力の値

    with tf.name_scope('loss'):
        # 各出力の値をsoftmax_entropyで求める
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        # バッチ内のxentropyの平均を求める
        loss = tf.reduce_mean(xentropy, name='loss')

    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        training_op = optimizer.minimize(loss)

    with tf.name_scope('eval'):
        # logitsの先頭k番目内にyをindexとするものがあるか
        # 存在したらTrue,　しなかったらFalse
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    saver = tf.train.Saver()
    if is_train:
        # 実行フェーズ
        # training
        init = tf.global_variables_initializer()

        writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
        summary_loss = tf.summary.scalar('loss', loss)

        with tf.Session() as sess:
            all_feed_dict = {x: mnist.validation.images, y: mnist.validation.labels}
            init.run()
            n_batches = mnist.train.num_examples // batch_size
            for epoch in range(n_epochs):
                for i in range(n_batches):
                    x_batch, y_batch = mnist.train.next_batch(batch_size)
                    sess.run(training_op, feed_dict={x: x_batch, y: y_batch})
                acc_val = accuracy.eval(feed_dict=all_feed_dict)
                print(epoch, " accuracy::", acc_val)

                summary_str = summary_loss.eval(feed_dict=all_feed_dict)
                writer.add_summary(summary_str, epoch)
            saver.save(sess, save_path)

        writer.close()
    else:
        # 予測部分
        with tf.Session() as sess:
            saver.restore(sess, save_path)
            x_new_scaled, y = mnist.test.next_batch(mnist.test.num_examples)
            z = logits.eval(feed_dict={x: x_new_scaled})
            y_pred = np.argmax(z, axis=1)
            print(accuracy_score(y, y_pred))


if __name__ == '__main__':
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('/tmp/data/')
    # mnist_dnn(mnist)
    mnist_dnn(mnist, False)
