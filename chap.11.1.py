from datetime import datetime
from functools import partial
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.base import Datasets
import numpy as np

seed = 42
now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
root_logdir = 'tf_logs'
logdir = f'{root_logdir}/run-{now}/'
save_path = './models/chap11_1_model_final.ckpt'


def net(mnist: Datasets,
        is_train=True,
        n_epochs=50,
        batch_size=100,
        n_inputs=28 * 28,
        n_hidden1=300,
        n_hidden2=100,
        n_outputs=10,
        lr=0.01,
        save_path='./models/chap11_1_model_final.ckpt'):
    # 構築phase
    x = tf.placeholder(tf.float32, shape=(None, n_inputs), name="x")
    training = tf.placeholder_with_default(False, shape=(), name='training')
    # これを実行するとUPDATE_OPSに当関数パラメータの最適化関数が追加される

    my_batch_norm_layer = partial(tf.layers.batch_normalization, training=training, momentum=0.9)
    # 外だしするのはlayersレベルにとどめておいたほうがグラフは見やすそう（みんみー）
    hidden1 = tf.layers.dense(x, n_hidden1, name='hidden1')
    bn1 = my_batch_norm_layer(hidden1)
    bn1_act = tf.nn.elu(bn1)

    hidden2 = tf.layers.dense(bn1_act, n_hidden2, name='hidden2')
    bn2 = my_batch_norm_layer(hidden2)
    bn2_act = tf.nn.elu(bn2)



    logits_before_bn = tf.layers.dense(bn2_act, n_outputs, name='outputs')
    logits = tf.layers.batch_normalization(logits_before_bn, training=training, momentum=0.9)

    y = tf.placeholder(tf.int64, shape=(None), name='y')

    with tf.name_scope('loss'):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name='loss')

    with tf.name_scope('optimize'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        training_op = optimizer.minimize(loss)

        # UPDATE_OPの関数コレクション。
        # runするとコレクション内の最適化が走る
        # batch_normalizer内の移動平均によりパラメータを更新する
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.name_scope('eval'):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

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
                    sess.run([training_op, extra_update_ops], feed_dict={x: x_batch, y: y_batch})

                print(epoch, ' accuracy:', accuracy.eval(feed_dict=validation_feed_dict))
                writer.add_summary(summary_loss.eval(feed_dict=validation_feed_dict), epoch)
            saver.save(sess, save_path)
        writer.close()
    else:
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
