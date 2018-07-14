from functools import partial
from my_util import *
import numpy as np
import tensorflow as tf

init_he = tf.contrib.layers.variance_scaling_initializer()
my_hidden_layer = partial(tf.layers.dense, activation=tf.nn.elu, kernel_initializer=init_he)


def inner_dnn_unit(input, name):
    with tf.variable_scope(name):
        hidden1 = my_hidden_layer(input, 100, name='hidden1')
        hidden2 = my_hidden_layer(hidden1, 100, name='hidden2')
        hidden3 = my_hidden_layer(hidden2, 100, name='hidden3')
        hidden4 = my_hidden_layer(hidden3, 100, name='hidden4')
        hidden5 = my_hidden_layer(hidden4, 100, name='hidden5')
        return hidden5


def dnn(train_x, train_y,
        validate_x, validate_y,
        test_x, test_y,
        n_epochs=20,
        batch_size=25,
        n_inputs=28 * 28,
        save_path='./models/chap_11_Q10_WDNN.ckpt'):
    # 構築フェーズ
    x_1 = tf.placeholder(tf.float32, shape=(None, n_inputs), name='x_1')
    x_2 = tf.placeholder(tf.float32, shape=(None, n_inputs), name='x_2')
    y = tf.placeholder(tf.int32, shape=(None), name='y')

    # 各数字画像のDNN
    dnn_unit_1 = inner_dnn_unit(x_1, 'dnn_1')
    dnn_unit_2 = inner_dnn_unit(x_2, 'dnn_2')

    # 2つのDNNの結合し、出力
    concanted = tf.concat([dnn_unit_1, dnn_unit_2], axis=1)
    hidden_concated = my_hidden_layer(concanted, 10, name='hidden_concated')
    # 0び確率と1の確率
    logit = tf.layers.dense(hidden_concated, 2)

    with tf.name_scope('loss'):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logit)
        loss = tf.reduce_mean(xentropy, name='loss')

    with tf.name_scope('optimize'):
        optimizer = tf.train.AdamOptimizer()
        training_op = optimizer.minimize(loss)

    with tf.name_scope('eval'):
        correct = tf.nn.in_top_k(logit, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')

    data_length = train_y.shape[0]

    saver = tf.train.Saver()
    loss_summary = tf.summary.scalar('loss', loss)
    writer = tf.summary.FileWriter(get_logdir(), graph=tf.get_default_graph())

    init = tf.global_variables_initializer()
    n_batches = data_length // batch_size

    # 訓練フェーズ

    def fetch_batch_in_pair(x, y, batch_size):
        """ペアの訓練セットを生成するバッチ"""
        # xはバッチサイズの2倍取得し、半分に分割する。
        tmp_x, tmp_y = fetch_batch(x, y, batch_size * 2)
        x_1, x_2 = tmp_x[:batch_size], tmp_x[batch_size:]

        # numpyはTrueが0、falseが1
        y = (tmp_y[:batch_size] != tmp_y[batch_size:]).astype(np.int32)

        return x_1, x_2, y

    with tf.Session() as sess:
        init.run()
        x_1_validate, x_2_validate, y_validate = fetch_batch_in_pair(validate_x, validate_y,
                                                                     int(validate_x.shape[0] // 2))
        validate_dict = {x_1: x_1_validate, x_2: x_2_validate, y: y_validate}

        x_1_test, x_2_test, y_test = fetch_batch_in_pair(test_x, test_y, int(test_x.shape[0] // 2))
        test_dict = {x_1: x_1_test, x_2: x_2_test, y: y_test}

        for epoch in range(n_epochs):
            for batch_index in range(n_batches):
                x_1_batch, x_2_batch, y_batch = fetch_batch_in_pair(train_x, train_y, batch_size)
                sess.run(training_op, feed_dict={x_1: x_1_batch, x_2: x_2_batch, y: y_batch})
                if batch_index % 10 == 0:
                    writer.add_summary(loss_summary.eval(feed_dict=validate_dict), epoch * n_batches + batch_index)
            print(epoch, ' epoch accuracy:', accuracy.eval(feed_dict=validate_dict))
        print('test data accuracy:', accuracy.eval(feed_dict=test_dict))
        saver.save(sess, save_path)
    writer.close()


def retrain_dnn(train_x, train_y,
                validate_x, validate_y,
                test_x, test_y,
                n_epochs=20,
                batch_size=25,
                n_inputs=28 * 28,
                n_outputs=10,
                load_path='./models/chap_11_Q10_WDNN.ckpt',
                save_path="./models/chap_11_Q10_ReTrain.ckpt"):
    # 構築フェーズ
    x = tf.placeholder(tf.float32, shape=(None, n_inputs), name='x')
    y = tf.placeholder(tf.int32, shape=(None), name='y')

    # variable含むレイヤ作成するときはvariable_scopeのほうがいいかもしれない
    with tf.variable_scope('network'):
        hidden1 = my_hidden_layer(x, 100, name='hidden1')
        hidden2 = my_hidden_layer(hidden1, 100, name='hidden2')
        hidden3 = my_hidden_layer(hidden2, 100, name='hidden3')
        hidden4 = my_hidden_layer(hidden3, 100, name='hidden4')
        hidden5 = my_hidden_layer(hidden4, 100, name='hidden5')
        logit = tf.layers.dense(hidden5, n_outputs, name='outputs')

    with tf.name_scope('loss'):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logit)
        loss = tf.reduce_mean(xentropy, name='loss')

    with tf.name_scope('optimize'):
        # logit層のみ更新対象にする
        optimizer = tf.train.AdamOptimizer()
        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='network/outputs')
        training_op = optimizer.minimize(loss, var_list=train_vars)

    with tf.name_scope('eval'):
        correct = tf.nn.in_top_k(logit, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')

    # 復元
    reuse_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="network/hidden[12345]")
    # Load対象変数名を指定. network=>dnn_1に変更
    reuse_vars_dict = dict([(var.op.name.replace('network', 'dnn_1'), var) for var in reuse_vars])
    restore_saver = tf.train.Saver(reuse_vars_dict)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    loss_summary = tf.summary.scalar('loss', loss)
    writer = tf.summary.FileWriter(get_logdir(), graph=tf.get_default_graph())
    data_length = train_y.shape[0]
    n_batches = data_length // batch_size

    # 学習フェーズ
    with tf.Session() as sess:
        init.run()
        restore_saver.restore(sess, load_path)
        h5_cache = sess.run(hidden5, feed_dict={x: train_x})
        validate_dict = {x: validate_x, y: validate_y}
        test_dict = {x: test_x, y: test_y}
        for epoch in range(n_epochs):
            shuffled_idx = np.random.permutation(train_x.shape[0])
            hidden5_batches = np.array_split(h5_cache[shuffled_idx], n_batches)
            y_batches = np.array_split(train_y[shuffled_idx], n_batches)

            for idx, (hidden5_batch, y_batch) in enumerate(zip(hidden5_batches, y_batches)):
                sess.run(training_op, feed_dict={hidden5: hidden5_batch, y: y_batch})
                writer.add_summary(loss_summary.eval(feed_dict=validate_dict), epoch * n_batches + idx)
            print(epoch, ' epoch accuracy:', accuracy.eval(feed_dict=validate_dict))
        print('test data accuracy:', accuracy.eval(feed_dict=test_dict))
        saver.save(sess, save_path)


if __name__ == '__main__':
    mnist = get_mnist_data()
    train_x, train_y, _, _ = slice_mnist(mnist.train.images, mnist.train.labels)
    validate_x, validate_y, _, _ = slice_mnist(mnist.validation.images, mnist.validation.labels)
    test_x, test_y, _, _ = slice_mnist(mnist.test.images, mnist.test.labels)


    def four_e():
        retrain_dnn(test_x, test_y,  validate_x, validate_y, train_x, train_y,)


    four_e()
