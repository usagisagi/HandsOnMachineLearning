import tensorflow as tf
from functools import partial
import numpy as np
from datetime import datetime
from sklearn.model_selection import cross_val_score
from chap11_util import *
import chap11_util


def dnn_change_softmax(train_x, train_y,
                       validate_x, validate_y,
                       n_epochs=100,
                       batch_size=10,
                       n_outputs=5,
                       save_path='./models/chap_11_Q8_num_5to9_change_softmax.ckpt',
                       base_dnn='./models/chap_11_Q8_num_0to4.ckpt'
                       ):
    n_inputs = 28 * 28

    # 構築フェーズ
    x = tf.placeholder(tf.float32, shape=(None, n_inputs), name='x')
    y = tf.placeholder(tf.int32, shape=(None), name='y')

    # variable含むレイヤ作成するときはvariable_scopeのほうがいいかもしれない
    with tf.variable_scope('network'):
        init_he = tf.contrib.layers.variance_scaling_initializer()
        my_hidden_layer = partial(tf.layers.dense, activation=tf.nn.elu, kernel_initializer=init_he)
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
        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       scope='network/outputs')
        training_op = optimizer.minimize(loss, var_list=train_vars)

    with tf.name_scope('eval'):
        correct = tf.nn.in_top_k(logit, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')

    data_length = train_y.shape[0]

    # 凍結部分のSaver
    reuse_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                   scope='network/hidden[12345]')
    reuse_vars_dict = dict([(var.op.name, var) for var in reuse_vars])
    restore_saver = tf.train.Saver(reuse_vars_dict)

    # 保存部分
    saver = tf.train.Saver()
    loss_summary = tf.summary.scalar('loss', loss)
    writer = tf.summary.FileWriter(get_logdir(), graph=tf.get_default_graph())

    init = tf.global_variables_initializer()
    n_batches = data_length // batch_size

    with tf.Session() as sess:
        init.run()

        # 凍結部分の復元
        restore_saver.restore(sess, base_dnn)

        validate_dict = {x: validate_x, y: validate_y}

        for epoch in range(n_epochs):
            for batch_index in range(n_batches):
                x_batch, y_batch = fetch_batch(train_x, train_y, batch_size)
                sess.run(training_op, feed_dict={x: x_batch, y: y_batch})
                writer.add_summary(loss_summary.eval(feed_dict=validate_dict), epoch * n_batches + batch_index)

            print(epoch, ' epoch accuracy:', accuracy.eval(feed_dict=validate_dict))
        saver.save(sess, save_path)
    writer.close()


def dnn_change_softmax_cached(train_x, train_y,
                              validate_x, validate_y,
                              n_epochs=100,
                              batch_size=10,
                              n_outputs=5,
                              save_path='./models/chap_11_Q8_num_5to9_change_softmax.ckpt',
                              base_dnn='./models/chap_11_Q8_num_0to4.ckpt'
                              ):
    n_inputs = 28 * 28

    # 構築フェーズ
    x = tf.placeholder(tf.float32, shape=(None, n_inputs), name='x')
    y = tf.placeholder(tf.int32, shape=(None), name='y')

    # variable含むレイヤ作成するときはvariable_scopeのほうがいいかもしれない
    with tf.variable_scope('network'):
        init_he = tf.contrib.layers.variance_scaling_initializer()
        my_hidden_layer = partial(tf.layers.dense, activation=tf.nn.elu, kernel_initializer=init_he)
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
        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       scope='network/outputs')
        training_op = optimizer.minimize(loss, var_list=train_vars)

    with tf.name_scope('eval'):
        correct = tf.nn.in_top_k(logit, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')

    data_length = train_y.shape[0]

    # 凍結部分のSaver
    reuse_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                   scope='network/hidden[12345]')
    reuse_vars_dict = dict([(var.op.name, var) for var in reuse_vars])
    restore_saver = tf.train.Saver(reuse_vars_dict)

    # 保存部分
    saver = tf.train.Saver()
    loss_summary = tf.summary.scalar('loss', loss)
    writer = tf.summary.FileWriter(get_logdir(), graph=tf.get_default_graph())

    init = tf.global_variables_initializer()
    n_batches = data_length // batch_size

    with tf.Session() as sess:
        init.run()

        # 凍結部分の復元
        restore_saver.restore(sess, base_dnn)
        top_frozen_layer = hidden5

        # cacheを求める
        hidden_cache = sess.run(top_frozen_layer, feed_dict={x: train_x})
        validate_dict = {x: validate_x, y: validate_y}

        for epoch in range(n_epochs):
            shuffled_idx = np.random.permutation(data_length)
            # xの代わりになる
            hidden_batches = np.array_split(hidden_cache[shuffled_idx], n_batches)
            y_batches = np.array_split(train_y[shuffled_idx], n_batches)

            for hidden_batch, y_batch in zip(hidden_batches, y_batches):
                sess.run(training_op, feed_dict={top_frozen_layer: hidden_batch, y: y_batch})

            writer.add_summary(loss_summary.eval(feed_dict=validate_dict), epoch * n_batches)

            print(epoch, ' epoch accuracy:', accuracy.eval(feed_dict=validate_dict))
        saver.save(sess, save_path)
    writer.close()


def dnn_change_softmax_cached_frozen4(train_x, train_y,
                                      validate_x, validate_y,
                                      n_epochs=100,
                                      batch_size=10,
                                      n_outputs=5,
                                      save_path='./models/chap_11_Q8_num_5to9_change_softmax.ckpt',
                                      base_dnn='./models/chap_11_Q8_num_0to4.ckpt'
                                      ):
    n_inputs = 28 * 28

    # 構築フェーズ
    x = tf.placeholder(tf.float32, shape=(None, n_inputs), name='x')
    y = tf.placeholder(tf.int32, shape=(None), name='y')

    # variable含むレイヤ作成するときはvariable_scopeのほうがいいかもしれない
    with tf.variable_scope('network'):
        init_he = tf.contrib.layers.variance_scaling_initializer()
        my_hidden_layer = partial(tf.layers.dense, activation=tf.nn.elu, kernel_initializer=init_he)
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
        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       scope='network/outputs|network/hidden5')
        training_op = optimizer.minimize(loss, var_list=train_vars)

    with tf.name_scope('eval'):
        correct = tf.nn.in_top_k(logit, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')

    data_length = train_y.shape[0]

    # 凍結部分のSaver
    reuse_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                   scope='network/hidden[1234]')
    reuse_vars_dict = dict([(var.op.name, var) for var in reuse_vars])
    restore_saver = tf.train.Saver(reuse_vars_dict)

    # 保存部分
    saver = tf.train.Saver()
    loss_summary = tf.summary.scalar('loss', loss)
    writer = tf.summary.FileWriter(get_logdir(), graph=tf.get_default_graph())

    init = tf.global_variables_initializer()
    n_batches = data_length // batch_size

    with tf.Session() as sess:
        init.run()

        # 凍結部分の復元
        restore_saver.restore(sess, base_dnn)
        top_frozen_layer = hidden4

        # cacheを求める
        hidden_cache = sess.run(top_frozen_layer, feed_dict={x: train_x})
        validate_dict = {x: validate_x, y: validate_y}

        for epoch in range(n_epochs):
            shuffled_idx = np.random.permutation(data_length)
            # xの代わりになる
            hidden_batches = np.array_split(hidden_cache[shuffled_idx], n_batches)
            y_batches = np.array_split(train_y[shuffled_idx], n_batches)

            for hidden_batch, y_batch in zip(hidden_batches, y_batches):
                sess.run(training_op, feed_dict={top_frozen_layer: hidden_batch, y: y_batch})

            writer.add_summary(loss_summary.eval(feed_dict=validate_dict), epoch * n_batches)

            print(epoch, ' epoch accuracy:', accuracy.eval(feed_dict=validate_dict))
        saver.save(sess, save_path)
    writer.close()


def dnn_change_softmax_cached_frozen3(train_x, train_y,
                                      validate_x, validate_y,
                                      n_epochs=100,
                                      batch_size=10,
                                      n_outputs=5,
                                      save_path='./models/chap_11_Q8_num_5to9_change_softmax.ckpt',
                                      base_dnn='./models/chap_11_Q8_num_0to4.ckpt'
                                      ):
    n_inputs = 28 * 28

    # 構築フェーズ
    x = tf.placeholder(tf.float32, shape=(None, n_inputs), name='x')
    y = tf.placeholder(tf.int32, shape=(None), name='y')

    # variable含むレイヤ作成するときはvariable_scopeのほうがいいかもしれない
    with tf.variable_scope('network'):
        init_he = tf.contrib.layers.variance_scaling_initializer()
        my_hidden_layer = partial(tf.layers.dense, activation=tf.nn.elu, kernel_initializer=init_he)
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
        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       scope='network/outputs|network/hidden[45]')
        training_op = optimizer.minimize(loss, var_list=train_vars)

    with tf.name_scope('eval'):
        correct = tf.nn.in_top_k(logit, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')

    data_length = train_y.shape[0]

    # 凍結部分のSaver
    reuse_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                   scope='network/hidden[123]')
    reuse_vars_dict = dict([(var.op.name, var) for var in reuse_vars])
    restore_saver = tf.train.Saver(reuse_vars_dict)

    # 保存部分
    saver = tf.train.Saver()
    loss_summary = tf.summary.scalar('loss', loss)
    writer = tf.summary.FileWriter(get_logdir(), graph=tf.get_default_graph())

    init = tf.global_variables_initializer()
    n_batches = data_length // batch_size

    with tf.Session() as sess:
        init.run()

        # 凍結部分の復元
        restore_saver.restore(sess, base_dnn)
        top_frozen_layer = hidden3

        # cacheを求める
        hidden_cache = sess.run(top_frozen_layer, feed_dict={x: train_x})
        validate_dict = {x: validate_x, y: validate_y}

        for epoch in range(n_epochs):
            shuffled_idx = np.random.permutation(data_length)
            # xの代わりになる
            hidden_batches = np.array_split(hidden_cache[shuffled_idx], n_batches)
            y_batches = np.array_split(train_y[shuffled_idx], n_batches)

            for hidden_batch, y_batch in zip(hidden_batches, y_batches):
                sess.run(training_op, feed_dict={top_frozen_layer: hidden_batch, y: y_batch})

            writer.add_summary(loss_summary.eval(feed_dict=validate_dict), epoch * n_batches)

            print(epoch, ' epoch accuracy:', accuracy.eval(feed_dict=validate_dict))
        saver.save(sess, save_path)
    writer.close()


def dnn_change_native(train_x, train_y,
                      validate_x, validate_y,
                      n_epochs=100,
                      batch_size=10,
                      n_outputs=5,
                      save_path='./models/chap_11_Q8_num_5to9_change_softmax.ckpt',
                      ):
    """凍結しない"""
    n_inputs = 28 * 28

    # 構築フェーズ
    x = tf.placeholder(tf.float32, shape=(None, n_inputs), name='x')
    y = tf.placeholder(tf.int32, shape=(None), name='y')

    # variable含むレイヤ作成するときはvariable_scopeのほうがいいかもしれない
    with tf.variable_scope('network'):
        init_he = tf.contrib.layers.variance_scaling_initializer()
        my_hidden_layer = partial(tf.layers.dense, activation=tf.nn.elu, kernel_initializer=init_he)
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
        training_op = optimizer.minimize(loss)

    with tf.name_scope('eval'):
        correct = tf.nn.in_top_k(logit, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')

    data_length = train_y.shape[0]

    # 保存部分
    saver = tf.train.Saver()
    loss_summary = tf.summary.scalar('loss', loss)
    writer = tf.summary.FileWriter(get_logdir(), graph=tf.get_default_graph())

    init = tf.global_variables_initializer()
    n_batches = data_length // batch_size

    with tf.Session() as sess:
        init.run()

        validate_dict = {x: validate_x, y: validate_y}

        for epoch in range(n_epochs):
            for batch_index in range(n_batches):
                x_batch, y_batch = fetch_batch(train_x, train_y, batch_size)
                sess.run(training_op, feed_dict={x: x_batch, y: y_batch})
                writer.add_summary(loss_summary.eval(feed_dict=validate_dict), epoch * n_batches + batch_index)
            print(epoch, ' epoch accuracy:', accuracy.eval(feed_dict=validate_dict))

        saver.save(sess, save_path)
    writer.close()


if __name__ == '__main__':
    mnist = get_mnist_data()

    _, _, train_x, train_y = slice_mnist(mnist.train.images, mnist.train.labels)
    _, _, validate_x, validate_y = slice_mnist(mnist.validation.images, mnist.validation.labels)
    _, _, test_x, test_y = slice_mnist(mnist.test.images, mnist.test.labels)

    # top_kで0-4とするため
    train_y = train_y - 5
    validate_y = validate_y - 5
    test_y = test_y - 5

    # a
    # dnn_change_softmax(train_x, train_y, validate_x, validate_y)

    # ここでlabelは0-4
    train_x_100, train_y_100 = extract_with_label_range(train_x, train_y, range(0, 5), 100)
    # b
    # dnn_change_softmax(train_x_100, train_y_100, validate_x, validate_y,
    #                    save_path='./models/chap_11_Q8_num_5to9_change_softmax_n_100.ckpt')
    #
    # c
    dnn_change_softmax_cached(train_x_100, train_y_100, validate_x, validate_y,
                              save_path='./models/chap_11_Q9_c.ckpt')
    tf.reset_default_graph()

    # d
    dnn_change_softmax_cached_frozen4(train_x_100, train_y_100, validate_x, validate_y,
                                      save_path='./models/chap_11_Q9_d.ckpt')
    tf.reset_default_graph()

    # e
    dnn_change_softmax_cached_frozen3(train_x_100, train_y_100, validate_x, validate_y,
                                      save_path='./models/chap_11_Q9_e.ckpt')
    tf.reset_default_graph()

    # native
    dnn_change_native(train_x_100, train_y_100, validate_x, validate_y, save_path='./models/chap_11_Q9_native.ckpt')
    tf.reset_default_graph()

    prob_list = []
    for e in ['c', 'd', 'e', 'native']:
        prob_list.append(predict(test_x, test_y, f'./models/chap_11_Q9_{e}.ckpt'))
        tf.reset_default_graph()

    print(prob_list)
