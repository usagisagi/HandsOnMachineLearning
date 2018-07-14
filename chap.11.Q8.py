from datetime import datetime
from functools import partial

import numpy as np
import tensorflow as tf

from my_util import predict, slice_mnist, get_mnist_data, get_logdir


def dnn(train_x, train_y,
        validate_x, validate_y,
        n_epochs=5,
        batch_size=25,
        n_inputs=28 * 28,
        n_outputs=5,
        save_path='./models/chap_11_Q8_num_0to4.ckpt'):
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
        optimizer = tf.train.AdamOptimizer()
        training_op = optimizer.minimize(loss)

    with tf.name_scope('eval'):
        correct = tf.nn.in_top_k(logit, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')

    data_length = train_y.shape[0]

    def fetch_batch(epoch, batch_index, batch_size):
        np.random.seed(epoch * batch_size + batch_index)
        indices = np.random.randint(data_length, size=batch_size)
        x_batch = train_x[indices]
        y_batch = train_y[indices]
        return x_batch, y_batch

    saver = tf.train.Saver()
    loss_summary = tf.summary.scalar('loss', loss)
    writer = tf.summary.FileWriter(get_logdir(), graph=tf.get_default_graph())

    init = tf.global_variables_initializer()
    n_batches = data_length // batch_size

    with  tf.Session() as sess:
        init.run()
        validate_dict = {x: validate_x, y: validate_y}

        for epoch in range(n_epochs):
            for batch_index in range(n_batches):
                x_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
                sess.run(training_op, feed_dict={x: x_batch, y: y_batch})
                writer.add_summary(loss_summary.eval(feed_dict=validate_dict), epoch * n_batches + batch_index)
            print(epoch, ' epoch accuracy:', accuracy.eval(feed_dict=validate_dict))
        saver.save(sess, save_path)
    writer.close()


def dnn_batch(train_x, train_y,
              validate_x, validate_y,
              n_epochs=5,
              batch_size=25,
              n_inputs=28 * 28,
              n_outputs=5,
              save_path='./models/chap_11_Q8_num_0to4_batch.ckpt'):
    # 構築フェーズ
    x = tf.placeholder(tf.float32, shape=(None, n_inputs), name='x')
    y = tf.placeholder(tf.int32, shape=(None), name='y')
    init_he = tf.contrib.layers.variance_scaling_initializer()
    training = tf.placeholder_with_default(False, (), 'training')

    def my_hidden_layer(input, n_output, name):
        with tf.variable_scope(name):
            hidden = tf.layers.dense(input, n_output, kernel_initializer=init_he)
            bn = tf.layers.batch_normalization(hidden, training=training)
            bn_act = tf.nn.elu(bn)
            return bn_act

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
        optimizer = tf.train.AdamOptimizer()
        training_op = optimizer.minimize(loss)
        extra_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.name_scope('eval'):
        correct = tf.nn.in_top_k(logit, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')

    data_length = train_y.shape[0]

    def fetch_batch(epoch, batch_index, batch_size):
        np.random.seed(epoch * batch_size + batch_index)
        indices = np.random.randint(data_length, size=batch_size)
        x_batch = train_x[indices]
        y_batch = train_y[indices]
        return x_batch, y_batch

    saver = tf.train.Saver()
    loss_summary = tf.summary.scalar('loss', loss)
    writer = tf.summary.FileWriter(get_logdir(), graph=tf.get_default_graph())

    init = tf.global_variables_initializer()
    n_batches = data_length // batch_size

    with  tf.Session() as sess:
        init.run()
        validate_dict = {x: validate_x, y: validate_y}

        for epoch in range(n_epochs):
            for batch_index in range(n_batches):
                x_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
                sess.run([extra_op, training_op], feed_dict={training: True, x: x_batch, y: y_batch})
                writer.add_summary(loss_summary.eval(feed_dict=validate_dict), epoch * n_batches + batch_index)
            print(epoch, ' epoch accuracy:', accuracy.eval(feed_dict=validate_dict))
        saver.save(sess, save_path)
    writer.close()


def dnn_batch_dropout(train_x, train_y,
                      validate_x, validate_y,
                      n_epochs=5,
                      batch_size=25,
                      n_inputs=28 * 28,
                      n_outputs=5,
                      dropout_rate=0.5,
                      save_path='./models/chap_11_Q8_num_0to4_batch_dropout.ckpt'):
    # 構築フェーズ
    x = tf.placeholder(tf.float32, shape=(None, n_inputs), name='x')
    y = tf.placeholder(tf.int32, shape=(None), name='y')
    training = tf.placeholder_with_default(False, (), 'training')

    def my_hidden_layer(input, n_output, name):
        with tf.variable_scope(name):
            init_he = tf.contrib.layers.variance_scaling_initializer()
            hidden = tf.layers.dense(input, n_output, kernel_initializer=init_he)
            bn = tf.layers.batch_normalization(hidden, training=training)
            bn_act = tf.nn.elu(bn)
            bn_drop = tf.layers.dropout(bn_act, rate=dropout_rate)
            return bn_drop

    with tf.variable_scope('network'):
        hidden1 = my_hidden_layer(x, 100, name='hidden1')
        hidden2 = my_hidden_layer(hidden1, 100, name='hidden2')
        hidden3 = my_hidden_layer(hidden2, 100, name='hidden3')
        hidden4 = my_hidden_layer(hidden3, 100, name='hidden4')
        hidden5 = my_hidden_layer(hidden4, 100, name='hidden5')
        logit = tf.layers.dense(hidden5, n_outputs, name='output')

    with tf.name_scope('loss'):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logit)
        loss = tf.reduce_mean(xentropy, name='loss')

    with tf.name_scope('optimize'):
        optimizer = tf.train.AdamOptimizer()
        training_op = optimizer.minimize(loss)
        extra_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.name_scope('eval'):
        correct = tf.nn.in_top_k(logit, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')

    data_length = train_y.shape[0]

    def fetch_batch(epoch, batch_index, batch_size):
        np.random.seed(epoch * batch_size + batch_index)
        indices = np.random.randint(data_length, size=batch_size)
        x_batch = train_x[indices]
        y_batch = train_y[indices]
        return x_batch, y_batch

    saver = tf.train.Saver()
    loss_summary = tf.summary.scalar('loss_sum', loss)
    writer = tf.summary.FileWriter(get_logdir(), graph=tf.get_default_graph())

    init = tf.global_variables_initializer()
    n_batches = data_length // batch_size

    with  tf.Session() as sess:
        init.run()
        validate_dict = {x: validate_x, y: validate_y}

        for epoch in range(n_epochs):
            for batch_index in range(n_batches):
                x_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
                sess.run([extra_op, training_op], feed_dict={training: True, x: x_batch, y: y_batch})
                writer.add_summary(loss_summary.eval(feed_dict=validate_dict), epoch * n_batches + batch_index)
            print(epoch, ' epoch accuracy:', accuracy.eval(feed_dict=validate_dict))
        saver.save(sess, save_path)
    writer.close()


if __name__ == '__main__':
    mnist = get_mnist_data()
    train_x, train_y, _, _ = slice_mnist(mnist.train.images, mnist.train.labels)
    validate_x, validate_y, _, _ = slice_mnist(mnist.validation.images, mnist.validation.labels)
    test_x, test_y, _, _ = slice_mnist(mnist.test.images, mnist.test.labels)
    # dnn(train_x, train_y, validate_x, validate_y)
    # tf.reset_default_graph()
    # dnn_batch(train_x, train_y, validate_x, validate_y)
    # tf.reset_default_graph()
    # dnn_batch_dropout(train_x, train_y, validate_x, validate_y)
    # tf.reset_default_graph()
    std = predict(test_x, test_y, './models/chap_11_Q8_num_0to4.ckpt')
    tf.reset_default_graph()
    batch = predict(test_x, test_y, './models/chap_11_Q8_num_0to4_batch.ckpt')
    tf.reset_default_graph()
    drop = predict(test_x, test_y, './models/chap_11_Q8_num_0to4_batch_dropout.ckpt')

    print(std)
    print(batch)
    print(drop)
