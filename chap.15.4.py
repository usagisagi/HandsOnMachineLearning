import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from my_util import get_mnist_data, get_logdir, show_multi_image

n_inputs = 28 * 28  # MNIST
n_hidden1 = 300
n_hidden2 = 150
n_hidden3 = n_hidden1
n_image_outputs = n_inputs
n_outputs = 10

learning_rate = 0.01
l2_reg = 0.0005
mnist = get_mnist_data()
he_init = tf.contrib.layers.variance_scaling_initializer()
l2_regularizer = tf.contrib.layers.l2_regularizer(l2_reg)

my_dense_layer = partial(tf.layers.dense,
                         activation=tf.nn.elu,
                         kernel_initializer=he_init,
                         kernel_regularizer=l2_regularizer)

x = tf.placeholder(tf.float32, shape=[None, n_inputs])
y = tf.placeholder(tf.int32, shape=[None])

with tf.variable_scope('common_layer'):
    hidden1 = my_dense_layer(x, n_hidden1)
    hidden2 = my_dense_layer(hidden1, n_hidden2)

with tf.variable_scope('pretrain_layer'):
    hidden3 = my_dense_layer(hidden2, n_hidden3)
    outputs = my_dense_layer(hidden3, n_image_outputs, activation=None)
    reconstruction_loss = tf.reduce_mean(tf.square(outputs - x))  # MSE
    pretrain_reg_lossess = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope='pretrain_layer/*')

    with tf.name_scope('optimize'):
        pretrain_loss = tf.add_n([reconstruction_loss] + pretrain_reg_lossess, name='pretrain_loss')
        pretrain_optimizer = tf.train.AdamOptimizer(learning_rate)
        pretraining_op = pretrain_optimizer.minimize(pretrain_loss)

with tf.variable_scope('classify_layer'):
    logits = my_dense_layer(hidden2, n_outputs, activation=None)
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
    classify_loss = tf.reduce_mean(xentropy)
    classify_reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope='classify_layer/*')

    with tf.name_scope('optimize'):
        classify_loss_sum = tf.add_n([classify_loss] + classify_reg_losses, name='classify_loss')
        classify_optimizer = tf.train.AdamOptimizer(learning_rate)
        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='classify_layer/*')
        classify_op = classify_optimizer.minimize(classify_loss_sum, var_list=train_vars)

with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

training_op = [pretraining_op, classify_op]


def train(n_epoches=None,
          batch_size=None,
          save_model_name='models/chap_15_4.ckpt'):
    if n_epoches is None:
        n_epoches = [1, 1]

    if batch_size is None:
        batch_size = [150, 150]

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(logdir=get_logdir(), graph=tf.get_default_graph())

    with tf.Session() as sess:
        init.run()
        for phase in range(2):
            print(f"phase {phase} start")
            n_batches = mnist.train.num_examples // batch_size[phase]
            if phase == 1:
                hidden2_cache = hidden2.eval({x: mnist.train.images})

            for epoch in range(n_epoches[phase]):
                for iteration in range(n_batches):
                    if phase == 0:
                        x_batch, _ = mnist.train.next_batch(batch_size[phase])
                        feed_dict = {x: x_batch}
                    else:
                        induces = np.random.permutation(mnist.train.num_examples)[:batch_size[phase]]
                        hidden2_batch = hidden2_cache[induces][:batch_size[phase]]
                        y_batch = mnist.train.labels[induces][:batch_size[phase]]
                        feed_dict = {hidden2: hidden2_batch, y: y_batch}

                    sess.run(training_op[phase], feed_dict=feed_dict)

                print(
                    f'{epoch} epoch classify loss :'
                    f'{classify_loss.eval(feed_dict={x: mnist.train.images, y: mnist.train.labels})}')
        print(f"accuracy: {accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels})}")
        saver.save(sess, save_model_name)
        writer.close()


if __name__ == '__main__':
    train(n_epoches=[1, 1])
