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
activation = tf.nn.elu

x = tf.placeholder(tf.float32, shape=[None, n_inputs])

weights1_init = he_init([n_inputs, n_hidden1])
weights2_init = he_init([n_hidden1, n_hidden2])
weights3_init = he_init([n_hidden2, n_hidden3])
weights4_init = he_init([n_hidden3, n_outputs])

weights1 = tf.Variable(weights1_init, dtype=tf.float32, name="weights1")
weights2 = tf.Variable(weights2_init, dtype=tf.float32, name="weights2")
weights3 = tf.Variable(weights3_init, dtype=tf.float32, name="weights3")
weights4 = tf.Variable(weights4_init, dtype=tf.float32, name="weights4")

biases1 = tf.Variable(tf.zeros(n_hidden1), name='biases1')
biases2 = tf.Variable(tf.zeros(n_hidden2), name='biases2')
biases3 = tf.Variable(tf.zeros(n_hidden3), name='biases3')
biases4 = tf.Variable(tf.zeros(n_outputs), name='biases4')

with tf.name_scope('full_network'):
    hidden1 = activation(tf.matmul(x, weights1) + biases1)
    hidden2 = activation(tf.matmul(hidden1, weights2) + biases2)
    hidden3 = activation(tf.matmul(hidden2, weights3) + biases3)
    outputs = tf.matmul(hidden3, weights4) + biases4

optimizer = tf.train.AdamOptimizer(learning_rate)

with tf.name_scope('phase1'):
    phase1_outputs = tf.matmul(hidden1, weights4) + biases4
    phase1_reconstruct_loss = tf.reduce_mean(tf.square(phase1_outputs - x))
    phase1_reg_loss = l2_regularizer(weights1) + l2_regularizer(weights4)
    phase1_loss = phase1_reconstruct_loss + phase1_reg_loss
    phase1_training_op = optimizer.minimize(phase1_loss)

with tf.name_scope('phase2'):
    phase2_reconstruct_loss = tf.reduce_mean(tf.square(hidden3 - hidden1))  # 出力の差
    phase2_reg_loss = l2_regularizer(weights2) + l2_regularizer(weights3)
    phase2_loss = phase2_reconstruct_loss + phase2_reg_loss
    train_vars = [weights2, weights3, biases2, biases3]
    phase2_training_op = optimizer.minimize(phase2_loss, var_list=train_vars)

with tf.name_scope('eval'):
    eval_reconstruction_loss = tf.reduce_mean(tf.square(outputs - x))  # MSE


def train(n_epoches=None,
          batch_size=None,
          save_model_name='models/chap_15_3_3.ckpt'):
    if n_epoches is None:
        n_epoches = [5, 5]

    if batch_size is None:
        batch_size = [150, 150]

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(logdir=get_logdir(), graph=tf.get_default_graph())

    training_op = [phase1_training_op, phase2_training_op]
    reconstruction_loss = [phase1_reconstruct_loss, phase2_reconstruct_loss]
    test_loss_summary = tf.summary.scalar('loss', eval_reconstruction_loss)

    with tf.Session() as sess:
        init.run()
        for phase in range(2):
            # phaseごとの初期化処理
            n_batches = mnist.train.num_examples // batch_size[phase]
            print(f'phase {phase+1} start')
            if phase == 1:
                hidden1_cache = hidden1.eval(feed_dict={x: mnist.train.images})  # キャッシュ

            for epoch in range(n_epoches[phase]):
                for iteration in range(n_batches):
                    if phase == 1:
                        indices = np.random.permutation(mnist.train.num_examples)  # ランダムでindexを選択する
                        hidden1_batch = hidden1_cache[indices[:batch_size[phase]]]
                        feed_dict = {hidden1: hidden1_batch}

                    else:
                        x_batch, _ = mnist.train.next_batch(batch_size[phase])
                        feed_dict = {x: x_batch}

                    sess.run(training_op[phase], feed_dict=feed_dict)

                loss_train = reconstruction_loss[phase].eval(feed_dict=feed_dict)
                print(epoch, " reconstruction loss:", loss_train)

                loss_test = eval_reconstruction_loss.eval({x: mnist.test.images})
                writer.add_summary(test_loss_summary.eval({x: mnist.test.images}),
                                   global_step=phase * n_epoches[phase] + epoch)
                print("TestMSE: ", loss_test)

        saver.save(sess, save_model_name)
        writer.close()


def predict(images,
            load_model_name='models/chap_15_3_3.ckpt'):
    restore_saver = tf.train.Saver()
    with tf.Session() as sess:
        restore_saver.restore(sess, load_model_name)
        reconstructed_images = outputs.eval(feed_dict={x: images})

    # show
    concated_images = np.concatenate([images, reconstructed_images], axis=0).reshape(images.shape[0] * 2, 28, 28)
    show_multi_image(2, images.shape[0], concated_images)


def view_neurons(load_model_name='models/chap_15_3_3.ckpt'):
    restore_saver = tf.train.Saver()
    with tf.Session() as sess:
        restore_saver.restore(sess, load_model_name)
        weights1_val = weights1.eval()
        weights1_imgs = weights1_val.T[:6].reshape(6, 28, 28)
        show_multi_image(2, 3, weights1_imgs)


if __name__ == '__main__':
    # train()
    # test_images = mnist.test.next_batch(10)[0]
    # predict(test_images)
    view_neurons()
