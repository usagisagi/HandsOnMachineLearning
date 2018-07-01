from _datetime import datetime

from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import numpy as np

# 定数
sns.set()
c_pallet = sns.color_palette(n_colors=2)
now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
root_logdir = 'tf_logs'
logdir = f'./{root_logdir}/run-{now}/'


def logistic_regression(x_train, y_train, n_epochs=10, batch_size=10, lr=0.1):
    # 構築フェーズ
    x = tf.placeholder(tf.float32, shape=(None, c + 1), name='x')  # N * c+1
    y = tf.placeholder(tf.float32, shape=(None, 1), name='y')
    theta = tf.Variable(tf.fill([c + 1, 1], 1.0e-4), name='theta')  # c+1 * 1
    t = tf.matmul(x, theta, name='t')  # 確率の対数, shape = (N * 1, 1)
    prob = tf.identity(1 / (1 + tf.exp(-t)), name='prob')

    with tf.name_scope('calc_loss'):
        # batchごとのloss
        label_0_mul = tf.matmul(y, tf.log(prob), transpose_a=True, name='positive_prob')
        label_1_mul = tf.matmul(1 - y, tf.log(1 - prob), transpose_a=True, name='negative_prob')
        loss = tf.identity(-1 * tf.reshape(label_0_mul + label_1_mul, ()) / batch_size, name='loss')

    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    training_op = optimizer.minimize(loss)

    n_batches = int(np.ceil(r / batch_size))

    def fetch_batch(epoch, index):
        np.random.seed(epoch * n_batches + index)
        indices = np.random.randint(r, size=batch_size)
        x_batch = x_train[indices]
        y_batch = y_train[indices]
        return x_batch, y_batch

    init = tf.global_variables_initializer()
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
    loss_summary = tf.summary.scalar('loss', loss)

    with tf.Session() as sess:
        sess.run(init)
        all_data_dict = {x: x_train, y: y_train}
        print('train start')
        for epoch in range(n_epochs):
            for i in range(n_batches):
                x_batch, y_batch = fetch_batch(epoch, i)
                sess.run(training_op, feed_dict={x: x_batch, y: y_batch})
            file_writer.add_summary(loss_summary.eval(feed_dict=all_data_dict), epoch)

        prob = sess.run(prob, feed_dict=all_data_dict)
        file_writer.close()
        return prob


def conv_prob_to_label(prob):
    arr = np.zeros(shape=prob.shape, dtype=np.int32)
    arr[prob >= 0.5] = 1
    return arr


def score(pred_label, label):
    return np.sum(pred_label == label) / label.shape[0]


if __name__ == '__main__':
    data = make_moons(n_samples=1000, random_state=42, noise=0.05)
    coordinate = data[0]
    label = data[1][:, np.newaxis]
    r, c = coordinate.shape
    coordinate_with_bias = np.c_[np.ones((r, 1)), coordinate]
    prob = logistic_regression(coordinate_with_bias, label, n_epochs=4000, lr=0.001, batch_size=100)
    pred_label = conv_prob_to_label(prob)
    print(score(pred_label, label))
    plt.scatter(coordinate[:, 0], coordinate[:, 1], c=[c_pallet[i] for i in pred_label.reshape((-1))])
    plt.show()
