import tensorflow as tf
from datetime import datetime

now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
root_logdir = 'tf_logs'
logdir = f'./{root_logdir}/run-{now}/'


def relu(x):
    with tf.name_scope('relu'):
        w_shape = (int(x.get_shape()[1]), 1)
        w = tf.Variable(tf.random_normal(w_shape), name='weights')
        b = tf.Variable(0.0, name='bias')
        z = tf.add(tf.matmul(x, w), b, name="z")
        # z or 0の大きい方を返す -> relu関数
        return tf.maximum(z, 0., name='relu')

n_features = 3
x = tf.placeholder(tf.float32, shape=(None, n_features), name='x')
relus = [relu(x) for _ in range(5)]
# tensorの合計を返す
output = tf.add_n(relus, name='output')

file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
file_writer.close()
