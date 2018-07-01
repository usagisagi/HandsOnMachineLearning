import tensorflow as tf
from datetime import datetime

now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
root_logdir = 'tf_logs'
logdir = f'./{root_logdir}/run-{now}/'


def relu(x):
    threshold = tf.get_variable('threshold', shape=(), initializer=tf.constant_initializer(0.))
    with tf.variable_scope("relu", reuse=True):
        w_shape = (int(x.get_shape()[1]), 1)
        w = tf.Variable(tf.random_normal(w_shape), name='weights')
        b = tf.Variable(0.0, name='bias')
        z = tf.add(tf.matmul(x, w), b, name="z")
        # z or 0の大きい方を返す -> relu関数
        return tf.maximum(z, threshold, name='max')


n_features = 3
x = tf.placeholder(tf.float32, shape=(None, n_features), name='x')
relus = []
for i in range(5):
    with tf.variable_scope('relu', reuse=(i >= True or None)):
        relus.append(relu(x))

output = tf.add_n(relus, name='output')

file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
file_writer.close()
