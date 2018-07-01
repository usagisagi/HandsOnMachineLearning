import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
import numpy as np
from datetime import datetime

now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
root_logdir = 'tf_logs'
logdir = f'./{root_logdir}/run-{now}/'

learning_rate = 0.01

housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
scaler = StandardScaler()
scaled_housing_data_plus_bias = scaler.fit_transform(housing_data_plus_bias)

# 構築フェーズ
# PlaceHolderの形だけ定義
x = tf.placeholder(tf.float32, shape=(None, n + 1), name='x')
y = tf.placeholder(tf.float32, shape=(None, 1), name='y')

# thetaの初期値
# thetaはVariable = 変数で変わりうる
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name='theta')

# 計算グラフ
y_pred = tf.matmul(x, theta, name="predictions")

with tf.name_scope('loss') as scope:
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")  # 誤差を2乗して平均を求める

# mseを小さくなるようにVariable ~> thetaを変化する
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()


def fetch_batch(epoch, batch_index, batch_size):
    np.random.seed(epoch * n_batches + batch_index)  # not shown in the book
    indices = np.random.randint(m, size=batch_size)  # not shown
    X_batch = scaled_housing_data_plus_bias[indices]  # not shown
    y_batch = housing.target.reshape(-1, 1)[indices]  # not shown
    return X_batch, y_batch


n_epochs = 10
batch_size = 100
n_batches = int(np.ceil(m / batch_size))

mse_summary = tf.summary.scalar('MSE', mse)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

with tf.Session() as sess:
    sess.run(init)

    print('train start')
    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            x_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)

            if batch_index % 10 == 0:
                summary_str = mse_summary.eval(feed_dict = {x: x_batch, y: y_batch})
                step = epoch * n_batches + batch_index
                file_writer.add_summary(summary_str, step)

            sess.run(training_op, feed_dict={x: x_batch, y: y_batch})

    print('MSE:',
          sess.run(mse, feed_dict={
              x: scaled_housing_data_plus_bias,
              y: housing.target.reshape(-1, 1)
          }))

    # thetaを評価する
    print('eval best_theta')
    best_theta = theta.eval()
    print(best_theta)

file_writer.close()