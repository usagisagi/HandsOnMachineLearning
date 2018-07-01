import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
import numpy as np

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
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name='theta')

# 計算グラフ
y_pred = tf.matmul(x, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")  # 誤差を2乗して平均を求める

# mseを小さくなるようにVariable ~> thetaを変化する
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()


def fetch_batch(epoch, batch_index, batch_size):
    np.random.seed(epoch * n_batches + batch_index)  # not shown in the book
    indices = np.random.randint(m, size=batch_size)  # not shown
    X_batch = scaled_housing_data_plus_bias[indices] # not shown
    y_batch = housing.target.reshape(-1, 1)[indices] # not shown
    return X_batch, y_batch


n_epochs = 10
batch_size = 100
n_batches = int(np.ceil(m / batch_size))

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)

    print('train start')
    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            x_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            # feed_dictを介してx, yにデータを食わせる
            sess.run(training_op, feed_dict={x: x_batch, y: y_batch})
            # mseはデータに依存するので、怒られが発生する
            # if epoch % 100 == 0:
            #     print("Epoch:", epoch, "MSE = ", mse.eval())

    print('MSE:',
          sess.run(mse, feed_dict={
              x: scaled_housing_data_plus_bias,
              y: housing.target.reshape(-1, 1)
          }))

    # thetaを評価する
    print('eval best_theta')
    best_theta = theta.eval()
    print(best_theta)

    saver.save(sess, './tmp/my_model_chap.9.8.ckpt')


