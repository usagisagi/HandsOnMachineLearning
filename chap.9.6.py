import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
import numpy as np

n_epochs = 1000
learning_rate = 0.01

housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
scaler = StandardScaler()
scaled_housing_data_plus_bias = scaler.fit_transform(housing_data_plus_bias)

# データはconstantで変わらない
x = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name='x')
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='y')

# thetaの初期値
# thetaはVariable = 変数で変わりうる
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name='theta')

# 計算グラフ
y_pred = tf.matmul(x, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")  # 誤差を2乗して平均を求める
# 手計算の微分
# gradients = 2/m * tf.matmul(tf.transpose(x), error)

# tfの微分, theta地点のmseのgradientを計算
# gradients = tf.gradients(mse, [theta])[0]

# theta <- theta - lr * gradients のノードを作成する
# training_op = tf.assign(theta, theta - learning_rate * gradients)

# mseを小さくなるようにVariable ~> thetaを変化する
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    print('train start')
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch:", epoch, "MSE = ", mse.eval())
        sess.run(training_op)   # ノード: training_opを実行

    # thetaを評価する
    print('eval best_theta')
    best_theta = theta.eval()

print(best_theta)


