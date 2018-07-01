import pickle

import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
m, n = housing.data.shape

# c_[xolumn]はsecond_axisでconcatnationする

housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
x = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="x")

# ベクトルを縦に並べる
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
x_t = tf.transpose(x)  # 転置
x_t_dot_x = tf.matmul(x_t, x)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(x_t_dot_x), x_t), y)

with tf.Session() as sess:
    theta_value = theta.eval() # theta_value.shape = (9, 1)

with open("chap.9.5.pkl", 'wb') as f:
    pickle.dump(theta_value, f)
