from datetime import datetime
from typing import List, Union

from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import tensorflow as tf

seed = 42
np.random.seed(seed)


def get_mnist_data():
    """mnistのデータを取得"""
    return input_data.read_data_sets('/tmp/data/')


def get_logdir():
    """logdirフォルダを取得"""
    now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    root_logdir = 'tf_logs'
    logdir = f'{root_logdir}/run-{now}/'
    return logdir


def predict(eval_x, eval_y, model_file='./models/chap_11_Q8_num_0to4_batch.ckpt'):
    """スコアを算出"""
    saver = tf.train.import_meta_graph(model_file + '.meta')

    with tf.Session() as sess:
        saver.restore(sess, model_file)
        x = tf.get_default_graph().get_tensor_by_name('x:0')
        y = tf.get_default_graph().get_tensor_by_name('y:0')
        accuracy = tf.get_default_graph().get_tensor_by_name('eval/accuracy:0')
        return accuracy.eval(feed_dict={x: eval_x, y: eval_y})


def slice_mnist(x, y, edge=4):
    """mnistを分割する"""
    x = x.astype(np.float32)
    y = y.astype(np.int32)
    lower_indexs = y <= edge
    upper_indexs = ~lower_indexs
    return x[lower_indexs], y[lower_indexs], x[upper_indexs], y[upper_indexs]


def fetch_batch(x, y, batch_size):
    """batch用データを取得"""
    data_length = y.shape[0]
    indices = np.random.randint(data_length, size=batch_size)
    x_batch = x[indices]
    y_batch = y[indices]
    return x_batch, y_batch


def extract_with_specific_label(x, y, label, size=100):
    """特定のlabelを持つデータをsize分抽出"""

    indexes = y == label
    target_x = x[indexes]
    target_y = y[indexes]
    return fetch_batch(target_x, target_y, size)


def extract_with_label_range(x, y, label_range, each_size=100):
    """特定のlabelを持つデータを各々size抽出"""

    data_list = [extract_with_specific_label(x, y, label, each_size) for label in label_range]
    target_x = np.concatenate([data_list[i][0] for i in range(len(data_list))])
    target_y = np.concatenate([data_list[i][1] for i in range(len(data_list))])
    n_datas = target_y.shape[0]
    indices = np.random.permutation(n_datas)
    return target_x[indices], target_y[indices]
