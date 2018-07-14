import re
import pandas as pd
from datetime import datetime
from typing import List, Optional, Any
import glob
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image

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


def get_model_params():
    """現グラフの変数をすべて保存"""
    gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    model_params = {gvar.op.name: value for gvar, value in zip(gvars, tf.get_default_session().run(gvars))}
    return model_params


def restore_model_params(model_params):
    """graphの変数をresore"""
    gvar_names = list(model_params.keys())

    # gvar中のAssign operateをすべて取得
    assign_ops = {gvar_name: tf.get_default_graph().get_operation_by_name(gvar_name + "/Assign")
                  for gvar_name in gvar_names}

    # 変数名　=> restore
    init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
    feed_doct = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
    tf.get_default_session().run(assign_ops, feed_dict=feed_doct)


def load_image(file_path: str, size: List[int] = None) -> tf.Tensor:
    """画像"""

    if size is None:
        size = [299, 299]

    with open(file_path, 'rb') as f:
        img = tf.image.decode_image(f.read(), channels=3)  # discard alpha channel

    img = tf.reshape(img, img.eval().shape)
    # -1.0 to 1.0
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize_images(img, size, tf.image.ResizeMethod.BICUBIC)
    img = img * 2 - 1.0

    # 誤差により範囲を少し超えることがある
    img = tf.clip_by_value(img, -1.0, 1.0)

    # to RGB
    return img


def load_class_names(file_neme='inception_v3_2016_08_28/image_net_classnames.txt'):
    p = re.compile('^(.*?) (.*)$', re.MULTILINE | re.UNICODE)
    with open(file_neme, 'r') as f:
        ret = [t[1] for t in p.findall(f.read())]
        ret.insert(0, 'background')
    return ret


class FlowerDataSet(object):
    """FlowerDataSetをLoadする"""

    def __init__(self, dir='flower_photos',
                 test_rate=0.1,
                 validate_rate=0.2,
                 batch_size=32,
                 validate_batch_size=32,
                 size=None,
                 image_data_generator=tf.keras.preprocessing.image.ImageDataGenerator(
                     rescale=1. / 255,
                     width_shift_range=0.25,
                     height_shift_range=0.25,
                     brightness_range=[0.5, 1.5],
                     shear_range=0.78 / 3,  # pi / 4
                     rotation_range=360,
                     zoom_range=0.2,
                     fill_mode='reflect'),
                 resize_method='bicubic',
                 ):

        if size is None:
            size = (299, 299)

        self.size = size

        if test_rate + validate_rate > 1.0:
            raise ValueError('too high rates')

        # FilePathsをロードする
        df = pd.DataFrame([(s.split('/')[1], s) for s in glob.glob(dir + "/*/*.jpg")],
                          columns=['flower', 'paths'])

        self.label_dic = {k: i for i, k in enumerate(set(df.flower))}
        flower_label_set = np.stack(df.flower.map(lambda s: self.label_dic[s]))

        def load_img(path):
            im = tf.keras.preprocessing.image \
                .load_img(path, target_size=size, interpolation=resize_method).convert('RGB')
            im_arr = tf.keras.preprocessing.image.img_to_array(im)
            return im_arr

        image_array_set = np.stack(df.paths.map(lambda p: load_img(p)))

        x_train, x_test_validate, y_train, y_test_validate \
            = train_test_split(image_array_set, flower_label_set,
                               test_size=test_rate + validate_rate,
                               random_state=seed)

        x_test, x_validate, y_test, y_validate \
            = train_test_split(x_test_validate, y_test_validate,
                               test_size=validate_rate + (validate_rate + test_rate),
                               random_state=seed)

        self.train_data_generator = \
            image_data_generator.flow(np.array(x_train), y_train, batch_size=batch_size, seed=seed)

        self.validate_data_generator = \
            image_data_generator.flow(np.array(x_validate), y_validate, batch_size=validate_batch_size, seed=seed)

        native_image_data_generator = tf.keras.preprocessing.image.ImageDataGenerator()

        self.x_test = x_test / 255.  # rescale
        self.y_test = y_test

    def _debug_export(self, export_dir='tmp'):
        self.train_data_generator.save_to_dir = export_dir
        next(self.train_data_generator)


def show_multi_image(n_rows, n_cols, images, figsize=(5, 5)):
    """
    画像表示を行う

    :param n_rows:
    :param n_cols:
    :param images:
        4次元の行列
    :param figsize:
    :return:
    """
    fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.subplots_adjust(hspace=0, wspace=0)

    for r in range(n_rows):
        for c in range(n_cols):
            ax[r, c].xaxis.set_major_locator(plt.NullLocator())
            ax[r, c].yaxis.set_major_locator(plt.NullLocator())
            ax[r, c].imshow(images[r * n_cols + c, :, :], cmap='bone')

    plt.show()


if __name__ == '__main__':
    data = FlowerDataSet()
