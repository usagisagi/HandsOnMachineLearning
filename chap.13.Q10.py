# https://github.com/tensorflow/tensorflow/blob/r0.10/tensorflow/examples/tutorials/deepdream/deepdream.ipynb
from __future__ import print_function

import os
from io import BytesIO
import numpy as np
from functools import partial
import PIL.Image

import tensorflow as tf
from IPython.core.display import HTML, display
import matplotlib.pyplot as plt
import seaborn as sns
from my_util import get_logdir

model_fn = 'inception_5h/tensorflow_inception_graph.pb'


def strip_consts(graph_def, max_const_size=32):
    """graph_defから定数を抽出"""
    strip_def = tf.GraphDef()  # 空のGraphDef

    for n0 in graph_def.node:
        n = strip_def.node.add()
        n.MergeFrom(n0)  # n0を結合

        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = b'<stripped {size} bytes>'
    return strip_def


def rename_nodes(graph_def, rename_func):
    res_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = res_def.node.add()
        n.MergeFrom(n0)
        n.name = rename_func(n.name)
        for i, s in enumerate(n.input):
            n.input[i] = rename_func(s) if s[0] != '^' else '^' + rename_func(s[1:])
    return res_def


def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()

    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph' + str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:800px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))


def show_array(a):
    plt.imshow(a)  # 0-1の間を切り取る
    plt.show()


def vis_std(a, s=0.1):
    """可視化のためにimageがぞの分布を正規分布にする"""
    ret = (a - a.mean()) / max(a.std(), 1e-4) * s + 0.5  # 平均0.5、分散1
    return np.uint8(np.clip(ret, 0, 1) * 255)


def T(layer):
    """tensorを取得するヘルパー関数"""
    return graph.get_tensor_by_name(f'import/{layer}:0')


img_noise = np.random.uniform(size=(224, 224, 3)) + 100.0


def render_naive(t_input, t_obj, img0=img_noise, iter_n=20, step=1.0):
    """最適化を行い可視化を行うためのarrayを作成する"""
    vis_img = generate_vis_img(img0, iter_n, step, t_input, t_obj)
    show_array(vis_img)


def render_multiple_naive(t_input, t_input_layer, n_rows=5, n_cols=5, img0=img_noise, iter_n=10, step=1.0):
    vis_img_list = [generate_vis_img(img0, iter_n, step, t_input, t_input_layer[:, :, :, channel])
                    for channel
                    in np.random.randint(t_input_layer.get_shape().as_list()[-1], size=n_rows * n_cols)]

    fig, ax = plt.subplots(n_rows, n_cols, figsize=(n_rows, n_cols))
    fig.subplots_adjust(hspace=0, wspace=0)

    for i in range(n_rows):
        for j in range(n_cols):
            ax[i, j].xaxis.set_major_locator(plt.NullLocator())
            ax[i, j].yaxis.set_major_locator(plt.NullLocator())
            ax[i, j].imshow(vis_img_list[i * n_rows + j])

    plt.show()


def generate_vis_img(img0, iter_n, step, t_input, t_obj):
    t_score = tf.reduce_mean(t_obj)
    t_grad = tf.gradients(t_score, t_input)[0]  # t_scoreをt_inputで微分
    img = img0.copy()
    for i in range(iter_n):
        g, score = sess.run([t_grad, t_score], {t_input: img})
        g /= g.std() + 1e-8  # 分散を1にする
        img += g * step  # gradient更新 => 傾きを**大きくする**
        print(score, end=' ')

    plt.show()
    return vis_std(img)


def tffunc(*argtypes):
    """
    Tf-graphを作成する関数を1つにする
    placeholderをfでwrapし、その値を評価(eval)する関数を生成する関数を返す

     :param argtypes:
        作成するplaceholderのタイプリスト
    """

    placeholders = list(map(tf.placeholder, argtypes))

    def wrap(f):
        """placeholderをfでwrapし、その値を評価(eval)する関数を返す"""
        out = f(*placeholders)

        def wrapper(*args, **kw):
            """
            outをevalする

            :param args:
                placeholderに渡す値
            :param kw:
                session
            :return:
                上階関数で指定したoutのeval methodを実行した返値
            """
            return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))

        return wrapper

    return wrap


def resize(img, size):
    img = tf.expand_dims(img, 0)
    return tf.image.resize_bilinear(img, size)[0, :, :, :]


def calc_grad_tiled(img, t_grad, tile_size=512):
    """
    タイル状にt_gradを計算する。イテレーション毎に呼び出される。
    タイルの境界をぼかすためにイテレーション毎にランダムにシフトしてからtileを切り出す
    """

    sz = tile_size
    h, w = img.shape[:2]
    sx, sy = np.random.randint(sz, size=2)
    img_shift = np.roll(np.roll(img, sx, 1), sy, 0)  # 画像のシフト
    grad = np.zeros_like(img)

    # szが大きい時、バッチサイズの半数以下の端部分は、gradを計算しない。
    for y in range(0, max(h - sz // 2, sz), sz):
        for x in range(0, max(w - sz // 2, sz), sz):
            sub = img_shift[y:y + sz, x:x + sz]
            g = sess.run(t_grad, {t_input: sub})
            grad[y:y + sz, x:x + sz] = g

    # shiftをもとに戻し、もと座標のgradを返す。
    return np.roll(np.roll(grad, -sx, 1), -sy, 0)


def render_multiscale(t_obj, img0=img_noise, iter_n=10, step=1.0, octave_n=3, octave_scale=1.4):
    """

    :param t_obj:
    :param img0:
    :param iter_n:
    :param step:
    :param octave_n:
        拡大を行う回数
    :param octave_scale:
        拡大率
    """
    t_score = tf.reduce_mean(t_obj)
    t_grad = tf.gradients(t_score, t_input)[0]

    img = img0.copy()

    # [np.float32のplaceholder, np.int32のplaceholder]を引数とするresize関数
    resize_method = tffunc(np.float32, np.int32)(resize)

    for octave in range(octave_n):
        if octave > 0:
            # octave_scale倍だけ画像サイズを引き伸ばす
            hw = np.float32(img.shape[:2]) * octave_scale
            img = resize_method(img, np.int32(hw))

        for i in range(iter_n):
            # 正規化の範囲で勾配を増加させる
            g = calc_grad_tiled(img, t_grad)
            g /= g.std() + 1e-8
            img += g * step

            show_array(vis_std(img))


k = np.float32([1, 4, 6, 4, 1])
k = np.outer(k, k)  # 外積

# (5, 5, 1, 1) / (256 / (3, 3)) -> (5, 5, 3, 3)
# 　第一項各々の成分に対し (256/3) * (3,3)の行列で割る
k5x5 = k[:, :, None, None] / k.sum() * np.eye(3, dtype=np.float32)


def lap_split(img):
    """
    :return:
        | lo: 低値の(5 * 5)　=> エッジ以外の部分
        | hi: 高値の(5 * 5)  => エッジ
    """
    with tf.name_scope('split'):
        lo = tf.nn.conv2d(img, k5x5, [1, 2, 2, 1], 'SAME')  # 同サイズでk5x5で畳み込む
        lo2 = tf.nn.conv2d_transpose(lo, k5x5 * 4, tf.shape(img), [1, 2, 2, 1])  # => loの成分を5*5に戻す
        hi = img - lo2  # loの成分から作成したlo2でhiを引くことにより、imgのhiの部分だけ抽出する
    return lo, hi


def lap_split_n(img, n):
    """
    n分割を行い、loとhiに交互に入れる
    """
    levels = []
    for i in range(n):
        img, hi = lap_split(img)
        levels.append(hi)
    levels.append(img)
    return levels[::-1]


def lap_merge(levels):
    """Laplacian Pyramidを結合する"""
    img = levels[0]
    for hi in levels[1:]:
        with tf.name_scope('merge'):
            # imgを拡大しhiを足す
            img = tf.nn.conv2d_transpose(img, k5x5 * 4, tf.shape(hi), [1, 2, 2, 1]) + hi
    return img


def normalize_std(img, eps=1e-10):
    """分散を1にする（平均0に限る）"""
    with tf.name_scope('normalize'):
        std = tf.sqrt(tf.reduce_mean(tf.square(img)))
        return img / tf.maximum(std, eps)


def lap_normarize(img, scale_n=4):
    """ラプラシアン正規化"""
    img = tf.expand_dims(img, 0)
    tlevels = lap_split_n(img, scale_n)
    tlevels = list(map(normalize_std, tlevels))
    out = lap_merge(tlevels)
    return out[0, :, :, :]


def render_lapnorm(t_obj, img0=img_noise,
                   iter_n=10, step=1.0, octave_n=4, octave_scale=1.4, lap_n=4):
    t_score = tf.reduce_mean(t_obj)
    t_grad = tf.gradients(t_score, t_input)[0]

    # scale_nを定めたlap_normalize関数を返す
    lap_norm_func = tffunc(np.float32)(partial(lap_normarize, scale_n=lap_n))

    # [np.float32のplaceholder, np.int32のplaceholder]を引数とするresize関数
    resize_method = tffunc(np.float32, np.int32)(resize)

    img = img0.copy()
    for octave in range(octave_n):
        if octave > 0:
            # imgを引き伸ばす
            hw = np.float32(img.shape[:2]) * octave_scale
            img = resize_method(img, np.int32(hw))
        for i in range(iter_n):
            g = calc_grad_tiled(img, t_grad)
            g = lap_norm_func(g)
            img += g * step

    img = vis_std(img)
    show_array(img)


def render_deepdream(t_obj, img0=img_noise,
                     iter_n=10, step=1.5, octave_n=10, octave_scale=1.1):
    """t_objの傾きが大きくなるように、img0を変化させていく"""

    t_score = tf.reduce_mean(t_obj)
    t_grad = tf.gradients(t_score, t_input)[0]

    img = img0
    # [np.float32のplaceholder, np.int32のplaceholder]を引数とするresize関数
    resize_method = tffunc(np.float32, np.int32)(resize)

    # 解像度が大きい、エッジ順に入る
    octaves = []
    for i in range(octave_n - 1):
        hw = img.shape[:2]
        lo = resize_method(img, np.int32(np.float32(hw) / octave_scale))
        hi = img - resize_method(lo, hw)
        img = lo
        octaves.append(hi)

    for octave in range(octave_n):
        if octave > 0:
            hi = octaves[-octave]
            img = resize_method(img, hi.shape[:2]) + hi
        for i in range(iter_n):
            g = calc_grad_tiled(img, t_grad)
            img += g * (step / (np.abs(g).mean() + 1e-7))

    img = vis_std(img)
    show_array(img)


if __name__ == '__main__':
    # 構築フェーズ
    # 初期値となる画像を投入するプレースホルダー
    t_input = tf.placeholder(np.float32, name='input')

    imagenet_mean = 177.
    t_preprocessed = tf.expand_dims(t_input - imagenet_mean, 0)

    with tf.gfile.FastGFile(model_fn, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # t_processedをスタートとした、googleNetの計算グラフを作成
    tf.import_graph_def(graph_def, {'input': t_preprocessed})

    graph = tf.get_default_graph()

    layers = [op.name for op in graph.get_operations() if op.type == 'Conv2D' and 'import/' in op.name]
    feature_nums = [int(graph.get_tensor_by_name(name + ':0').get_shape()[-1]) for name in layers]


    # Picking some internal layer. Note that we use outputs before applying the ReLU nonlinearity
    # to have non-zero gradients for features with negative initial activations.
    # ReLuでカットされる特徴も抽出するためpre_reluを用いる
    def render(layer='mixed4d_3x3_bottleneck_pre_relu',  # 可視化するlayer名
               channel=139  # 何番目のchannelを取得するか
               ):
        t_layer_channel = T(layer)
        img0 = PIL.Image.open('mountain.jpeg')
        img0 = np.float32(img0)
        render_lapnorm(tf.square(T('mixed4c')))
        render_deepdream(tf.square(T('mixed4c')), img0=img0)


    with tf.Session() as sess:
        render()

    writer = tf.summary.FileWriter(get_logdir(), tf.get_default_graph())
    writer.close()
