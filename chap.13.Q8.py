import tensorflow as tf
from tensorflow.contrib.slim.nets import inception
import tensorflow.contrib.slim as slim
from my_util import load_image, load_class_names
import numpy as np

CLASS_NAME_DICT = load_class_names()

def predict_inception(file_path):
    x = tf.placeholder(tf.float32, shape=[None, 299, 299, 3], name='x')
    with slim.arg_scope(inception.inception_v3_arg_scope()):

        # (logits, 層の情報の集合)
        logits, _ = inception.inception_v3(x, num_classes=1001, is_training=False)
        softmax = tf.nn.softmax(logits)
        top_labels = tf.nn.top_k(logits, 5)[1]

    # predictions = end_points['Predictions']
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        init.run()
        saver.restore(sess, "inception_v3_2016_08_28/inception_v3.ckpt")
        img = load_image(file_path).eval()[np.newaxis, :, :, :]
        print(img.shape)
        print(img.dtype)
        pred_i = top_labels.eval(feed_dict={x: img}).flatten()
        pred_logits = softmax.eval(feed_dict={x: img}).flatten()[pred_i]

        return [(CLASS_NAME_DICT[i], p) for i, p in zip(pred_i, pred_logits)]

if __name__ == '__main__':
    p = predict_inception("inception_v3_2016_08_28/cropped_panda.jpg")
    print('\n'.join([f'{t[0]}\t{t[1]}' for t in p]))
    pass
