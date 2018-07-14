import tensorflow as tf
from tensorflow.contrib.slim.nets import inception
import tensorflow.contrib.slim as slim
from my_util import load_image, load_class_names, get_logdir, get_model_params, restore_model_params
import numpy as np
from my_util import FlowerDataSet

seed = 42

CLASS_NAME_DICT = load_class_names()


def inception_net(
        ckpt="inception_v3_2016_08_28/inception_v3.ckpt",
        n_epochs=1000,
        n_batchs=20,
        batch_size=32 * 16,
        validation_batch_size=32 * 16,
        check_batch_interval=5,
        max_checks_without_progress=10,
        save_path="models/chap13_Q9.ckpt"):
    x = tf.placeholder(tf.float32, shape=[None, 299, 299, 3], name='x')
    y = tf.placeholder(tf.int32, shape=[None], name='y')

    with slim.arg_scope(inception.inception_v3_arg_scope()):
        # (logits, 層の情報の集合)
        inception.inception_v3(x, num_classes=1001, is_training=False)

        # 新しい層を組む前にload用のsaverを作成すること
        # saverを呼び出した時点でload対象の変数scopeを決定する
        inception_saver = tf.train.Saver()

        pre_logits = tf.get_default_graph().get_tensor_by_name('InceptionV3/Logits/Dropout_1b/Identity:0')
        # 凍結
        pre_logits_stop = tf.squeeze(tf.stop_gradient(pre_logits), [1, 2])
        logits = tf.layers.dense(pre_logits_stop, 5,
                                 kernel_initializer=tf.keras.initializers.he_uniform(seed),
                                 name='flower_logits')

    with tf.name_scope('loss'):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
        loss = tf.reduce_mean(xentropy)
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(loss)

    with tf.name_scope('eval'):
        pred_y = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(pred_y, tf.float32), name='accuracy')

    # 実行フェーズ
    saver = tf.train.Saver()
    loss_summary = tf.summary.scalar('loss', loss)

    init = tf.global_variables_initializer()
    writer = tf.summary.FileWriter(get_logdir(), tf.get_default_graph())

    # early_stopping用の変数
    best_loss_val = np.infty
    checks_since_last_progress = 0
    best_model_params = None

    # 画像のデータ作成
    data_set = FlowerDataSet(batch_size=batch_size, validate_batch_size=validation_batch_size)

    with tf.Session() as sess:
        init.run()
        inception_saver.restore(sess, ckpt)

        print('train start')
        for epoch in range(n_epochs):
            for batch_index in range(n_batchs):
                x_batch, y_batch = next(data_set.train_data_generator)
                sess.run(train_op, feed_dict={x: x_batch, y: y_batch})

                if batch_index % check_batch_interval == 0:
                    print(epoch * n_batchs + batch_index, ' step', ' best loss :', best_loss_val)
                    x_validate, y_validate = next(data_set.validate_data_generator)
                    validate_feed_dict = {x: x_validate, y: y_validate}
                    loss_val = loss.eval(feed_dict=validate_feed_dict)
                    # print(epoch, ' epoch accuracy :', accuracy.eval(feed_dict=validate_feed_dict))
                    writer.add_summary(loss_summary.eval(validate_feed_dict), epoch * n_batchs + batch_index)

                    # update
                    if loss_val < best_loss_val:
                        best_loss_val = loss_val
                        best_model_params = get_model_params()
                        checks_since_last_progress = 0
                    else:
                        checks_since_last_progress += 1

            # epochの最終時のステップ
            x_validate, y_validate = next(data_set.validate_data_generator)
            validate_feed_dict = {x: x_validate, y: y_validate}
            print(epoch, ' epoch accuracy :', accuracy.eval(feed_dict=validate_feed_dict))
            print('test data accuracy :', accuracy.eval(feed_dict={x: data_set.x_test, y: data_set.y_test}))

            if checks_since_last_progress > max_checks_without_progress:
                print('early stopping')
                break

        if best_model_params:
            restore_model_params(best_model_params)

        print('test data accuracy :', accuracy.eval(feed_dict={x: data_set.x_test, y: data_set.y_test}))
        saver.save(sess, save_path)


if __name__ == '__main__':
    inception_net()
