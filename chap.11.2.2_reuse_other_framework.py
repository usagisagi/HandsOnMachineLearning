import tensorflow as tf

n_inputs = 2
n_hidden1 = 3

original_w = [[1., 2., 3.], [4., 5., 6.]]
original_b = [7., 8., 9.]

x = tf.placeholder(tf.float32, shape=(None, n_inputs), name='x')
hidden1 = tf.layers.dense(x, n_hidden1, activation=tf.nn.elu, name='hidden1')

graph = tf.get_default_graph()
assign_kernel = graph.get_operation_by_name('hidden1/kernel/Assign')
assign_bias = graph.get_operation_by_name('hidden1/bias/Assign')
init_kernel = assign_kernel.inputs[1]
init_bias = assign_bias.inputs[1]
init = tf.global_variables_initializer()

with tf.Session() as sess:
    # 代入
    sess.run(init, feed_dict={init_kernel: original_w, init_bias: original_b})
    tvars = tf.trainable_variables(scope='hidden1/kernel')
    print("kernel", sess.run(tvars))

    tvars = tf.trainable_variables(scope='hidden1/bias')
    print("bias", sess.run(tvars))
