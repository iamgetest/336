# coding:utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    # tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)
    # x(input)  : [batch, in_height, in_width, in_channels]
    # W(filter) : [filter_height, filter_width, in_channels, out_channels]
    # strides   : The stride of the sliding window for each dimension of input.
    #             For the most common case of the same horizontal and vertices strides, strides = [1, stride, stride, 1]


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')
    # tf.nn.max_pool(value, ksize, strides, padding, data_format='NHWC', name=None)
    # x(value)              : [batch, height, width, channels]
    # ksize(pool大小)        : A list of ints that has length >= 4. The size of the window for each dimension of the input tensor.
    # strides(pool滑动大小)   : A list of ints that has length >= 4. The stride of the sliding window for each dimension of the input tensor.


def first_cnn_layer(x):
    x_image = tf.reshape(x, [-1, 28, 28, 1])  # 最后一维代表通道数目，如果是rgb则为3
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    # x_image -> [batch, in_height, in_width, in_channels]
    #            [batch, 28, 28, 1]
    # W_conv1 -> [filter_height, filter_width, in_channels, out_channels]
    #            [5, 5, 1, 32]
    # output  -> [batch, out_height, out_width, out_channels]
    #            [batch, 28, 28, 32]
    h_pool1 = max_pool_2x2(h_conv1)
    return h_pool1
    # h_conv1 -> [batch, in_height, in_weight, in_channels]
    #            [batch, 28, 28, 32]
    # output  -> [batch, out_height, out_weight, out_channels]
    #            [batch, 14, 14, 32]


def second_cnn_layer():
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_pool1 = first_cnn_layer(x)
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    return h_pool2
    # h_conv2 -> [batch, 14, 14, 64]
    # output  -> [batch, 7, 7, 64]
    #123456


def full_connection():
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(second_cnn_layer(), [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    return h_fc1


def out_layer(keep_prob):
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    # keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(full_connection(), keep_prob)
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    return y_conv


y_conv = out_layer(0.5)
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))  # 计算交叉熵
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)  # 使用adam优化器来以0.0001的学习率来进行微调
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))  # 判断预测标签和实际标签是否匹配
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(1000):  # 开始训练模型，循环训练5000次
        batch = mnist.train.next_batch(50)  # batch大小设置为50
        if i % 100 == 0:
            train_accuracy = accuracy.eval(session=sess,
                                           feed_dict={x: batch[0], y_: batch[1]})
            print("step %d, train_accuracy %g" % (i, train_accuracy))
            sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})  # 神经元输出保持不变的概率 keep_prob 为0.5

    print("test accuracy %g" % accuracy.eval(session=sess,feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    # 神经元输出保持不变的概率 keep_prob 为 1，即不变，一直保持输出