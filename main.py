import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def main():
    fashion = pd.read_csv('fashion-mnist_train.csv')
    labels = fashion['label']
    fashion = fashion.drop('label', 1)

    fashion_test = pd.read_csv('fashion-mnist_test.csv')
    labels_test = fashion_test['label']
    fashion_test = fashion_test.drop('label', 1)

    labelsOHE = np.zeros((60000, 10))

    test_labelsOHE = np.zeros((10000, 10))

    for i, index in enumerate(labels):
        labelsOHE[i, index] = 1

    for i, index in enumerate(labels_test):
        print(index)
        test_labelsOHE[i, index] = 1

    scaler = MinMaxScaler()

    train_scaled = scaler.fit_transform(fashion)
    test_scaled = scaler.fit_transform(fashion_test)

    labels = {'0': 'T-shirt/top',
              '1': 'Trouser',
              '2': 'Pullover',
              '3': 'Dress',
              '4': 'Coat',
              '5': 'Sandal',
              '6': 'Shirt',
              '7': 'Sneaker',
              '8': 'Bag',
              '9': 'Ankle boot'}

    x = tf.placeholder(tf.float32, shape=[None, 784])

    y_true = tf.placeholder(tf.float32, shape=[None, 10])

    # layers
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    convo_1 = convolutional_layer(x_image, shape=[5, 5, 1, 32])
    convo_1_pooling = max_pool_2by2(convo_1)

    convo_2 = convolutional_layer(convo_1_pooling, shape=[5, 5, 32, 64])
    convo_2_pooling = max_pool_2by2(convo_2)

    convo_2_flat = tf.reshape(convo_2_pooling, [-1, 7 * 7 * 64])
    # 1024 is the number of neurons
    full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat, 1024))

    # DROPOUT
    hold_prob = tf.placeholder(tf.float32)
    full_one_dropout = tf.nn.dropout(full_layer_one, keep_prob=hold_prob)

    y_pred = normal_full_layer(full_one_dropout, 10)

    # loss function
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))

    # optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train = optimizer.minimize(cross_entropy)

    init = tf.global_variables_initializer()

    steps = 5000

    with tf.Session() as sess:
        sess.run(init)

        for i in range(steps):
            batch_x, batch_y = next_batch(50, train_scaled, labelsOHE)
            #         print(batch_x.shape)
            #         print(batch_x)
            #         print(batch_y.shape)
            #         print(batch_y)

            sess.run(train, feed_dict={x: batch_x, y_true: batch_y, hold_prob: 0.5})

            if i % 100 == 0:
                print("running")
                print("ON STEP: {}".format(i))
                print("Accuracy: ")
                matches = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))

                acc = tf.reduce_mean(tf.cast(matches, tf.float32))
                print(sess.run(acc, feed_dict={x: test_scaled, y_true: test_labelsOHE, hold_prob: 1.0}))
                print('\n')


def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_dist)


def init_bias(shape):
    init_bias_vals = tf.constant(0, 1, shape=shape)
    return tf.Variable(init_bias_vals)


def conv2d(x, W):
    # x --> [batch, H, w, Channels]
    # w --> [filter H, filter W, Channels IN, Channels OUT]

    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2by2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# CONVOLUTIONAL LAYER
def convolutional_layer(input_x, shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x, W) + b)


# NORMAL (FULLY CONNECTED)
def normal_full_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer, W) + b


def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = data[idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

if __name__ == "__main__":
    main()