# 数据加载

import _pickle as pickle
import gzip
import random

import numpy as np

import matplotlib.pyplot as plt

mnist_filename = 'mnist.pkl.gz'


def vectorized_result(j):
    result = np.zeros((10,1))
    result[j] = 1.0
    return result


def load_data():
    f = gzip.open(mnist_filename, 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    f.close()
    return training_data, validation_data, test_data


def load_data_by_batch(batch_size):
    tr_d, va_d, te_d = load_data()
    m = len(tr_d[0])
    index_list = list(range(m))
    random.shuffle(index_list)
    # vectorization input X
    training_inputs = [np.reshape(tr_d[0][i], (784,)) for i in index_list]
    training_results = [np.reshape(vectorized_result(tr_d[1][i]), (10,)) for i in index_list]
    mini_batches = [(np.array(training_inputs[k:k+batch_size]).T, np.array(training_results[k:k+batch_size]).T) for k in range(0, m, batch_size)]

    # init validation data
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])

    # init test data
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return mini_batches, validation_data, test_data


def get_show_image(count):
    tr_d, va_d, te_d = load_data()
    images = tr_d[0][:count]
    labels = tr_d[1][:count]
    return images, labels


def show_fashion_mnist(image, labels):
    _, figs = plt.subplots(1, len(image), figsize=(12, 12))
    for f, img, lbl in zip(figs, image, labels):
        f.imshow(img.reshape((28, 28)), cmap='Greys', interpolation='nearest')
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    mini_batches, validation_data, test_data = load_data_by_batch(64)
    print("total train data count :", len(mini_batches)*64)
    mini_batch,labels = mini_batches[0]
    print(mini_batch.shape, labels.shape)
    for i in range(10):
        print(labels[i][0],end=' ')

    # show mnist hand-written digit image
    images, labels = get_show_image(20)
    show_fashion_mnist(images, labels)
