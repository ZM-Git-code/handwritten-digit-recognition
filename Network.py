import numpy as np


class CrossEntropyCost(object):

    @staticmethod
    def cost(a, y):
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    @staticmethod
    def delta(a, y):
        return a - y


class Network(object):
    def __init__(self, sizes, batch_size, cost=CrossEntropyCost):
        self.sizes = sizes
        self.batch_size = batch_size
        self.num_layers = len(sizes)
        self.cost = cost
        # weight and biases init
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def accuracy(self, data, convert=False):
        result_accuracy = 0
        if convert:
            for (x, y) in data:
                if np.argmax(self.feedforward(x)) == np.argmax(y):
                    result_accuracy += 1
        else:
            for x, y in data:
                if np.argmax(self.feedforward(x)) == y:
                    result_accuracy += 1

        return result_accuracy

    def train(self, mini_batches, epochs, learning_rate, lmbda, evaluation_data=None):
        if evaluation_data:
            evaluation_data = list(evaluation_data)
            n_data = len(evaluation_data)

        if evaluation_data:
            accuracy = self.accuracy(evaluation_data)
            print("Non train accuracy on evaluation data: {} / {}".format(accuracy, n_data))

        m = epochs * self.batch_size

        for mini_batch, labels in mini_batches[:epochs]:
            self.update_mini_batch(mini_batch, labels, learning_rate, lmbda, m)
            if evaluation_data:
                accuracy = self.accuracy(evaluation_data)
                print("Accuracy on evaluation data: {} / {}".format(accuracy, n_data))

    def bias_grad_decent(self, layer_index, learning_rate, db):
        db = np.sum(db, axis=1).reshape(self.biases[layer_index].shape)
        self.biases[layer_index] = self.biases[layer_index] - (learning_rate / self.batch_size) * db

    def weight_grad_decent(self, layer_index, learning_rate, dw, lmbda, m):
        self.weights[layer_index] = (1 - learning_rate * (lmbda / m)) * self.weights[layer_index] - (
                    learning_rate / self.batch_size) * dw

    def update_mini_batch(self, mini_batch, labels, learning_rate, lmbda, m):
        activations, zs = self.forward_compute(mini_batch)

        delta = self.cost.delta(activations[-1], labels)
        self.bias_grad_decent(-1, learning_rate, delta)
        self.weight_grad_decent(-1, learning_rate, np.dot(delta, activations[-2].transpose()), lmbda, m)

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            self.bias_grad_decent(-l, learning_rate, delta)
            self.weight_grad_decent(-l, learning_rate, np.dot(delta, activations[-l - 1].transpose()), lmbda, m)

    def update_mini_batch_by_vectorization(self, mini_batch, labels, learning_rate, lmbda, m):
        delta_nabla_b, delta_nabla_w = self.backprop_vectorization(mini_batch, labels)
        self.weights = [(1 - learning_rate * (lmbda / m)) * w - (learning_rate / self.batch_size) * dw for w, dw in
                        zip(self.weights, delta_nabla_w)]
        self.biases = [b - (learning_rate / self.batch_size) * db for b, db in zip(self.biases, delta_nabla_b)]

    def forward_compute(self, mini_batch):
        a = mini_batch
        activations = [a]
        zs = []
        for W, B in zip(self.weights, self.biases):
            z = np.dot(W, a) + B
            zs.append(z)
            a = sigmoid(z)
            # print("activation shape : ", a.shape)
            activations.append(a)
        return activations, zs

    def backprop_vectorization(self, mini_batch, labels):
        activations, zs = self.forward_compute(mini_batch)

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        delta = self.cost.delta(activations[-1], labels)
        nabla_b[-1] = np.sum(delta, axis=1).reshape((10, 1))
        # print("db 3: ", nabla_b[-1].shape)
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # print("dw 3: ", nabla_w[-1].shape)
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = np.sum(delta, axis=1).reshape(nabla_b[-l].shape)
            # print("db 2: ", nabla_b[-l].shape)
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
            # print("dw 2: ", nabla_w[-l].shape)
        return nabla_b, nabla_w


def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))
