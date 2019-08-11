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

    def train(self, mini_batches, epochs, learning_rate, lmbda, evaluation_data=None,
              monitor_evaluation_cost=False,
              monitor_evaluation_accuracy=False,
              monitor_training_cost=False):
        if evaluation_data:
            evaluation_data = list(evaluation_data)
            n_data = len(evaluation_data)

        if evaluation_data:
            accuracy = self.accuracy(evaluation_data)
            print("Non train accuracy on evaluation data: {} / {}".format(accuracy, n_data))

        m = epochs * self.batch_size

        for mini_batch, labels in mini_batches[:epochs]:
            self.update_mini_batch(mini_batch, labels, learning_rate, lmbda, m)

            if monitor_training_cost:
                cost = self.total_cost_on_batch(mini_batch, labels, lmbda)
                print("Cost on training data: {}".format(cost))
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                print("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                print("Accuracy on evaluation data: {} / {}".format(accuracy, n_data))

        if evaluation_data:
            accuracy = self.accuracy(evaluation_data)
            print("accuracy on evaluation data: {} / {}".format(accuracy, n_data))

        show_image()

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

    def forward_compute(self, mini_batch):
        a = mini_batch
        activations = [a]
        zs = []
        for W, B in zip(self.weights, self.biases):
            z = np.dot(W, a) + B
            zs.append(z)
            a = sigmoid(z)
            activations.append(a)
        return activations, zs

    def total_cost_on_batch(self, mini_batch, labels, lmbda):
        cost = 0.0
        a = self.feedforward(mini_batch)
        for y_hat,y in zip(a, labels):
            cost += self.cost.cost(y_hat, y) / self.batch_size
            # L2正则化 加上的一项
            cost += 0.5 * (lmbda / self.batch_size) * sum(np.linalg.norm(w) ** 2 for w in self.weights)
        return cost

    def total_cost(self, data, lmbda, convert=False):
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectorized_result(y)
            cost += self.cost.cost(a, y) / len(data)
            # L2正则化 加上的一项
            cost += 0.5 * (lmbda / len(data)) * sum(np.linalg.norm(w) ** 2 for w in self.weights)
        return cost


def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))
