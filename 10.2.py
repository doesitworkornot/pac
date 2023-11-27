import numpy as np


class Neuron:
    def __init__(self, act_func, features, learning_rate, is_inner):
        self.act_func = act_func
        self.features = features
        self.w = np.array([np.random.random() for i in range(features)])
        self.learning_rate = learning_rate
        self.out = 0
        self.net = 0
        self.inner = is_inner
        self.delta = 0.001
        self.x = 0

    def forward(self, x, bias):
        net = np.dot(x, np.transpose(self.w))
        self.net = net
        self.out = self.act_func(self.net)
        self.x = x
        return self.out

    def backward(self, y, target):
        der = ((self.act_func(self.net + self.delta) - self.out) / self.delta)
        if self.inner:
            delta = der * self.net
        else:
            loss_der = ((loss_function(target, self.out + self.delta) - loss_function(target, self.out)) /
                       (self.delta * len(target)))
            delta = loss_der * der
        new_weight = np.dot(delta, self.x) * self.learning_rate
        self.w += new_weight
        return delta


class Network:
    def __init__(self, inner_layers, outer_layer, lr):
        self.outer = [Neuron(sigmoid, 2, lr, 0) for j in range(outer_layer)]
        self.inner = [[Neuron(relu, 2, lr, 1) for j in range(inner_layers[0])] for i in range(inner_layers[1])]
        self.bias = [np.random.random() for i in range(inner_layers[1] + 1)]
        self.learning_rate = lr

    def forward(self, x):
        bias_cnt = 0
        for layer in self.inner:
            new_vec = []
            for neuron in layer:
                out = neuron.forward(x, self.bias[bias_cnt])
                new_vec.append(out)
            bias_cnt += 1
            x = new_vec
        output = []
        x = np.transpose(x)
        for neuron in self.outer:
            out = neuron.forward(x, self.bias[-1])
            output.append(out)
        output = np.mean(output, axis=0)
        if hasattr(output, '__iter__'):
            output = [1 if x > 0.5 else 0 for x in output]
        else:
            output = 1 if output > 0.5 else 0
        return output

    def backward(self, target, y):
        outer_sum = 0
        for neuron in self.outer:
            delta = neuron.backward(y, target)
            outer_sum += delta
        self.bias[-1] += np.sum(outer_sum) * self.learning_rate
        bias_ind = -2
        for layer in reversed(self.inner):
            inner_sum = 0
            for neuron in layer:
                delta = neuron.backward(outer_sum, target)
                inner_sum += delta
            outer_sum = inner_sum
            bias_delta = np.sum(outer_sum) * self.learning_rate
            self.bias[bias_ind] += bias_delta
            bias_ind -= 1


def main():
    net = Network((2, 1), 1, 0.1)
    input_xor = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.], [0., 0.], [0., 1.]])
    test_xor = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.], [0., 0.]])
    output_xor = np.array([0, 1, 1, 0, 0, 1])
    test_output = np.array([0, 1, 1, 0, 0])
    iterations = 2000
    for i in range(iterations):
        y = net.forward(input_xor)
        print(loss_function(y, output_xor), 'loss')
        print(y, 'get')
        net.backward(output_xor, y)
    print(net.forward(test_xor))
    print(test_output)


def loss_function(t, out):
    return np.array((t - out)**2)


def sigmoid(x):
    if hasattr(x, '__iter__'):
        return np.array([1/(1 + np.exp(-y)) for y in x])
    else:
        return 1/(1 + np.exp(-x))


def relu(x):
    if hasattr(x, '__iter__'):
        return np.array([max(0, y) for y in x])
    else:
        return max(0, x)


if __name__ == "__main__":
    main()
