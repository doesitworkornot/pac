import numpy as np


class Neuron:
    def __init__(self, act_func, features, learning_rate, is_inner):
        self.act_func = act_func
        self.w = np.array([np.random.random() for i in range(features)])
        self.bias = np.random.random()
        self.learning_rate = learning_rate
        self.out = 0
        self.net = 0
        self.inner = is_inner
        self.delta = 0.001
        self.x = 0

    def forward(self, x):
        net = np.dot(x, np.transpose(self.w))
        self.net = net + self.bias
        self.out = self.act_func(self.net)
        self.x = x
        return self.out

    def backward(self, prev, target):
        der = ((self.act_func(self.net + self.delta) - self.out) / self.delta)
        if self.inner:
            delta = prev * der
        else:
            loss_der = ((loss_function(target, self.out + self.delta) - loss_function(target, self.out)) /
                        (self.delta))
            delta = loss_der * der
        new_weight = np.dot(delta, self.x) * self.learning_rate
        delta_return = delta * self.w
        self.w -= new_weight
        self.bias -= delta * self.learning_rate
        return delta_return


class Network:
    def __init__(self, inner_layers, outer_layer, lr):
        self.outer = [Neuron(sigmoid, 2, lr, 0) for j in range(outer_layer)]
        self.inner = [[Neuron(relu, 2, lr, 1) for j in range(inner_layers[0])] for i in range(inner_layers[1])]
        self.learning_rate = lr

    def forward(self, x):
        for layer in self.inner:
            new_vec = []
            for neuron in layer:
                out = neuron.forward(x)
                new_vec.append(out)
            x = new_vec
        output = []
        x = np.transpose(x)
        for neuron in self.outer:
            out = neuron.forward(x)
            output.append(out)
        return output

    def backward(self, target, y):
        outer_sum = [0, 0]
        for neuron in self.outer:
            delta = neuron.backward(y, target)
            outer_sum += delta
        for layer in reversed(self.inner):
            ind = 0
            inner_sum = [0, 0]
            for neuron in layer:
                delta = neuron.backward(outer_sum[ind], target)
                ind += 1
                inner_sum += delta
            outer_sum = inner_sum


def main():
    net = Network((2, 1), 1, 0.1)
    input_xor = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.], [0., 0.], [0., 1.]])
    test_xor = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.], [0., 0.]])
    output_xor = np.array([0, 1, 1, 0, 0, 1])
    test_output = np.array([0, 1, 1, 0, 0])
    iterations = 20000
    for i in range(iterations):
        for j in range(len(input_xor)):
            y = net.forward(input_xor[j])
            # print(loss_function(y, output_xor[j]), 'loss')
            # print(y, 'get')
            net.backward(output_xor[j], y)

    cnt = 0
    for i in range(len(test_xor)):
        y = net.forward(input_xor[i])[0]
        y = 1 if y > 0.5 else 0
        if y == test_output[i]:
            cnt += 1
    print(f'Current accuracy is {cnt/len(test_xor)}')


def loss_function(t, out):
    return np.array((t - out)**2)


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def relu(x):
    return max(0, x)


if __name__ == "__main__":
    main()
