import torch
import torch.nn.functional as f


class NeuralNetwork:
    def __init__(self):
        self.w1 = torch.randn(256, 128, requires_grad=True)
        self.b1 = torch.randn(1, 128, requires_grad=True)

        self.w2 = torch.randn(128, 64, requires_grad=True)
        self.b2 = torch.randn(1, 64, requires_grad=True)

        self.w3 = torch.randn(64, 4, requires_grad=True)
        self.b3 = torch.randn(1, 4, requires_grad=True)

    def forward(self, x):
        x = f.softmax((x @ self.w1) + self.b1, dim=1)
        x = f.tanh((x @ self.w2) + self.b2)
        x = (x @ self.w3) + self.b3  # IDK but using third activation function reduces accuracy
        return x

    def study(self, x, target, learning_rate=0.001, num_epochs=1000):
        for epoch in range(num_epochs):
            output = self.forward(x)
            loss = f.mse_loss(output, target)

            self.w1.grad = None
            self.b1.grad = None
            self.w2.grad = None
            self.b2.grad = None
            self.w3.grad = None
            self.b3.grad = None

            loss.backward()

            with torch.no_grad():
                self.w1 -= learning_rate * self.w1.grad
                self.b1 -= learning_rate * self.b1.grad
                self.w2 -= learning_rate * self.w2.grad
                self.b2 -= learning_rate * self.b2.grad
                self.w3 -= learning_rate * self.w3.grad
                self.b3 -= learning_rate * self.b3.grad

            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item()}')


def main():
    model = NeuralNetwork()
    inputs = torch.randn(1, 256)
    targets = torch.randn(1, 4)
    model.study(inputs, targets)


if __name__ == "__main__":
    main()
