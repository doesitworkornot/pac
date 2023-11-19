import torchvision
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def main():
    # Downloading the MNIST dataset
    train_dataset = torchvision.datasets.MNIST(
        root="./MNIST/train", train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True)

    test_dataset = torchvision.datasets.MNIST(
        root="./MNIST/test", train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True)

    # Shaping into most inconvenient form
    train = shape_data(train_dataset)
    test = shape_data(test_dataset)
    train = list(train)
    test = list(test)

    # Making average form for each digit from 0 to 9
    avg = [average_digit(train, x) for x in range(10)]

    # Calculating dot product of test image and average one
    # And choosing the biggest one, making at index of that 1 else 0
    predicted = predict(test, avg)
    # Or you could just use sigmoid with bias but not sure if this help
    predicted_ans = [np.array([1 if s == max(x) else 0 for s in x]) for x in predicted]

    # Accuracy check
    test_ans = [x[1] for x in test]
    acc(predicted_ans, test_ans, 'Some convolution')

    # Some visualization
    x = np.array([np.reshape(x[0][0].numpy(), (784,)) for x in train_dataset])
    y = np.array([y[1] for y in train_dataset])
    x_subset = x[0:10000]
    y_subset = y[0:10000]
    tsne_plot(x_subset, y_subset, 'Before')

    test_ans = np.array([np.argmax(x) for x in test_ans])
    predicted = np.array(predicted)
    n_samples, nx, ny, nz = predicted.shape
    predicted = predicted.reshape(n_samples, nx)
    tsne_plot(predicted, test_ans, 'After')


def encode_label(j):
    # 5 -> [[0], [0], [0], [0], [0], [1], [0], [0], [0], [0]]
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def shape_data(data):
    features = [np.reshape(x[0][0].numpy(), (784, 1)) for x in data]
    labels = [encode_label(y[1]) for y in data]
    return zip(features, labels)


def average_digit(data, digit):
    filtered_data = [x[0] for x in data if np.argmax(x[1]) == digit]
    filtered_array = np.asarray(filtered_data)
    return np.average(filtered_array, axis=0)


def predict(test, avg):
    test = [x[0] for x in test]
    ans = []
    for x in test:
        x = np.transpose(x)
        ans.append([np.dot(x, av) for av in avg])

    return ans


def acc(pred, target, alg):
    accuracy = np.sum([np.dot(target[i].T, pred[i]) for i in range(len(pred))])/len(pred)
    print(f'{alg} accuracy is {accuracy}')


def tsne_plot(x_subset, y_subset, name):
    tsne = TSNE(random_state=42, n_components=2, verbose=0, perplexity=5, n_iter=300).fit_transform(x_subset)
    plt.scatter(tsne[:, 0], tsne[:, 1], s=5, c=y_subset, cmap='Spectral')
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries=np.arange(11) - 0.5).set_ticks(np.arange(10))
    plt.title(name, fontsize=20)
    plt.show()


if __name__ == "__main__":
    main()
