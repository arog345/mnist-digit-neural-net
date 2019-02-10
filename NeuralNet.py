import numpy as np
from pathlib import Path
from collections.abc import Iterable

from MNISTDataLoader import MNISTDataLoader

TRAINING_IMAGES = Path("data/train-images-idx3-ubyte.gz")
TRAINING_LABELS = Path("data/train-labels-idx1-ubyte.gz")
TEST_IMAGES = Path("data/t10k-images-idx3-ubyte.gz")
TEST_LABELS = Path("data/t10k-labels-idx1-ubyte.gz")
MNIST_PATHS = [TRAINING_IMAGES, TRAINING_LABELS, TEST_IMAGES, TEST_LABELS]


def main():
    dataLoader = MNISTDataLoader(MNIST_PATHS)

    x, y, label = dataLoader.trainingData[0]

    print("Example of loaded data:")
    printWithLabel("x.T", x.T)
    printWithLabel("yVector", y)
    printWithLabel("y", label)

    layerSizes = [dataLoader.inputSize, dataLoader.inputSize, dataLoader.outputSize]
    learningRate = 0.1
    numIterations = 1000

    np.set_printoptions(threshold=50)

    nn = NeuralNet(layerSizes)
    # For now, pass in a slice of it so that it runs faster.
    nn.train(numIterations, learningRate, dataLoader.trainingData[0:50])


class NeuralNet:
    def __init__(self, layerSizes):
        if len(layerSizes) < 2:
            raise Exception("Must specify at least the input and output layer sizes")

        self.depth = len(layerSizes)

        # No bias for the input layer
        self.__biasMatrices = [np.random.randn(rows, 1) for rows in layerSizes[1:]]
        self.__thetaMatrices = [
            np.random.randn(rows, cols)
            for (rows, cols) in zip(layerSizes[1:], layerSizes[:-1])
        ]

    # Feed-forward algo for NN
    def feedForward(self, x, returnHiddenLayerValues=False):
        a = x

        # The z and a values for the hidden and ouput layers
        zValues = []
        aValues = []

        # Run the input through the network
        for theta, b in zip(self.__thetaMatrices, self.__biasMatrices):
            z = np.matmul(theta, a) + b
            a = sigmoid(z)

            if returnHiddenLayerValues:
                zValues.append(z)
                aValues.append(a)

        if returnHiddenLayerValues:
            return (a, zValues, aValues)
        else:
            return a

    def train(self, numIterations, learningRate, trainingData):
        """
        Trains the NN using the given training data. 
        The training data array elements should be tuples of the form (inputVector, outputVector, label).
        """

        a, zValues, aValues = self.feedForward(trainingData[0][0], True)

        print("NeuralNet.feedForward return values:")
        printWithLabel("a", a)
        printWithLabel("zValues", zValues)
        printWithLabel("aValues", aValues)


# Elementwise sigmoid function
def sigmoid(vector):
    return 1 / (np.exp(vector) + 1)


# Elementwise sigmoid derivative function
def sigmoidDeriv(vector):
    sig = sigmoid(vector)
    return sig * (1 - sig)


def printWithLabel(label, val):
    print(label)
    printList(val)
    print()


def printList(l):
    if isinstance(l, Iterable) and not isinstance(l, np.ndarray):
        for a in l:
            print(a)
    else:
        print(l)


if __name__ == "__main__":
    main()
