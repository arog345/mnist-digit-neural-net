import numpy as np
from pathlib import Path
from collections.abc import Iterable
import random

from MNISTDataLoader import MNISTDataLoader

TRAINING_IMAGES = Path("data/train-images-idx3-ubyte.gz")
TRAINING_LABELS = Path("data/train-labels-idx1-ubyte.gz")
TEST_IMAGES = Path("data/t10k-images-idx3-ubyte.gz")
TEST_LABELS = Path("data/t10k-labels-idx1-ubyte.gz")
MNIST_PATHS = [TRAINING_IMAGES, TRAINING_LABELS, TEST_IMAGES, TEST_LABELS]


def main():
    dataLoader = MNISTDataLoader(MNIST_PATHS)

    layerSizes = [dataLoader.inputSize, 100, dataLoader.outputSize]
    learningRate = 1
    batchSize = 500
    numEpochs = 10
    showDiagnostics = False

    np.set_printoptions(threshold=50)

    nn = NeuralNet(layerSizes)
    nn.train(
        numEpochs, 
        batchSize, 
        learningRate, 
        dataLoader.trainingData, 
        dataLoader.testData,
        showDiagnostics
    )


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

    def predict(self, x):
        a = self.feedForward(x)
        return np.argmax(a)

    def feedForward(self, x, returnHiddenLayerValues=False):
        """Feed forward algo for the NN"""

        a = x

        # The z and a values for the hidden and ouput layers
        zValues = []
        aValues = [a]

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

    def train(self, epochs, batchSize, learningRate, trainingData, testData, showDiagnostics):
        """
        Trains the NN using the given training data. 
        The training data array elements should be tuples of the form (inputVector, outputVector, label).
        """

        for i in range(epochs):
            self.__train_epoch(learningRate, batchSize, trainingData)
            if showDiagnostics:
                self.__printErrorStats(i, testData)
            else:
                print(f'Epoch {i}: Complete')
        self.__printErrorStats('End', testData)

    def __train_epoch(self, learningRate, batchSize, trainingData):
        """Splits the training data into batches of the specified sizes and runs gradient descent on each one."""
        n = len(trainingData)

        random.shuffle(trainingData)
        batches = [
            trainingData[i : min(i + batchSize, n)]
            for i in range(0, n, batchSize)
        ]

        for b in batches:
            self.__train_batch(learningRate, b)

    def __train_batch(self, learningRate, trainingBatch):
        """Obtains the average gradient of all the input/output pairs in the batch."""

        # The gradient for the bias and theta matrices. It will be the sum of the values
        # from each back prop run
        grad_bias = [np.zeros(b.shape) for b in self.__biasMatrices]
        grad_thetas = [np.zeros(t.shape) for t in self.__thetaMatrices]

        m = len(trainingBatch)

        # Perform back prop on each entry in the batch
        for x, y, _ in trainingBatch:
            del_bias, del_thetas = self.__backProp(x, y)

            grad_bias = [gb + db for gb, db in zip(grad_bias, del_bias)]
            grad_thetas = [gt + dt for gt, dt in zip(grad_thetas, del_thetas)]

        # Update the bias and weights based off the current gradient
        self.__biasMatrices = [
            b - (learningRate / m) * gb for b, gb in zip(self.__biasMatrices, grad_bias)
        ]
        self.__thetaMatrices = [
            t - (learningRate / m) * gt
            for t, gt in zip(self.__thetaMatrices, grad_thetas)
        ]

    def __backProp(self, x, y):
        """Finds the derivated of the MSE for the given input/output pair for each bias and theta in the network."""

        # The gradient for the bias and theta matrices for each layer
        del_bias = [np.zeros(b.shape) for b in self.__biasMatrices]
        del_thetas = [np.zeros(t.shape) for t in self.__thetaMatrices]

        a, zValues, aValues = self.feedForward(x, True)

        # Derivatives for the output layer
        delta = (a - y) * sigmoidDeriv(zValues[-1])
        del_bias[-1] = delta
        del_thetas[-1] = np.matmul(delta, aValues[-2].T)

        # Propograte the error back through the network
        for l in range(2, self.depth):
            z = zValues[-l]
            deriv = sigmoidDeriv(z)

            delta = np.matmul(self.__thetaMatrices[-l + 1].T, delta) * deriv
            del_bias[-l] = delta
            del_thetas[-l] = np.matmul(delta, aValues[-l - 1].T)

        return del_bias, del_thetas

    def __printErrorStats(self, epoch, testData):
        """Prints some diagnostic data."""

        numExamples = len(testData)
        testResults = [(self.predict(x), label) for x, _, label in testData]
        numRight = sum(int(y == label) for (y, label) in testResults)

        print(
            f"Epoch {epoch}: {numRight} out of {numExamples} correct. Accuracy: {(100 * numRight) / numExamples:.2f}%"
        )


def sigmoid(vector):
    """Elementwise sigmoid function"""
    return 1.0 / (np.exp(-vector) + 1.0)


def sigmoidDeriv(vector):
    """Elementwise sigmoid derivative function"""
    sig = sigmoid(vector)
    return sig * (1 - sig)


def printWithLabel(label, val):
    """Prints the label and val, each on its own line, followed by a blank line."""
    print(label)
    printList(val)
    print()


def printList(l):
    """
    Workaround for print a list of np.ndarray. Each npdarray in a list is printed 
    as if were printed on its own.
    """
    if isinstance(l, Iterable) and not isinstance(l, np.ndarray):
        for a in l:
            print(a)
    else:
        print(l)


if __name__ == "__main__":
    main()
