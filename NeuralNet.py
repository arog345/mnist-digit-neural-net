import numpy as np


class NeuralNet:
    def __init__(self, layerSizes):
        if len(layerSizes) < 2:
            raise Exception("Must specify at least the input and output layer sizes")

        self.depth = len(layerSizes) - 1

        self.__thetaMatrices = []
        for i in range(self.depth):
            rows = layerSizes[i + 1]
            cols = layerSizes[i]
            theta = np.random.rand(rows, cols)

            # Hidden layers have a constant entry added to them (for bias?)
            if not self.__isLastLayerIndex(i):
                theta = np.column_stack((theta, np.ones((rows, 1))))

            self.__thetaMatrices.append(theta)

    def predict(self, x):
        h = self.feedForward(x)
        return max([(x, i) for i, x in enumerate(h)])

    # Feed-forward algo for NN
    def feedForward(self, x):
        # TODO: maybe have a flag to return all the intermediate values to for the back prop algo to use?
        a = np.copy(x)

        for i, theta in enumerate(self.__thetaMatrices):
            if not self.__isLastLayerIndex(i):
                a = np.append(a, np.ones([a.shape[0], 1]), 1)

            z = np.matmul(a, theta.T)
            a = self.__sigmoid(z)

        return a

    def train(self, numIterations, trainingData, learningRate):
        """
        Trains the NN using the given training data. 
        The training data array elements should be tuples of the form (input, label).
        """
        # TODO: everything
        inputData = np.row_stack(tuple(i for i, _ in trainingData))
        labelData = np.row_stack(tuple(l for _, l in trainingData))

        print(inputData)
        outputs = self.feedForward(inputData)
        print(outputs)
        print(labelData)
        pass

    # Elementwise sigmoid function
    def __sigmoid(self, vector):
        return 1 / (np.exp(vector) + 1)

    def __isLastLayerIndex(self, i):
        return i + 1 == self.depth


if __name__ == "__main__":
    layerSizes = [4, 3, 4, 2]

    nn = NeuralNet(layerSizes)

    nn.train(
        1000,
        [
            (np.array([1, 1, 0, 0], dtype="float"), [0, 1]),
            (np.array([1, 0, 0, 0], dtype="float"), [0, 1]),
            (np.array([0, 1, 0, 0], dtype="float"), [0, 1]),
            (np.array([0, 0, 1, 1], dtype="float"), [1, 0]),
            (np.array([0, 0, 1, 0], dtype="float"), [1, 0]),
            (np.array([0, 0, 0, 1], dtype="float"), [1, 0]),
        ],
    )
