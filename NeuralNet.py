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

    # Feed-forward algo for NN
    def predict(self, x):
        a = np.copy(x)

        for i, theta in enumerate(self.__thetaMatrices):
            if not self.__isLastLayerIndex(i):
                a = np.append(a, 1)

            z = np.matmul(theta, a)
            a = self.__sigmoid(z)

        return max([(x, i) for i, x in enumerate(a)])

    def train(self, numIterations, trainingData):
        # TODO: everything
        pass

    # Elementwise sigmoid function
    def __sigmoid(self, vector):
        return 1 / (np.exp(vector) + 1)

    def __isLastLayerIndex(self, i):
        return i + 1 == self.depth


if __name__ == "__main__":
    layerSizes = [4, 3, 4, 2]

    nn = NeuralNet(layerSizes)

    inVector = np.array([[1, 1, 1, 1]], dtype="float")
    outVector = nn.predict(inVector)
    print(outVector)
