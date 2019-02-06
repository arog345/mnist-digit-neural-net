import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image

import MNISTDataLoader

TRAINING_IMAGES = Path('data/train-images-idx3-ubyte.gz')
TRAINING_LABELS = Path('data/train-labels-idx1-ubyte.gz')
TEST_IMAGES = Path('data/t10k-images-idx3-ubyte.gz')
TEST_LABELS = Path('data/t10k-labels-idx1-ubyte.gz')

def getImages():
    #read the image in
    im3d=plt.imread('imgBW.png')
    
    #convert the image to a 2d image with averaged values, assumes image will be
    #black and white, otherwise this step is not necessary
    im2d = (im3d[:,:,0] + im3d[:,:,1] + im3d[:,:,2])/3
    return im2d

class neuralNet: #assuming no hidden layer
    def __init__(self,inputs,outputs,hiddenLayers):
        np.random.seed(1)
        self.input = inputs.astype(float)
        self.output = outputs
        self.hiddenLayers = hiddenLayers
        self.weights = [] #list of lists to store all the weights
        row = self.input.shape[1]
        for i in range(self.hiddenLayers.shape[0]): #loop through all layers and assign weights
            col = self.hiddenLayers[i,1]
            self.weights.append(self.getWeights(row,col))
            row = col
        
    
    #function to define weights
    def getWeights(self,x,y):
        return 2*np.random.rand(x,y) - 1
    
    #activation function    
    def sigmoid(self,x,deriv=False):
        if deriv:
            return x*(1-x)
        return 1/(1+np.exp(-x))
    
    #calculate prediction - need to include layers, missing for loop
    def linearPredict(self,z,currentWeights):
        z = np.dot(z,currentWeights)
        return z
    
    def activation(self,z):
        sigma = self.sigmoid(z)
        return sigma
        
    def forwardProp(self,currentInput,currentOutput):
        z = []
        sigma = [currentInput.reshape((1,currentInput.shape[0]))]
        for j in range(self.hiddenLayers.shape[0]):
            z.append(self.linearPredict(sigma[j],self.weights[j]))
            sigma.append(np.array(self.activation(z[j])))
        #derivative of the error function
        error = currentOutput.reshape(1,currentOutput.shape[0]) - sigma[-1] #last sigma should be the last output
        return (z,sigma,error)
    
    def backwardProp(self,z,sigma,error):
        #first iteration - coming from back to front
        auxWeight = self.weights[-1]
        endDerivative = np.dot(error,self.sigmoid(sigma[-1],True))
        backProp = np.dot(endDerivative,sigma[-2])
        self.weights[-1] += backProp.T #last weight
        memoization = endDerivative
        #looping through, need to start 1 index later since initialized
        for i in range(self.hiddenLayers.shape[0]-1-1,0,-1):
            intermediateCalc = auxWeight*self.sigmoid(sigma[i+1],True)
            memoization= memoization * intermediateCalc
            backProp = np.dot(sigma[i-1], 
                              backProp*memoization)
            auxWeight = self.weights[i]
            self.weights[i] += backProp
            
                
    def train(self,iterations):
        for i in range(iterations):
            #two types of ways to train (i) batch or (ii) iterative?
            #choosing iterative because I think it has some benefits
            for j in range(self.input.shape[0]):
                (z,sigma,error) = self.forwardProp(self.input[j],self.output[j])
                self.backwardProp(z,sigma,error)
                
    def evaluate(self,userIn):
        #because sigmoid, output is going to be between 0 and 1, 
        #must scale outputs accordingly
        sigma = userIn
        for j in range(self.hiddenLayers.shape[0]):
            z = self.linearPredict(sigma,self.weights[j])
            sigma = self.activation(z)
        return sigma
    
if __name__ == "__main__":
    #inputFig = getImages()

    mnistData = MNISTDataLoader.MNISTDataLoader([TRAINING_IMAGES, TRAINING_LABELS, TEST_IMAGES, TEST_LABELS])

    print(mnistData.testData[0])

    image = Image.fromarray(mnistData.testData[0][0])
    image.show()

    #sample training set, output is equal to first element of the inputs ith row
    x = np.array([[0,0,0],
              [0,1,0],
              [0,0,1],
              [1,1,1],
              [1,0,0]])
    y = np.array([[0,0,0,1,1]]).T
    
    #specify numbers of layers
    layers = 3
    layerID = [x for x in range(layers)] #provide a layer number
    numberNodes = np.array([4,3,y.shape[1]]) #how many nodes are in a layer, last layer is output
    hiddenLayers = np.hstack([(layerID,numberNodes)]).T #bring together - actually not needed since number of rows is number of layers, keeping anyway
    iterations = 10000
    net = neuralNet(x,y,hiddenLayers)
    
    print(net.weights)
    net.train(iterations)
    print("--------")
    print(net.weights)
    testInput = np.array([[1,0,1]])
    print("--------")
    print(net.evaluate(testInput))
    
