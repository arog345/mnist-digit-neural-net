import matplotlib.pyplot as plt
import numpy as np


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
        self.weights = [[]*hiddenLayers[:,0]] #list of lists to store all the weights
        for i in range(self.hiddenLayers.shape[0]): #loop through all layers and assign weights
            self.weights[i] = self.getWeights()
        
    
    #function to define weights
    def getWeights(self):
        return 2*np.random.rand(self.input.shape[1],1) - 1
    
    #activation function    
    def sigmoid(self,x,deriv=False):
        if deriv:
            return x*(1-x)
        return 1/(1+np.exp(-x))
    
    #calculate prediction - need to include layers, missing for loop
    def predict(self,currentWeights):
        expectedOutput = np.dot(self.input,currentWeights)
        return self.sigmoid(expectedOutput)
    
    def train(self,iterations):
        for i in range(iterations):
            for j in range(self.hiddenLayers.shape[0]):
                currentOutput = self.predict(self.weights[j])
                error = self.output - currentOutput
                #this should be broken up into a different loop
                backProp = np.dot(self.input.T,error*self.sigmoid(currentOutput,deriv=True))
                self.weights[j] += backProp
                
    def evaluate(self,input):
        output = np.dot(input,self.weights)
        return self.sigmoid(output)
    
if __name__ == "__main__":
    #inputFig = getImages()
    
    #sample training set, output is equal to first element of the inputs ith row
    x = np.array([[0,0,0],
              [0,1,0],
              [0,0,1],
              [1,1,1],
              [1,0,0]])
    y = np.array([[0,0,0,1,1]]).T
    
    #specify numbers of layers
    layers = 1
    layerID = [x for x in range(layers)] #provide a layer number
    numberNodes = np.array([y.shape[1]]) #how many nodes are in a layer, last layer is output
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