import gzip
import numpy as np


class MNISTDataLoader:
    """Loads the MNIST dataset. Get the files from 'http://yann.lecun.com/exdb/mnist/'"""

    def __init__(self, paths):
        """Creates a class that handles obtaining and loading the MNIST data set.
        
        paths: List of strings/Path objects in the order of [training images, trianing labels, test images, test labels] 
               where the data set files can be found.
        """

        # Need to verify this more, but this works for now
        if len(paths) != 4:
            raise Exception("Should be 4 path objects")

        self.trainingData = self.__readImageSet(paths[0], paths[1])
        self.testData = self.__readImageSet(paths[2], paths[3])

        self.inputSize = 0
        self.outputSize = 0

        if len(self.trainingData):
            x, y, _ = self.trainingData[0]
            self.inputSize = x.size
            self.outputSize = y.size

    def __readImageSet(self, imagesPath, labelsPath):
        images = self.__readImageFile(imagesPath)
        labels = self.__readLabelsFile(labelsPath)

        labeledData = [(image, labelVector, label) for image, (labelVector, label) in zip(images, labels)]

        return labeledData

    def __readImageFile(self, imagesPath):
        """
        Reads the image file. The format is:

        32-bit uint: magic number (2051)
        32-bit uint: number of images
        32-bit uint: number of pixels (uint) per row (28)
        32-bit uint: number of pixels (uint) per column (28)
               uint: pixel
        ...
               uint: pixel
        
        Returns a list of images as vectors
        """
        images = []

        with gzip.open(imagesPath, "rb") as f:
            magicNumber = self.__readInt(f, 4)
            numImages = self.__readInt(f, 4)
            numRowsOfPixels = self.__readInt(f, 4)
            numColsOfPixels = self.__readInt(f, 4)

            self.__checkMagicNumber("image", 2051, magicNumber)

            pixelsPerImage = numRowsOfPixels * numColsOfPixels

            for _ in range(numImages):
                image = f.read(pixelsPerImage)
                image = np.fromstring(image, dtype="uint8").reshape((len(image), 1))
                images.append(image)

        return images

    def __readLabelsFile(self, labelsPath):
        """
        Reads the label file. The format is:

        32-bit uint: magic number (2049)
        32-bit uint: number of labels
               uint: label
        ...
               uint: label

        Returns a list of tuples where each tuple is of the form: (label, label as vector). 
        The label vector is in R^10 where every element is 0 except v_label = 1.
        """
        labels = []

        with gzip.open(labelsPath, "rb") as f:
            magicNumber = self.__readInt(f, 4)
            numItems = self.__readInt(f, 4)

            self.__checkMagicNumber("label", 2049, magicNumber)

            for _ in range(numItems):
                label = self.__readInt(f, 1)

                labelVector = np.zeros((10, 1), dtype="uint8")
                labelVector[label] = 1

                labels.append((labelVector, label))

        return labels

    def __readInt(self, f, n):
        """Reads 'n' integers stored in big-endian format (most significant byte first)."""
        return int.from_bytes(f.read(n), byteorder="big")

    def __checkMagicNumber(self, filename, expectedMagicNumber, actualMagicNumber):
        if actualMagicNumber != expectedMagicNumber:
            warning = f"WARNING: The magic number for the {filename} did not match. Exected={expectedMagicNumber}, Actual={actualMagicNumber}."
            print(warning)
