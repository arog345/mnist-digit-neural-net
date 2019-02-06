import gzip
import numpy as np


class MNISTDataLoader:
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

    def __readImageSet(self, imagesPath, labelsPath):
        images = self.__readImageFile(imagesPath)
        labels = self.__readLabelsFile(labelsPath)

        labeledData = list(zip(images, labels))

        return labeledData

    def __readImageFile(self, imagesPath):
        images = []

        with gzip.open(imagesPath, "rb") as f:
            magicNumber = self.__readInt(f, 4)
            numImages = self.__readInt(f, 4)
            numRowsOfPixels = self.__readInt(f, 4)
            numColsOfPixels = self.__readInt(f, 4)

            pixelsPerImage = numRowsOfPixels * numColsOfPixels

            for _ in range(numImages):
                image = f.read(pixelsPerImage)
                image = np.fromstring(image, dtype='uint8')
                image = np.resize(image, (numRowsOfPixels, numColsOfPixels))
                images.append(image)

        return images

    def __readLabelsFile(self, labelsPath):
        labels = []

        with gzip.open(labelsPath, "rb") as f:
            magicNumber = self.__readInt(f, 4)
            numItems = self.__readInt(f, 4)

            for _ in range(numItems):
                labels.append(self.__readInt(f, 1))

        return labels

    def __readInt(self, f, n):
        return int.from_bytes(f.read(n), byteorder='big')
