import matplotlib.pyplot as plt

#read the image in
im3d=plt.imread('imgBW.png')

#convert the image to a 2d image with averaged values, assumes image will be
#black and white, otherwise this step is not necessary
im2d = (im3d[:,:,0] + im3d[:,:,1] + im3d[:,:,2])/3

