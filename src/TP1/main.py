# jpeg project
import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt
import matplotlib.colors as clr
 

debug = False


class jpeg:
    def __init__(self, filename):
        self.filename = filename

    def readImage(self):
        self.data = img.imread(self.filename)

        if debug:
            print(self.data)


    def showImage(self, *args):
        plt.figure()
        if len(args) > 0:
            plt.imshow(args[0])
        else:
            plt.imshow(self.data)
        plt.axis('off')
        plt.show()
    
    def showColorMap(self, data, colorMap):
        plt.figure()
        plt.axis('off')
        plt.imshow(data, colorMap)
        plt.show()

    def splitChannels(self):
        self.R = self.data[:, :, 0]
        self.G = self.data[:, :, 1]
        self.B = self.data[:, :, 2]

    def merge_channels(self):
        return  np.dstack((self.R,self.G,self.B))

    def showChannels(self):
        red = self.gradient('myRed', (1,0,0))
        green = self.gradient('myGreen', (0,1,0))
        blue = self.gradient('myBlue', (0,0,1))

        self.splitChannels()

        self.showColorMap(self.R, red)
        self.showColorMap(self.G, green)
        self.showColorMap(self.B, blue)


    def gradient(self, name, color=(1,1,1)):
        return clr.LinearSegmentedColormap.from_list(name, [(0,0,0), color], 256)

    def encoder(self):
        pass

    def decoder(self):
        pass


if __name__ == "__main__":
    
    a = jpeg('../../Assets/peppers/peppers.bmp')
    a.readImage()
    a.showImage()
    a.showChannels()