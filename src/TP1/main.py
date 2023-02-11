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


    def showImage(self):
        plt.figure()
        plt.imshow(self.data)
        plt.axis('off')
        plt.show()


    def gradient(self, color):
        return clr.LinearSegmentedColormap.from_list('custom', [(0,0,0), (0,1, 0)], 256)

    def encoder(self):
        pass

    def decoder(self):
        pass


if __name__ == "__main__":
    
    a = jpeg('../../Assets/peppers/peppers.bmp')
    a.readImage()
    a.showImage()
    # g = a.gradient("blue")
    # print(type(g))
