# jpeg project
import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import cv2
 

debug = False
debugSampling = True
RGB_YCBCR=np.array([[0.299,0.587,0.114],[-0.168736, -0.331264, 0.5],[0.5, -0.418688, -0.081312]])
YCBCR_RGB=np.linalg.inv(RGB_YCBCR)


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
    
    def showColorMap(self, data, colorMap, title=""):
        plt.figure()
        plt.axis('off')
        plt.title(title)
        plt.imshow(data, colorMap)
        plt.show()

    def splitChannels(self):
        self.R = self.data[:, :, 0]
        self.G = self.data[:, :, 1]
        self.B = self.data[:, :, 2]

    def splitChannelsPadding(self):
        self.R_P = self.dataPadding[:, :, 0]
        self.G_P = self.dataPadding[:, :, 1]
        self.B_P = self.dataPadding[:, :, 2]

    def merge_channels(self):
        return  np.dstack((self.R,self.G,self.B))

    def showChannels(self):
        red = self.colormap('myRed', (1,0,0))
        green = self.colormap('myGreen', (0,1,0))
        blue = self.colormap('myBlue', (0,0,1))

        self.splitChannels()

        self.showColorMap(self.R, red)
        self.showColorMap(self.G, green)
        self.showColorMap(self.B, blue)


    def padding(self):
        lines = np.shape(self.data)[0]
        cols = np.shape(self.data)[1]
        self.dataPadding = self.data

        nl, nc = 0, 0 
        if lines%32:
            nl = 32 - (lines % 32)
        
        if cols%32:
            nc = 32 - (cols % 32)
            
        # concat lines
        ll = self.dataPadding[-1,:][np.newaxis,:]
        rep = ll.repeat(nl, axis=0)
        self.dataPadding = np.vstack((self.dataPadding, rep))

        # concat cols
        cc = self.dataPadding[:, -1][:, np.newaxis]
        rep = cc.repeat(nc, axis=1)
        self.dataPadding = np.hstack((self.dataPadding, rep))

        if debug:
            self.showImage(self.dataPadding)


    def remove_padding(self):
        lines = np.shape(self.data)[0]
        cols = np.shape(self.data)[1]
        self.dataPadding = self.dataPadding[:lines,:cols,:]

        if debug:
            self.showImage(self.dataPadding)


    def colormap(self, name, color=(1,1,1)):
        return clr.LinearSegmentedColormap.from_list(name, [(0,0,0), color], 256)
    

    def rgbToYCbCr(self):
        self.Y  = RGB_YCBCR[0][0] * self.R_P + RGB_YCBCR[0][1] * self.G_P + RGB_YCBCR[0][2] * self.B_P
        self.CB = RGB_YCBCR[1][0] * self.R_P + RGB_YCBCR[1][1] * self.G_P + RGB_YCBCR[1][2] * self.B_P + 128
        self.CR = RGB_YCBCR[2][0] * self.R_P + RGB_YCBCR[2][1] * self.G_P + RGB_YCBCR[2][2] * self.B_P + 128

        #self.CB[self.CB > 255] = 255
        #self.CB[self.CB < 0] = 0

        gray  = self.colormap('myGray', (1,1,1))

        if debug:
            self.showColorMap(self.Y, gray)
            self.showColorMap(self.CB, gray)
            self.showColorMap(self.CR, gray)

        
        

    def YCbCrTorgb(self):
        self.R_ycbcr = (YCBCR_RGB[0][0] * self.Y) + (YCBCR_RGB[0][1] * (self.CB -128))  + (YCBCR_RGB[0][2] * (self.CR -128))
        self.G_ycbcr = (YCBCR_RGB[1][0] * self.Y) + (YCBCR_RGB[1][1] * (self.CB -128))  + (YCBCR_RGB[1][2] * (self.CR -128))
        self.B_ycbcr = (YCBCR_RGB[2][0] * self.Y) + (YCBCR_RGB[2][1] * (self.CB -128))  + (YCBCR_RGB[2][2] * (self.CR -128))

        self.R_ycbcr = np.round(self.R_ycbcr).astype(np.uint8)
        self.G_ycbcr = np.round(self.G_ycbcr).astype(np.uint8)
        self.B_ycbcr = np.round(self.B_ycbcr).astype(np.uint8)

        self.R_ycbcr[self.R_ycbcr>255] = 255
        self.R_ycbcr[self.R_ycbcr<0] = 0
        
        self.G_ycbcr[self.G_ycbcr>255] = 255
        self.G_ycbcr[self.G_ycbcr<0] = 0

        self.B_ycbcr[self.B_ycbcr>255] = 255
        self.B_ycbcr[self.B_ycbcr<0] = 0
        
        if debug:
            self.showImage(self.R_ycbcr)
            self.showImage(self.G_ycbcr)
            self.showImage(self.B_ycbcr)

        self.dataPadding = np.zeros(np.shape(self.dataPadding), np.uint8)
        self.dataPadding[:,:,0] = self.R_ycbcr
        self.dataPadding[:,:,1] = self.G_ycbcr
        self.dataPadding[:,:,2] = self.B_ycbcr

        self.remove_padding()
        if debug:
            self.showImage(self.dataPadding)


    def sampling(self, factor="4:1:0"):
        gray  = self.colormap('MyGray', (1,1,1))
        shape = np.shape(self.CB)
        if factor[-1] == '0':
            widh   = int(factor[2])/4
            cbDim  = (int(shape[1]*widh), int(shape[0]*widh))
            crDim  = (int(shape[1]*widh), int(shape[0]*widh))
        else:
            widhCB = int(factor[2])/4
            widhCR = int(factor[-1])/4
            cbDim  = (int(shape[1]*widhCB), int(shape[0]))
            crDim  = (int(shape[1]*widhCR), int(shape[0]))

        self.CB = cv2.resize(self.CB, cbDim, interpolation=cv2.INTER_CUBIC)
        self.CR = cv2.resize(self.CR, crDim, interpolation=cv2.INTER_CUBIC)

        if debugSampling:
            self.showColorMap(self.CB, gray, "Sampling CB")
            self.showColorMap(self.CR, gray, "Sampling CR")

    
    def upSampling(self, factor="4:1:0"):
        gray  = self.colormap('myGray', (1,1,1))
        shape = np.shape(self.CB)

        if factor[-1] == '0':
            widh   = int(4 / int(factor[2]))
            cbDim  = (int(shape[1]*widh), int(shape[0]*widh))
            crDim  = (int(shape[1]*widh), int(shape[0]*widh))
        else:
            widhCB = int(4 / int(factor[2]))
            widhCR = int(4 / int(factor[-1]))
            cbDim  = (int(shape[1])*widhCB, int(shape[0]))
            crDim  = (int(shape[1])*widhCR, int(shape[0]))
            
        self.CB = cv2.resize(self.CB, cbDim, interpolation=cv2.INTER_CUBIC)
        self.CR = cv2.resize(self.CR, crDim, interpolation=cv2.INTER_CUBIC)

        if debugSampling:
            print("CB shape: ", np.shape(self.CB))
            print("Padding shape: ", np.shape(self.dataPadding))
            self.showColorMap(self.dataPadding, gray, "dataPaddging")
            self.showColorMap(self.CB, gray, "UnSampling CB")
            self.showColorMap(self.CR, gray, "UnSampling CR")


    def encoder(self):
        self.readImage()
        # self.showImage()
        # self.showChannels()
        self.padding()

        self.splitChannelsPadding()
        self.rgbToYCbCr()
        self.sampling()


    def decoder(self):
        self.upSampling()
        self.YCbCrTorgb()


if __name__ == "__main__":
    
    a = jpeg('../../Assets/barn_mountains/barn_mountains.bmp')
    a.encoder()
    a.decoder()
