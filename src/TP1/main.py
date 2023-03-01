# jpeg project
import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import cv2
from scipy.fftpack import dct, idct


debug = False
debugSampling = False
debugDCT = True
debugDPCM = True




class jpeg:


    RGB_YCBCR=np.array([[0.299,0.587,0.114],[-0.168736, -0.331264, 0.5],[0.5, -0.418688, -0.081312]])
    YCBCR_RGB=np.linalg.inv(RGB_YCBCR)
    
    q_y = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                    [12, 12, 14, 19, 26, 58, 60, 55],
                                    [14, 13, 16, 24, 40, 57, 69, 56],
                                    [14, 17, 22, 29, 51, 87, 80, 62],
                                    [18, 22, 37, 56, 68, 109, 103, 77],
                                    [24, 35, 55, 64, 81, 104, 113, 92],
                                    [49, 64, 78, 87, 103, 121, 120, 101],
                                    [72, 92, 95, 98, 112, 100, 103, 99]])

    q_cbcr = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
          [18, 21, 26, 66, 99, 99, 99, 99],
          [24, 26, 56, 99, 99, 99, 99, 99],
          [47, 66, 99, 99, 99, 99, 99, 99],
          [99, 99, 99, 99, 99, 99, 99, 99],
          [99, 99, 99, 99, 99, 99, 99, 99],
          [99, 99, 99, 99, 99, 99, 99, 99],
          [99, 99, 99, 99, 99, 99, 99, 99]])

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
        self.Y  = self.RGB_YCBCR[0][0] * self.R_P + self.RGB_YCBCR[0][1] * self.G_P + self.RGB_YCBCR[0][2] * self.B_P
        self.CB = self.RGB_YCBCR[1][0] * self.R_P + self.RGB_YCBCR[1][1] * self.G_P + self.RGB_YCBCR[1][2] * self.B_P + 128
        self.CR = self.RGB_YCBCR[2][0] * self.R_P + self.RGB_YCBCR[2][1] * self.G_P + self.RGB_YCBCR[2][2] * self.B_P + 128

        #self.CB[self.CB > 255] = 255
        #self.CB[self.CB < 0] = 0

        gray  = self.colormap('myGray', (1,1,1))

        if debug:
            self.showColorMap(self.Y, gray)
            self.showColorMap(self.CB, gray)
            self.showColorMap(self.CR, gray)

        
        

    def YCbCrTorgb(self):
        self.R_ycbcr = (self.YCBCR_RGB[0][0] * self.Y) + (self.YCBCR_RGB[0][1] * (self.CB -128))  + (self.YCBCR_RGB[0][2] * (self.CR -128))
        self.G_ycbcr = (self.YCBCR_RGB[1][0] * self.Y) + (self.YCBCR_RGB[1][1] * (self.CB -128))  + (self.YCBCR_RGB[1][2] * (self.CR -128))
        self.B_ycbcr = (self.YCBCR_RGB[2][0] * self.Y) + (self.YCBCR_RGB[2][1] * (self.CB -128))  + (self.YCBCR_RGB[2][2] * (self.CR -128))

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


    def sampling(self, factor="4:2:2"):
        gray  = self.colormap('MyGray', (1,1,1))
        shape = np.shape(self.CB)
        if factor[-1] == '0':
            widh   = int(factor[2])/int(factor[0])
            cbDim  = (int(shape[1]*widh), int(shape[0]*widh))
            crDim  = (int(shape[1]*widh), int(shape[0]*widh))
        else:
            widhCB = int(factor[2])/int(factor[0])
            widhCR = int(factor[-1])/int(factor[0])
            cbDim  = (int(shape[1]*widhCB), int(shape[0]))
            crDim  = (int(shape[1]*widhCR), int(shape[0]))

        self.CB = cv2.resize(self.CB, cbDim, interpolation=cv2.INTER_CUBIC)
        self.CR = cv2.resize(self.CR, crDim, interpolation=cv2.INTER_CUBIC)

        if debugSampling:
            print("CB shape: ", np.shape(self.CB))
            print("Padding shape: ", np.shape(self.dataPadding))
            self.showColorMap(self.CB, gray, "Sampling CB")
            self.showColorMap(self.CR, gray, "Sampling CR")

    
    def upSampling(self, factor="4:2:2"):
        gray  = self.colormap('myGray', (1,1,1))
        shape = np.shape(self.CB)

        if factor[-1] == '0':
            widh   = int(int(factor[0]) / int(factor[2]))
            cbDim  = (int(shape[1]*widh), int(shape[0]*widh))
            crDim  = (int(shape[1]*widh), int(shape[0]*widh))
        else:
            widhCB = int(int(factor[0]) / int(factor[2]))
            widhCR = int(int(factor[0]) / int(factor[-1]))
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
   

    def DCT(self, channel):
        # dct(dct(X, norm=”ortho”).T, norm=”ortho”).T
        c_dct = dct(dct(channel, norm='ortho').T, norm='ortho').T

        return c_dct

    
    def IDCT(self, channel):
        # c_idct = idct(idct(channel, norm="ortho").T, norm="ortho").T
        c_idct = idct(idct(channel, axis = 0, norm = 'ortho', type = 2), axis = 1, norm = 'ortho', type = 2)

        return c_idct
    

    def dctBlock(self, blocks=8):
        length = self.Y.shape
        Y_dct = np.zeros(length)

        for i in range(0, length[0], blocks):
            for j in range(0, length[1], blocks):
                slice = self.Y[i:i+blocks, j:j+blocks]
                Y_dct[i:i+blocks, j:j+blocks] = self.DCT(slice)

        length = self.CB.shape
        CB_dct = np.zeros(length)
        CR_dct = np.zeros(length)

        for i in range(0, length[0], blocks):
            for j in range(0, length[1], blocks):
                slice = self.CB[i:i+blocks, j:j+blocks]
                CB_dct[i:i+blocks, j:j+blocks] = self.DCT(slice)
                slice = self.CR[i:i+blocks, j:j+blocks]
                CR_dct[i:i+blocks, j:j+blocks] = self.DCT(slice)

        self.Y  = Y_dct
        self.CB = CB_dct
        self.CR = CR_dct

        if debugDCT:
            gray  = self.colormap('myGray', (1,1,1))

            y_dct = np.log(abs(self.Y)+0.00001)
            self.showColorMap(y_dct, gray, 'Y dct')

            cb_dct = np.log(abs(self.CB)+0.00001)
            self.showColorMap(cb_dct, gray, 'CB dct')

            cr_dct = np.log(abs(self.CR)+0.00001)
            self.showColorMap(cr_dct, gray, 'CR dct')


    def idctBlock(self, blocks=8):
        length = self.Y.shape

        for i in range(0, length[0], blocks):
            for j in range(0, length[1], blocks):
                slice = self.Y[i:i+blocks, j:j+blocks]
                self.Y[i:i+blocks, j:j+blocks] = self.IDCT(slice)

        length = self.CB.shape

        for i in range(0, length[0], blocks):
            for j in range(0, length[1], blocks):
                slice = self.CB[i:i+blocks, j:j+blocks]
                self.CB[i:i+blocks, j:j+blocks] = self.IDCT(slice)

                slice = self.CR[i:i+blocks, j:j+blocks]
                self.CR[i:i+blocks, j:j+blocks] = self.IDCT(slice)

        if debugDCT:
            gray  = self.colormap('myGray', (1,1,1))

            y_dct = np.log(abs(self.Y)+0.00001)
            self.showColorMap(self.Y, gray, 'Y idct')

            cb_dct = np.log(abs(self.CB)+0.00001)
            self.showColorMap(self.CB, gray, 'CB idct')

            cr_dct = np.log(abs(self.CR)+0.00001)
            self.showColorMap(self.CB, gray, 'CR idct')
            

    def quantDCT(self, blocks=8, quality=100):
        self.calcQuality(quality)

        length = self.Y.shape
        for i in range(0, length[0], blocks):
            for j in range(0, length[1], blocks):
                slice = self.Y[i:i+blocks, j:j+blocks]
                # self.Y_dct[i:i+blocks, j:j+blocks] = slice / self.q_y
                self.Y[i:i+blocks, j:j+blocks] = slice / self.qualityQ_Y

        length = self.CB.shape
        for i in range(0, length[0], blocks):
            for j in range(0, length[1], blocks):
                slice = self.CB[i:i+blocks, j:j+blocks]
                # self.CB_dct[i:i+blocks, j:j+blocks] = slice / self.q_cbcr
                self.CB[i:i+blocks, j:j+blocks] = slice / self.qualityQ_CBCR

                slice = self.CR[i:i+blocks, j:j+blocks]
                # self.CR_dct[i:i+blocks, j:j+blocks] = slice / self.q_cbcr
                self.CR[i:i+blocks, j:j+blocks] = slice / self.qualityQ_CBCR

        self.Y= np.round(self.Y)
        self.CB = np.round(self.CB)
        self.CR = np.round(self.CR)

        if debugDCT:
            gray  = self.colormap('myGray', (1,1,1))
            self.showColorMap(np.log(abs(self.Y) + 0.0001), gray, 'Y quantization')
            self.showColorMap(np.log(abs(self.CB) + 0.0001), gray, 'CB quantization')
            self.showColorMap(np.log(abs(self.CR) + 0.0001), gray, 'CR quantization')


    def iQuantDCT(self, blocks=8, quality=100):
        self.calcQuality(quality)

        length = self.Y.shape
        for i in range(0, length[0], blocks):
            for j in range(0, length[1], blocks):
                slice = self.Y[i:i+blocks, j:j+blocks]
                # self.Y_dct[i:i+blocks, j:j+blocks] = slice * self.q_y
                self.Y[i:i+blocks, j:j+blocks] = slice * self.qualityQ_Y

        length = self.CB.shape
        for i in range(0, length[0], blocks):
            for j in range(0, length[1], blocks):
                slice = self.CB[i:i+blocks, j:j+blocks]
                # self.CB_dct[i:i+blocks, j:j+blocks] = slice * self.q_cbcr
                self.CB[i:i+blocks, j:j+blocks] = slice * self.qualityQ_CBCR

                slice = self.CR[i:i+blocks, j:j+blocks]
                # self.CR_dct[i:i+blocks, j:j+blocks] = slice * self.q_cbcr
                self.CR[i:i+blocks, j:j+blocks] = slice * self.qualityQ_CBCR

        self.Y  = np.round(self.Y)
        self.CB = np.round(self.CB)
        self.CR = np.round(self.CR)

        if debugDCT:
            gray  = self.colormap('myGray', (1,1,1))
            self.showColorMap(np.log(abs(self.Y) + 0.0001), gray,  'Y iquantization')
            self.showColorMap(np.log(abs(self.CB) + 0.0001), gray, 'CB iquantization')
            self.showColorMap(np.log(abs(self.CR) + 0.0001), gray, 'CR iquantization')


    def calcQuality(self, quality):
        if quality >= 50:
            scaleFactor = (100 - quality) / 50
        else:
            scaleFactor = 50 / quality

        qualityQ_Y = self.q_y * scaleFactor
        qualityQ_CBCR = self.q_cbcr * scaleFactor


        qualityQ_CBCR = np.round(qualityQ_CBCR).astype(np.uint8)
        qualityQ_Y = np.round(qualityQ_Y).astype(np.uint8)

        qualityQ_Y[qualityQ_Y < 1] = 1
        qualityQ_Y[qualityQ_Y > 255] = 255

        qualityQ_CBCR[qualityQ_CBCR < 1] = 1
        qualityQ_CBCR[qualityQ_CBCR > 255] = 255

        self.qualityQ_Y, self.qualityQ_CBCR = qualityQ_Y, qualityQ_CBCR


    
    def dpcm(self):
        self.Y_dpcm = np.copy(self.Y)
        self.Cb_dpcm = np.copy(self.CB)
        self.Cr_dpcm = np.copy(self.CR)

        
        for i in range(int(self.Y.shape[0]/8)):
            for j in range(int(self.Y.shape[1]/8)):
                if (j != 0):
                    self.Y_dpcm[i*8, j*8] = self.Y[i*8, j*8] - self.Y[i*8, j*8-8]
                else:
                    if(i != 0):
                        self.Y_dpcm[i*8, j*8] = self.Y[i*8, j*8] - self.Y[i*8-8, int(self.Y.shape[1])-8]
        

        for i in range(int(self.CB.shape[0]/8)):
            for j in range(int(self.CB.shape[1]/8)):
                if (j != 0):
                    self.Cb_dpcm[i*8, j*8] = self.CB[i*8, j*8] - self.CB[i*8, j*8-8]
                    self.Cr_dpcm[i*8, j*8] = self.CR[i*8, j*8] - self.CR[i*8, j*8-8]
                else:
                    if(i != 0):
                        self.Cb_dpcm[i*8, j*8] = self.CB[i*8, j*8] - self.CB[i*8-8, int(self.CB.shape[1])-8]
                        self.Cr_dpcm[i*8, j*8] = self.CR[i*8, j*8] - self.CR[i*8-8, int(self.CB.shape[1])-8]
        

        if debugDPCM:
            gray  = self.colormap('myGray', (1,1,1))
            self.showColorMap(np.log(abs(self.Y_dpcm ) + 0.0001), gray, 'Y DPCM')
            self.showColorMap(np.log(abs(self.Cb_dpcm) + 0.0001), gray, 'CB DPCM')
            self.showColorMap(np.log(abs(self.Cr_dpcm) + 0.0001), gray, 'CR DPCM')


    def reverse_dpcm(self):

        self.Y = np.copy(self.Y_dpcm)
        self.CB = np.copy(self.Cb_dpcm)
        self.CR = np.copy(self.Cr_dpcm)
        
        for i in range(int(self.Y_dpcm.shape[0]/8)):
            for j in range(int(self.Y_dpcm.shape[1]/8)):
                if (j != 0):
                    self.Y[i*8, j*8] = self.Y[i*8, j*8-8] + self.Y_dpcm[i*8, j*8]
                else:
                    if(i != 0):
                        self.Y[i*8, j*8] = self.Y[i*8-8, int(self.Y_dpcm.shape[1])-8] + self.Y_dpcm[i*8, j*8]
                

        for i in range(int(self.Cb_dpcm.shape[0]/8)):
            for j in range(int(self.Cb_dpcm.shape[1]/8)):
                if (j != 0):
                    self.CB[i*8, j*8] = self.CB[i*8, j*8-8] + self.Cb_dpcm[i*8, j*8] 
                    self.CR[i*8, j*8] = self.CR[i*8, j*8-8] + self.Cr_dpcm[i*8, j*8] 
                else:
                    if(i != 0):
                        self.CB[i*8, j*8] = self.CB[i*8-8, int(self.Cb_dpcm.shape[1])-8] + self.Cb_dpcm[i*8, j*8] 
                        self.CR[i*8, j*8] = self.CR[i*8-8, int(self.Cb_dpcm.shape[1])-8] + self.Cr_dpcm[i*8, j*8] 


    def encoder(self): 
        self.readImage()
        # self.showImage()
        # self.showChannels()
        self.padding()

        self.splitChannelsPadding()
        self.rgbToYCbCr()
        self.sampling()
        self.dctBlock()
        self.quantDCT(8,75)
        self.dpcm()

    def decoder(self):
        self.reverse_dpcm()
        self.iQuantDCT(8,75)
        self.idctBlock()
        self.upSampling()
        self.YCbCrTorgb()


if __name__ == "__main__":
    
    a = jpeg('../../Assets/barn_mountains/barn_mountains.bmp')
    a.encoder()
    a.decoder()
