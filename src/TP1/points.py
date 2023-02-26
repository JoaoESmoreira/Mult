"""

coeficientes (canto esquerdo proporcional à media)

tansiçoes abruptas -> canto direito com valores altos
tansiçoes suaves -> canto direito com valores baixos

"""
    def quantdct(self, blocks=8):

        length = self.Y_dct.shape
        for i in range(0, length[0], blocks):
            for j in range(0, length[1], blocks):
                slice = self.Y_dct[i:i+blocks, j:j+blocks]
                self.Y_dct[i:i+blocks, j:j+blocks] = slice / self.q_y

        length = self.CB_dct.shape
        for i in range(0, length[0], blocks):
            for j in range(0, length[1], blocks):
                slice = self.CB_dct[i:i+blocks, j:j+blocks]
                self.CB_dct[i:i+blocks, j:j+blocks] = slice / self.q_cbcr

                slice = self.CR_dct[i:i+blocks, j:j+blocks]
                self.CR_dct[i:i+blocks, j:j+blocks] = slice / self.q_cbcr

        self.Y_dct = np.round(self.Y_dct).astype(np.uint8)
        self.CB_dct = np.round(self.CB_dct).astype(np.uint8)
        self.CR_dct = np.round(self.CR_dct).astype(np.uint8)

        gray  = self.colormap('myGray', (1,1,1))
        self.showColorMap(self.Y_dct, gray, 'Y quantization')
        self.showColorMap(self.CB_dct, gray, 'CB quantization')
        self.showColorMap(self.CR_dct, gray, 'CR quantization')


    def iquantdct(self, blocks=8):

        length = self.Y_dct.shape
        for i in range(0, length[0], blocks):
            for j in range(0, length[1], blocks):
                slice = self.Y_dct[i:i+blocks, j:j+blocks]
                self.Y_dct[i:i+blocks, j:j+blocks] = slice * self.q_y

        length = self.CB_dct.shape
        for i in range(0, length[0], blocks):
            for j in range(0, length[1], blocks):
                slice = self.CB_dct[i:i+blocks, j:j+blocks]
                self.CB_dct[i:i+blocks, j:j+blocks] = slice * self.q_cbcr

                slice = self.CR_dct[i:i+blocks, j:j+blocks]
                self.CR_dct[i:i+blocks, j:j+blocks] = slice * self.q_cbcr

        self.Y_dct = np.round(self.Y_dct).astype(np.uint8)
        self.CB_dct = np.round(self.CB_dct).astype(np.uint8)
        self.CR_dct = np.round(self.CR_dct).astype(np.uint8)

        gray  = self.colormap('myGray', (1,1,1))
        # y_dct = np.log(abs(self.Y_dct)+0.00001)
        # cb_dct = np.log(abs(self.CB_dct)+0.00001)
        # cr_dct = np.log(abs(self.CR_dct)+0.00001)
        self.showColorMap(self.Y_dct, gray, 'Y iQuantization')
        self.showColorMap(self.CB_dct, gray, 'CB iQuantization')
        self.showColorMap(self.CR_dct, gray, 'CR iQuantization')