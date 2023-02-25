"""

coeficientes (canto esquerdo proporcional à media)

tansiçoes abruptas -> canto direito com valores altos
tansiçoes suaves -> canto direito com valores baixos

"""

pip install -r requirements.txt


    def sampling(self, factor="4:1:1"):
        gray  = self.colormap('mygray', (1,1,1))
        shape = np.shape(self.cb)
        self.shape = shape

        if factor[-1] == '0':
            widh   = int(factor[2])/4
            cbdim  = (int(shape[1]*widh), int(shape[0]*widh))
            crdim  = (int(shape[1]*widh), int(shape[0]*widh))
        else:        
            widhcb = int(factor[2])/4
            widhcr = int(factor[-1])/4
            cbdim  = (int(shape[1]*widhcb), int(shape[0]))
            crdim  = (int(shape[1]*widhcr), int(shape[0]))

        self.cb = cv2.resize(self.cb, cbdim, interpolation=cv2.inter_cubic)
        self.cr= cv2.resize(self.cr, crdim, interpolation=cv2.inter_cubic)

        if debugsampling:
            self.showcolormap(self.cb, gray)
            self.showcolormap(self.cb, gray)
            self.showcolormap(self.cr, gray)
            self.showcolormap(self.cr, gray)

    
    def upsampling(self, factor="4:1:1"):
        gray  = self.colormap('mygray', (1,1,1))
        shape = np.shape(self.cb)

        if factor[-1] == '0':
            widh   = int(factor[2])
            cbdim  = (int(shape[1]*widh), int(shape[0]*widh))
            crdim  = (int(shape[1]*widh), int(shape[0]*widh))
        
        else:
            widhcb = 4/int(factor[2]) 
            widhcr = 4/int(factor[-1])
            cbdim  = (int(shape[1])*widhcb, int(shape[0]))
            crdim  = (int(shape[1])*widhcr, int(shape[0]))
            


        self.cb = cv2.resize(self.cb, cbdim, interpolation=cv2.inter_cubic)
        self.cr= cv2.resize(self.cr, crdim, interpolation=cv2.inter_cubic)

        print(cbdim)
        print(np.shape(self.cb))
        print(np.shape(self.datapadding))

        self.showcolormap(self.datapadding, gray)
        self.showcolormap(self.cb, gray)
        self.showcolormap(self.cr, gray)