import numpy as np


class feature():

    def __init__(self, path):
        self.path = path
        self.__readFile()
        self.__normalization(0, 1)
        self.__saveFeatures("./feature.csv")


    def __readFile(self):
        features = np.genfromtxt(self.path, delimiter=',')
        features = features[1:, 1:-1].astype('float')

        self.features = features


    def __normalization(self, a, b):
        fMin = self.features.min()
        fMax = self.features.max()
        
        if fMax - fMin == 0:
            self.features[:, :] = 0

        self.features = a + ((self.features-fMin) * (b-a)) / (fMax-fMin)


    def __saveFeatures(self, path):
        self.showFeatures()
        np.savetxt(path, self.features, delimiter=',')


    def showFeatures(self):
        for row in self.features:
            print(row)
            break
    

if __name__ == "__main__":
    file = feature('./assets/top100_features.csv')
    # file.showFeatures()

