import numpy as np
# import librosa

# from os import listdir
# from os.path import isfile, join

class feature():

    

    def __init__(self, path):

        self.sr = 22050
        self.mono = True

        self.path = path
        self.__readFile()
        self.features = self.__normalization(0, 1, self.features)
        self.__saveFeatures("./feature.csv")

        # self.readDirectory("./assets/songs/")


    def __readFile(self):
        np.set_printoptions(suppress=True)
        features = np.genfromtxt(self.path, delimiter=',')
        features = features[1:, 1:-1].astype('float')

        self.features = features


    def __normalization(self, a, b, data):
        columns = np.shape(data)[1]
        for column in range(columns):
            fMin = data[:, column].min()
            fMax = data[:, column].max()

            if fMax - fMin == 0:
                data[:, column] = 0

            data[:, column] = a + ((data[:, column]-fMin) * (b-a)) / (fMax-fMin)

        return data


    def __saveFeatures(self, path):
        # self.showFeatures()
        np.savetxt(path, self.features, delimiter=',')


    # def readDirectory(self, path):
    #     self.songsNames = [f for f in listdir(path) if isfile(join(path, f))]

    #     song1 = self.songsNames[0]
    #     # warnings.filterwarnings("ignore")
    #     y, fs = librosa.load(path + song1, sr=self.sr, mono=self.mono)
    #     mfcc = librosa.feature.mfcc(y=y, sr=self.sr)
    #     print(mfcc)


    def showFeatures(self):
        for row in self.features:
            print(row)
            break
    

if __name__ == "__main__":
    file = feature('./assets/top100_features.csv')
    # file.showFeatures()

