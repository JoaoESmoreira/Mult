import numpy as np
import librosa

from os import listdir
from os.path import isfile, join

class feature():

    

    def __init__(self, path):

        self.sr = 22050
        self.mono = True
        self.windowLength = 92.88
        self.frameLength = 92.88
        self.hopLength = 23.22

        self.path = path
        self.__readFile()
        self.features = self.__normalization(0, 1, self.features)
        self.__saveFeatures("./feature.csv")

        self.__readDirectory("./assets/songs/")
        self.__getFeatures("./assets/songs/")


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


    def __readDirectory(self, path):
        self.songsNames = [f for f in listdir(path) if isfile(join(path, f))]


    def __getFeatures(self, path):
        song1 = self.songsNames[0]
        y, fs = librosa.load(path + song1, sr=self.sr, mono=self.mono)

        lenSongs = int(np.shape(self.songsNames)[0])
        self.features2 = np.arange(lenSongs*10, dtype=object).reshape((lenSongs, 10))

        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=13)
        # print(mfcc)

        centroid = librosa.feature.spectral_centroid(y=y, sr=self.sr)[0]
        # print(centroid)
        
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=self.sr)[0]
        # print(bandwidth)

        contrast = librosa.feature.spectral_contrast(y=y, sr=self.sr)[0]
        # print(contrast)

        flatness = librosa.feature.spectral_flatness(y=y)[0]
        # print(flatness)

        rolloff = librosa.feature.spectral_rolloff(y=y, sr=self.sr)[0]
        # print(rolloff)

        f0 = librosa.yin(y=y, fmin=20, fmax=fs/2)
        f0[f0==fs/2] = 0

        rms = librosa.feature.rms(y=y)[0,:]

        zero_cross = librosa.feature.zero_crossing_rate(y=y)[0]

        # time = librosa.beat.tempo(y=y)
        time = librosa.feature.tempo(y=y)

        self.features2[0] = [mfcc, centroid, bandwidth, contrast, flatness, rolloff, f0, rms, zero_cross, time]
        for feature in self.features2[0]:
            print(feature)


    def showFeatures(self):
        for row in self.features:
            print(row)
            break
    

if __name__ == "__main__":
    file = feature('./assets/top100_features.csv')
    # file.showFeatures()

