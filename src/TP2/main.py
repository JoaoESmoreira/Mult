import numpy as np
import librosa
import scipy 

from os import listdir
from os.path import isfile, join

class feature():
    def __init__(self, path):

        self.sr = 22050
        self.mono = True
        self.windowLength = 92.88
        self.frameLength = 92.88
        self.hopLength = 23.22
        self.fmin = 20
        self.fmax = self.sr // 2

        self.features = self.__readTop100features(path)
        self.features = self.__normalization(0, 1, self.features)
        self.__saveFeatures("./feature.csv", self.features)

        # self.__readDirectory("./assets/songs/")
        # self.__getFeatures("./assets/songs/")
        # self.__saveFeatures("./featuresStates.csv", self.featuresStats)

        self.featuresStats = self.__readFile('./featuresStates.csv')
        self.featuresStatsNormalizated = self.__normalization(0, 1, self.featuresStats)
        self.__saveFeatures("./featuresStatesNormalizated.csv", self.featuresStatsNormalizated)


    def __readTop100features(self, path):
        np.set_printoptions(suppress=True)
        features = np.genfromtxt(path, delimiter=',')
        features = features[1:, 1:-1].astype('float')

        return features


    def __readFile(self, path):
        np.set_printoptions(suppress=True)
        return np.genfromtxt(path, delimiter=',')


    def __normalization(self, a, b, data):
        columns = np.shape(data)[1]
        for column in range(columns):
            fMin = data[:, column].min()
            fMax = data[:, column].max()

            if fMax == fMin:
                data[:, column] = 0

            try:
                data[:, column] = a + ((data[:, column]-fMin) * (b-a)) / (fMax-fMin)
            except:
                print(data[:, column])


        return np.round(data, 5)


    def __saveFeatures(self, path, data):
        # self.showFeatures()
        np.savetxt(path, data, delimiter=',')


    def __readDirectory(self, path):
        self.songsNames = [f for f in listdir(path) if isfile(join(path, f))]


    def extrationStats(self, data):
        mean = np.mean(data)
        stddv = scipy.stats.tstd(data)
        skew = scipy.stats.skew(data)
        kurtosis = scipy.stats.kurtosis(data)
        median = np.median(data)
        max = np.max(data)
        min = np.min(data)
        # print(mean)
        # print(stddv)
        # print(skew)
        # print(kurtosis)
        # print(median)
        # print(max)
        # print(min)
        
        return np.array([mean, stddv, skew, kurtosis, median, min, max], dtype=object)


    def __getFeatures(self, path):
        lenSongs = int(np.shape(self.songsNames)[0])
        self.featuresStats = np.arange(lenSongs*190, dtype=object).reshape((lenSongs, 190))

        for song in range(lenSongs):
            songName = self.songsNames[song]
            y, fs = librosa.load(path + songName, sr=self.sr, mono=self.mono)

            # =============== MFCC ===========================
            mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=13)
            for i in range(len(mfcc)):
                self.featuresStats[song, i*7:i*7+7] = self.extrationStats(mfcc[i])

            # =============== Centroid ===========================
            centroid = librosa.feature.spectral_centroid(y=y, sr=self.sr)[0]
            self.featuresStats[song, 91:91+7] = self.extrationStats(centroid)
            
            # =============== BandWidth ===========================
            bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=self.sr)[0]
            self.featuresStats[song, 98:98+7] = self.extrationStats(bandwidth)

            # =============== Contrast ===========================
            contrast = librosa.feature.spectral_contrast(y=y, sr=self.sr)
            for i in range(len(contrast)):
                self.featuresStats[song, 105+i*7:105+i*7+7] = self.extrationStats(contrast[i])

            # =============== flatness ===========================
            flatness = librosa.feature.spectral_flatness(y=y)[0]
            self.featuresStats[song, 154:154+7] = self.extrationStats(flatness)

            # =============== rolloff ===========================
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=self.sr)[0]
            self.featuresStats[song, 161:161+7] = self.extrationStats(rolloff)

            # =============== f0 ===========================
            f0 = librosa.yin(y=y, fmin=self.fmin, fmax=self.fmax)
            f0[f0==self.fmax] = 0
            self.featuresStats[song, 168:168+7] = self.extrationStats(f0)  
            
            # =============== RMS ===========================
            rms = librosa.feature.rms(y=y)[0]
            self.featuresStats[song, 175:175+7] = self.extrationStats(rms)

            # =============== Cross ===========================
            zero_cross = librosa.feature.zero_crossing_rate(y=y)[0]
            self.featuresStats[song, 182:182+7] = self.extrationStats(zero_cross)
            
            # =============== Time ===========================
            # time = librosa.beat.tempo(y=y)
            time = librosa.feature.tempo(y=y)
            self.featuresStats[song, -1] = time[0]


    def showFeatures(self):
        for row in self.features:
            print(row)
            break
    

if __name__ == "__main__":
    file = feature('./assets/top100_features.csv')
    # file.showFeatures()
