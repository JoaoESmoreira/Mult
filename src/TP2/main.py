import numpy as np
import librosa
import scipy 
from scipy.spatial.distance import cityblock , cosine

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

        #self.features = self.__readFile('./feature.csv')

        self.songsNames = self.__readDirectory("./assets/songs/")
        self.songsNames2 = self.__readDirectory("./assets/Queries/")

        self.__getFeatures("./assets/songs/")
        

        if isfile("./featuresStatesNormalized.csv"):
            self.featuresStatsNormalizated = self.__readFile('./featuresStatesNormalized.csv')
        else:
            self.featuresStatsNormalizated = self.__normalization(0, 1, self.featuresStats)
            self.__saveFeatures("./featuresStatesNormalized.csv", self.featuresStatsNormalizated)
        
        self.metricasSim()
        

        self.ranking()
        self.metadata()
        self.metadataRankingCalc()

        self.precision()
        
        
        
    def precision(self):
        metrics = ["Euclidian_top100", "Manhattan_top100","Cosine_top100","Euclidian_features","Manhattan_features","Cosine_features"]

        with open("./precision.txt", "w") as file:
            for i in range(4):
                file.write(self.songsNames2[i] + "\n\n")  
                for j in range(6):
                    file.write(metrics[j]+ " " +  str((len(np.intersect1d(self.metadataRanking[i], self.rankings[i*6+j]))-1)/20) + "\n")
                file.write("\n\n")


    def metadata(self):
        if isfile("./similaridade.csv"):
            self.meta_mat = self.__readFile("./similaridade.csv")
        else:
            self.metadados = np.genfromtxt('./assets/panda_dataset_taffc_metadata.csv', dtype="str", delimiter=",")[1:, [1, 3, 9, 11]]
            self.meta_mat = np.zeros((900,900))

            #ALTERAR RANGE
            for x in range(900):
                for y in range(900):
                    cont = 0
                    if(self.metadados[x][0] == self.metadados[y][0]):
                        cont+=1
                    if(self.metadados[x][1] == self.metadados[y][1]):
                        cont+=1

                    moods1 = self.metadados[x][2][1:-1].split("; ")
                    moods2 = self.metadados[y][2][1:-1].split("; ")
                    for z in range(len(moods1)):
                        if(moods1[z] in moods2):
                            cont+=1

                    genres1 = self.metadados[x][3][1:-1].split("; ")
                    genres2 = self.metadados[y][3][1:-1].split("; ")
    
                    for z in range(len(genres1)):
                        if(genres1[z] in genres2):
                            cont+=1

                    self.meta_mat[x][y] = cont
            np.savetxt("./similaridade.csv", self.meta_mat, fmt = "%d", delimiter= ",")
        #with open('ranking_meta.txt', 'w') as f:
        #    for song in range(len(self.songsNames2)):
        #        f.write('\n\nquery = ' + self.songsNames2[song] + '\n')
        #        indexSong = self.songsNames.index(self.songsNames2[song])
        #        first20 = np.argsort(self.meta_mat[indexSong , : ])[-21:].astype(int)[::-1]
        #        for pos in range(len(first20)):
        #            if(pos%3 == 0):
        #                f.write('\n')
        #            f.write("\'"+self.songsNames[first20[pos]]+ "\' ")
        #        f.write('\n[')
        #        for pos in range(len(first20)):
        #            f.write(str(self.meta_mat[indexSong,first20[pos]]) + ", ")
        #        f.write(']\n')


    def metadataRankingCalc(self):
        self.metadataRanking = np.zeros((4, 21))
        for i in range(len(self.songsNames2)):
            indexSong = self.songsNames.index(self.songsNames2[i])
            self.metadataRanking[i] = np.argsort(self.meta_mat[indexSong , : ])[-21:].astype(int)[::-1]


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
            else:
                try:
                    data[:, column] = a + ((data[:, column]-fMin) * (b-a)) / (fMax-fMin)
                except:
                    print(data[:, column])

        return data


    def __saveFeatures(self, path, data):
        # self.showFeatures()
        np.savetxt(path, data, fmt="%lf", delimiter=',')


    def __readDirectory(self, path):
        return [f for f in listdir(path) if isfile(join(path, f))]

    def extrationStats(self, data):
        mean = np.mean(data)
        stddv = np.std(data)
        skew = scipy.stats.skew(data)
        kurtosis = scipy.stats.kurtosis(data)
        median = np.median(data)
        max = np.max(data)
        min = np.min(data)
        
        return np.array([mean, stddv, skew, kurtosis, median, max, min], dtype=object)


    def __getFeatures(self, path):
        if isfile("./featuresStates.csv"):
            self.featuresStats = self.__readFile('./featuresStates.csv')
        else:
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


            self.__saveFeatures("./featuresStates.csv", self.featuresStats)


    def showFeatures(self):
        for row in self.features:
            print(row)
            break
    

    def metricasSim(self):
        if isfile("./euclidian_top100.csv"):
            self.euclidian100 = self.__readFile('./euclidian_top100.csv')
            self.manhattan100 = self.__readFile('./manhattan_top100.csv')
            self.cosine100 = self.__readFile('./cosine_top100.csv')
            self.euclidianF = self.__readFile('./euclidian_features.csv')
            self.manhattanF = self.__readFile('./manhattan_features.csv')
            self.cosineF = self.__readFile('./cosine_features.csv')
        else:
            self.euclidian100 = np.zeros((900,900))
            self.manhattan100 = np.zeros((900,900))
            self.cosine100 = np.zeros((900,900))
            self.euclidianF = np.zeros((900,900))
            self.manhattanF = np.zeros((900,900))
            self.cosineF = np.zeros((900,900))

            for i in range(900):
                for j in range(900):
                    self.euclidian100[i][j] = np.linalg.norm(self.features[i] - self.features[j])
                    self.manhattan100[i][j] = cityblock(self.features[i], self.features[j])
                    self.cosine100[i][j] = cosine(self.features[i],self.features[j])

                    self.euclidianF[i][j] = np.linalg.norm(self.featuresStatsNormalizated[i] - self.featuresStatsNormalizated[j])
                    self.manhattanF[i][j] = cityblock(self.featuresStatsNormalizated[i], self.featuresStatsNormalizated[j])
                    self.cosineF[i][j] = cosine(self.featuresStatsNormalizated[i],self.featuresStatsNormalizated[j])
            
            self.__saveFeatures("./euclidian_top100.csv", self.euclidian100)
            self.__saveFeatures("./euclidian_features.csv", self.euclidianF)
            
            self.__saveFeatures("./manhattan_top100.csv", self.manhattan100)
            self.__saveFeatures("./manhattan_features.csv", self.manhattanF)
            
            self.__saveFeatures("./cosine_top100.csv", self.cosine100)
            self.__saveFeatures("./cosine_features.csv", self.cosineF)


    def ranking(self):
        if isfile("./ranking.csv") and isfile("./ranking_songs.csv"):
            self.rankings = self.__readFile("./ranking.csv")
            #self.rankingSongs = self.__readFile("./ranking_songs.csv")
        else:
            self.rankings = np.zeros((24, 21))
            # self.rankingSongs = np.array((24, 21), dtype=str)
            for i in range(len(self.songsNames2)):
                indexSong = self.songsNames.index(self.songsNames2[i])
                
                self.rankings[i*6] = np.argsort(self.euclidian100[indexSong, : ])[0:21].astype(int) 
                self.rankings[i*6+1] = np.argsort(self.manhattan100[indexSong, : ])[0:21].astype(int)   
                self.rankings[i*6+2] = np.argsort(self.cosine100[indexSong, : ])[0:21].astype(int)
                self.rankings[i*6+3] = np.argsort(self.euclidianF[indexSong, : ])[0:21].astype(int)
                self.rankings[i*6+4] = np.argsort(self.manhattanF[indexSong, : ])[0:21].astype(int)
                self.rankings[i*6+5] = np.argsort(self.cosineF[indexSong, : ])[0:21].astype(int)
               

            self.rankingSongs = []
            np.savetxt("./ranking.csv", self.rankings, fmt = "%d", delimiter=",")
            for i in range(len(self.rankings)):
                # aux = [self.songsNames[int(self.rankings[i][j])] for j in range(len(self.rankings[i]))]
                aux = []
                for j in range(len(self.rankings[i])):
                    aux.append((self.songsNames[int(self.rankings[i][j])]))
                self.rankingSongs.append(aux)

            self.rankingSongs = np.array(self.rankingSongs)
            np.savetxt("./ranking_songs.csv", self.rankingSongs, fmt= "%s", delimiter=",")

            
            
            
                    

if __name__ == "__main__":
    file = feature('./assets/top100_features.csv')
    