import numpy as np


class file():

    def __init__(self, path):
        self.file = self.__readFile(path)


    def __readFile(self, path):
        np.set_printoptions(suppress=True)
        return np.genfromtxt(path, delimiter=',')
    

    def getData(self):
        return self.file

    
    def compare(self, dataFile):
        # diff = np.zeros(np.shape(self.file))
        diff = np.abs(self.file - dataFile) > 0.1

        return diff
    
    

if __name__ == "__main__":
    a = file('featuresStatesNormalizated.csv')
    b = file('resultados/FMrosa.csv')

    output = a.compare(b.getData()[0])
    print(np.shape(output))
    for i in range(len(output[0])):
        if output[0, i] == True:
            print(i, end=" ")

    print("\n")