from keras.models import Sequential
from keras.layers import Dense
import numpy

class NeuralNetwork:

    def __init__(self, fileName="", cols=0):
        print("Neural network starting")
        # Dataset
        print("Loading dataset file, please wait...")
        dataset = numpy.loadtxt(fileName, delimiter=",")

        # Entradas X y salidas Y esperadas
        self.x = dataset[:, 0:cols]
        self.y = dataset[:, cols]

        self.model = Sequential()
        self.model.add(Dense(16, input_dim=8, kernel_initializer='uniform', activation='relu'))
        self.model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
        self.model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


    def trainNetwork(self, nepochs=1, batch=10):
        print("Training started")
        self.model.fit(self.x, self.y, epochs=nepochs, batch_size=batch)

    def predict(self, predictData):
        return self.model.predict(predictData).round()