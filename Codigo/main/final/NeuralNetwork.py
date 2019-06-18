from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy

class NeuralNetwork:
    cols = 0
    def __init__(self, cols=0):
        print("Neural network starting")
        self.cols = cols

    def loadTrainingSamples(self, fileName=""):
        print("Loading dataset file, please wait...")
        dataset = numpy.loadtxt(fileName, delimiter=",")

        # Entradas X y salidas Y esperadas
        self.x = dataset[:, 0:self.cols]
        self.y = dataset[:, self.cols]

        self.model = Sequential()
        self.configureNetwork()

    def configureNetwork(self):
        self.model.add(Dense(16, input_dim=self.cols, kernel_initializer='uniform', activation='relu'))
        self.model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
        self.model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def saveModel(self, fileName):
        model_json = self.model.to_json()
        with open(fileName, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights("model.h5")

    def loadModel(self, fileName):
        # load json and create model
        json_file = open(fileName, 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights("model.h5")

    def trainNetwork(self, nepochs, batch=10):
        print("Training started")
        self.model.fit(self.x, self.y, epochs=nepochs, batch_size=batch)

    def predict(self, predictData):
        return self.model.predict(predictData).round()