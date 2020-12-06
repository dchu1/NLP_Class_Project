import numpy as np
from tensorflow.keras.utils import Sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from Vectorizer import Vectorizer
from keras.models import Sequential
from keras.layers import Dense
from keras.losses import BinaryCrossentropy
from Model import makeLabelArr
from Model import makePredArr

class DataLoader(Sequence):
    def __init__(self, data, labels, vec, batchSize, toFit=True, shuffle=True):
        self.data = data
        self.labels = labels
        self.vec = vec
        self.batchSize = batchSize
        self.toFit = toFit
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        # Returns number of batches per epoch
        return(int(np.floor(len(self.data) / self.batchSize)))

    def on_epoch_end(self):
        # Shuffle indexes every epoch
        self.indexes = np.arange(len(self.data))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, batchIndex):
        # Generate one batch of data for the given batchIndex
        indexes = self.indexes[batchIndex * self.batchSize:(batchIndex + 1) * self.batchSize]
        data = self.data[indexes]
        if self.toFit:
            labels = self.labels[indexes]
            return(self.vec.transform(data).toarray(), makeLabelArr(labels))
        else:
            return(self.vec.transform(data).toarray())

class OptimNN():
    def __init__(
        self, vecMode = 'TFIDF', maxVecFeatures = None, batchSize = 128, epochs = 3, modelN = 128, modelNFactor = 2, showModelSummary = True, valSplit = 0.1):
        self.vecMode = vecMode
        self.maxVecFeatures = maxVecFeatures
        self.batchSize = batchSize
        self.epochs = epochs
        self.modelN = modelN
        self.modelNFactor = modelNFactor
        self.showModelSummary = showModelSummary
        self.valSplit = valSplit

    def fit(self, trainData, trainLabels):
        trainData=np.array(trainData)
        trainLabels=np.array(trainLabels)

        # split training set into training and validation sets
        from sklearn.model_selection import train_test_split
        trainData, valData, trainLabels, valLabels = train_test_split(trainData, trainLabels, test_size=self.valSplit, random_state=42)

        self.vec = Vectorizer(mode = self.vecMode, maxFeatures=self.maxVecFeatures)
        self.vec.fit(trainData)

        #print(self.vec.transform([trainData[0]])[0].shape[1])
        numFeatures = len(self.vec.vec.vocabulary_.keys())
        print('Vectorizer fit complete with',numFeatures,'features in each vector')

        trainDataGenerator = DataLoader(
            data = trainData,
            labels = trainLabels,
            vec = self.vec,
            batchSize = self.batchSize
        )

        valDataGenerator = DataLoader(
            data = valData,
            labels = valLabels,
            vec = self.vec,
            batchSize = self.batchSize
        )

        loss = BinaryCrossentropy(from_logits=True)
        self.model = Sequential()
        self.model.add(Dense(self.modelN*self.modelNFactor*2, input_shape= (numFeatures,), activation='relu'))
        self.model.add(Dense(self.modelN*self.modelNFactor, activation='relu'))
        self.model.add(Dense(self.modelN, activation='relu'))
        self.model.add(Dense(2, activation='softmax'))
        self.model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])
        if(self.showModelSummary):
            self.model.summary()
        callbacks = [
            ModelCheckpoint("model.h5", verbose=1, save_best_model=True),
            ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.1, verbose=1, min_lr=1e-6),
            EarlyStopping(monitor="val_loss", patience=5, verbose=1)
        ]

        print('Shape of the training dataset: (%i,%i)'%(len(trainData),numFeatures))

        print('Training and validation data loaders initialized with batch size:',self.batchSize)
        results = self.model.fit(
            trainDataGenerator, 
            validation_data=valDataGenerator,
            workers = 8,
            callbacks=callbacks,
            epochs=self.epochs
        )

        return(results)
        

    def predict(self, testData, testLabels = None, verbose = True):
        testData = np.array(testData)
        testLabels = np.array(testLabels)

        testDataGenerator = DataLoader(
            data = testData,
            labels = testLabels,
            vec = self.vec,
            batchSize = self.batchSize
        )

        labels = []
        predictions = []

        for batchData,batchLabels in testDataGenerator:
            labels.extend(list(makePredArr(batchLabels)))
            predictions.extend(list(makePredArr(self.model.predict(batchData))))

        if(testLabels is None):
            return(predictions)

        labels = np.array(labels)
        predictions = np.array(predictions)

        from sklearn.metrics import precision_recall_fscore_support
        results = precision_recall_fscore_support(labels, predictions)
        acc = (list(predictions==labels).count(True))/(len(predictions))
        print('Accuracy of the model:\t %0.3f'%(acc))
        print('Precision wrt. class 0:\t %0.3f'%(results[0][0]))
        print('Recall wrt. class 0:\t %0.3f'%(results[1][0]))
        print('F1 Score wrt. class 0:\t %0.3f'%(results[2][0]))
        return(results,acc)
