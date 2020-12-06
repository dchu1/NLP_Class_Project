import pdb
from scipy.sparse.csr import csr_matrix
'''
Models:
    Naive Bayes: nb | Note: needs input shape which has to be manually configured
    Logistic Regression: lr
    Support Vector Machine: svm
'''
saveModel = True # For scikit learn models only

import numpy as np
def makeLabelArr(arr):
    res = np.zeros((arr.shape[0],2))
    res[arr == 0] = (1,0)
    res[arr == 1] = (0,1)
    return(res)

def makePredArr(arr):
    res = np.zeros((arr.shape[0]))
    for i,x in enumerate(arr):
        if(x[0]<x[1]):
            res[i] = 1
    return(res)

class Model:
    model = None
    modelType = None

    def __init__(self, modelType = 'nb', opt = None, inputDim = None):
        if(modelType == 'nb'):
            from sklearn.naive_bayes import MultinomialNB
            self.model = MultinomialNB()
            self.modelType = 'Naive Bayes'
        if(modelType == 'svm'):
            from sklearn import svm
            kernelType = 'rbf' #default of svms
            if opt:
                kernel = opt
            self.model = svm.SVC(kernel=kernelType)
            self.modelType = 'Support Vector Machine'
        if(modelType == 'lr'):
            from sklearn.linear_model import LogisticRegression
            self.model = LogisticRegression(max_iter=1000)
            self.modelType = 'Logistic Regression'
        if(modelType == 'nn'):
            from keras.models import Sequential
            from keras.layers import Dense
            from keras.losses import BinaryCrossentropy
            loss = BinaryCrossentropy(from_logits=True)
            self.model = Sequential()
            self.model.add(Dense(512, input_shape= (inputDim,), activation='relu'))
            self.model.add(Dense(256, activation='relu'))
            self.model.add(Dense(128, activation='relu'))
            self.model.add(Dense(2, activation='softmax'))
            self.model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])
            self.model.summary()
            self.modelType = 'Neural Network'
    
    def __str__(self):
        return('Model of type: '+self.modelType+' - '+str(type(self.model)))

    def fit(self, trainData, trainLabels):
        if(self.modelType == 'Neural Network'):
            trainLabels = makeLabelArr(trainLabels)
            if(isinstance(trainData, csr_matrix)): # Consider dataloader
                trainData = trainData.toarray()
            return(self.model.fit(trainData, trainLabels, epochs = 150, batch_size=int(trainData.shape[0]/5)))
        res = self.model.fit(trainData, list(trainLabels))
        if(saveModel):
            from joblib import dump
            dump(self.model, 'saved/model.sav')
        return(res)
    
    def validate(self,labels,predictions, predictions_proba = False, verbose = True):
        from sklearn.metrics import precision_recall_fscore_support
        if(self.modelType == 'Neural Network'):
            predictions = makePredArr(predictions)
        if predictions_proba:
            predictions_proba = predictions
            predictions = np.argmax(predictions_proba, axis=1)
        results = precision_recall_fscore_support(labels, predictions)
        acc = (list(predictions==labels).count(True))/(len(predictions))
        if(verbose):
            # print('Accuracy of the model:\t %0.3f'%(acc))
            # print('Precision wrt. class 0:\t %0.3f'%(results[0][0]))
            # print('Recall wrt. class 0:\t %0.3f'%(results[1][0]))
            # print('F1 Score wrt. class 0:\t %0.3f'%(results[2][0]))
            from sklearn.metrics import classification_report
            from sklearn.metrics import confusion_matrix
            print(classification_report(labels, predictions))
            print('Confusion matrix:')
            cm = confusion_matrix(labels, predictions)
            for row in cm:
                print(row)

            # get most suprising misclassifications
            misclassified = ([],[])
            # go through each prediction
            for i,pred_label in enumerate(predictions):
                # if the prediction was incorrect, put it in that labels list along with probability
                if pred_label != labels[i]:
                    misclassified[labels[i]].append((i, predictions_proba[i][pred_label]))
        return(results,acc,misclassified)

    def predict(self, testData, testLabels = None, verbose = True):
        if(self.modelType == 'Neural Network' and isinstance(testData, csr_matrix)):
            testData = testData.toarray()
        # if verbose and nb or lr, get proba estimates to determine which were the worst classified
        if verbose and (self.modelType == 'Logistic Regression' or self.modelType == 'Naive Bayes'):
            predictions = self.model.predict_proba(testData)
        else:
            predictions = self.model.predict(testData)
        return predictions
        # if(testLabels is not None):
        #     return(self.validate(list(testLabels), predictions, predictions_proba = proba, verbose = verbose))