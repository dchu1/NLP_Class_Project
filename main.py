from Vectorizer import Vectorizer
from Model import Model
from Downloader import Downloader
from Pipeline import Pipeline
from Tokenizer import Tokenizer
from Stemmer import Stemmer
from Splitter import splitRows
from OptimNN import OptimNN
import heapq
import pdb
import numpy as np
import pandas as pd
import sys, getopt

'''
Arguments (comma separated, in order):
    Data Transformations:
        toke                -> Tokenizer
        stem                -> Stemmer [Optional]
        split-sentences     -> Split rows via sentences
        vect                -> Vectorizer

    Model Selection:
        svm                 -> Sets model as SVM 
        nb                  -> Sets model as NB

    Misc:
        no-verb             -> (No Verbose): Command to not print intermediate steps to terminal.

Example:
    python main.py toke,stem,vect,nb
'''

### Globals
verbose = True
norm = False

'''
def run(vectorizer, model, train_data, train_labels, test_data, test_labels):
    vecTrainData = v.fitTransform(trainData)
    vecTestData = v.transform(testData)
    clf.fit(vecTrainData, trainLabels)
    clf.predict(vecTestData, testLabels, verbose = True)
    return 0
'''

def main(argv):
    # construct our pipeline list reading from command line args
    # still need to figure out best way to pass parameters on command
    # line

    global verbose
    global norm
    split = None

    transforms = []
    for arg in argv[0].split(","):
        if arg == "toke":
            transforms.append(Tokenizer())
        elif arg == "stem":
            transforms.append(Stemmer())
        elif arg == "stem-porter":
            transforms.append(Stemmer(mode = 'Porter'))
        elif arg == "stem-lancaster":
            transforms.append(Stemmer(mode = 'Lancaster'))
        elif arg == "stem-lemmatize":
            transforms.append(Stemmer(mode = 'Lemmatize'))
        elif arg == "vect":
            transforms.append(Vectorizer())
        elif arg == "vect-tfidf":
            transforms.append(Vectorizer(mode='TFIDF'))
        elif arg == "vect-count":
            transforms.append(Vectorizer(mode='Count'))
        elif arg == "vect-lda-2":
            transforms.append(Vectorizer(mode='LDA', ldaSplits=2))
        elif arg == "vect-lda-10":
            transforms.append(Vectorizer(mode='LDA', ldaSplits=10))
        elif arg == "vect-lda-25":
            transforms.append(Vectorizer(mode='LDA', ldaSplits=25))
        elif arg == "vect-lda-50":
            transforms.append(Vectorizer(mode='LDA', ldaSplits=50))
        elif arg == "vect-lda-150":
            transforms.append(Vectorizer(mode='LDA', ldaSplits=150))
        elif arg == "vect-lda-500":
            transforms.append(Vectorizer(mode='LDA', ldaSplits=500))
        elif arg == "svm":
            transforms.append(Model('svm'))
        elif arg == "nb":
            transforms.append(Model('nb'))
        elif arg == "lr":
            transforms.append(Model('lr'))
        elif arg == "nn":
            transforms.append(Model('nn', inputDim = 10000)) #Configured for Vectorizer with vectors limited to 1000
        elif arg == "norm":
            norm = True
        elif arg == "no-verb":
            verbose =  False
        elif arg == "split-sentences":
            split = "sentences"
        elif arg == "nn-optim":
            # Memory optimized neural network.
            transforms.append(OptimNN(vecMode='TFIDF',epochs=2,batchSize=2048))
        else:
            raise Exception(f"Invalid transformer {arg}")
    pipe = Pipeline(transforms, norm=norm)

    # read our data (hardcoded for now)
    df0 = pd.read_pickle("./data/democrat_comments.pkl")#.sample(frac = 0.05) # DEBUG ONLY
    df1 = pd.read_pickle("./data/republican_comments.pkl")#.sample(frac = 0.05) # DEBUG ONLY

    if(split is not None):
        if(verbose):
            print('Splitting Democrat comments')
        df0 = splitRows(df0, mode=split, verbose=verbose)

        if(verbose):
            print('Splitting Republican comments')
        df1 = splitRows(df1, mode=split, verbose=verbose)

    label0 = df0.subreddit.iloc[0]
    label1 = df1.subreddit.iloc[0]

    # concatenate and clean our data
    X = pd.concat([df0.body, df1.body], ignore_index=True)
    y = pd.concat([df0.subreddit, df1.subreddit], ignore_index=True).replace(to_replace=[label0, label1], value=[0, 1])

    # split into training and test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    if(verbose):
        print('Applying Transforms and Training Model')
        print('Train Data:', train_path)
        print('Test Data:', test_path)
        print('Transforms:', argv[0])
    # fit our data
    pipe.fit_transform(X_train, y_train)

    # do the prediction
    y_pred = pipe.predict(X_test)
    results = pipe.validate(y_pred, y_test, True, True)
    
    # get most suprising misclassifications for class 0
    print("Most suprising texts misclassified as class 0")
    idx_list = heapq.nlargest(5, results[2][0], key = lambda x: x[1])
    for i,(idx,prob) in enumerate(idx_list):
        print(f"{i}) probability class 1 = {prob}\n{X_test[idx]}, \n")

    # get most suprising misclassifications for class 1
    print("Most suprising texts misclassified as class 1")
    idx_list = heapq.nlargest(5, results[2][1], key = lambda x: x[1])
    for i,(idx,prob) in enumerate(idx_list):
        print(f"{i}) probability class 0 = {prob}\n{X_test[idx]}\n")

if __name__ == "__main__":
    main(sys.argv[1:])