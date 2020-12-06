from joblib import load
from Tokenizer import Tokenizer
from Stemmer import Stemmer
from Vectorizer import Vectorizer
import numpy as np
import sys

# Config: toke,stem-lemmatize,vect-tfidf,lr
tokenizer = Tokenizer()
stemmer = Stemmer('Lemmatize')
vectorizer = load('saved/vectorizer.sav')
model = load('saved/model.sav')

'''
############### DEBUG
import pandas as pd

df0 = pd.read_pickle("./data/democrat_comments.pkl").sample(frac = 0.05) # DEBUG ONLY
df1 = pd.read_pickle("./data/republican_comments.pkl").sample(frac = 0.05) # DEBUG ONLY

label0 = df0.subreddit.iloc[0]
label1 = df1.subreddit.iloc[0]

# concatenate and clean our data
X = pd.concat([df0.body, df1.body], ignore_index=True)
y = pd.concat([df0.subreddit, df1.subreddit], ignore_index=True).replace(to_replace=[label0, label1], value=[0, 1])

# split into training and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
############### DEBUG
predData = X_test
predData = vectorizer.transform(stemmer.transform((tokenizer.transform(np.array(predData)))))
predictions = model.predict(predData)
labels = np.array(y_test)
'''

def makePredictions(predData):
    predData = vectorizer.transform(stemmer.transform((tokenizer.transform(np.array(predData)))))
    predictions = model.predict(predData)
    return(predictions)

if __name__ == "__main__":
    '''
    Argument format: <mode> <input>
    mode:
        c -> comment. 2nd argument should be the comment string (spaces are allowed)
        u -> username. 2nd argument should be the username.
    '''
    args = sys.argv[1:]
    if(args[0]) == 'c':
        comments = [' '.join(args[1:])]
        print('Making prediction for the comment:',comments[0])
        
    if(args[0]) == 'u':
        username = ' '.join(args[1:])
        print('Fetching comments for user:',username)
        comments = [] #fetch comments of user
        if(len(comments) == 0):
            print('No comments found. Exiting')
            exit()
        print('Making predictions for',len(comments),'comments:')
        
    predictions = makePredictions(comments)
    print(predictions)