from joblib import load
from Tokenizer import Tokenizer
from Stemmer import Stemmer
from Vectorizer import Vectorizer
import numpy as np
import pandas as pd
from Downloader import Downloader
from Model import Model
import sys

# Config: toke,stem-lemmatize,vect-tfidf,lr
tokenizer = Tokenizer()
stemmer = Stemmer('Lemmatize')
vectorizer = load('saved/vectorizer.sav')
model = load('saved/model.sav')

def makePredictions(predData):
    predData = vectorizer.transform(stemmer.transform((tokenizer.transform(np.array(predData)))))
    predictions = model.predict(predData)
    return(predictions)

def getCommentPreds(comment):
    print('Making prediction for the comment:',comment)
    predictions = makePredictions([comment])
    print(predictions)
    return(predictions)

def getUserPreds(username):
    print('Fetching comments for user:',username)
    comments = [] #fetch comments of user

    downloader = Downloader(save_local=False,max_comment_count=1000) #DEBUG
    res = downloader.fetch_subreddit('user',username)

    data = res[['subreddit','score','body']]
    comments = data['body'].to_numpy()
    if(len(comments) == 0):
        print('No comments found. Exiting')
        exit()
    print('Making predictions for',len(comments),'comments:')
    predictions = makePredictions(comments)
    data.insert(0,'predictions',predictions,True)

    res = {
        'predDem': list(predictions).count(0),
        'predRep':list(predictions).count(1),
        'countDem':len(data[data['subreddit']=='democrats']),
        'countRep': len(data[data['subreddit']=='Republican'])
    }

    print('Total classified as Democrat:',res['predDem'])
    print('Total classified as Republican:',res['predRep'])
    print('Total posts in r/democrat',res['countDem'])
    print('Total posts in r/republican',res['countRep'])

    repData = data[data['subreddit']=='Republican']
    demData = data[data['subreddit']=='democrats']
    
    demPredComments = []
    repPredComments = []
    labels = []
    predictions = []
    for _,x in demData.iterrows():
        labels.append(0)
        demPredComments.append(x['body'])
        predictions.append(x['predictions'])
    for _,x in repData.iterrows():
        labels.append(1)
        repPredComments.append(x['body'])
        predictions.append(x['predictions'])

    res['demPredComments'] = demPredComments
    res['repPredComments'] = repPredComments

    m = Model()
    correctPreds = list(np.array(labels)==np.array(predictions)).count(True)
    print('Correcly Predicted %i/%i comments in r/democrats or r/republican subreddits'%(correctPreds,len(predictions)))
    if(len(predictions) > 0):
        res['valAcc'] = correctPreds/len(predictions)
        print('The validation accuracy is:',res['valAcc'])
    return(res)

if __name__ == "__main__":
    '''
    Argument format: <mode> <input>
    mode:
        c -> comment. 2nd argument should be the comment string (spaces are allowed)
        u -> username. 2nd argument should be the username.
    examples:
        python predictions.py u IBiteYou
        python predictions.py u TrumpizzaTraitor
    '''
    args = sys.argv[1:]
    if(args[0]) == 'c':
        getCommentPreds(' '.join(args[1:]))
        
    if(args[0]) == 'u':
        getUserPreds(' '.join(args[1:]))
        
'''
Test usernames
True Democrat: TrumpizzaTraitor
True Republican: IBiteYou
'''