from joblib import load
from Tokenizer import Tokenizer
from Stemmer import Stemmer
from Vectorizer import Vectorizer
import numpy as np
import pandas as pd
from Downloader import Downloader
from Model import Model
import sys

politicalSubreddits = [
    "Democrats",
    "Republican",
    "WorldNews",
    "News",
    "Business",
    "Economics",
    "Environment",
    "Energy",
    "Law",
    "Education",
    "Government",
    "Wikileaks",
    "SOPA",
    "Libertarian",
    "Anarchism",
    "Socialism",
    "Progressive",
    "Conservative",
    "Liberal",
    "Egalitarian",
    "DemocraticSocialism",
    "Republicans",
    "Egalitarianism",
    "AnarchaFeminism",
    "Communist",
    "Conspiracy",
    "USpolitics",
    "JoeBiden",
    "Trump"
]
politicalSubreddits = [x.lower() for x in politicalSubreddits]
politicalSubredditsOnly = True

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

    downloader = Downloader(save_local=False,max_comment_count=999)
    res = downloader.fetch_subreddit('user',username)

    data = res[['subreddit','score','body']]
    if(politicalSubredditsOnly):
        data = data[data.apply(lambda x: (x['subreddit']).lower() in politicalSubreddits, axis = 1)]

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

    res['demPredComments'] = (data[data['predictions']==0]['body']).tolist()
    res['repPredComments'] = (data[data['predictions']==1]['body']).tolist()

    print('Total classified as Democrat:',res['predDem'])
    print('Total classified as Republican:',res['predRep'])
    print('Total posts in r/democrat',res['countDem'])
    print('Total posts in r/republican',res['countRep'])

    repData = data[data['subreddit']=='Republican']
    demData = data[data['subreddit']=='democrats']
    
    labels = []
    predictions = []
    for _,x in demData.iterrows():
        labels.append(0)
        predictions.append(x['predictions'])
    for _,x in repData.iterrows():
        labels.append(1)
        predictions.append(x['predictions'])

    m = Model()
    correctPreds = list(np.array(labels)==np.array(predictions)).count(True)
    print('Correcly Predicted %i/%i comments in r/democrats or r/republican subreddits'%(correctPreds,len(predictions)))
    if(len(predictions) > 0):
        res['valAcc'] = '%0.2f'%(correctPreds/len(predictions))
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
True Democrat: walter1950, TrumpizzaTraitor, Juvisy7
True Republican: imsquidward4032, IBiteYou
'''