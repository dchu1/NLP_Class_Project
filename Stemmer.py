import pdb
'''
Modes:
    Porter
    Lancaster
    Lemmatize

    https://www.tutorialspoint.com/natural_language_toolkit/natural_language_toolkit_stemming_lemmatization.htm
'''
class Stemmer:
    stemmer = None
    mode = None
    join = True

    def __init__(self, mode = 'Porter', join = True):
        if(mode == 'Porter'):
            from nltk.stem import PorterStemmer
            self.stemmer = PorterStemmer()
        elif(mode == 'Lancaster'):
            from nltk.stem import LancasterStemmer
            self.stemmer = LancasterStemmer()
        elif(mode == 'Lemmatize'):
            from nltk.stem import WordNetLemmatizer
            self.stemmer = WordNetLemmatizer()
        elif(mode == 'Snowball'):
            raise Exception("TODO")
        elif(mode == 'Regexp'):
            raise Exception("TODO")
        self.mode = mode
        self.join = join

    def __str__(self):
        return('NLTK Stemmer using '+self.mode+' Stemming')

    # data is a list of strings or a list of list of strings
    # returns either a list of words or a joined list
    def fitTransform(self, data):
        return self.transform(data)


    # data is a list of list of words
    # returns either a list of words or a joined list
    def transform(self, data):
        # for each list of words in data list, lemmatize/stem each word
        if(self.mode == "Lemmatize"):
            result = [[self.stemmer.lemmatize(word) for word in doc] for doc in data]
        else:
            result = [[self.stemmer.stem(word) for word in doc] for doc in data]
        
        # if necessary join words in each list of words in result list
        if (self.join):
            result = [' '.join(doc) for doc in result]
        return result