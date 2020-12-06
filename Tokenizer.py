import pdb
'''
Modes:
    WordPunct 'won’t' = ['won', "'", 't']
    Treebank 'won’t' = ['wo', "n't"]

https://www.tutorialspoint.com/natural_language_toolkit/natural_language_toolkit_tokenizing_text.htm

Tokenizing may require the nltk punkt lib:
import nltk
nltk.download('punkt')
'''
class Tokenizer:
    tok = None
    mode = None
    stopwords = None

    def __init__(self, mode = 'WordPunct', stopwords = True):
        if(mode == 'WordPunct'):
            from nltk.tokenize import WordPunctTokenizer
            self.tok = WordPunctTokenizer()
        elif(mode == 'Treebank'):
            from nltk.tokenize import TreebankWordTokenizer
            self.vec = TreebankWordTokenizer()
        elif(mode == 'Regexp'):
            raise Exception("TODO")
        self.mode = mode

        if(stopwords):
            from nltk.corpus import stopwords
            self.stopwords = stopwords.words('english')

    def __str__(self):
        return('Custom Vectorizer using '+self.mode+' Vectorization')

    def fitTransform(self, data):
        return self.transform(data)

    def transform(self, data):
        tok = [self.tok.tokenize(doc) for doc in data]
        if(self.stopwords):
            tok = self._removeStopwords(tok)
        return tok

    def _removeStopwords(self, data):
        return([[word for word in doc if word not in self.stopwords] for doc in data])