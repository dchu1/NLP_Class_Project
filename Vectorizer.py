'''
Modes:
    TFIDF
    Count
    LDA
'''
class Vectorizer:
    vec = None
    mode = None

    def __init__(self, mode = 'TFIDF', ldaSplits = None, maxFeatures=None):
        if(mode == 'TFIDF'):
            from sklearn.feature_extraction.text import TfidfVectorizer
            if(maxFeatures is None):
                self.vec = TfidfVectorizer()
            else:
                self.vec = TfidfVectorizer(max_features=maxFeatures)
        elif(mode == 'Count'):
            from sklearn.feature_extraction.text import CountVectorizer
            self.vec = CountVectorizer()
        elif(mode == 'LDA'):
            from sklearn.feature_extraction.text import CountVectorizer
            from sklearn.decomposition import LatentDirichletAllocation
            self.preVec = CountVectorizer()
            print('Initializing LDA with',ldaSplits,'splits')
            self.vec = LatentDirichletAllocation(n_components=ldaSplits)

        self.mode = mode

    def __str__(self):
        return('Custom Vectorizer using '+self.mode+' Vectorization')

    def features(self):
        return(self.vec.get_feature_names())

    def fit(self, data):
        if(self.mode == 'LDA'):
            print('Generating LDA Vector')
            data = self.preVec.fit_transform(data)
        res = self.vec.fit(data)
        from joblib import dump
        dump(self.vec, 'saved/vectorizer.sav')
        return(res)

    def fitTransform(self, data):
        if(self.mode == 'LDA'):
            print('Generating LDA Vector')
            data = self.preVec.fit_transform(data)
        res = self.vec.fit_transform(data)
        from joblib import dump
        dump(self.vec, 'saved/vectorizer.sav')
        return(res)

    def transform(self, data):
        if(self.mode == 'LDA'):
            data = self.preVec.transform(data)
        return(self.vec.transform(data))