import numpy as np
from scipy import int32
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import nltk.stem

class Data(object):
    '''
    Class to load and pre-process paragraphs from a json file
    for the document identification task.
    '''
    def __init__(self, paragraphs, max_features=3000, lemmatize=True):
        '''
        Constructor
        :param paragraphs: list of paragraphs
        :param max_features: max number of words in the vocabulary
        :param lemmatize: flag to determine if words should be lemmatized
        '''
        self.load_doc(paragraphs, max_features, lemmatize)
    
    def process_doc(self, paragraphs):
        rho = []
        sents = []
        for paragraph in paragraphs:
            sents.append(paragraph["paragraph"])
            doc_start = -1
            if "document_start" in paragraph:
                doc_start = int(paragraph["document_start"])
            rho.append(doc_start)
        rho[0] = 0
        return rho, sents
                       
    def load_doc(self, doc_path, max_features, lemmatize):
        self.rho, sents = self.process_doc(doc_path)
        self.n_sents = len(sents)
        self.docs_index = [self.n_sents] #Assumes a single file for all paragraphs
        self.rho_eq_1 = np.append(np.nonzero(self.rho)[0], [self.n_sents-1])
        self.n_segs = len(self.rho_eq_1)
        
        if lemmatize:
            vectorizer = ENLemmatizerCountVectorizer(stopwords.words("english"),
                                                     max_features=max_features)
        else:
            vectorizer = CountVectorizer(analyzer = "word",\
                                         strip_accents = "unicode",\
                                         stop_words = stopwords.words("english"),\
                                         max_features=max_features)
            
        self.U_W_counts = vectorizer.fit_transform(sents).toarray()
        self.vocab = vectorizer.vocabulary_
        self.inv_vocab =  {v: k for k, v in self.vocab.items()}
        self.W = len(self.vocab)
        self.sents_len = np.sum(self.U_W_counts, axis=1)
        self.U_I_words = np.zeros((self.n_sents, max(self.sents_len)), dtype=int32)
        self.W_I_words = []
        self.d_u_wi_indexes = []
        
        analyzer = vectorizer.build_analyzer()
        word_count = 0
        doc_i_u = []
        for u_index, u in enumerate(sents):
            u_w_indexes = []
            u_I = analyzer(u)
            i = 0
            for w_ui in u_I:
                if w_ui in self.vocab:
                    u_w_indexes.append(word_count)
                    word_count += 1
                    vocab_index = self.vocab[w_ui]
                    self.U_I_words[u_index, i] = vocab_index
                    self.W_I_words.append(vocab_index)
                    i += 1
                    
            if len(u_w_indexes) > 0:
                doc_i_u.append(u_w_indexes)
            if u_index+1 in self.docs_index:
                self.d_u_wi_indexes.append(doc_i_u)
                doc_i_u = []
        self.W_I_words = np.array(self.W_I_words)
                                
class ENLemmatizerCountVectorizer(CountVectorizer):
    def __init__(self, stopwords_list=None, max_features=None):
        CountVectorizer.__init__(self,analyzer="word",\
                                 strip_accents="unicode",\
                                 stop_words=stopwords_list,\
                                 max_features=max_features)
        self.en_lemmatizer = nltk.stem.WordNetLemmatizer()
        
    def build_analyzer(self):
        analyzer = super(ENLemmatizerCountVectorizer, self).build_analyzer()
        return lambda doc: ([self.en_lemmatizer.lemmatize(w) for w in analyzer(doc)])
