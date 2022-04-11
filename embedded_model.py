from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity 
from collections import Counter 
from data_model import DataModel
import numpy as np
class W2Vec:

    def __init__(self):
        self.data            = DataModel()
        self.xtrain_q        = self.input_feature(self.data.X_train()[0])   
        print(len(self.xtrain_q))
        self.xtrain_qa       = self.input_feature(self.data.X_train()[1]) 
        self.models          = [Word2Vec(self.xtrain_q, min_count = 1, size = 100, workers = 4),
                                Word2Vec(self.xtrain_qa, min_count = 1, size = 100, workers = 4)
                               ]
        self.train_wv        = [np.array([self.word_vectors(q, self.models[0]) for q in self.xtrain_q]),
                                np.array([self.word_vectors(q, self.models[1]) for q in self.xtrain_qa])
                                ] 
        #print(self.train_wv[0])
        self.qa              = self.data.getQA()
        
    def word2vec_data(self, data):
        return[[word for word in row.split()] for row in data]
    
    def input_feature(self, q):
        q = [q]
        q = self.data.text_preprocessor(q)
        q = self.word2vec_data(q)
        return q
    
    def word_vectors(self, q, model, test = False):
        if test:
            q = self.input_feature(q)
        wv = []       
        for row in q:
            vec = np.zeros(100)
            count = 0
            for word in row:
                try:                        
                    vec += model[word]
                    count += 1
                except:
                    pass               
            wv.append(vec/count)
        return np.array(wv)
    
    def predict(self, q):
        answers   = []
        for index, vectors  in enumerate(self.train_wv):   
            print(vectors.shape)
            #print([self.train_wv[0][i] for i in range(self.data.rows)])
            
            #rank = np.array([cosine_similarity(self.word_vectors(q, self.models[index], test = True), vectors[i].reshape(1,100)) for i in range(self.data.rows)])       
            #answers.append(np.argmax(rank))        
        #majority = Counter(answers).most_common()[0][0]
        #return self.qa[majority][0]
                                
m = W2Vec()
print(m.predict("coronavirus"))
