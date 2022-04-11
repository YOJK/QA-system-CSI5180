"""
step1 : 将文本转为特征向量
用到两个接口 CountVectorizer, TfidfVectorizer。
前者将文档转为一系列对词的词频的统计矩阵， 后者将文档转为一些列词的TF-IDF值的统计矩阵。
    Examples
    --------
    >>> corpus = [
    ...     'This is the first document.',
    ...     'This document is the second document.',
    ...     'And this is the third one.',
    ...     'Is this the first document?',
    ... ]
    >>> vectorizer = CountVectorizer()
    >>> X = vectorizer.fit_transform(corpus)
    >>> print(vectorizer.get_feature_names())
    ['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']
    >>> print(X.toarray())  # doctest: +NORMALIZE_WHITESPACE
    [[0 1 1 1 0 0 1 0 1]
     [0 2 0 1 0 1 1 0 1]
     [1 0 0 1 1 0 1 1 1]
     [0 1 1 1 0 0 1 0 1]]

    >>> vectorizer = TfidfVectorizer()
    >>> X = vectorizer.fit_transform(corpus)
    >>> print(vectorizer.get_feature_names())
    ['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']
    >>> print(X.shape)
    (4, 9)

step2 : 将query的句子向量和已经收集的语料的query向量做比较
这里使用余弦相似度，找到最相近的query，并将与之配对的 answer 返回。

"""
import csv
import numpy as np
from sklearn.feature_extraction import stop_words
import os
from nltk.corpus import stopwords 
from sklearn.metrics.pairwise import cosine_similarity 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from collections import Counter
from data_model import *


class Models:
    def __init__(self):
        self.data      = DataModel()
        self.xtrain_q  = self.data.X_train()[0]    
        self.xtrain_qa = self.data.X_train()[1]       
        self.qa        = self.data.getQA()
        self.q_models   = [ #CountVectorizer(), # bow with qa                           
                          TfidfVectorizer(ngram_range=(1,2))                          
                          ]
        self.qa_models  = [ CountVectorizer(),                    
                           TfidfVectorizer(ngram_range=(1,2)), 
                           
                          ]
        self.features_q  =  [m.fit_transform(self.xtrain_q) for m in self.q_models]
        self.features_qa =  [m.fit_transform(self.xtrain_qa) for m in self.qa_models]
    
    def get_q_models(self):
        return self.q_models
        
    def get_qa_models(self):
        return self.qa_models

    def get_q_features (self):
        return self.features_q
    
    def get_qa_features(self):
        return self.features_q

    # 计算句子的特征向量
    def input_feature(self , q):
        q = [q]
        q = self.data.text_preprocessor(q)
        return [m.transform(q) for m in self.q_models+self.qa_models]

    def predict(self,q):
        q_transform  = self.input_feature(q)
        answers   = []
        # 计算相似度
        for model_id, f in enumerate(self.features_q+self.features_qa):
            rank = np.array([cosine_similarity(q_transform[model_id], f[i]) for i in range(self.data.rows)])       
            answers.append(np.argmax(rank))        
        majority = Counter(answers).most_common()[0][0]

        # 返回最相近的句子
        return self.qa[majority][0]

    def predict_test(self,filename):
        with open(filename, 'rt', encoding='GBK') as f:
            data_corpus = csv.reader(f)
            data = list(data_corpus)
            row_count = len(data) - 1
        f.close()
        count = 0
        with open(filename, 'rt', encoding='GBK') as f:
            data_corpus = csv.reader(f)
            header = next(data_corpus)
            for words in data_corpus:
                q_transform  = self.input_feature(words[0])
                answers   = []
                # 计算相似度
                for model_id, f in enumerate(self.features_q+self.features_qa):
                    rank = np.array([cosine_similarity(q_transform[model_id], f[i]) for i in range(self.data.rows)])
                    answers.append(np.argmax(rank))
                majority = Counter(answers).most_common()[0][0]
                if self.qa[majority][0] == words[1]:
                    print('Current accuracy',count,'/200')
                    count = count + 1
        return count/row_count


if __name__ == "__main__":
    model = Models()
    print("Test in process")
    filename = 'TestData.csv'
    print("The accuracy of the system:",model.predict_test(filename)*100,'%')
