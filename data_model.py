"""
数据预处理代码，
1. 空格切分句子变为词列表
2. 将句子中的一些标点符号去掉，如 [?!'"#].,)|(|
3. 将句子中的停用词去掉，就是一些高频词，但是没有什么意义，比如 a the 之类
去除这些词和标点的原因是，这些东西对句子的语义没有什么影响，但是又高频出现，相当于noise。

"""

import pandas as pd
import numpy as np
import os
import re 
import nltk

class DataModel:

    def __init__(self):
        self.stop_words = set(['?',',',':','@','is','the', 'it',';'])
        self.path = ['/Users/zhuxiao/Desktop/Project Center/QA system-CSI5180/ExtendedData.xlsx']
        self.df = self.load_df()  # dataframe
        self.rows = self.df.shape[0]
        self.cols = self.df.shape[1]
        self.final_df()    

    def load_df(self):
        dataframe = [pd.read_excel(p, names = ['q','a']) for p in self.path]
        temp_df = pd.DataFrame()
        for d in dataframe:
            if d.shape[0]>0:
                temp_df = pd.concat([temp_df,d])
        return temp_df
    
    def text_preprocessor(self,q_a):
        temp_list = []
        stemmer = nltk.stem.SnowballStemmer('english')
        for sentence in q_a:
            sentence = str(sentence)
            sentence = sentence.lower()                 # Converting to lowercase
            cleanr = re.compile('<.*?>')
            sentence = re.sub(cleanr, ' ', sentence)        #Removing HTML tags
            sentence = re.sub(r'[?|!|\'|"|#]\d',r'',sentence)
            sentence = re.sub(r'[.|,|)|(|\|/]@',r' ',sentence)        #Removing Punctuations
            words = [stemmer.stem(word) for word in sentence.split() if word not in self.stop_words]   # Stemming and removing stopwords
            temp_list.append(words)
        q_a = temp_list  
        return self.clean(q_a)

    def clean(self, rows):
        temp_list = []
        for row in rows:
            sequ = ''
            for word in row:
                sequ = sequ + ' ' + word
            temp_list.append(sequ)
        rows = temp_list
        return rows

    def combine_cols(self, l1, l2):
        r = []
        for q, a in  zip(l1, l2):
            r.append(q+" "+a)
        return r
    
    def final_df(self):
        q_new =  self.text_preprocessor(self.df['q'])
        a_new =  self.text_preprocessor(self.df['a'])
        qa_combined = self.combine_cols(q_new, a_new)
        self.df['q_new'] =  q_new
        self.df['a_new'] =  a_new
        #if self.combined:
        self.df['qa_combined'] = qa_combined  
   
    def X_train(self):
        #if self.combined:
        return [list(self.df['qa_combined'].values), list(self.df['q_new'].values)] # q & qa combined
   
    def getQA(self):
        qa_dict = {}
        for i, (q , a) in enumerate(zip(self.df['a'], self.df['q'])):
            qa_dict[i] = [q, a]  
        return qa_dict                  
