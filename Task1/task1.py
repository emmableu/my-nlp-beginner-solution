import os
import torch
import numpy as np
import pandas as pd
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from scipy.sparse import hstack
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

dir_all_data = 'data/task1_all_data.tsv'
data_all = pd.read_csv(dir_all_data, sep='\t')
x_all = data_all['Phrase']
y_all = data_all['Sentiment']
# print(x_all.head(), y_all.head())
x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2)

"""接下来要提取几个特征：文本计数特征、word级别的TF-IDF特征、ngram级别的TF-IDF特征"""
# 提取文本计数特征 -- 每个单词的数量
# 对文本的单词进行计数，包括文本的预处理, 分词以及过滤停用词
count_transformer = CountVectorizer().fit(x_train)
x_train_count = count_transformer.transform(x_train)
x_test_count = count_transformer.transform(x_test)
print(x_train_count.shape, x_test_count.shape)
# 在词汇表中一个单词的索引值对应的是该单词在整个训练的文集中出现的频率。
# print(count_vect.vocabulary_.get(u'good'))    #5812     count_vect.vocabulary_是一个词典：word-id


# 提取TF-IDF特征 -- word级别的TF-IDF
# 将各文档中每个单词的出现次数除以该文档中所有单词的总数：这些新的特征称之为词频tf。
tfidf_transformer = TfidfVectorizer(analyzer='word').fit(x_train)
x_train_tfidf = tfidf_transformer.transform(x_train)
x_test_tfidf = tfidf_transformer.transform(x_test)
print(x_train_tfidf.shape, x_test_tfidf.shape)

# 提取TF-IDF特征 - ngram级别的TF-IDF
# 将各文档中每个单词的出现次数除以该文档中所有单词的总数：这些新的特征称之为词频tf。
ngram_tfidf_transformer = TfidfVectorizer(analyzer='word', ngram_range=(2, 3), max_features=50000)
ngram_tfidf_transformer.fit(x_train)
x_train_tfidf_ngram = ngram_tfidf_transformer.transform(x_train)
x_test_tfidf_ngram = ngram_tfidf_transformer.transform(x_test)
print(x_train_tfidf_ngram.shape, x_test_tfidf_ngram.shape)

x_train = hstack([x_train_count, x_train_tfidf, x_train_tfidf_ngram])
x_test = hstack([x_test_count, x_test_tfidf, x_test_tfidf_ngram])

model = SGDClassifier().fit(x_train, y_train)
y_pred = model.predict(x_test)
print(confusion_matrix(y_test, y_pred))
print(precision_recall_fscore_support(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
# tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
# print(tn, fp, fn, tp)