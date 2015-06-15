# -*- coding: utf-8 -*-
"""
Created on Fri May  1 03:00:59 2015

@author: surat_000
"""
from sklearn.datasets import load_files

#categories = ['alt.atheism', 'soc.religion.christian',
 #             'comp.graphics', 'sci.med']
#train_path = "C:\\Users\\surat_000\\Downloads\\scikit-learn-0.15.2\\doc\\tutorial\\text_analytics\\data\\twenty_newsgroups\\20news-bydate-train\\"
#test_path = "C:\\Users\\surat_000\\Downloads\\scikit-learn-0.15.2\\doc\\tutorial\\text_analytics\\data\\twenty_newsgroups\\20news-bydate-test\\"


categories = ['G06F3', 'G06F5', 'G06F7', 'G06F15', 'G06F17']

global_path = "C:\\Users\\surat_000\\Documents\\Visual Studio 2013\\Projects\\searchDB_CS\\searchDB_CS\\bin\\Debug\\"
global_path = global_path + "text-data-G06F-20000-1\\"
train_path = global_path + "train\\"
test_path = global_path +  "test\\"


#stop_words_list = ['it', 'a', 'is', ]

#Trying to use PorterStemmer
import nltk
from nltk import word_tokenize          
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import EnglishStemmer

#######
# based on http://www.cs.duke.edu/courses/spring14/compsci290/assignments/lab02.html
stemmer = EnglishStemmer()
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

import string
import unicodedata
import re
def tokenize(text):
#    text = ''.join(ch for ch in unicodedata.normalize('NFD', text.lower()) if unicodedata.category(ch) != 'Mn')
    text = "".join([ch for ch in text if ch not in (string.punctuation + '“'+ '”' + string.digits)])
    tokens = nltk.word_tokenize(text)
    print(text)
    stems = stem_tokens(tokens, stemmer)
    return stems
######## 

#Trying to use Lemmatizer
#from nltk import word_tokenize          
#from nltk.stem import WordNetLemmatizer 
#class LemmaTokenizer(object):
#    def __init__(self):
#        self.wnl = WordNetLemmatizer()
#    def __call__(self, doc):
#        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


twenty_train = load_files(train_path, None, categories)
print("Train set is loaded")

_ = twenty_train.target_names
print(_)

#Tokenizing text with scikit-learn

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(decode_error='ignore', stop_words='english', tokenizer=tokenize)
X_train_counts = count_vect.fit_transform(twenty_train.data)
X_train_counts.shape


#From occurrences to frequencies

from sklearn.feature_extraction.text import TfidfTransformer
#tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
#X_train_tf = tf_transformer.transform(X_train_counts)
#_ = X_train_tf.shape
#print(_)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
_ = X_train_tfidf.shape
print(_)

#Training a classifier
from sklearn.naive_bayes import MultinomialNB
#from sklearn.naive_bayes import GaussianNB
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)
#clf_gauss = GaussianNB().fit(X_train_tfidf.toarray(), twenty_train.target)

docs_new = ['God is love', 'OpenGL on the GPU is fast']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

print("Multinomial NB")
predicted = clf.predict(X_new_tfidf)
for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, twenty_train.target_names[category]))

'''				
print("Gaussian NB")
predicted = clf_gauss.predict(X_new_tfidf.toarray())
for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, twenty_train.target_names[category]))
'''

#Building a pipeline
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
text_clf = Pipeline([('vect', CountVectorizer(decode_error='ignore', stop_words='english', tokenizer=tokenize)),
                     ('chi2', SelectKBest(chi2, k=1000)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),
])

text_clf = text_clf.fit(twenty_train.data, twenty_train.target)

#Evaluation of the performance on the test set

import numpy as np
twenty_test = load_files(test_path, None, categories)
print("Test set is loaded")


docs_test = twenty_test.data
predicted = text_clf.predict(docs_test)
_ = np.mean(predicted == twenty_test.target)
print(_)

from sklearn import metrics
print(metrics.classification_report(twenty_test.target, predicted,
    target_names=twenty_test.target_names))
_ = metrics.confusion_matrix(twenty_test.target, predicted)
print(_)

from sklearn.linear_model import SGDClassifier
text_clf = Pipeline([('vect', CountVectorizer(decode_error='ignore', stop_words='english', tokenizer=tokenize)),
                     ('chi2', SelectKBest(chi2, k=1000)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                           alpha=1e-3, n_iter=5, random_state=42)),
])
_ = text_clf.fit(twenty_train.data, twenty_train.target)
predicted = text_clf.predict(docs_test)
_ = np.mean(predicted == twenty_test.target)
print(_)

from sklearn import metrics
print(metrics.classification_report(twenty_test.target, predicted,
    target_names=twenty_test.target_names))
_ = metrics.confusion_matrix(twenty_test.target, predicted)
print(_)

'''
text_clf = Pipeline([('vect', CountVectorizer(decode_error='ignore', stop_words='english', tokenizer=tokenize)),
                     ('chi2', SelectKBest(chi2, k=1000)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', BernoulliNB()),
])

text_clf = text_clf.fit(twenty_train.data, twenty_train.target)
_ = text_clf.fit(twenty_train.data, twenty_train.target)
predicted = text_clf.predict(docs_test)
_ = np.mean(predicted == twenty_test.target)
print(_)

from sklearn import metrics
print(metrics.classification_report(twenty_test.target, predicted,
    target_names=twenty_test.target_names))
_ = metrics.confusion_matrix(twenty_test.target, predicted)
print(_)
'''
'''
from sklearn.svm import NuSVC
pipeline = Pipeline([('vect', CountVectorizer(decode_error='ignore', stop_words='english', tokenizer=tokenize)),
                     ('chi2', SelectKBest(chi2, k=1000)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', NuSVC()),
])

from nltk.classify.scikitlearn import SklearnClassifier
classif = SklearnClassifier(pipeline)

text_clf = classif.train([twenty_train.data, twenty_train.target])
#_ = text_clf.fit(twenty_train.data, twenty_train.target)
#predicted = text_clf.predict(docs_test)
#_ = np.mean(predicted == twenty_test.target)
#print(_)
'''
'''
from sklearn import metrics
print(metrics.classification_report(twenty_test.target, predicted,
    target_names=twenty_test.target_names))
_ = metrics.confusion_matrix(twenty_test.target, predicted)
print(_)
'''