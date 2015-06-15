# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 08:14:41 2015

@author: surat_000
"""

#########
#Using English Snowball Stemmer
import nltk
from nltk import word_tokenize
from nltk.stem.snowball import EnglishStemmer

stemmer = EnglishStemmer()
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

import string
def tokenize(text):
    #Delete punctuation
    special_ch = '“'+ '”' + '°' +'±' + 'µ' + '¼' + '½' + 'å' + '–' + '’' + '‘' + '®' + '¾' + '˜' + '®' + '™'
    special_ch = special_ch + string.punctuation + string.digits
    replace_punctuation = str.maketrans(special_ch, ' '*len(special_ch))
    text = text.translate(replace_punctuation)
    tokens = nltk.word_tokenize(text)
#    print(text)
    stems = stem_tokens(tokens, stemmer)
    return stems
##########

global_path = "C:\\Users\\surat_000\\Documents\\Visual Studio 2013\\Projects\\"
global_path += "searchDB_CS\\searchDB_CS\\bin\\Debug\\"

train_fn = "train.dat"
test_fn = "test.dat"

count_vect_fn = "count_vect.dat"
X_train_counts_fn = "X_train_counts.dat"
X_test_counts_fn = "X_test_counts.dat"

tfidf_trans_fn = "tfidf_trans.dat"
X_train_fn = "X_train.dat"
X_test_fn = "X_test.dat"

X_train_ch2_fn = "X_train_ch2.dat"
X_test_ch2_fn = "X_test_ch2.dat"

train = "train\\"
test = "test\\"

ch2_fn = "ch2.dat"
text_clf_sgd_fn = "text_clf_sgd.dat"

G06F = "text-data-G06F-20000-CV-5000\\step-3\\"
G06F3 = "text-data-G06F3-regen\\"
G06F7 = "text-data-G06F7-regen-2\\"
G06F15 = "text-data-G06F15-regen-3\\"
G06F17 = "text-data-G06F17-regen\\"

report_fn = "report_training_.log"

def size_mb(docs):
	return sum(len(s) for s in docs) / 1e6
				
def reporting(string, dest_filename):
	print(string)
	dest_file = open(dest_filename, "a", encoding="utf-8")
	print(string, file=dest_file)
	dest_file.close()	
	return

import pickle

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import SGDClassifier

from time import time

import numpy as np
from sklearn import metrics

import ntpath

main_clf_path = global_path + G06F
main_clf_train = pickle.load(open(main_clf_path + train_fn, "rb"))
main_clf_test = pickle.load(open(main_clf_path + test_fn, "rb"))
main_clf_vectorizer = pickle.load(open(main_clf_path + count_vect_fn, "rb"))
main_clf_tfidf_trasformer = pickle.load(open(main_clf_path + tfidf_trans_fn, "rb"))
main_clf_ch2 = pickle.load(open(main_clf_path + ch2_fn, "rb"))
main_clf_X_train = pickle.load(open(main_clf_path + X_train_ch2_fn, "rb"))
main_clf_X_test = pickle.load(open(main_clf_path + X_test_ch2_fn, "rb"))
main_clf = pickle.load(open(main_clf_path + text_clf_sgd_fn, "rb"))

doc = main_clf_test.data
main_clf_X_new_counts = main_clf_vectorizer.transform(doc)
main_clf_X_new_tfidf = main_clf_tfidf_trasformer.transform(main_clf_X_new_counts)
main_clf_X_new_tfidf_ch2 = main_clf_ch2.transform(main_clf_X_new_tfidf)
#main_clf_predicted = main_clf.predict(main_clf_X_new_tfidf_ch2)

pipeline = Pipeline([
	('vect', main_clf_vectorizer),
	('chi2', main_clf_ch2),
	('tfidf', main_clf_tfidf_trasformer),
	('clf', main_clf),
])


main_clf_predicted = pipeline.predict(main_clf_test.data)

print("Evaluation")
_ = np.mean(main_clf_predicted == main_clf_test.target)
print(_)

print(metrics.classification_report(main_clf_test.target, main_clf_predicted,
    target_names=main_clf_test.target_names))
_ = metrics.confusion_matrix(main_clf_test.target, main_clf_predicted)
print(_)
