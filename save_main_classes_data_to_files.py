# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 23:20:57 2015

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
		x = stemmer.stem(item)
		if len(str(x)) > 2:
			stemmed.append(x)
	return stemmed

import string
def tokenize(text):
	#Delete punctuation
	special_ch = '“'+ '”' + '°' +'±' + 'µ' + '¼' + '½' + 'å' + '–' + '’' + '‘' + '®' + '¾' + '˜' + '®' + '™'
	special_ch += '§' + '·' + '—'
	special_ch = special_ch + string.punctuation + string.digits
	replace_punctuation = str.maketrans(special_ch, ' '*len(special_ch))
	text = text.translate(replace_punctuation)
	tokens = nltk.word_tokenize(text)
				
	stems = stem_tokens(tokens, stemmer)
	return stems
##########

global_path = "C:\\Users\\surat_000\\Documents\\Visual Studio 2013\\Projects\\"
global_path += "searchDB_CS\\searchDB_CS\\bin\\Debug\\text-data-G06F-20000-CV-1000\\"

step_path = "step-" + str(1) + "\\"
#print("We're in {0}".format(step_path))

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

#train = "train\\"
#test = "test\\"


train = "test\\"
test = "train\\"


ch2_fn = "ch2.dat"
text_clf_sgd_fn = "text_clf_sgd.dat"

report_fn = "report_training_.log"

categories = None

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


pth = global_path + step_path
report_dest = pth + report_fn
categories = None

reporting('=' * 80, report_dest)
reporting("We're in {0}".format(pth), report_dest)
reporting("", report_dest)

t0 = time()	
train_data = load_files(pth + train, None, categories)	
train_load_time = time() - t0
reporting("train load time: %0.3fs" % train_load_time, report_dest)

pickle.dump(train_data, open(pth+ train_fn, "wb"))

t0 = time()	
test_data = load_files(pth + test, None, categories)	
test_load_time = time() - t0
reporting("test load time: %0.3fs" % test_load_time, report_dest)

pickle.dump(test_data, open(pth+ test_fn, "wb"))


data_train_size_mb = size_mb(train_data.data)
data_test_size_mb = size_mb(test_data.data)

categories = train_data.target_names

reporting("", report_dest)
reporting("%d documents - %0.3fMB (training set)" % (len(train_data.data), data_train_size_mb), report_dest)
reporting("%d documents - %0.3fMB (test set)" % (len(test_data.data), data_test_size_mb), report_dest)
reporting("%d categories" % len(categories), report_dest)
reporting("", report_dest)

y_train, y_test = train_data.target, test_data.target

n_gram_range = (1,1)
count_vectorizer = CountVectorizer(decode_error='ignore', stop_words='english', tokenizer=tokenize, ngram_range=n_gram_range)

reporting("Extracting features from the training data using a sparse vectorizer", report_dest)
t0 = time()	
X_train_counts = count_vectorizer.fit_transform(train_data.data)
duration = time() - t0
reporting("done in %fs at %0.3fMB/s" % (duration, data_train_size_mb / duration), report_dest)
reporting("n_samples: %d, n_features: %d" % X_train_counts.shape, report_dest)
reporting("", report_dest)

reporting("Extracting features from the test data using the same vectorizer", report_dest)
t0 = time()
X_test_counts = count_vectorizer.transform(test_data.data)
duration = time() - t0
reporting("done in %fs at %0.3fMB/s" % (duration, data_test_size_mb / duration), report_dest)
reporting("n_samples: %d, n_features: %d" % X_test_counts.shape, report_dest)
reporting("", report_dest)

pickle.dump(count_vectorizer, open(pth + count_vect_fn, "wb"))
pickle.dump(X_train_counts, open(pth + X_train_counts_fn, "wb"))
pickle.dump(X_test_counts, open(pth + X_test_counts_fn, "wb"))

tfidf_transformer = TfidfTransformer()

reporting("Extracting features from the training data using a tfidf_transformer", report_dest)
t0 = time()	
X_train = tfidf_transformer.fit_transform(X_train_counts)
duration = time() - t0
reporting("done in %fs" % (duration), report_dest)
reporting("n_samples: %d, n_features: %d" % X_train.shape, report_dest)
reporting("", report_dest)

reporting("Extracting features from the test data using the same tfidf_transformer", report_dest)
t0 = time()
X_test = tfidf_transformer.transform(X_test_counts)
duration = time() - t0
reporting("done in %fs" % (duration), report_dest)
reporting("n_samples: %d, n_features: %d" % X_test.shape, report_dest)
reporting("", report_dest)


pickle.dump(tfidf_transformer, open(pth + tfidf_trans_fn, "wb"))
pickle.dump(X_train, open(pth + X_train_fn, "wb"))
pickle.dump(X_test, open(pth + X_test_fn, "wb"))

X_train_ch2	= None
X_test_ch2 = None

feature_names = count_vectorizer.get_feature_names()
feature_names = np.asarray(feature_names)

'''
k_best = 1000
ch2 = SelectKBest(chi2, k=k_best)

reporting("Extracting %d best features by a chi-squared test" % k_best, report_dest)
t0 = time()
X_train_ch2 = ch2.fit_transform(X_train, y_train)		
X_test_ch2 = ch2.transform(X_test)
duration = time() - t0
reporting("done in %fs" % (duration), report_dest)
reporting("", report_dest)
	
pickle.dump(ch2, open(pth + ch2_fn, "wb"))

# keep selected feature names
feature_names = [feature_names[i] for i in ch2.get_support(indices=True)]
'''	

X_train_ch2	= X_train
X_test_ch2 = X_test
	
pickle.dump(X_train_ch2, open(pth + X_train_ch2_fn, "wb"))
pickle.dump(X_test_ch2, open(pth + X_test_ch2_fn, "wb"))

__penalty = 'elasticnet'
__n_iter = 10

reporting('_' * 80, report_dest)	
text_clf_sgd = SGDClassifier(penalty=__penalty, alpha=1e-4, n_iter=__n_iter)
reporting(text_clf_sgd, report_dest)	

reporting("Training: ", report_dest)
t0 = time()
text_clf_sgd = text_clf_sgd.fit(X_train_ch2, y_train)
train_time = time() - t0
reporting("train time: %0.3fs" % train_time, report_dest)

pickle.dump(text_clf_sgd, open(pth + text_clf_sgd_fn, "wb"))	

reporting("Predicting: ", report_dest)
t0 = time()
predicted = text_clf_sgd.predict(X_test_ch2)
test_time = time() - t0
reporting("test time:  %0.3fs" % test_time, report_dest)


score = metrics.accuracy_score(y_test, predicted)
reporting("accuracy:   %0.3f" % score, report_dest)

reporting("Evaluating: ", report_dest)
reporting(metrics.classification_report(test_data.target, predicted, target_names=test_data.target_names), report_dest)
_ = metrics.confusion_matrix(test_data.target, predicted)
reporting(_, report_dest)

feature_names = np.asarray(feature_names)

for i, category in enumerate(categories):
	top10 = np.argsort(text_clf_sgd.coef_[i])[-10:]
	reporting("%s: %s"  % (category, " ".join(feature_names[top10])), report_dest)