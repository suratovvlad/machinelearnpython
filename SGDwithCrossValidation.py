# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 18:19:34 2015

@author: surat_000
"""

from __future__ import print_function

from pprint import pprint
from time import time
import logging
print(__doc__)

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

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
#from sklearn.grid_search import GridSearchCV
import numpy as np
from sklearn import metrics

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

from sklearn.datasets import load_files
###############################################################################
# Load some categories from the training set
categories = ['G06F3', 'G06F7', 'G06F15', 'G06F17']
# Uncomment the following to do the analysis on all the categories
#categories = None



MAX_ = 20000
MAX_step = 2
MAX_train = MAX_/MAX_step

global_path = "C:\\Users\\surat_000\\Documents\\Visual Studio 2013\\Projects\\"
global_path += "searchDB_CS\\searchDB_CS\\bin\\Debug\\text-data-G06F-20000-CV-"
global_path += str(round(MAX_train)) + "\\"


report_file_name = global_path + "report_SGD_CV_max_step_" + str(MAX_step) + ".txt"
report_file = open(report_file_name, "w")


print("Loading patent dataset for categories:", file=report_file)
print(categories, file=report_file)

for step in range(0, MAX_step):
	print(file=report_file)
	print("We're in step {0}".format(str(step)), file=report_file)
	print("We're in step {0}".format(str(step)))
	print(file=report_file)
	step_path = global_path + "step-" + str(step) + "\\"
	print("We're in {0}".format(step_path), file=report_file)
	
	train_path = step_path + "train\\"
	twenty_train = load_files(train_path, None, categories)
	print("Train set is loaded", file=report_file)
	print("Train set is loaded")
	print("{0} documents".format(len(twenty_train.filenames)), file=report_file)
	print("{0} categories".format(len(twenty_train.target_names)), file=report_file)
	print(file=report_file)
	
	
	test_path = step_path + "test\\"
	twenty_test = load_files(test_path, None, categories)
	docs_test = twenty_test.data
	print("Test set is loaded", file=report_file)
	print("Test set is loaded")
	print("{0} documents".format(len(twenty_test.filenames)), file=report_file)
	print("{0} categories".format(len(twenty_test.target_names)), file=report_file)
	print(file=report_file)
	
	###############################################################################
	# define a pipeline combining a text feature extractor with a simple
	# classifier
	pipeline = Pipeline([
		('vect', CountVectorizer(decode_error='ignore', stop_words='english', tokenizer=tokenize)),
		('chi2', SelectKBest(chi2, k=1000)),
		('tfidf', TfidfTransformer()),
		('clf', SGDClassifier(n_jobs = -1, verbose=1, loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)),
	])

	
	print("Fit", file=report_file)
	text_clf_sgd = pipeline.fit(twenty_train.data, twenty_train.target)
	
	print("Evaluation", file=report_file)
	predicted = text_clf_sgd.predict(docs_test)
	_ = np.mean(predicted == twenty_test.target)
	print("Accuracy: {0}".format(_), file=report_file)
	print("Accuracy: {0}".format(_))
	
	print(metrics.classification_report(twenty_test.target, predicted, target_names=twenty_test.target_names), file=report_file)
	_ = metrics.confusion_matrix(twenty_test.target, predicted)
	print(_, file=report_file)	
	
	print(file=report_file)
	print("END step {0}".format(str(step)), file=report_file)
	print("END step {0}".format(str(step)))
	print(file=report_file)


report_file.close()	