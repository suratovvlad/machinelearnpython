# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 19:36:58 2015

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

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
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

#print("Loading patent dataset for categories:")
#print(categories)

global_path = "C:\\Users\\surat_000\\Documents\\Visual Studio 2013\\Projects\\"
global_path += "searchDB_CS\\searchDB_CS\\bin\\Debug\\"

#global_path += "text-data-G06F-20000-1\\"
#global_path += "text-data-G06F3-regen\\"
#global_path += "text-data-G06F7-regen-2\\"
#global_path += "text-data-G06F15-regen-3\\"
#global_path += "text-data-G06F17-regen\\"
global_path += "text-data-G06F-20000-CV-1000\\"

#step_path = global_path + "step-" + str(3) + "\\"
#step_path = global_path

step_path = global_path + "step-" + str(1) + "\\"
print("We're in {0}".format(step_path))

#for step in range(3, 4):
#	print("We're in step {0}".format(str(step)))
#print()
#step_path = global_path + "step-" + str(step) + "\\"
#print("We're in {0}".format(step_path))

train_path = step_path + "test\\"
twenty_train = load_files(train_path, None, categories)
#print("Train set is loaded")
#print()

#print("{0} documents".format(len(twenty_train.filenames)))
#print("{0} categories".format(len(twenty_train.target_names)))
#print()

'''
test_path = step_path + "test\\"
twenty_test = load_files(test_path, None, categories)
docs_test = twenty_test.data
print("Test set is loaded")
'''
###############################################################################
# define a pipeline combining a text feature extractor with a simple
# classifier
#	('chi2', SelectKBest(chi2, k=1000)),
pipeline = Pipeline([
	('vect', CountVectorizer(decode_error='ignore', stop_words='english', tokenizer=tokenize)),
	('tfidf', TfidfTransformer()),
	('clf', SGDClassifier()),
])

# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
parameters = {
	#'vect__tokenizer' : (None, tokenize),
	'vect__max_df': (0.75, 1.0),
	#'vect__max_features': (None, 5000, 10000, 50000),
	#'vect__ngram_range': ((1, 1), (1, 2), (1, 3)),  # unigrams or bigrams or trigrams
	#'tfidf__use_idf': (True, False),
	#'tfidf__norm': ('l1', 'l2'),
	'clf__alpha': (0.001, 0.0001),
	'clf__penalty': ('l2', 'elasticnet'),
	'clf__n_iter': (5, 10, 50),
}

if __name__ == "__main__":
	# multiprocessing requires the fork to happen in a __main__ protected
	# block

	# find the best parameters for both the feature extraction and the
	# classifier
	grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)
	
	print("Performing grid search...")
	print("pipeline:", [name for name, _ in pipeline.steps])
	print("parameters:")
	pprint(parameters)
	t0 = time()
	grid_search.fit(twenty_train.data, twenty_train.target)
	print("done in {0}".format((time() - t0)))
	print()
	
	print("Best score: {0}".format(grid_search.best_score_))
	print("Best parameters set:")
	best_parameters = grid_search.best_estimator_.get_params()
	for param_name in sorted(parameters.keys()):
		print("\t{0}: {1}".format(param_name, best_parameters[param_name]))