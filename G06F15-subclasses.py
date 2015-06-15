# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 06:02:43 2015

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


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
#from sklearn.grid_search import GridSearchCV
import numpy as np
from sklearn import metrics


##########
#Load train and test data			
from sklearn.datasets import load_files
categories = None

global_path = "C:\\Users\\surat_000\\Documents\\Visual Studio 2013\\Projects\\"
global_path += "searchDB_CS\\searchDB_CS\\bin\\Debug\\text-data-G06F15-regen-3\\"

step_path = global_path
print("We're in {0}".format(step_path))

train_path = step_path + "train\\"
twenty_train = load_files(train_path, None, categories)
print("Train set is loaded")
print("{0} documents".format(len(twenty_train.filenames)))
print("{0} categories".format(len(twenty_train.target_names)))
print()

test_path = step_path + "test\\"
twenty_test = load_files(test_path, None, categories)
docs_test = twenty_test.data
print("Test set is loaded")
print("{0} documents".format(len(twenty_test.filenames)))
print("{0} categories".format(len(twenty_test.target_names)))
print()
##########


##########
#Count Vectorizer , ngram_range=(1,2), tokenizer=tokenize
from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer = CountVectorizer(decode_error='ignore', max_df=0.75, stop_words='english', tokenizer=tokenize)
#count_vectorizer = count_vectorizer.fit(twenty_train.data)
# tf-idf transformer
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
##########

##########
print("SGDClassifier")
#SGDClassifier
from sklearn.linear_model import SGDClassifier
pipeline = Pipeline([
	('vect', count_vectorizer),
	('chi2', SelectKBest(chi2, k=1500)),
	('tfidf', tfidf_transformer),
	('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)),
])

#Fit
print("Fit")
text_clf_sgd = pipeline.fit(twenty_train.data, twenty_train.target)

#Evaluation of the performance on the test set
print("Evaluation")
predicted = text_clf_sgd.predict(docs_test)
_ = np.mean(predicted == twenty_test.target)
print(_)

print(metrics.classification_report(twenty_test.target, predicted,
    target_names=twenty_test.target_names))
_ = metrics.confusion_matrix(twenty_test.target, predicted)
print(_)
print("END SGDClassifier")
###########