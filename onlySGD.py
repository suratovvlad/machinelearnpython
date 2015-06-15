# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 22:07:08 2015

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
categories = ['G06F3', 'G06F7', 'G06F15', 'G06F17']

global_path = "C:\\Users\\surat_000\\Documents\\Visual Studio 2013\\Projects\\"
global_path += "searchDB_CS\\searchDB_CS\\bin\\Debug\\"

#global_path += "text-data-G06F-20000-1\\"
#global_path += "text-data-G06F3-regen\\"
#global_path += "text-data-G06F7-regen-2\\"
#global_path += "text-data-G06F15-regen-3\\"
#global_path += "text-data-G06F17-regen\\"
#global_path += "text-data-G06F-20000-CV-1000\\"
global_path += "new-proc\\"
global_path += "text-data-G06F15-regen\\"

step_path = global_path
#step_path = global_path + "step-" + str(1) + "\\"
print("We're in {0}".format(step_path))

train_path = step_path + "train\\"
test_path = step_path + "test\\"

#train_path = step_path + "test\\"
#test_path = step_path + "train\\"


twenty_train = load_files(train_path)
#twenty_train = load_files(test_path, None, categories)
print("Train set is loaded")
print("{0} documents".format(len(twenty_train.filenames)))
print("{0} categories".format(len(twenty_train.target_names)))
print()


twenty_test = load_files(test_path)
#twenty_test = load_files(train_path, None, categories)
docs_test = twenty_test.data
print("Test set is loaded")
print("{0} documents".format(len(twenty_test.filenames)))
print("{0} categories".format(len(twenty_test.target_names)))
print()
##########


##########
#Count Vectorizer , ngram_range=(1,2), tokenizer=tokenize
from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer = CountVectorizer(decode_error='ignore', stop_words='english', tokenizer=tokenize)
#count_vectorizer = count_vectorizer.fit(twenty_train.data)
# tf-idf transformer
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
##########
'''
##########
print("SGDClassifier")
#SGDClassifier loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42
#('chi2', SelectKBest(chi2, k=1000)),
from sklearn.linear_model import SGDClassifier
pipeline = Pipeline([
	('vect', count_vectorizer),
	('chi2', SelectKBest(chi2, k=1000)),
	('tfidf', tfidf_transformer),
	('clf', SGDClassifier( penalty='elasticnet', n_iter=5, alpha=1e-4)),
])
print("=" * 80)
print(pipeline)
print("=" * 80)

#Fit
print("Fit")
text_clf_sgd = pipeline.fit(twenty_train.data, twenty_train.target)

#Evaluation of the performance on the test set
print("Evaluation")
predicted = text_clf_sgd.predict(docs_test)
_ = np.mean(predicted == twenty_test.target)
print(_)

accuracy_score = metrics.accuracy_score(twenty_test.target, predicted)
print("accuracy: %0.4f" % (accuracy_score))


macro_precision = metrics.precision_score(twenty_test.target, predicted, average='macro')
macro_recall = metrics.recall_score(twenty_test.target, predicted, average='macro')
macro_fscore = metrics.f1_score(twenty_test.target, predicted, average='macro')

my_macro_fscore = 2 * macro_precision * macro_recall / (macro_precision + macro_recall)

print("macro precision: %0.4f" % (macro_precision))
print("macro recall: %0.4f" % (macro_recall))
print("macro f1-score: %0.4f" % (macro_fscore))
print("my macro f1-score: %0.4f" % (my_macro_fscore))
print()

micro_precision = metrics.precision_score(twenty_test.target, predicted, average='micro')
micro_recall = metrics.recall_score(twenty_test.target, predicted, average='micro')
micro_fscore = metrics.f1_score(twenty_test.target, predicted, average='micro')

my_micro_fscore = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)

print("micro precision: %0.4f" % (micro_precision))
print("micro recall: %0.4f" % (micro_recall))
print("micro f1-score: %0.4f" % (micro_fscore))
print("my micro f1-score: %0.4f" % (my_micro_fscore))
print()


weighted_precision = metrics.precision_score(twenty_test.target, predicted, average='weighted')
weighted_recall = metrics.recall_score(twenty_test.target, predicted, average='weighted')
weighted_fscore = metrics.f1_score(twenty_test.target, predicted, average='weighted')

my_weighted_fscore = 2 * weighted_precision * weighted_recall / (weighted_precision + weighted_recall)

print("weighted precision: %0.4f" % (weighted_precision))
print("weighted recall: %0.4f" % (weighted_recall))
print("weighted f1-score: %0.4f" % (weighted_fscore))
print("my weighted f1-score: %0.4f" % (my_weighted_fscore))
print()

print(metrics.classification_report(twenty_test.target, predicted,
    target_names=twenty_test.target_names))
_ = metrics.confusion_matrix(twenty_test.target, predicted)
print(_)
print("END SGDClassifier")
###########
'''
'''
from sklearn.grid_search import GridSearchCV
parameters_ = {'vect__ngram_range': [(1, 1), (1, 2)],
              'tfidf__use_idf': (True, False),
              'clf__alpha': (1e-2, 1e-3),
}
if __name__ == "__main__":
	gs_clf = GridSearchCV(pipeline, parameters_, n_jobs=-1)
	gs_clf = gs_clf.fit(twenty_train.data[:400], twenty_train.target[:400])
'''
'''
doc_new_ = ['An architecture is provided for data mining of electronic messages to extract information relating to relevancy and popularity of websites and/or web pages for ranking of web pages or other documents. A monitor component monitors information of a message for a reference to a web page or other document, and a ranking component computes rank of the web page based in part on the reference.',
	'The present invention provides a configurable field definition document as well as a method, system and program product for configuring a field definition document. Specifically, under the present invention, fields of the field definition document are configured to store values of data elements used by a computer application. The computer application is then mapped to the fields. As needed, values of data elements for the computer application are accessed from the field definition document based on the mapping and at least one selection criterion. The at least one selection criterion can include, for example, a customer identity, a location, a language, a project type, etc.',
	'A system for collecting and processing data includes a device ( 102 ) for producing output data describing deformation of a surface ( 104 ), which may be part of a golf driving mat. The system also includes a device ( 206 ) for obtaining ( 402 ) data describing deformation of the surface over a period of time. The system can process the obtained data as a time series to produce ( 406 ) data describing characteristics of the deformation, and/or classify ( 408 ) the obtained data according to one or more of a set of data describing characteristics of deformation of the surface.',
	'A WHOIS record of a domain name is retrieved at a first time, the WHOIS record including an expiry date of a second time, a time difference value can be calculated between the first time and the second time, and the time difference value provided to a user. Time difference value can be determined to satisfy at least one condition including a threshold value. An indication can be provided to the user that the at least one condition has been satisfied such as notifying the user of domain name expiration status, storing the domain name in a user expiration watch list, monitoring the domain name for expiration upon or after the second time, and attempting to register the domain name with a selected domain name registration provider after the second time or upon determining that either the domain name may soon be available for registration or available for registration. The WHOIS record can be retrieved in response to receiving or obtaining a request such as a resource location request, domain name resolution request, search engine request, WHOIS request, domain name availability request, and domain name registration request. ']
#X_new_counts = count_vectorizer.transform(doc_new_)
#X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted = text_clf_sgd.predict(doc_new_)
for doc, category in zip(doc_new_, predicted):
	print('%r => %s' % (doc_new_, twenty_train.target_names[category]))
'''


from time import time

y_train, y_test = twenty_train.target, twenty_test.target


X_train_counts = count_vectorizer.fit_transform(twenty_train.data)
X_test_counts = count_vectorizer.transform(twenty_test.data)

X_train = tfidf_transformer.fit_transform(X_train_counts)
X_test = tfidf_transformer.transform(X_test_counts)

ch2 = SelectKBest(chi2, k=1000)
X_train = ch2.fit_transform(X_train, y_train)
X_test = ch2.transform(X_test)

print('_' * 80)
print("Training: ")
clf_sgd = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)
print(clf_sgd)
t0 = time()
clf_sgd = clf_sgd.fit(X_train, y_train)
train_time = time() - t0
print("train time: %0.3fs" % train_time)

t0 = time()
pred = clf_sgd.predict(X_test)
test_time = time() - t0
print("test time:  %0.3fs" % test_time)

score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)



#####
# COSINE SIMILATIRY
#from sklearn.metrics.pairwise import cosine_similarity
