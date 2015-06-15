# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 22:26:09 2015

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
				
				
##########
class Densifier(object):
    def fit(self, X, y=None):
        pass
    def fit_transform(self, X, y=None):
        return self.transform(X)
    def transform(self, X, y=None):
        return X.toarray()
##########
	

##########
#Load train and test data			
from sklearn.datasets import load_files
#categories = ['G06F3', 'G06F7', 'G06F15', 'G06F17']
categotires = None

global_path = "C:\\Users\\surat_000\\Documents\\Visual Studio 2013\\Projects\\"
global_path += "searchDB_CS\\searchDB_CS\\bin\\Debug\\"
#global_path += "text-data-G06F-20000-1\\"
#global_path += "text-data-G06F3-regen\\"
#global_path += "text-data-G06F7-regen-2\\"
#global_path += "text-data-G06F15-regen-3\\"
#global_path += "text-data-G06F17-regen\\"


#global_path += "text-data-G06F-20000-CV-1000\\"
global_path += "new-proc\\"
global_path += "text-data-G06F17-regen\\"

#step_path = global_path + "step-" + str(3) + "\\"
step_path = global_path
#step_path = global_path + "step-" + str(1) + "\\"

print("-"*80)
print("We're in {0}".format(step_path))

train_path = step_path + "train\\"
test_path = step_path + "test\\"

#train_path = step_path + "test\\"
#test_path = step_path + "train\\"

twenty_train = load_files(train_path)
print("Train set is loaded")
print("{0} documents".format(len(twenty_train.filenames)))
print("{0} categories".format(len(twenty_train.target_names)))
print()

twenty_test = load_files(test_path)
docs_test = twenty_test.data
print("Test set is loaded")
print("{0} documents".format(len(twenty_test.filenames)))
print("{0} categories".format(len(twenty_test.target_names)))
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

import numpy as np
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2

print("="*80)

##########
print("Multinomial")
#MultinomialNB
from sklearn.naive_bayes import MultinomialNB
text_clf_mnb = Pipeline([('vect', count_vectorizer),
                     ('tfidf', tfidf_transformer),
                     ('clf', MultinomialNB()),
])

#Fit
print("Fit")
text_clf_mnb = text_clf_mnb.fit(twenty_train.data, twenty_train.target)

#Evaluation of the performance on the test set
print("Evaluation")
predicted = text_clf_mnb.predict(docs_test)
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


'''
samples_precision = metrics.precision_score(twenty_test.target, predicted, average='samples')
samples_recall = metrics.recall_score(twenty_test.target, predicted, average='samples')
samples_fscore = metrics.f1_score(twenty_test.target, predicted, average='samples')

print("samples precision: {0}".format(weighted_precision))
print("samples recall: {0}".format(weighted_recall))
print("samples f1-score: {0}".format(weighted_fscore))


binary_precision = metrics.precision_score(twenty_test.target, predicted, average='binary')
binary_recall = metrics.recall_score(twenty_test.target, predicted, average='binary')
binary_fscore = metrics.f1_score(twenty_test.target, predicted, average='binary')

print("binary precision: {0}".format(binary_precision))
print("binary recall: {0}".format(binary_recall))
print("binary f1-score: {0}".format(binary_fscore))
'''


print(metrics.classification_report(twenty_test.target, predicted,
    target_names=twenty_test.target_names))
_ = metrics.confusion_matrix(twenty_test.target, predicted)
print(_)
print("END Multinomial")
###########



print("="*80)

##########
print("Bernoulli")
#BernoulliNB
from sklearn.naive_bayes import BernoulliNB
text_clf_gnb = Pipeline([('vect', count_vectorizer),
                     ('tfidf', tfidf_transformer),
                     ('clf', BernoulliNB()),
])

#Fit
print("Fit")
text_clf_gnb = text_clf_gnb.fit(twenty_train.data, twenty_train.target)

#Evaluation of the performance on the test set
print("Evaluation")
predicted = text_clf_gnb.predict(docs_test)
_ = np.mean(predicted == twenty_test.target)
print(_)

accuracy_score = metrics.accuracy_score(twenty_test.target, predicted)
print("accuracy: {0}".format(accuracy_score))


macro_precision = metrics.precision_score(twenty_test.target, predicted, average='macro')
macro_recall = metrics.recall_score(twenty_test.target, predicted, average='macro')
macro_fscore = metrics.f1_score(twenty_test.target, predicted, average='macro')

my_macro_fscore = 2 * macro_precision * macro_recall / (macro_precision + macro_recall)

print("macro precision: {0}".format(macro_precision))
print("macro recall: {0}".format(macro_recall))
print("macro f1-score: {0}".format(macro_fscore))
print("my macro f1-score: {0}".format(my_macro_fscore))
print()

micro_precision = metrics.precision_score(twenty_test.target, predicted, average='micro')
micro_recall = metrics.recall_score(twenty_test.target, predicted, average='micro')
micro_fscore = metrics.f1_score(twenty_test.target, predicted, average='micro')

my_micro_fscore = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)

print("micro precision: {0}".format(micro_precision))
print("micro recall: {0}".format(micro_recall))
print("micro f1-score: {0}".format(micro_fscore))
print("my micro f1-score: {0}".format(my_micro_fscore))
print()


weighted_precision = metrics.precision_score(twenty_test.target, predicted, average='weighted')
weighted_recall = metrics.recall_score(twenty_test.target, predicted, average='weighted')
weighted_fscore = metrics.f1_score(twenty_test.target, predicted, average='weighted')

my_weighted_fscore = 2 * weighted_precision * weighted_recall / (weighted_precision + weighted_recall)

print("weighted precision: {0}".format(weighted_precision))
print("weighted recall: {0}".format(weighted_recall))
print("weighted f1-score: {0}".format(weighted_fscore))
print("my weighted f1-score: {0}".format(my_weighted_fscore))
print()
print(metrics.classification_report(twenty_test.target, predicted,
    target_names=twenty_test.target_names))
_ = metrics.confusion_matrix(twenty_test.target, predicted)
print(_)
print("END Bernoulli")
###########

print("="*80)

##########
print("SGDClassifier")
#SGDClassifier loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42
from sklearn.linear_model import SGDClassifier
text_clf_sgd = Pipeline([('vect', count_vectorizer),
                     ('tfidf', tfidf_transformer),
                     ('clf', SGDClassifier()),
])

#Fit
print("Fit")
text_clf_sgd = text_clf_sgd.fit(twenty_train.data, twenty_train.target)

#Evaluation of the performance on the test set
print("Evaluation")
predicted = text_clf_sgd.predict(docs_test)
_ = np.mean(predicted == twenty_test.target)
print(_)

accuracy_score = metrics.accuracy_score(twenty_test.target, predicted)
print("accuracy: {0}".format(accuracy_score))


macro_precision = metrics.precision_score(twenty_test.target, predicted, average='macro')
macro_recall = metrics.recall_score(twenty_test.target, predicted, average='macro')
macro_fscore = metrics.f1_score(twenty_test.target, predicted, average='macro')

my_macro_fscore = 2 * macro_precision * macro_recall / (macro_precision + macro_recall)

print("macro precision: {0}".format(macro_precision))
print("macro recall: {0}".format(macro_recall))
print("macro f1-score: {0}".format(macro_fscore))
print("my macro f1-score: {0}".format(my_macro_fscore))
print()

micro_precision = metrics.precision_score(twenty_test.target, predicted, average='micro')
micro_recall = metrics.recall_score(twenty_test.target, predicted, average='micro')
micro_fscore = metrics.f1_score(twenty_test.target, predicted, average='micro')

my_micro_fscore = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)

print("micro precision: {0}".format(micro_precision))
print("micro recall: {0}".format(micro_recall))
print("micro f1-score: {0}".format(micro_fscore))
print("my micro f1-score: {0}".format(my_micro_fscore))
print()


weighted_precision = metrics.precision_score(twenty_test.target, predicted, average='weighted')
weighted_recall = metrics.recall_score(twenty_test.target, predicted, average='weighted')
weighted_fscore = metrics.f1_score(twenty_test.target, predicted, average='weighted')

my_weighted_fscore = 2 * weighted_precision * weighted_recall / (weighted_precision + weighted_recall)

print("weighted precision: {0}".format(weighted_precision))
print("weighted recall: {0}".format(weighted_recall))
print("weighted f1-score: {0}".format(weighted_fscore))
print("my weighted f1-score: {0}".format(my_weighted_fscore))
print()
print(metrics.classification_report(twenty_test.target, predicted,
    target_names=twenty_test.target_names))
_ = metrics.confusion_matrix(twenty_test.target, predicted)
print(_)
print("END SGDClassifier")
###########

print("="*80)

##########
print("LinearSVC")
#LinearSVC
from sklearn.svm import LinearSVC
text_clf_lsvc = Pipeline([('vect', count_vectorizer),
                     ('tfidf', tfidf_transformer),
                     ('clf', LinearSVC()),
])

#Fit
print("Fit")
text_clf_lsvc = text_clf_lsvc.fit(twenty_train.data, twenty_train.target)

#Evaluation of the performance on the test set
print("Evaluation")
predicted = text_clf_lsvc.predict(docs_test)
_ = np.mean(predicted == twenty_test.target)
print(_)

accuracy_score = metrics.accuracy_score(twenty_test.target, predicted)
print("accuracy: {0}".format(accuracy_score))


macro_precision = metrics.precision_score(twenty_test.target, predicted, average='macro')
macro_recall = metrics.recall_score(twenty_test.target, predicted, average='macro')
macro_fscore = metrics.f1_score(twenty_test.target, predicted, average='macro')

my_macro_fscore = 2 * macro_precision * macro_recall / (macro_precision + macro_recall)

print("macro precision: {0}".format(macro_precision))
print("macro recall: {0}".format(macro_recall))
print("macro f1-score: {0}".format(macro_fscore))
print("my macro f1-score: {0}".format(my_macro_fscore))
print()

micro_precision = metrics.precision_score(twenty_test.target, predicted, average='micro')
micro_recall = metrics.recall_score(twenty_test.target, predicted, average='micro')
micro_fscore = metrics.f1_score(twenty_test.target, predicted, average='micro')

my_micro_fscore = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)

print("micro precision: {0}".format(micro_precision))
print("micro recall: {0}".format(micro_recall))
print("micro f1-score: {0}".format(micro_fscore))
print("my micro f1-score: {0}".format(my_micro_fscore))
print()


weighted_precision = metrics.precision_score(twenty_test.target, predicted, average='weighted')
weighted_recall = metrics.recall_score(twenty_test.target, predicted, average='weighted')
weighted_fscore = metrics.f1_score(twenty_test.target, predicted, average='weighted')

my_weighted_fscore = 2 * weighted_precision * weighted_recall / (weighted_precision + weighted_recall)

print("weighted precision: {0}".format(weighted_precision))
print("weighted recall: {0}".format(weighted_recall))
print("weighted f1-score: {0}".format(weighted_fscore))
print("my weighted f1-score: {0}".format(my_weighted_fscore))
print()
print(metrics.classification_report(twenty_test.target, predicted,
    target_names=twenty_test.target_names))
_ = metrics.confusion_matrix(twenty_test.target, predicted)
print(_)
print("END LinearSVC")
###########

print("="*80)

##########
print("KNeighborsClassifier")
#KNeighborsClassifier n_neighbors=15, weights='distance'
from sklearn.neighbors import KNeighborsClassifier
text_clf_knn = Pipeline([('vect', count_vectorizer),
                     ('tfidf', tfidf_transformer),
                     ('clf', KNeighborsClassifier()),
])

#Fit
print("Fit")
text_clf_knn = text_clf_knn.fit(twenty_train.data, twenty_train.target)

#Evaluation of the performance on the test set
print("Evaluation")
predicted = text_clf_knn.predict(docs_test)
_ = np.mean(predicted == twenty_test.target)
print(_)

accuracy_score = metrics.accuracy_score(twenty_test.target, predicted)
print("accuracy: {0}".format(accuracy_score))


macro_precision = metrics.precision_score(twenty_test.target, predicted, average='macro')
macro_recall = metrics.recall_score(twenty_test.target, predicted, average='macro')
macro_fscore = metrics.f1_score(twenty_test.target, predicted, average='macro')

my_macro_fscore = 2 * macro_precision * macro_recall / (macro_precision + macro_recall)

print("macro precision: {0}".format(macro_precision))
print("macro recall: {0}".format(macro_recall))
print("macro f1-score: {0}".format(macro_fscore))
print("my macro f1-score: {0}".format(my_macro_fscore))
print()

micro_precision = metrics.precision_score(twenty_test.target, predicted, average='micro')
micro_recall = metrics.recall_score(twenty_test.target, predicted, average='micro')
micro_fscore = metrics.f1_score(twenty_test.target, predicted, average='micro')

my_micro_fscore = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)

print("micro precision: {0}".format(micro_precision))
print("micro recall: {0}".format(micro_recall))
print("micro f1-score: {0}".format(micro_fscore))
print("my micro f1-score: {0}".format(my_micro_fscore))
print()


weighted_precision = metrics.precision_score(twenty_test.target, predicted, average='weighted')
weighted_recall = metrics.recall_score(twenty_test.target, predicted, average='weighted')
weighted_fscore = metrics.f1_score(twenty_test.target, predicted, average='weighted')

my_weighted_fscore = 2 * weighted_precision * weighted_recall / (weighted_precision + weighted_recall)

print("weighted precision: {0}".format(weighted_precision))
print("weighted recall: {0}".format(weighted_recall))
print("weighted f1-score: {0}".format(weighted_fscore))
print("my weighted f1-score: {0}".format(my_weighted_fscore))
print()
print(metrics.classification_report(twenty_test.target, predicted,
    target_names=twenty_test.target_names))
_ = metrics.confusion_matrix(twenty_test.target, predicted)
print(_)
print("END KNeighborsClassifier")
###########
