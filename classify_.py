# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 23:26:33 2015

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


ch2_fn = "ch2.dat"
text_clf_sgd_fn = "text_clf_sgd.dat"

G06F = "text-data-G06F-20000-CV-1000\\step-1\\"
G06F3 = "new-proc\\text-data-G06F3-regen\\"
G06F7 = "new-proc\\text-data-G06F7-regen\\"
G06F15 = "new-proc\\text-data-G06F15-regen\\"
G06F17 = "new-proc\\text-data-G06F17-regen\\"

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
#main_clf_ch2 = pickle.load(open(main_clf_path + ch2_fn, "rb"))
main_clf_X_train = pickle.load(open(main_clf_path + X_train_ch2_fn, "rb"))
main_clf_X_test = pickle.load(open(main_clf_path + X_test_ch2_fn, "rb"))
main_clf = pickle.load(open(main_clf_path + text_clf_sgd_fn, "rb"))

clf_17_path = global_path + G06F17
clf_17_train = pickle.load(open(clf_17_path + train_fn, "rb"))
clf_17_vectorizer = pickle.load(open(clf_17_path + count_vect_fn, "rb"))
clf_17_tfidf_trasformer = pickle.load(open(clf_17_path + tfidf_trans_fn, "rb"))
#clf_17_ch2 = pickle.load(open(clf_17_path + ch2_fn, "rb"))
clf_17 = pickle.load(open(clf_17_path + text_clf_sgd_fn, "rb"))

clf_15_path = global_path + G06F15
clf_15_train = pickle.load(open(clf_15_path + train_fn, "rb"))
clf_15_vectorizer = pickle.load(open(clf_15_path + count_vect_fn, "rb"))
clf_15_tfidf_trasformer = pickle.load(open(clf_15_path + tfidf_trans_fn, "rb"))
#clf_15_ch2 = pickle.load(open(clf_15_path + ch2_fn, "rb"))
clf_15 = pickle.load(open(clf_15_path + text_clf_sgd_fn, "rb"))

clf_7_path = global_path + G06F7
clf_7_train = pickle.load(open(clf_7_path + train_fn, "rb"))
clf_7_vectorizer = pickle.load(open(clf_7_path + count_vect_fn, "rb"))
clf_7_tfidf_trasformer = pickle.load(open(clf_7_path + tfidf_trans_fn, "rb"))
#clf_7_ch2 = pickle.load(open(clf_7_path + ch2_fn, "rb"))
clf_7 = pickle.load(open(clf_7_path + text_clf_sgd_fn, "rb"))

clf_3_path = global_path + G06F3
clf_3_train = pickle.load(open(clf_3_path + train_fn, "rb"))
clf_3_vectorizer = pickle.load(open(clf_3_path + count_vect_fn, "rb"))
clf_3_tfidf_trasformer = pickle.load(open(clf_3_path + tfidf_trans_fn, "rb"))
#clf_3_ch2 = pickle.load(open(clf_3_path + ch2_fn, "rb"))
clf_3 = pickle.load(open(clf_3_path + text_clf_sgd_fn, "rb"))

list_of_all_docs = []
count_by_class = {}

max_length_of_top_word_list = 10

print("processing test")
for i in range(0, len(main_clf_test.data)):
	doc = []
	doc.append(main_clf_test.data[i])
	main_clf_X_new_counts = main_clf_vectorizer.transform(doc)
	main_clf_X_new_tfidf = main_clf_tfidf_trasformer.transform(main_clf_X_new_counts)
	#main_clf_X_new_tfidf_ch2 = main_clf_ch2.transform(main_clf_X_new_tfidf)
	main_clf_X_new_tfidf_ch2 = main_clf_X_new_tfidf
	
	main_clf_predicted = main_clf.predict(main_clf_X_new_tfidf_ch2)
	#print('%r => %s' % (doc, main_clf_train.target_names[main_clf_predicted]))
	
	result_class = 'NONE'
	result_id = ntpath.basename(main_clf_test.filenames[i])	
	
	sum_of_weights = 0
	list_of_properties = []
	
	if main_clf_train.target_names[main_clf_predicted] == 'G06F15':
		sub_clf_X_new_counts = clf_15_vectorizer.transform(doc)
		sub_clf_X_new_tfidf = clf_15_tfidf_trasformer.transform(sub_clf_X_new_counts)
		#sub_clf_X_new_tfidf_ch2 = clf_15_ch2.transform(sub_clf_X_new_tfidf)
		sub_clf_X_new_tfidf_ch2 = sub_clf_X_new_tfidf
		
		clf_15_predicted = clf_15.predict(sub_clf_X_new_tfidf_ch2)
		#print('%r => %s' % (doc, clf_15_train.target_names[clf_15_predicted]))
		
		result_class = clf_15_train.target_names[clf_15_predicted]
		
		#selecting top words
		top_words_index = np.argsort(clf_15.coef_[clf_15_predicted[0]])
		feature_names = clf_15_vectorizer.get_feature_names()
		#feature_names = [feature_names[i] for i in clf_15_ch2.get_support(indices=True)]
		feature_names = np.asarray(feature_names)
		
		temp_doc = tokenize(str(doc[0]))
		top_words_for_class = feature_names[top_words_index]
		
		word_list_for_document_from_top = []
		
		#decrement range
		for i in range(len(top_words_for_class) - 1, 0, -1):
			if any(top_words_for_class[i] in s for s in temp_doc):
				word_list_for_document_from_top.append(top_words_for_class[i])
				if len(word_list_for_document_from_top) >= max_length_of_top_word_list:
					break
				
		word_weights_ = []
		
		for i in range(0, len(word_list_for_document_from_top)):		
			word_weights_.append(
				clf_15.coef_[clf_15_predicted[0]][top_words_index[
					top_words_for_class.tolist().index(word_list_for_document_from_top[i])]
				]
			)
		
		for i in range(0, len(word_list_for_document_from_top)):
			sum_of_weights += word_weights_[i]
			list_of_properties.append({'word' : word_list_for_document_from_top[i], 'weight' : word_weights_[i]})
		
	elif main_clf_train.target_names[main_clf_predicted] == 'G06F17':		
		sub_clf_X_new_counts = clf_17_vectorizer.transform(doc)
		sub_clf_X_new_tfidf = clf_17_tfidf_trasformer.transform(sub_clf_X_new_counts)
		#sub_clf_X_new_tfidf_ch2 = clf_17_ch2.transform(sub_clf_X_new_tfidf)
		sub_clf_X_new_tfidf_ch2 = sub_clf_X_new_tfidf
		
		clf_17_predicted = clf_17.predict(sub_clf_X_new_tfidf_ch2)
		#print('%r => %s' % (doc, clf_17_train.target_names[clf_17_predicted]))
		
		result_class = clf_17_train.target_names[clf_17_predicted]
		
		#selecting top words
		top_words_index = np.argsort(clf_17.coef_[clf_17_predicted[0]])
		feature_names = clf_17_vectorizer.get_feature_names()
		#feature_names = [feature_names[i] for i in clf_17_ch2.get_support(indices=True)]
		feature_names = np.asarray(feature_names)
		
		temp_doc = tokenize(str(doc[0]))
		top_words_for_class = feature_names[top_words_index]
		
		word_list_for_document_from_top = []
		
		#decrement range
		for i in range(len(top_words_for_class) - 1, 0, -1):
			if any(top_words_for_class[i] in s for s in temp_doc):
				word_list_for_document_from_top.append(top_words_for_class[i])
				if len(word_list_for_document_from_top) >= max_length_of_top_word_list:
					break
				
		word_weights_ = []
		
		for i in range(0, len(word_list_for_document_from_top)):		
			word_weights_.append(
				clf_17.coef_[clf_17_predicted[0]][top_words_index[
					top_words_for_class.tolist().index(word_list_for_document_from_top[i])]
				]
			)
		
		for i in range(0, len(word_list_for_document_from_top)):
			sum_of_weights += word_weights_[i]
			list_of_properties.append({'word' : word_list_for_document_from_top[i], 'weight' : word_weights_[i]})
		
	elif main_clf_train.target_names[main_clf_predicted] == 'G06F7':
		sub_clf_X_new_counts = clf_7_vectorizer.transform(doc)
		sub_clf_X_new_tfidf = clf_7_tfidf_trasformer.transform(sub_clf_X_new_counts)
		#sub_clf_X_new_tfidf_ch2 = clf_7_ch2.transform(sub_clf_X_new_tfidf)
		sub_clf_X_new_tfidf_ch2 = sub_clf_X_new_tfidf
		
		clf_7_predicted = clf_7.predict(sub_clf_X_new_tfidf_ch2)
		#print('%r => %s' % (doc, clf_17_train.target_names[clf_17_predicted]))
		
		result_class = clf_7_train.target_names[clf_7_predicted]
		
		#selecting top words
		top_words_index = np.argsort(clf_7.coef_[clf_7_predicted[0]])
		feature_names = clf_7_vectorizer.get_feature_names()
		#feature_names = [feature_names[i] for i in clf_7_ch2.get_support(indices=True)]
		feature_names = np.asarray(feature_names)
		
		temp_doc = tokenize(str(doc[0]))
		top_words_for_class = feature_names[top_words_index]
		
		word_list_for_document_from_top = []
		
		#decrement range
		for i in range(len(top_words_for_class) - 1, 0, -1):
			if any(top_words_for_class[i] in s for s in temp_doc):
				word_list_for_document_from_top.append(top_words_for_class[i])
				if len(word_list_for_document_from_top) >= max_length_of_top_word_list:
					break
				
		word_weights_ = []
		
		for i in range(0, len(word_list_for_document_from_top)):		
			word_weights_.append(
				clf_7.coef_[clf_7_predicted[0]][top_words_index[
					top_words_for_class.tolist().index(word_list_for_document_from_top[i])]
				]
			)
		
		for i in range(0, len(word_list_for_document_from_top)):
			sum_of_weights += word_weights_[i]
			list_of_properties.append({'word' : word_list_for_document_from_top[i], 'weight' : word_weights_[i]})
		
	else:
		sub_clf_X_new_counts = clf_3_vectorizer.transform(doc)
		sub_clf_X_new_tfidf = clf_3_tfidf_trasformer.transform(sub_clf_X_new_counts)
		sub_clf_X_new_tfidf_ch2 = sub_clf_X_new_tfidf
		
		clf_3_predicted = clf_3.predict(sub_clf_X_new_tfidf_ch2)
		#print('%r => %s' % (doc, clf_17_train.target_names[clf_17_predicted]))
		
		result_class = clf_3_train.target_names[clf_3_predicted]
		
		#selecting top words
		top_words_index = np.argsort(clf_3.coef_[clf_3_predicted[0]])
		feature_names = clf_3_vectorizer.get_feature_names()
		#feature_names = [feature_names[i] for i in clf_7_ch2.get_support(indices=True)]
		feature_names = np.asarray(feature_names)
		
		temp_doc = tokenize(str(doc[0]))
		top_words_for_class = feature_names[top_words_index]
		
		word_list_for_document_from_top = []
		
		#decrement range
		for i in range(len(top_words_for_class) - 1, 0, -1):
			if any(top_words_for_class[i] in s for s in temp_doc):
				word_list_for_document_from_top.append(top_words_for_class[i])
				if len(word_list_for_document_from_top) >= max_length_of_top_word_list:
					break
				
		word_weights_ = []
		
		for i in range(0, len(word_list_for_document_from_top)):		
			word_weights_.append(
				clf_3.coef_[clf_3_predicted[0]][top_words_index[
					top_words_for_class.tolist().index(word_list_for_document_from_top[i])]
				]
			)
		
		for i in range(0, len(word_list_for_document_from_top)):
			sum_of_weights += word_weights_[i]
			list_of_properties.append({'word' : word_list_for_document_from_top[i], 'weight' : word_weights_[i]})
	
	if count_by_class.get(result_class) == None:
		temp = {result_class : 1}
		count_by_class.update(temp)
	else:
		count_by_class[result_class] += 1
	
	list_of_all_docs.append({
		'id': result_id,
		'class' : result_class,
		'words_list' : list_of_properties,
		'sum_of_weights' : sum_of_weights
	})

all_train_data = [clf_3_train, clf_7_train, clf_15_train, clf_17_train]
all_clf = [clf_3, clf_7, clf_15, clf_17]
all_clf_vectorizer = [clf_3_vectorizer, clf_7_vectorizer, clf_15_vectorizer, clf_17_vectorizer]
#all_clf_ch2 = [clf_7_ch2, clf_15_ch2, clf_17_ch2]

print("processing trains")
index_of_clf = 0
for dataset in all_train_data:
	for index_of_data in range(0, len(dataset.data)):
		doc = dataset.data[index_of_data]
		result_class = dataset.target_names[dataset.target[index_of_data]]
		result_id = ntpath.basename(dataset.filenames[index_of_data])
		
		if count_by_class.get(result_class) == None:
			temp = {result_class : 1}
			count_by_class.update(temp)
		else:
			count_by_class[result_class] += 1
			
		#selecting top words
		top_words_index = np.argsort(all_clf[index_of_clf].coef_[dataset.target[index_of_data]])
		feature_names = all_clf_vectorizer[index_of_clf].get_feature_names()
		
		#if index_of_clf > 0:
		#	feature_names = [feature_names[i] for i in all_clf_ch2[index_of_clf-1].get_support(indices=True)]
			
		feature_names = np.asarray(feature_names)
		
		sum_of_weights = 0
		list_of_properties = []
		
		doc = []
		doc.append(all_train_data[index_of_clf].data[index_of_data])
		temp_doc = tokenize(str(doc[0]))
		top_words_for_class = feature_names[top_words_index]
		
		word_list_for_document_from_top = []
		
		#decrement range
		for i in range(len(top_words_for_class) - 1, 0, -1):
			if any(top_words_for_class[i] in s for s in temp_doc):
				word_list_for_document_from_top.append(top_words_for_class[i])
				if len(word_list_for_document_from_top) >= max_length_of_top_word_list:
					break
				
		word_weights_ = []
		
		for i in range(0, len(word_list_for_document_from_top)):		
			word_weights_.append(
				all_clf[index_of_clf].coef_[dataset.target[index_of_data]][top_words_index[
					top_words_for_class.tolist().index(word_list_for_document_from_top[i])]
				]
			)
		
		for i in range(0, len(word_list_for_document_from_top)):
			sum_of_weights += word_weights_[i]
			list_of_properties.append({'word' : word_list_for_document_from_top[i], 'weight' : word_weights_[i]})
			
		list_of_all_docs.append({
			'id': result_id,
			'class' : result_class,
			'words_list' : list_of_properties,
			'sum_of_weights' : sum_of_weights
		})
	index_of_clf += 1
	
	
pickle.dump(list_of_all_docs, open(global_path + "__list_of_all_docs.dat", "wb"))
pickle.dump(count_by_class, open(global_path + "__count_by_class.dat", "wb"))
