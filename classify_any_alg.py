# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 17:26:50 2015

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

train = "train\\"
test = "test\\"

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

max_length_of_top_word_list = 10

sorted_data_set = pickle.load(open(global_path + "__sorted_collection.dat", "rb"))


#list_of_all_docs = []
#count_by_class = {}

#doc_str = "Method and System for Enabling Item-Level Approval of Payment Card\nA method includes the steps of facilitating obtaining indicia identifying individual items to be purchased at a point of interaction, in conjunction with an inbound authorization request; facilitating translation of the indicia into a form understandable by a third party transaction approver, to obtain translated indicia; and facilitating transfer of the translated indicia to the third patty transaction approver for item-by-item validation on the individual items The transfer of the translated indicia is in conjunction with an outbound authorization request. A system implementing the method can make use of an inventive transfer engine. "
#doc_str = "METHOD AND APPARATUS FOR NETWORK-BASED SALES FORCE MANAGEMENT\nA method and apparatus for network-based sales force automation are provided that meet objectives of increasing sales and marketing efficiency and efficiency of technical and customer support by providing multi-dimensional displays of transactional information to users. Transactional information of deals, contacts, accounts, and leads is provided over the Internet using a Web browser. The information of related transactions is electronically linked, and the transactional information is electronically searchable using custom profiles. The transactional information is accessed and shared among host organization members according to a hierarchy and predefined territories. A Radar Screen Opportunity Display (RSOD) may be selected on which deal objects are displayed that represent the stages in a sales pipeline of corresponding deals. New business information may be selected, wherein automatic notification is provided of new information and changed information relating to transactions, wherein the new business information comprises information on at least one monitored customer Web site. A communication capability is provided that comprises electronic mail, facsimile, telephones, and paging devices, wherein communication is automatically established using transactional information."
#doc_str = "Method and System for Identifying an Operating System Running on a Computer System\nIdentifying an operating system running on a computer system. In one aspect of the invention, an enumeration pattern is collected, the enumeration pattern describing an enumeration of a device that has been performed between the device and the operating system running on a host computer system. The type of the operating system running on the host computer system is identified based on the collected enumeration pattern. "
#doc_str = "Method and system for the protected storage of downloaded media content via a virtualized platform\nA method and system for the protected storage of downloaded media content via a virtualized platform. A method comprises downloading content to a special purpose virtual machine and then storing the downloaded content at a location, where the location is only accessible via the special purpose virtual machine. The stored content is then streamed over a virtual network to a general purpose virtual machine, where the special purpose virtual machine and the general purpose virtual machine exist on the same personal computer (PC). "

doc_str = "VIDEO GAME METHOD AND SYSTEM WITH CONTENT-RELATED OPTIONS\nA method for providing media content-related options in conjunction with video game use is provided. The method includes identifying video game content of a video game and identifying commercial media content that is related to the video game content. The method further includes displaying an interactive menu through a screen of the video game. The interactive menu itemizes selectable commercial media content and provides a link to a commercial Internet website to enable purchase of the commercial media content."  

doc = []
doc.append(doc_str)
main_clf_X_new_counts = main_clf_vectorizer.transform(doc)
main_clf_X_new_tfidf = main_clf_tfidf_trasformer.transform(main_clf_X_new_counts)
#main_clf_X_new_tfidf_ch2 = main_clf_ch2.transform(main_clf_X_new_tfidf)
main_clf_X_new_tfidf_ch2 = main_clf_X_new_tfidf

main_clf_predicted = main_clf.predict(main_clf_X_new_tfidf_ch2)
#print('%r => %s' % (doc, main_clf_train.target_names[main_clf_predicted]))

result_class = 'NONE'
#result_id = ntpath.basename(main_clf_test.filenames[i])	

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


print(result_class)
print(sum_of_weights)

relevant = []


list_of_ = sorted_data_set[result_class]

print("relevant:")
count_of_ = 0
for index_of_ in range(0, len(list_of_)):
	word_list = []
	for word in list_of_[index_of_]['words_list']:
		word_list.append(word['word'])
	
	count_of_matches = 0
	score_of_matches = 0
	matches_ = []
	
	#for word_1 in word_list_for_document_from_top:
	for iiii in range(0, len(word_list_for_document_from_top)):
		word_1 = word_list_for_document_from_top[iiii]
		for word_2 in word_list:
			if (word_1 == word_2):
				count_of_matches += 1
				score_of_matches += word_weights_[iiii]
				matches_.append(word_1)
	
	if count_of_matches > 0:
		temp_dict = {
			'count' : count_of_matches,
			'matches' : matches_,
			'document_id': list_of_[index_of_]['id'],
			#'score' : list_of_[index_of_]['sum_of_weights']
			'score' : score_of_matches
		}
		relevant.append(temp_dict)


from operator import itemgetter
sorted_relevant =sorted(relevant, key=lambda x: (-x['count'], -x['score']))


top10 = sorted_relevant[:10]
top10_toReturn = []
for doc_ in top10:
	print("{0} : {1} : {2}".format(doc_['count'], doc_['document_id'], doc_['score']))
	top10_toReturn.append(doc_['document_id'])