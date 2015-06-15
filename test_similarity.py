# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 04:45:15 2015

@author: surat_000
"""
'''
global_path = "C:\\Users\\surat_000\\Documents\\Visual Studio 2013\\Projects\\"
global_path += "searchDB_CS\\searchDB_CS\\bin\\Debug\\text-data-G06F-20000-CV-5000\\"

step_path = global_path + "step-" + str(3) + "\\"
print("We're in {0}".format(step_path))

#pickle
import pickle

X_train_counts = pickle.load(open(step_path + "count_matrix.dat", "rb"))
X_test_counts = pickle.load(open(step_path + "count_matrix_test.dat", "rb"))
X_train = pickle.load(open(step_path + "tfidf_matrix.dat", "rb"))
X_test = pickle.load(open(step_path + "tfidf_matrix_test.dat", "rb"))
'''




#####
# COSINE SIMILATIRY
from sklearn.metrics.pairwise import cosine_similarity

for k in range(0, 10):
	print(twenty_test.target_names[twenty_test.target[0]])
	for j, category in enumerate(categories):
		sum = 0.0
		count = 0
		for i in range(0, 5000):
		    if twenty_train.target[i] == j:
		        sum += cosine_similarity(X_test[k], X_train[i])
		        count += 1
		print("{0} => {1}, {2}, {3}".format(category, sum, count, sum/count))

