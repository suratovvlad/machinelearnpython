# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 14:53:05 2015

@author: surat_000
"""

global_path = "C:\\Users\\surat_000\\Documents\\Visual Studio 2013\\Projects\\"
global_path += "searchDB_CS\\searchDB_CS\\bin\\Debug\\"

import pickle

list_of_all_docs = pickle.load(open(global_path + "__list_of_all_docs.dat", "rb"))
count_by_class = pickle.load(open(global_path + "__count_by_class.dat", "rb"))


sorted_data_set = {}

for index_of_doc in range(0, len(list_of_all_docs)):
	
	class_name = list_of_all_docs[index_of_doc]['class']
	
	if sorted_data_set.get(class_name) == None:
		
		list_of_items = []
		item = list_of_all_docs[index_of_doc]		
		list_of_items.append(item)
		
		temp = {class_name : list_of_items}
		sorted_data_set.update(temp)
	else:
		item = list_of_all_docs[index_of_doc]	
		sorted_data_set[class_name].append(item)


from operator import itemgetter

for key__ in sorted_data_set.keys():
	newlist = sorted(sorted_data_set[key__], key=itemgetter('sum_of_weights'), reverse=True)
	temp = {class_name : newlist}
	sorted_data_set.update(temp)
	
pickle.dump(sorted_data_set, open(global_path + "__sorted_collection.dat", "wb"))