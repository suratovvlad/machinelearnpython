# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 09:02:34 2015

@author: surat_000
"""

import os
import pymssql
from lxml import etree

tree = etree.parse('./fullset.xml')
nodes = tree.xpath('/docs/section/doc')

conn = pymssql.connect("192.168.1.129", 'DEGREE-PC\\suratovvlad@gmail.com', 'Fcnhjabpbrf95', "sophia_index_patents_g06")
cursor = conn.cursor()

#idsA = []
text_folder = u'documents'

if not os.path.exists(text_folder):
    os.makedirs(text_folder)
				
for node in nodes:	
#	print("%s" % node.get('class'))
#	idsA.append(int(node.text))
	class_cur = node.get('class')
	if (class_cur != 'G' and class_cur != 'H'):
		continue
	text_lang_folder = os.path.join(text_folder, class_cur)
	if not os.path.exists(text_lang_folder):
		os.makedirs(text_lang_folder)
	text_filename = os.path.join(text_lang_folder, 'ID_%s.txt' % (node.text))
	
	cursor.execute('SELECT Doc_ID, Doc_Title, Doc_Body FROM [dbo].[search_Documents] WHERE Doc_ID=%d', int(node.text))
	row = cursor.fetchone()
	while row:
#		print("Doc_ID=%d, Doc_Title=%s" % (row[0], row[1]))
		content = row[2]
#		print("Writing %s" % text_filename)
		open(text_filename, 'wb').write(content.encode('utf-8', 'ignore'))
		row = cursor.fetchone()
		
print("Well done")
	
	
	

	


#fullset = []
#for doc_id in idsA:
#	cursor.execute('SELECT Doc_ID, Doc_Title, Doc_Body FROM [dbo].[search_Documents] WHERE Doc_ID=%d', doc_id)
#	row = cursor.fetchone()
#	while row:
#		print("Doc_ID=%d, Doc_Title=%s" % (row[0], row[1]))
#		trainingSet.append(row[2])
#		row = cursor.fetchone()

	