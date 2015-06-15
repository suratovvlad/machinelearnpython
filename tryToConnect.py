# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 08:59:58 2015

@author: surat_000
"""

from os import getenv
import pymssql

conn = pymssql.connect(
	"192.168.1.129",
	'DEGREE-PC\\suratovvlad@gmail.com',
	'Fcnhjabpbrf95',
	"sophia_index_patents_g06"
)

doc_ids = ['538796', '477283', '262383', '526610', '294174', '249480', '2500', '7553', '550109', '278480']

int_ids = []
for id__ in doc_ids:
	int_ids.append(int(id__))
	
toReturn = []

for id__ in int_ids:
	cursor = conn.cursor()
	cursor.execute('SELECT Doc_Place, Doc_Author, Doc_Title, Doc_Body  FROM [dbo].[search_Documents] WHERE Doc_ID = %s;', id__)
	row = cursor.fetchone()
	while row:
		#print("Doc_Author=%s, Doc_Topics=%s" % (row[0], row[1]))
		temp_dict = {
			'place' : str(row[0]),
			'author' : row[1],
			'title': row[2],
			'patent' : row[3]
		}
		toReturn.append(temp_dict)
		row = cursor.fetchone()
		

conn.close()