# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 21:05:06 2015

@author: surat_000
"""

from sklearn.feature_extraction.text import CountVectorizer

vocab = ['The swimmer likes swimming so he swims.']
vec = CountVectorizer().fit(vocab)

sentence1 = vec.transform(['The swimmer likes swimming.'])
sentence2 = vec.transform(['The swimmer swims.'])

print('Vocabulary: {0}'.format(vec.get_feature_names()))
print('Sentence 1: {0}'.format(sentence1.toarray()))
print('Sentence 2: {0}'.format(sentence2.toarray()))


from nltk import word_tokenize          
from nltk.stem.porter import PorterStemmer

#######
# based on http://www.cs.duke.edu/courses/spring14/compsci290/assignments/lab02.html
stemmer = PorterStemmer()
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    text = "".join([ch for ch in text if ch not in string.punctuation])
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems
######## 

vect = CountVectorizer(tokenizer=tokenize, stop_words='english') 

vect.fit(vocab)

sentence1 = vect.transform(['The swimmer likes swimming.'])
sentence2 = vect.transform(['The swimmer swims.'])

print('Vocabulary: {0}'.format(vect.get_feature_names()))
print('Sentence 1: {0}'.format(sentence1.toarray()))
print('Sentence 2: {0}'.format(sentence2.toarray()))
