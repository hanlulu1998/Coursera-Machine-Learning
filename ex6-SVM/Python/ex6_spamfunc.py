import matplotlib.pyplot as plt
import numpy as np
import re
from sklearn import svm
from nltk.stem.porter import PorterStemmer


def getVocabList():
    data = np.loadtxt('vocab.txt', dtype=str, delimiter='\t')
    vocabList = np.array(data[:, 1])
    return vocabList


def processEmail(email_contents):
    vocabList = getVocabList()
    email_contents = email_contents.lower()
    strinfo = re.compile('<[^<>]+>')
    email_contents = strinfo.sub(' ', email_contents)
    strinfo = re.compile('[0-9]+')
    email_contents = strinfo.sub('number', email_contents)
    strinfo = re.compile('(http|https)://[^\s]*')
    email_contents = strinfo.sub('httpaddr', email_contents)
    strinfo = re.compile('[^\s]+@[^\s]+')
    email_contents = strinfo.sub('emailaddr', email_contents)
    strinfo = re.compile('[$]+')
    email_contents = strinfo.sub('dollar', email_contents)
    print('\n==== Processed Email ====\n\n')

    email_contents = re.split('[^a-z]', email_contents)
    notEmpty = lambda s: s and s.strip()
    email_contents = list(filter(notEmpty, email_contents))
    porter_stemmer = PorterStemmer()
    email_contents = list(map(porter_stemmer.stem, email_contents))
    print(email_contents)
    word_indices = []
    for str in email_contents:
        index = np.argwhere(vocabList == str)
        if index.size == 0:
            continue
        word_indices.append(index[0][0])
    word_indices = np.array(word_indices)
    return word_indices


def emailFeatures(word_indices):
    n = 1899
    x = np.zeros((n, 1))
    x[word_indices] = 1
    return x
