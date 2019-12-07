import nltk
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import svm
import re
import csv

    
trainDocuments = []
trainLabels = []

with open('E:/github/SI-561-Project/train.csv', 'r', encoding='UTF-8') as f:
    csvreader = csv.reader(f)
    headers = next(f)
    for line in csvreader:
        trainDocuments.append(line[4])
        trainLabels.append(line[0])
        # trainLabels.append(int(line[0])>4)
        
testDocuments = []
testLabels = []
c2 = 0
with open('E:/github/SI-561-Project/test.csv', 'r', encoding='UTF-8') as f:
    csvreader = csv.reader(f)
    for line in csvreader:
        testDocuments.append(line[4])
        testLabels.append(line[0])
        # testLabels.append(int(line[0])>4)
        # if int(line[0])<4: 
            # c2 += 1
# print(c2)

vectorizer = TfidfVectorizer(analyzer = "word", stop_words = 'english', preprocessor = None, encoding='utf-8', ngram_range=(1,2), sublinear_tf = True)
documentsVectors = vectorizer.fit_transform(trainDocuments+testDocuments)
size1 = len(trainDocuments)
clf = svm.LinearSVC(C = 1.5)
clf.fit(documentsVectors[0:size1],trainLabels[0:size1])

prediction = clf.predict(documentsVectors[size1:])
count = 0
for ans, truth in zip(prediction, testLabels):
    if ans == truth:
        count += 1
    # if ans != '5':
        # print(ans)
print(count/len(testLabels))