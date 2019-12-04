import nltk
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import SGDClassifier
# from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier 
from sklearn import svm
import re
import csv
import os
import pickle


EMBED_FILE = 'E:/github/SI-561-Project/glove.6B.300d.txt'
EMBEDDING_DIM = 300

def loadEmbedModel(embedFile):
    outname = embedFile + '.pkl'
    if os.path.isfile(outname):
        model = pickle.load(open(outname,'rb'))
    else:
        f = open(embedFile,'rb')
        model = {}
        # f.readline()
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            # embedding = np.array([float(val) for val in splitLine[1:]])
            embedding = [float(val) for val in splitLine[1:]]
            model[word] = embedding
        f.close()
        pickle.dump(model, open(outname,'wb'))
    return model


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
with open('E:/github/SI-561-Project/test.csv', 'r', encoding='UTF-8') as f:
    csvreader = csv.reader(f)
    for line in csvreader:
        testDocuments.append(line[4])
        testLabels.append(line[0])


size1 = len(trainDocuments)
sentences = trainDocuments+testDocuments
labels = trainLabels+testLabels
model = loadEmbedModel(EMBED_FILE)

embeddings = []
for sentence in sentences:
    sum = np.zeros(EMBEDDING_DIM)
    for word in sentence:
        if word.lower().encode() in model.keys():
            sum = sum + np.array(model[word.lower().encode()])
    embeddings.append(sum/len(sentence))

clf = RandomForestClassifier()
clf.fit(embeddings[0:size1],trainLabels[0:size1])

prediction = clf.predict(embeddings[size1:])
count = 0
for ans, truth in zip(prediction, testLabels):
    if ans == truth:
        count += 1
    if ans != '5':
        print(ans)
print(count/len(testLabels))