import nltk
from nltk import FreqDist
import math
import pickle

# 计算mylist中每个元素出现的概率 log scale
def frequency(mylist):
    p = {}
    freq = FreqDist(mylist)
    length = len(mylist)
    keys = list(freq.keys())
    for key in keys:
        p[key] = math.log(freq[key]/length)
    return p

filename1 = 'E:\SI 561\hw1\wsj1-18.training'
fileTrain = open(filename1, 'r').read()

collection = fileTrain.split('\n')
freq = FreqDist(fileTrain.split())
num = len(collection)

# calculate initial probabilities
# the list of the tags of the initial word
initialList = []
# calculate transition probabilities
# keys: tag, values: next tag
transitionDict = {}
transitionP = {}
# calculate observation probabilities
# keys: tag, values: corresponding word
observationDict = {}
observationP = {}

i=1
wordSet = []
for document in collection:
    tokens = document.split()
    for index in range(0,len(tokens),2):
        if freq[tokens[index]]<3:
            tokens[index] = 'UNKA'    
    tags = []
    words = []
    for index in range(1,len(tokens),2):
        tags.append(tokens[index])
        words.append(tokens[index-1])
    initialList.append(tags[0])
    wordSet = wordSet + words
    for tagIndex in range(len(tags)-1):
        if tags[tagIndex] not in transitionDict:
            transitionDict[tags[tagIndex]] = []
        transitionDict[tags[tagIndex]].append(tags[tagIndex+1])
        if tags[tagIndex] not in observationDict:
            observationDict[tags[tagIndex]] = []
        observationDict[tags[tagIndex]].append(words[tagIndex])
        
    i = i+1

# calculate observation probabilities
for tag in observationDict.keys():
    wordList = observationDict[tag]
    wordP = frequency(wordList)
    observationP[tag] = wordP
 
# calculate transition probabilities
for tag in transitionDict.keys():
    nextTagList = transitionDict[tag]
    nextP = frequency(nextTagList)
    transitionP[tag] = nextP

# calculate initial probabilities
initialP = frequency(initialList)


# Add smoothing
#wordSet = set(wordSet)
#totalWordCount = len(wordSet)

#for tag in observationP.keys():
#    wordP = observationP[tag]
#    valCount = len(wordP)
#    for word in wordP.keys():
#        wordP[word] = math.log(math.exp(wordP[word])-0.75/len(observationDict[tag]))
#    observationP[tag] = wordP
#    for word in wordSet-set(wordP.keys()):
#        observationP[tag][word] = math.log(0.75 * valCount/(totalWordCount-valCount)/len(observationDict[tag]))

#tagSet = set(transitionP.keys())
#totalTagCount = len(transitionP.keys())        
#for tag in transitionP.keys():
#    nextP = transitionP[tag]
#    for nextTag in nextP.keys():
#        nextP[nextTag] = math.log(math.exp(nextP[nextTag])-0.75/len(transitionDict[tag]))
#    transitionP[tag] = nextP
#    for nextTag in tagSet-set(nextP.keys()):
#        transitionP[tag][nextTag] = math.log(0.75 * len(nextP)/(totalTagCount - len(nextP))/len(transitionDict[tag]))

filename2 = 'E:\SI 561\hw1\model.pyc'
fileModel = open(filename2, 'wb')
pickle.dump(initialP, fileModel)
pickle.dump(transitionP, fileModel)
pickle.dump(observationP, fileModel)
