import sys
import pickle


def viterbi(observe, initialP, transitionP, observationP):
    v = [] # the value is a dictionary (state, probability)
    order = [] # the predicted order
    # calculate v1
    first = observe[0]
    tempv = {}
    for tag in initialP.keys():
        pi = initialP[tag]
        b = float("-inf")
        if first in observationP[tag]:
            b = observationP[tag][first]  
        p = pi + b
        tempv[tag] = p
    #order.append(max(tempv,key=tempv.get))
    for diffTag in set(transitionP.keys())-set(initialP.keys()):
        tempv[diffTag] = float("-inf")
    v.append(tempv)    
    
    lastBest = []
    for index in range(1,len(observe)):        
        tempv = {}
        tempLastBest = {}
        for tag in transitionP.keys():
            #aijtemp = []
            aijtemp = {}
            for state in transitionP.keys():
                if tag in transitionP[state]:
                    aij = transitionP[state][tag]
                else:
                    #print(tag)
                    #print('a')
                    aij = float("-inf")
                aijtemp[state] = v[index-1][state]+aij
                #aijtemp.append(v[index-1][state]+aij)
            b = float("-inf")
            if observe[index] in observationP[tag]:
                b = observationP[tag][observe[index]]
            p = aijtemp[max(aijtemp, key=aijtemp.get)] + b
            tempLastBest[tag] = max(aijtemp, key=aijtemp.get)
            #p = max(aijtemp) + b
            tempv[tag] = p
        #if observe[index] in [',', '.', "''", '$', '``']:
        #    order.append(observe[index])
        #elif observe[index] in ['?', '!']:
        #    order.append('.')
        #elif observe[index] in ['--']:
        #    order.append(':')
        #else:
        #    order.append(max(tempv,key=tempv.get))
        lastBest.append(tempLastBest)
        v.append(tempv)
        #print(tempv)
        
    lastv = v[len(observe)-1]
    order.append(max(lastv,key=lastv.get))
    nextTag = order[0]
    for index in range(len(lastBest)):
        prevTag = lastBest[len(lastBest)-1-index][nextTag]
        order.append(prevTag)
        nextTag = prevTag
    order.reverse()
    for index in range(len(observe)):
        if observe[index] in [',', '.', "''", '$', '``']:
            order[index] = observe[index]
        elif observe[index] in ['?', '!']:
            order[index] = '.'
        elif observe[index] in ['--']:
            order[index] = ':'
        
    return order  

    
testname=sys.argv[1]
truthname=sys.argv[2]

#testname = 'E:\SI 561\hw1\wsj19-21.testing'
#truthname = 'E:\SI 561\hw1\wsj19-21.truth'

#filename = 'E:\SI 561\hw1\model.pyc'
filename = 'model.pyc'
fileModel = open(filename, 'rb')
initialP = pickle.load(fileModel)
transitionP = pickle.load(fileModel)
observationP = pickle.load(fileModel)

prediction = []
fileTest = open(testname, 'r').read()
testCollection = fileTest.split('\n')
fileTruth = open(truthname, 'r').read()
truth = fileTruth.split()


#m=0
for document in testCollection:
    if document != '':
        prediction = prediction + viterbi(document.split(), initialP, transitionP, observationP)
        
    #if m==4:
    #    print(prediction)
    #    break
    #m=m+1

count = 0
for index in range(len(prediction)):
    if prediction[index] == truth[index*2+1]:
        count = count + 1
print('The accuracy is ', count/len(prediction))