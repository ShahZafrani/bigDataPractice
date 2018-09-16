# python 3.6
# decisionTree.py

import numpy as np
import math
import json

# Hyper Params
MAX_DEPTH = 2


training = np.genfromtxt('car.training.csv', dtype=None, encoding="utf-8", delimiter=',')
print("training shape: {}, {}".format(training.shape[0], training.shape[1]))
test = np.genfromtxt('car.test.csv', dtype=None, encoding="utf-8", delimiter=',')
print("test shape: {}, {}".format(test.shape[0], test.shape[1]))

def accuracy(preds, labels):
    return float(sum(preds == labels[:,-1])) / len(labels)

def entropy(dataSplit): # formatted as an array of counts: [5,9]
    total = sum(dataSplit)
    numClasses = len(dataSplit)
    entropy = 0
    for d in dataSplit:
        prob = float(d) / total
        entropy += -1 * (prob * math.log(prob, 2))
    return entropy

def attrEntropy(labeledData, featCol):
    labeledData = np.array(labeledData)
    dataSize = len(labeledData[:,featCol])
    uniqueAttrs = set(labeledData[:, featCol])
    attrEntropySplit = {}
    attrDataSplit = {}
    combinedFeatureEntropy = 0
    for u in uniqueAttrs:
        uOnly = labeledData[labeledData[:, featCol] == u]
        attrDataSplit[u] = countSplit(uOnly)
        attrEntropySplit[u] =  entropy(attrDataSplit[u])
        combinedFeatureEntropy += float(attrEntropySplit[u]) * (float(sum(attrDataSplit[u])) / dataSize)
    return attrEntropySplit, combinedFeatureEntropy


def countSplit(labeledData): # this is assuming the labels are in the last column
    # Entropy : summation(C[], i): -1 * probability(C[i]) * log2(probability(C[i]))
    labeledData = np.array(labeledData)
    totalSamples = len(labeledData[:,-1])
    uniqueClasses = set(labeledData[:,-1])
    dataSplit = []
    for u in uniqueClasses:
        dataSplit.append(sum(labeledData[:,-1]==u))
    return dataSplit

def performSplit(labeledData, featCol, featVal):
    matchBranch = []
    notMatchBranch = []
    for d in labeledData:
        dval = d[featCol]
        if (dval == featVal):
            matchBranch.append(d)
        else:
            notMatchBranch.append(d)
    return matchBranch, notMatchBranch

def guessFromMajority(data):
    uniqClasses = data[:,-1]
    bestClass = ""
    highestCount = 0
    for u in uniqClasses:
        uCount = sum(data[:,-1]==u)
        if uCount >= highestCount:
            bestClass = u
            highestCount = uCount
    return bestClass


def buildTree(tree, data, currDepth):
    if(currDepth > MAX_DEPTH):
        return guessFromMajority(np.array(data))
    currDepth += 1
    currentEntropy = entropy(countSplit(data))
    bestInfoGain = 0
    tree["splitCol"] = 0
    tree["splitVal"] = ""
    lowestSplit = 1
    for featCol in range(0, len(data[0]) - 1):
        featAttrEntropy, combFeatEntropy = attrEntropy(data, featCol)
        featInfoGain = currentEntropy - combFeatEntropy
        if bestInfoGain < featInfoGain:
            bestInfoGain = featInfoGain
            tree["splitCol"] = featCol
            for k in featAttrEntropy.keys():
                if featAttrEntropy[k] < lowestSplit:
                    tree["splitVal"] = k
                    lowestSplit = featAttrEntropy[k]
    matchData, noMatchData = performSplit(data, tree["splitCol"], tree["splitVal"])
    if (len(matchData) > 0 and entropy(countSplit(matchData)) == 0):
        tree["true"] = np.array(matchData)[0,-1]
    elif (len(matchData) > 0):
        tree["true"] = buildTree({}, matchData, currDepth)
    if (len(noMatchData) > 0 and entropy(countSplit(noMatchData)) == 0):
        tree["false"] = np.array(noMatchData)[0,-1]
    elif (len(noMatchData) > 0):
        tree["false"] = buildTree({}, noMatchData, currDepth)

    return tree

def predictFromTree(tree, dataSample): # takes one row of data at a time
    if isinstance(tree, str): # is it a leaf?
        return tree
    if (dataSample[tree["splitCol"]] == tree["splitVal"]):
        return predictFromTree(tree["true"], dataSample)
    else:
        return predictFromTree(tree["false"], dataSample)

def evaluateTree(tree, testData):
    predictions = []
    for d in testData:
        predictions.append(predictFromTree(tree, d))
    return accuracy(predictions, testData)

#  node: {
#         splitCol: num,
#         splitVal: any,
#         trueNode: node,
#         falseNode: node
#  }
print("building decision tree with maximum depth of {}".format(MAX_DEPTH))
dTree = buildTree({}, training, 0)
print("evaluating decision tree with test data. \n \t accuracy: {}%".format(evaluateTree(dTree, test) * 100))

# Uncomment below to output tree to file as a json
# with open('tree.json', 'w') as fp:
#   json.dump(dTree, fp)











    #
