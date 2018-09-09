# python 3.6
# decisionTree.py

# Resources:
#   - http://saedsayad.com/decision_tree.htm
#      - Used for help understanding entropy split and information gain
#

import numpy as np
import math

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
        if d[featCol] == featVal:
            matchBranch.append(d)
        else:
            notMatchBranch.append(d)
    return matchBranch, notMatchBranch

print("starting entropy of training set: \n\t" + str(entropy(countSplit(training))))
print("if split on col1=\"vhigh\":")
vhigh, nvhigh = performSplit(training, 0, "vhigh")
print("vhigh:\n\t" + str(entropy(countSplit(vhigh))))
print("not vhigh:\n\t" + str(entropy(countSplit(nvhigh))))

















    #
