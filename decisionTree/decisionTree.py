# python 3.6
# decisionTree.py

# Resources:
#   - http://saedsayad.com/decision_tree.htm
#      - Used for help understanding entropy split and information gain
#

import numpy as np

training = np.genfromtxt('car.training.csv', dtype=None, encoding="utf-8", delimiter=',')
print("training shape: {}, {}".format(training.shape[0], training.shape[1]))
test = np.genfromtxt('car.test.csv', dtype=None, encoding="utf-8", delimiter=',')
print("test shape: {}, {}".format(test.shape[0], test.shape[1]))

def featureAttributeOccurance(attr, featCol, data):
    count = 0
    for d in range(len(data[:,featCol])):
        if data[d, featCol] == attr:
            if data[d, -1] == 'acc':
                count+=1
    return count, sum(data[:,featCol]==attr) - count

def accuracy(preds, labels):
    return float(sum(preds == labels[:,-1])) / len(labels)

def trainDecisionTree():
    return "hoopla"
def giniIndex(dataset):
    #assuming labels are at the end
    labels = dataset[:,-1]
    uniq_classes = set(labels)
    return uniq_classes

def entropy(labeledData): # this is assuming the labels are in the last column
    # Entropy :
    totalSamples = len(labeledData[:,-1])
    return 0.5

# class DecisionTree:
#     def __init__(self, featCol, featAttr, leftTree, rightTree):
#         self.featCol = featCol
#         self.featAttr = featAttr
#         self.leftTree = leftTree
#         self.rightTree = rightTree
#
# def bruteForceSplit(data): # greedy strategy that locally optimizes
#
#
# def trainTree(): # based on Hunt's Algorithm





















    #
