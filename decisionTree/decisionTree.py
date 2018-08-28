# python 3.6
# decisionTree.py

import numpy as np

training = np.genfromtxt('car.training.csv', dtype=None, delimiter=',')
print("training shape: ",  training.shape)
test = np.genfromtxt('car.test.csv', dtype=None, delimiter=',')
print("test shape: ",  test.shape)

stupid_prediction = training[0, -1]

print(stupid_prediction)
predictions = []
for i in range(test.shape[0]):
    predictions.append(stupid_prediction)

def accuracy(preds, labels):
    return float(sum(preds == labels[:,-1])) / len(labels)

def trainDecisionTree():
    return "hoopla"


print(accuracy(predictions, test))
