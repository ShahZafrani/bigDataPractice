import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

def normalize(X):
    return X / 255

def rescale(Y):
    return Y - 5

def ridge_regression(X, y, l):
    return np.dot(np.dot(np.dot(np.linalg.pinv(np.dot(X.transpose(), X)), X.transpose()), y), l * np.identity(X.shape[1]))

def predict(w, b, X, threshold):
    return np.dot(X, w) + b > threshold

def calculate_rmse(predictions, Y):
    return (sum((Y - predictions)**2)/len(Y))**.5

def calculate_tpr_and_fpr(test_y, predictions):
    tpr = 0
    fpr = 0
    tnr = 0
    fnr = 0
    num_positive = sum(test_y == 1)
    num_negative = len(test_y) - num_positive
    assert (len(predictions) == len(test_y))
    for i in range(len(test_y)):
        if (test_y[i] == 1):
            if(predictions[i] > threshold):
                tpr += 1
            else:
                fnr += 1
        if (test_y[i] == 0):
            if(predictions[i] <= threshold):
                tnr += 1
            else:
                fpr += 1
    tpr = float(tpr) / num_positive
    fpr = float(fpr) / num_negative
    return tpr, fpr

if __name__ == '__main__':
    num_folds = 10
    base_lambda = .75
    max_lambda = 1.25
    lambda_val = 1.125
    base_threshold = .4
    max_threshold = .6
    mnist_features = np.genfromtxt('MNIST_15_15.csv', delimiter=',', dtype=int)
    mnist_labels = np.genfromtxt('MNIST_LABEL.csv', delimiter=',', dtype=int)

    kf = KFold(n_splits=num_folds)
    kf.get_n_splits(mnist_labels)
    average_tpr = []
    average_fpr = []
    colors = ['red', 'green', 'blue', 'purple', 'orange']
    thresholds = np.linspace(base_threshold, max_threshold, len(colors))
    for threshold in thresholds:
        # for lambda_val in np.linspace(base_lambda, max_lambda, 5):
        tpr_arr = []
        fpr_arr = []
        print("lambda= {}, Threshold of {}: (TPR, FPR)".format(lambda_val, threshold))
        for train_index, test_index in kf.split(mnist_labels):
            folded_training_X = normalize(mnist_features[train_index])
            folded_training_Y = rescale(mnist_labels[train_index])
            folded_test_X = normalize(mnist_features[test_index])
            folded_test_Y = rescale(mnist_labels[test_index])

            w = ridge_regression(folded_training_X, folded_training_Y, lambda_val)
            b = calculate_rmse(predict(w, 0, folded_training_X, threshold), folded_training_Y)

            predictions = predict(w, b, folded_test_X, threshold)

            # print("\t {}".format(calculate_tpr_and_fpr(folded_test_Y, predictions)))
            tpr, fpr = calculate_tpr_and_fpr(folded_test_Y, predictions)
            tpr_arr.append(tpr)
            fpr_arr.append(fpr)
        avgtpr = sum(tpr_arr)/len(tpr_arr)
        average_tpr.append(tpr_arr)
        avgfpr = sum(fpr_arr)/len(fpr_arr)
        average_fpr.append(fpr_arr)
        print("\t {}, {}".format(avgtpr, avgfpr))
    plt.title('Receiver Operating Characteristic, lambda={}'.format(lambda_val))
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    for t in range(len(thresholds)):
        tprs = average_tpr[t]
        tprs += [0,1]
        tprs.sort()
        fprs = average_fpr[t]
        fprs += [0,1]
        fprs.sort()
        plt.plot(fprs, tprs, colors[t], label="Threshold: {}".format(thresholds[t]))
    plt.legend()
    plt.show()