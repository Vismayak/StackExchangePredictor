import numpy as np
import pandas as pd
import math
import operator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
# import xgboost as xgb
from hpsklearn import HyperoptEstimator

if __name__ == "__main__":
    #load the dataset
    data = pd.read_csv('Accepted_answer_prediction_data_train.txt', sep="\t", header=None)
    labels = pd.read_csv('Accepted_answer_prediction_labels_train.txt', sep="\t", header=None)
    data = data.iloc[:,[1,2,4]]
    labels = labels.iloc[:,1]
    # print(data.head, data.isna().sum())
    # print(labels.head)
    classWeight = {1:0.8,0:0.2}

    dataNew = list()
    #preprocess the data
    allWords = set()
    for i in range(data.shape[0]):
        # print(data.iloc[i,0])
        if not isinstance(data.iloc[i,0], float):
            allWords.update(data.iloc[i,0].split(" "))
    # print(allWords, len(allWords))
    allWordsList = sorted(list(allWords))
    wordsToIdx = [(i[1],i[0]) for i in enumerate(allWordsList)]
    wordsToIdx = dict(wordsToIdx)
    # print(wordsToIdx)
    for i in range(data.shape[0]):
        #convert to array
        arr = np.zeros(len(allWordsList))
        if not isinstance(data.iloc[i,0], float):
            for word in data.iloc[i,0].split(" "):
                arr[wordsToIdx[word]] = 1
            # print(arr)
        np.append(arr,data.iloc[i,1])
        # print("i,1:",data.iloc[i,1])
        np.append(arr,data.iloc[i,2])
        dataNew.append(arr)
    # print(dataNew)

    X_train, X_test, y_train, y_test = train_test_split(dataNew, labels, test_size=0.15, random_state=42)
    print("Training LG...")
    # curr_classifier = HyperoptEstimator(classifier = LogisticRegression(random_state=0,class_weight = classWeight))
    lg = LogisticRegression(random_state=0,class_weight = classWeight).fit(X_train, y_train)
    print("Training CNB...")
    cnb = ComplementNB().fit(X_train, y_train)
    print("Training GNB...")
    gnb = GaussianNB().fit(X_train, y_train)
    print("Training KNN...")
    curr_classifier = KNeighborsClassifier(n_neighbors=10)
    # curr_classifier = HyperoptEstimator(classifier=KNeighborsClassifier(n_neighbors=5))
    neigh = curr_classifier.fit(X_train, y_train)
    print("Training MLP...")
    mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1).fit(X_train, y_train)
    print("Training DTC...")
    dtc = DecisionTreeClassifier().fit(X_train,y_train)
    print("Training GBM...")
    curr_classifier=GradientBoostingClassifier()
    # curr_classifier = HyperoptEstimator(classifier = GradientBoostingClassifier())
    gbm = curr_classifier.fit(X_train,y_train)

    print("Predicting LG...")
    y_predLG = lg.predict(X_test)
    print("Predicting CNB...")
    y_predCNB = cnb.predict(X_test)
    print("Predicting GNB...")
    y_predGNB = gnb.predict(X_test)
    print("Predicting KNN...")
    y_predNEIGH = neigh.predict(X_test)
    print("Predicting MLP...")
    y_predMLP = mlp.predict(X_test)
    print("Predicting DTC...")
    y_predDTC = dtc.predict(X_test)
    print("Predicting GBM...")
    y_predGBM = gbm.predict(X_test)
    # y_pred = np.zeros(len(y_pred))
    predictors = [y_predLG,y_predCNB,y_predNEIGH,y_predMLP,y_predDTC,y_predGBM]

    accLG = accuracy_score(y_test, y_predLG)
    accCNB = accuracy_score(y_test, y_predCNB)
    accGNB = accuracy_score(y_test, y_predGNB)
    accNEIGH = accuracy_score(y_test,y_predNEIGH)
    accMLP = accuracy_score(y_test,y_predMLP)
    accDTC = accuracy_score(y_test,y_predDTC)
    accGBM = accuracy_score(y_test,y_predGBM)
    accuracies = [accLG, accCNB, accNEIGH, accMLP, accDTC, accGBM]

    print("accuracy (LG, CNB, KNN, MLP, DTC, GBM):", accLG, accCNB, accNEIGH, accMLP, accDTC, accGBM)

    y_pred = list()
    for i in range(len(y_predLG)):
        predictions = {0:0, 1:0}
        for idx, predictor in enumerate(predictors):
            predictions[predictor[i]] += accuracies[idx]
            # predictions[predictor[i]] += 1
        # print(predictions)
        k = max(predictions.items(), key=operator.itemgetter(1))[0]
        y_pred.append(k)
    # y_pred = np.sum(np.multiply(y_predLG,accLG*100),np.multiply(y_predCNB,accCNB*100))
    # print(y_pred)
    finalAcc = accuracy_score(y_test,y_pred)

    # scores = cross_val_score(lg, X_train, y_train, cv=3)
    # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()
    print(tp,fp,fn,tn)
    precision = precision_score(y_test,y_pred)
    recall =  recall_score(y_test,y_pred)

    print("combined accuracy:",finalAcc)
    print("False Positives:",fp)
    print("False Negatives:",fn)
    print("Correct Predictions:", tn + tp)
    print("Precision",precision)
    print("Recall",recall)
    print("10 nearest neighbors and swapped weights")
