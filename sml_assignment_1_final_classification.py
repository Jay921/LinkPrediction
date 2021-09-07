import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
import random
import csv
import pickle
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
import json

train_df = pd.read_csv('train_df.csv')
test_from_train_df = pd.read_csv('dev_df.csv')
test_df = pd.read_csv('test_df.csv')

print(train_df)
print(test_from_train_df)
print(test_df)

# Classification--------------------------------------------------
lr_classifier = LogisticRegression(solver='lbfgs', max_iter=1000)
rf_classifier = RandomForestClassifier(max_depth=10, random_state=0, n_estimators=100)
svm_classifier = SVC(gamma='auto')
mlp_classifier = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(10, 10, 10, 10), activation='relu', solver='adam', max_iter=1000, random_state=0)
nb_classifier = GaussianNB()

# columns = ['ra', 'jcc', 'adamic', 'pref_a', 'katz', 'page_rank', ] 
columns = ['Keywords', 'Venues','cn', 'Shortest_Path', 'ra', 'jcc', 'adamic', 'page_rank', 'katz', 'pref_a', 'hubs']
# columns = ['ra', 'jcc', 'adamic', 'pref_a']

X = train_df[columns]
y = train_df['Labels']


lr_classifier.fit(X,y)
rf_classifier.fit(X,y)
svm_classifier.fit(X,y)
mlp_classifier.fit(X,y)
nb_classifier.fit(X,y)

prediction_probs = lr_classifier.predict_proba(test_from_train_df[columns])
predict = list(prediction_probs)
pred = []
for i in predict:
    pred.append(i[1])

prediction_probs_rf = rf_classifier.predict_proba(test_from_train_df[columns])
predict_rf = list(prediction_probs_rf)
pred_rf = []
for i in predict_rf:
    pred_rf.append(i[1])

prediction_svm = svm_classifier.predict(test_from_train_df[columns])
pred_svm = list(prediction_svm)

prediction_probs_mlp = mlp_classifier.predict_proba(test_from_train_df[columns])
predict_mlp = list(prediction_probs_mlp)
pred_mlp = []
for i in predict_mlp:
    pred_mlp.append(i[1])

prediction_probs_nb = nb_classifier.predict_proba(test_from_train_df[columns])
predict_nb = list(prediction_probs_nb)
pred_nb = []
for i in predict_nb:
    pred_nb.append(i[1])

result_df = pd.DataFrame()
result_df = test_from_train_df[["Source", "Sink"]]
result_df["Predicted"] = pred_rf

predict_unlabeled = lr_classifier.predict_proba(test_df[columns])
pred_unlabeled = list(predict_unlabeled)
pred_u = []
for i in pred_unlabeled:
    pred_u.append(i[1])

predict_unlabeled_rf = rf_classifier.predict_proba(test_df[columns])
pred_unlabeled_rf = list(predict_unlabeled_rf)
pred_u_rf = []
for i in pred_unlabeled_rf:
    pred_u_rf.append(i[1])

prediction_u_svm = svm_classifier.predict(test_df[columns])
pred_u_svm = list(prediction_u_svm)

predict_unlabeled_mlp = mlp_classifier.predict_proba(test_df[columns])
pred_unlabeled_mlp = list(predict_unlabeled_mlp)
pred_u_mlp = []
for i in pred_unlabeled_mlp:
    pred_u_mlp.append(i[1])

predict_unlabeled_nb = nb_classifier.predict_proba(test_df[columns])
pred_unlabeled_nb = list(predict_unlabeled_nb)
pred_u_nb = []
for i in pred_unlabeled_nb:
    pred_u_nb.append(i[1])

result_unlabeled_df = pd.DataFrame()
result_unlabeled_df["Id"] = test_df["Id"]
result_unlabeled_df["Predicted"] = pred_u

print(result_unlabeled_df)

result_unlabeled_df.to_csv('final_prediction.csv', index=False)

def evaluate_model(predictions, actual):
    return pd.DataFrame({
        "Measure": ["Accuracy", "Precision", "Recall", "F1_Score", "AUC_Score"],
        "Score": [accuracy_score(actual, predictions),
            precision_score(actual, predictions),
            recall_score(actual, predictions),
            f1_score(actual, predictions),
            roc_auc_score(actual, predictions)]
    })

print('Logistic Regression:')
p = [i.round() for i in pred]
print(evaluate_model(p, test_from_train_df["Labels"]))

print('Random Forest:')
p = [i.round() for i in pred_rf]
print(evaluate_model(p, test_from_train_df["Labels"]))
print(rf_classifier.feature_importances_)

print('SVM:')
p = [i.round() for i in pred_svm]
print(evaluate_model(p, test_from_train_df["Labels"]))

print('MLP:')
p = [i.round() for i in pred_mlp]
print(evaluate_model(p, test_from_train_df["Labels"]))

print('Naive Bayes:')
p = [i.round() for i in pred_nb]
print(evaluate_model(p, test_from_train_df["Labels"]))



