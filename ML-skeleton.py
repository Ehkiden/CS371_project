#Program:   ML-skeleton.py
#Authors:   Daniel Weigle, David Mercado, Jared Rigdon
#Purpose:   Perform various machine learning algorithms and runs them against the collected flows from the scapy-skeleton.py to calculate the prediction accuracy and other performance metrics for each ML algorithm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn import tree
from sklearn.neural_network import MLPClassifier


try:
    # importing the csv file into a data structure known as pandas DataFrame
    df = pd.read_csv('flowData.csv', header=None)
    feat = []
    line = []
    with open('features.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            feat.append(int(float(row[0])))
except ValueError:
    print("Error... no csv file or file is curruppted")

#These are the names that will be assigned to the fields of the csv --> make sure to name these correctly
columns_list = ['srcIP', 'dstIP', 'srcPort', 'destPort', 'proto', 'totalPkts', 'srcPkts', 'destPkts', 'totalBytes', 'srcBytes', 'destBytes', 'currTime', 'durrTime', 'label']
df.columns = columns_list

#These are the actual features that will be used to feed the machine learning model
features = ['totalPkts', 'srcPkts', 'destPkts', 'totalBytes', 'srcBytes', 'destBytes', 'durrTime']

X = df[features]    #This are features, needed for training and testing sets
y = df['label']     #This are labels, needed for training and testing sets

print('features: ', feat)
WebB = []
VidSt = []
VidCon = []
FileDL = []

for x in range(0,len(feat),4):
    WebB.append(feat[x])
    VidSt.append(feat[x+1])
    VidCon.append(feat[x+2])
    FileDL.append(feat[x+3])

#Will run the machine learning model 10 times (cross validation) and store the values
TreeAccScores = []
TreePrecsionScores = []
TreeRecallScores = []
TreeF1Scores = []

mlpAccScores = []
mlpPrecsionScores = []
mlpRecallScores = []
mlpF1Scores = []

svcAccScores = []
svcPrecsionScores = []
svcRecallScores = []
svcF1Scores = []

for i in range(0, 10):
    
    #Splitting the dataset into training set and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

    #MACHINE LEARNING MODEL 1---------------------------------------------------
    #Decision Trees
    
    clf = tree.DecisionTreeClassifier()                                         #initialize the Decision Tree
    clf.fit(X_train, y_train)                                                   #Build a decision tree classifier from the training set (X, y).
    y_pred_class = clf.predict(X_test)                                          #Predict class or regression value for X. Returns the predicted classes/values
    #print('y_pred_class is: ', y_pred_class)
    
    TreeAccScores.append(accuracy_score(y_test, y_pred_class))                              #accuracy scores
    TreePrecsionScores.append(precision_score(y_test, y_pred_class, average='weighted'))    #precsion scores
    TreeRecallScores.append(recall_score(y_test, y_pred_class, average='weighted'))         #recall scores
    TreeF1Scores.append(f1_score(y_test, y_pred_class, average='weighted'))                 #f1 scores
    
    #MACHINE LEARNING MODEL 2---------------------------------------------------
    # Neural network (MultiPerceptron Classifier)

    clf = MLPClassifier()
    clf.fit(X_train, y_train)
    y_pred_class = clf.predict(X_test)                                          #Predict class or regression value for X. Returns the predicted classes/values
    
    mlpAccScores.append(accuracy_score(y_test, y_pred_class))                              #accuracy scores
    mlpPrecsionScores.append(precision_score(y_test, y_pred_class, average='weighted'))    #precsion scores
    mlpRecallScores.append(recall_score(y_test, y_pred_class, average='weighted'))         #recall scores
    mlpF1Scores.append(f1_score(y_test, y_pred_class, average='weighted'))                 #f1 scores

    #MACHINE LEARNING MODEL 3---------------------------------------------------
    #SVM's
    clf = SVC(gamma='auto')     #SVC USE THIS
    clf = LinearSVC()  #Linear SVC
    clf.fit(X_train, y_train)
    y_pred_class = clf.predict(X_test)                                          #Predict class or regression value for X. Returns the predicted classes/values
    
    svcAccScores.append(accuracy_score(y_test, y_pred_class))                              #accuracy scores
    svcPrecsionScores.append(precision_score(y_test, y_pred_class, average='weighted'))    #precsion scores
    svcRecallScores.append(recall_score(y_test, y_pred_class, average='weighted'))         #recall scores
    svcF1Scores.append(f1_score(y_test, y_pred_class, average='weighted'))                 #f1 scores

#===============================================================================
#EVALUATION PORTION-------------------------------------------------------------
#Graph results for machine learning models and features for each label

n_bins = 10
ind = np.arange(n_bins)
ind2 = np.arange(len(WebB))
width = 0.35
executions = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
fig, axes = plt.subplots(nrows=2, ncols=4, constrained_layout=True)
ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7 = axes.flatten()

#Web browsing-------------------------------------------------------------------
print('web browsing: ', WebB)
ax0.bar(ind2, WebB, width, color='gray')
ax0.set_title('Web Browsing', fontweight='bold')
ax0.set_ylabel('Occurrences', size='8')
ax0.set_xticks(ind2)
ax0.set_xticklabels(features, size='5')
ax0.set_xlabel('Features', size='8')
ax0.set_yscale('symlog')

#Video streaming (ie Youtube)---------------------------------------------------
print('video streaming: ', VidSt)
ax1.bar(ind2, VidSt, width, color='red')
ax1.set_title('Video Streaming', fontweight='bold')
ax1.set_ylabel('Occurrences', size='8')
ax1.set_xticks(ind2)
ax1.set_xticklabels(features, size='5')
ax1.set_xlabel('Features', size='8')
ax1.set_yscale('symlog')

#Video conference (ie skype)----------------------------------------------------
print('video conference: ', VidCon)
ax2.bar(ind2, VidCon, width, color='black')
ax2.set_title('Video Conference', fontweight='bold')
ax2.set_ylabel('Occurrences', size='8')
ax2.set_xticks(ind2)
ax2.set_xticklabels(features, size='5')
ax2.set_xlabel('Features', size='8')
ax2.set_yscale('symlog')

#File download-----------------------------------------------------------------------
print('File Download: ', FileDL)
ax3.bar(ind2, FileDL, width, color='purple')
ax3.set_title('File Download', fontweight='bold')
ax3.set_ylabel('Occurrences', size='8')
ax3.set_xticks(ind2)
ax3.set_xticklabels(features, size='5')
ax3.set_xlabel('Features', size='8')
ax3.set_yscale('symlog')

#Accuracy-----------------------------------------------------------------------
treeBarAcc = ax4.bar(ind, TreeAccScores, width)
mlpBarAcc = ax4.bar(ind, mlpAccScores, width, bottom=TreeAccScores)
svcBarAcc = ax4.bar(ind, svcAccScores, width, bottom=mlpAccScores)

ax4.set_title('Accuracy', fontweight='bold')
ax4.set_ylabel('Accuracy Score', size='8')
ax4.set_xticks(ind)
ax4.set_xticklabels(executions, size='5')
ax4.set_xlabel('Executions', size='8')
ax4.legend((treeBarAcc[0], mlpBarAcc[0], svcBarAcc[0]), ('Decision Tree', 'Netural Network', 'Support Vector Machine'), prop={'size': 4})

#Precision----------------------------------------------------------------------
treeBarPre = ax5.bar(ind, TreePrecsionScores, width)
mlpBarPre = ax5.bar(ind, mlpPrecsionScores, width, bottom=TreePrecsionScores)
svcBarPre = ax5.bar(ind, svcPrecsionScores, width, bottom=mlpPrecsionScores)

ax5.set_title('Precision', fontweight='bold')
ax5.set_ylabel('Precision Score', size='8')
ax5.set_xticks(ind)
ax5.set_xticklabels(executions, size='5')
ax5.set_xlabel('Executions', size='8')
ax5.legend((treeBarPre[0], mlpBarPre[0], svcBarPre[0]), ('Decision Tree', 'Netural Network', 'Support Vector Machine'), prop={'size': 4})

#Recall-------------------------------------------------------------------------
treeBarRecall = ax6.bar(ind, TreeRecallScores, width)
mlpBarRecall = ax6.bar(ind, mlpRecallScores, width, bottom=TreeRecallScores)
svcBarRecall = ax6.bar(ind, svcRecallScores, width, bottom=mlpRecallScores)

ax6.set_title('Recall', fontweight='bold')
ax6.set_ylabel('Recall Score', size='8')
ax6.set_xticks(ind)
ax6.set_xticklabels(executions, size='5')
ax6.set_xlabel('Executions', size='8')
ax6.legend((treeBarRecall[0], mlpBarRecall[0], svcBarRecall[0]), ('Decision Tree', 'Netural Network', 'Support Vector Machine'), prop={'size': 4})

#F1-----------------------------------------------------------------------------
treeBarF1 = ax7.bar(ind, TreeF1Scores, width)
mlpBarF1 = ax7.bar(ind, mlpF1Scores, width, bottom=TreeF1Scores)
svcBarF1 = ax7.bar(ind, svcF1Scores, width, bottom=mlpF1Scores)

ax7.set_title('F1', fontweight='bold')
ax7.set_ylabel('F1 Score', size='8')
ax7.set_xticks(ind)
ax7.set_xticklabels(executions, size='5')
ax7.set_xlabel('Executions', size='8')
ax7.legend((treeBarF1[0], mlpBarF1[0], svcBarF1[0]), ('Decision Tree', 'Netural Network', 'Support Vector Machine'), prop={'size': 4})

plt.show()
