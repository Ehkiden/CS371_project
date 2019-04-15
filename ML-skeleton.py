import pandas as pd     #library
import numpy as np      #library
import matplotlib.pyplot as plt   #library
import csv

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.svm import SVC                                     #machine learning model
from sklearn.svm import LinearSVC                               #machine learning model
from sklearn import tree                                        #machine learning model
from sklearn.neural_network import MLPClassifier                #machine learning model


try:
    # importing the csv file into a data structure known as pandas DataFrame
    df = pd.read_csv('test1.csv', header=None)
    print(df)
    #df2 = pd.read_csv("features.csv", header=None)
    #print(df2)

    feat = []
    with open('features.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            feat.append(row)
#print(row)
#print(row[0])
#print(row[0],row[1],row[2],)
except ValueError:
    print("Error... no csv file or file is curruppted")

# You might not need this next line if you do not care about losing information about flow_id etc.
#These are the names that will be assigned to the fields of the csv --> make sure to name these correctly
columns_list = ['srcIP', 'dstIP', 'srcPort', 'destPort', 'proto', 'totalPkts', 'srcPkts', 'destPkts', 'totalBytes', 'srcBytes', 'destBytes','currTime', 'durrTime', 'label']
df.columns = columns_list

#columns_list2 = ['totalPkts', 'srcPkts', 'destPkts', 'totalBytes', 'srcBytes', 'destBytes','currTime', 'durrTime', 'label']
#df2.columns = columns_list2

#These are the actual features that will be used to feed the machine learning model
features = ['totalPkts', 'srcPkts', 'destPkts', 'totalBytes', 'srcBytes', 'destBytes', 'currTime', 'durrTime']

X = df[features]    #This are features, needed for training and testing sets
y = df['label']     #This are labels, needed for training and testing sets
#feat_X = df2[features]
#feat_Y = df2['label']

print('feat 0:', feat[0], '\n')
print('feat 1:', feat[1], '\n')
print('feat 2:', feat[2], '\n')
print('feat 3:', feat[3], '\n')
#print('feat Y:', feat_Y)
#Labels----------------------------------------------------------------
# 1 = Web browsing                      NEED 25 FLOWS FOR THIS CASE
# 2 = Video streaming (ie Youtube)      NEED 25 FLOWS FOR THIS CASE
# 3 = Video conference (ie skype)       NEED 25 FLOWS FOR THIS CASE
# 4 = File download                     NEED 25 FLOWS FOR THIS CASE
#                                       FOR A TOTAL OF 100 SAMPLES (FLOWS)
labelWeb = 1
labelVStream = 2
labelVConfrence = 3
labelFileDL = 4

#Will run the machine learning model 10 times (cross validation)
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
    #Reference: https://scikit-learn.org/stable/modules/tree.html#tree
    #Reference: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
    #Reference: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score
    #Reference: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score
    #Reference: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    
    clf = tree.DecisionTreeClassifier()                                         #initialize the Decision Tree
    clf.fit(X_train, y_train)                                                   #Build a decision tree classifier from the training set (X, y).
    y_pred_class = clf.predict(X_test)                                          #Predict class or regression value for X. Returns the predicted classes/values
    #print('y_pred_class is: ', y_pred_class)

    #acc = accuracy_score(y_test, y_pred_class)
    #print('accScores is: ', acc)
    #result = clf.score(X_test, y_test)
    #print('results is: ', result)
    
    TreeAccScores.append(accuracy_score(y_test, y_pred_class))                              #accuracy scores
    TreePrecsionScores.append(precision_score(y_test, y_pred_class, average='weighted'))    #precsion scores
    TreeRecallScores.append(recall_score(y_test, y_pred_class, average='weighted'))         #recall scores
    TreeF1Scores.append(f1_score(y_test, y_pred_class, average='weighted'))                 #f1 scores
    
    #MACHINE LEARNING MODEL 2---------------------------------------------------
    # Neural network (MultiPerceptron Classifier)

    clf = MLPClassifier()
    clf.fit(X_train, y_train)

    y_pred_class = clf.predict(X_test)                                          #Predict class or regression value for X. Returns the predicted classes/values
    #print('y_pred_class is: ', y_pred_class)
    
    #acc = accuracy_score(y_test, y_pred_class)
    #print('accScores is: ', acc)
    #result = clf.score(X_test, y_test)
    #print('results is: ', result)
    
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
    #print('y_pred_class is: ', y_pred_class)
    
    #acc = accuracy_score(y_test, y_pred_class)
    #print('accScores is: ', acc)
    #result = clf.score(X_test, y_test)
    #print('results is: ', result)
    
    svcAccScores.append(accuracy_score(y_test, y_pred_class))                              #accuracy scores
    svcPrecsionScores.append(precision_score(y_test, y_pred_class, average='weighted'))    #precsion scores
    svcRecallScores.append(recall_score(y_test, y_pred_class, average='weighted'))         #recall scores
    svcF1Scores.append(f1_score(y_test, y_pred_class, average='weighted'))                 #f1 scores

#resultsAcc = (acc_scores/10)
#print('avg score', resultsAcc)
#===============================================================================
#EVALUATION PORTION-------------------------------------------------------------
    #here you are supposed to calculate the evaluation measures indicated in the project proposal (accuracy, F-score etc)

print('Complete TreeAccScores array: ', TreeAccScores)
print('Complete TreePrecsionScores array: ', TreePrecsionScores)
print('Complete TreeRecallScores array: ', TreeRecallScores)
print('Complete TreeF1Scores array: ', TreeF1Scores)

print('Complete mplAccScores array: ', mlpAccScores)
print('Complete mplPrecsionScores array: ', mlpPrecsionScores)
print('Complete mplRecallScores array: ', mlpRecallScores)
print('Complete mplF1Scores array: ', mlpF1Scores)

print('Complete svcAccScores array: ', svcAccScores)
print('Complete svcPrecsionScores array: ', svcPrecsionScores)
print('Complete svcRecallScores array: ', svcRecallScores)
print('Complete svcF1Scores array: ', svcF1Scores)

#===============================================================================
#Graph results for machine learning models and features for each label
#var for features
f1 = 'TotalPkts'
f2 = 'SrcPkts'
f3 = 'DestPkts'
f4 = 'TotalBytes'
f5 = 'SrcBytes'
f6 = 'DestBytes'
f7 = 'DurrTime'

#sample data
#sampleDATA = (3,4,1,3,4,6,6,7,5,4)
np.random.seed(19680801)

n_bins = 10
x = np.random.randn(1000, 4)
ind = np.arange(n_bins)
width = 0.25

fig, axes = plt.subplots(nrows=1, ncols=5, constrained_layout=True)
#fig, axes = plt.subplots(nrows=1, ncols=3, constrained_layout=True)
ax0, ax1, ax2, ax3, ax4 = axes.flatten()

#All Features-------------------------------------------------------------------
colors = ['red', 'blue', 'green', 'purple']
labelType = ['Web Browsing', 'Video Streaming', 'Video Conference', 'File Download']
ax0.hist(feat, histtype='bar', color=colors, label=labelType)
ax0.legend() #prop={'size': 5}
ax0.set_title('Feature Details')
ax0.set_ylabel('Occurrences')
#ax0.set_xticks(7, ('totalPkts', 'srcPkts', 'destPkts', 'totalBytes', 'srcBytes', 'destBytes', 'currTime', 'durrTime'))
ax0.set_xlabel('Features')

#Accuracy-----------------------------------------------------------------------
treeBarAcc = ax1.bar(ind, TreeAccScores, width)
mlpBarAcc = ax1.bar(ind, mlpAccScores, width, bottom=TreeAccScores)
svcBarAcc = ax1.bar(ind, svcAccScores, width, bottom=mlpAccScores)

ax1.set_title('Accuracy')
ax1.set_ylabel('Accuracy Score')
ax1.set_xticks(ind, ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10'))
ax1.set_xlabel('Executions')
ax1.legend((treeBarAcc[0], mlpBarAcc[0], svcBarAcc[0]), ('Decision Tree', 'Netural Network', 'Support Vector Machine'), prop={'size': 5})

#Precision----------------------------------------------------------------------
treeBarPre = ax2.bar(ind, TreePrecsionScores, width)
mlpBarPre = ax2.bar(ind, mlpPrecsionScores, width, bottom=TreePrecsionScores)
svcBarPre = ax2.bar(ind, svcPrecsionScores, width, bottom=mlpPrecsionScores)

ax2.set_title('Precision')
ax2.set_ylabel('Precision Score')
ax2.set_xticks(ind, ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10'))
ax2.set_xlabel('Executions')
ax2.legend((treeBarPre[0], mlpBarPre[0], svcBarPre[0]), ('Decision Tree', 'Netural Network', 'Support Vector Machine'), prop={'size': 5})

#Recall-------------------------------------------------------------------------
treeBarRecall = ax3.bar(ind, TreeRecallScores, width)
mlpBarRecall = ax3.bar(ind, mlpRecallScores, width, bottom=TreeRecallScores)
svcBarRecall = ax3.bar(ind, svcRecallScores, width, bottom=mlpRecallScores)

ax3.set_title('Recall')
ax3.set_ylabel('Recall Score')
ax3.set_xticks(ind, ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10'))
ax3.set_xlabel('Executions')
ax3.legend((treeBarRecall[0], mlpBarRecall[0], svcBarRecall[0]), ('Decision Tree', 'Netural Network', 'Support Vector Machine'), prop={'size': 5})

#F1-----------------------------------------------------------------------------
treeBarF1 = ax4.bar(ind, TreeF1Scores, width)
mlpBarF1 = ax4.bar(ind, mlpF1Scores, width, bottom=TreeF1Scores)
svcBarF1 = ax4.bar(ind, svcF1Scores, width, bottom=mlpF1Scores)

ax4.set_title('F1')
ax4.set_ylabel('F1 Score')
ax4.set_xticks(ind, ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10'))
ax4.set_xlabel('Executions')
ax4.legend((treeBarF1[0], mlpBarF1[0], svcBarF1[0]), ('Decision Tree', 'Netural Network', 'Support Vector Machine'), prop={'size': 5})

#fig.tight_layout()
plt.show()
'''
#-----------------------------------------------------------------------------
treeBar = ax5.bar(ind, TreeAccScores, width)
mlpBar = ax5.bar(ind, MLP, width, bottom=TreeAccScores)
svcBar = ax5.bar(ind, SVC, width, bottom=MLP)

ax5.set_title('Accuracy')
ax5.set_ylabel('Accuracy Score')
ax5.set_xticks(ind, ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10'))
ax5.set_xlabel('Executions')
ax5.legend((treeBar[0], mlpBar[0], svcBar[0]), ('Decision Tree', 'Netural Network', 'Support Vector Machine'))
'''
'''
ax2.hist(x, n_bins, histtype='step', stacked=True, fill=False)
ax2.set_title('stack step (unfilled)')

# Make a multiple-histogram of data-sets with different length.
x_multi = [np.random.randn(n) for n in [10000, 5000, 2000]]
ax3.hist(x_multi, n_bins, histtype='bar')
ax3.set_title('different sample sizes')

ax4.hist(x, n_bins, density=True, histtype='bar', stacked=True)
ax4.set_title('stacked bar')

ax5.hist(x, n_bins, density=True, histtype='bar', stacked=True)
ax5.set_title('stacked bar')
'''




'''
#Graph results from machine learning model--Accuracy
N = 10
ind = np.arange(N)
width = 0.25

#Tree = (1,2,3,4,5,6,7,8,9,1)
#MLP = (3,4,1,3,4,6,6,7,5,4)
#SVC = (9,6,4,7,5,3,5,7,8,6)
plt.figure()

treeBar = plt.bar(ind, TreeAccScores, width)
#mlpBar = plt.bar(ind, MLP, width, bottom=TreeAccScores)
#svcBar = plt.bar(ind, SVC, width, bottom=MLP)

plt.title('Accuracy')
plt.ylabel('Accuracy Score')
plt.xticks(ind, ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10'))
plt.xlabel('Executions')
#plt.legend((treeBar[0], mlpBar[0], svcBar[0]), ('Decision Tree', 'Netural Network', 'Support Vector Machine'))
plt.show()
'''
'''
#Graph results from machine learning model--Precision
N = 10
ind = np.arange(N)
width = 0.25

Tree = (1,2,3,4,5,6,7,8,9,1)
MLP = (3,4,1,3,4,6,6,7,5,4)
SVC = (9,6,4,7,5,3,5,7,8,6)
plt.figure()

treeBar = plt.bar(ind, Tree, width)
mlpBar = plt.bar(ind, MLP, width, bottom=Tree)
svcBar = plt.bar(ind, SVC, width, bottom=MLP)

plt.title('Precision')
plt.ylabel('Precision Score')
plt.xticks(ind, ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10'))
plt.xlabel('Executions')
plt.legend((treeBar[0], mlpBar[0], svcBar[0]), ('Decision Tree', 'Netural Network', 'Support Vector Machine'))
plt.show()

#Graph results from machine learning model--Recall
N = 10
ind = np.arange(N)
width = 0.25

Tree = (1,2,3,4,5,6,7,8,9,1)
MLP = (3,4,1,3,4,6,6,7,5,4)
SVC = (9,6,4,7,5,3,5,7,8,6)
plt.figure()

treeBar = plt.bar(ind, Tree, width)
mlpBar = plt.bar(ind, MLP, width, bottom=Tree)
svcBar = plt.bar(ind, SVC, width, bottom=MLP)

plt.title('Recall')
plt.ylabel('Recall Score')
plt.xticks(ind, ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10'))
plt.xlabel('Executions')
plt.legend((treeBar[0], mlpBar[0], svcBar[0]), ('Decision Tree', 'Netural Network', 'Support Vector Machine'))
plt.show()

#Graph results from machine learning model--F1 score
N = 10
ind = np.arange(N)
width = 0.25

Tree = (1,2,3,4,5,6,7,8,9,1)
MLP = (3,4,1,3,4,6,6,7,5,4)
SVC = (9,6,4,7,5,3,5,7,8,6)
plt.figure()

treeBar = plt.bar(ind, Tree, width)
mlpBar = plt.bar(ind, MLP, width, bottom=Tree)
svcBar = plt.bar(ind, SVC, width, bottom=MLP)

plt.title('F1')
plt.ylabel('F1 Score')
plt.xticks(ind, ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10'))
plt.xlabel('Executions')
plt.legend((treeBar[0], mlpBar[0], svcBar[0]), ('Decision Tree', 'Netural Network', 'Support Vector Machine'))
plt.show()
'''
