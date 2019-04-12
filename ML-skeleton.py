import pandas as pd     #library
import numpy as np      #library
import matplotlib.pyplot as plt   #library
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC             #machine learning model
from sklearn.svm import LinearSVC       #machine learning model
from sklearn import tree                #machine learning model


try:
    # importing the csv file into a data structure known as pandas DataFrame
    df = pd.read_csv("test.csv", header=None)
    print(df)
except ValueError:
    print("Error... no csv file or file is curruppted")

# You might not need this next line if you do not care about losing information about flow_id etc.
#These are the names that will be assigned to the fields of the csv --> make sure to name these correctly
columns_list = ['srcIP', 'dstIP', 'srcPort', 'destPort', 'proto', 'totalPkts', 'srcPkts', 'destPkts', 'totalBytes', 'srcBytes', 'destBytes','currTime', 'durrTime', 'label']
df.columns = columns_list

#These are the actual features that will be used to feed the machine learning model
features = ['totalPkts', 'srcPkts', 'destPkts', 'totalBytes', 'srcBytes', 'destBytes', 'currTime', 'durrTime']

X = df[features]    #This are features, needed for training and testing sets
y = df['label']     #This are labels, needed for training and testing sets

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
mlpAccScores = []
svcAccScores = []

TreePrecsionScores = []
mlpPrecsionScores = []
svcPrecsionScores = []

TreeRecallScores = []
mlpRecallScores = []
svcRecallScores = []

TreeF1Scores = []
mlpF1Scores = []
svcF1Scores = []

for i in range(0, 10):
    
    #Splitting the dataset into training set and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

    #MACHINE LEARNING MODEL 1---------------------------------------------------
    #Decision Trees
    #Reference: https://scikit-learn.org/stable/modules/tree.html#tree
    
    clf = tree.DecisionTreeClassifier()            #initialize the Decision Tree
    clf.fit(X_train, y_train)                      #train the model with training sets
    y_pred_class = clf.predict(X_test)             #test the model
    TreeAccScores.append(accuracy_score(y_test, y_pred_class))
#TODO
#Precision
#recall
#F1 score

    #results = clf.score(X_test, y_test)
    #print(empty(results))
    #print(bool(results))
    #print(item(results))
    #print(all(results))
    
    #MACHINE LEARNING MODEL 2---------------------------------------------------
    # Neural network (MultiPerceptron Classifier)
    # clf = MLPClassifier()
    # clf.fit(X_train, y_train)

    #MACHINE LEARNING MODEL 3---------------------------------------------------
    #SVM's
    # clf = SVC(gamma='auto')     #SVC USE THIS
    # clf = LinearSVC()  #Linear SVC
    # clf.fit(X_train, y_train)

#resultsAcc = (acc_scores/10)
#print('avg score', resultsAcc)
#===============================================================================
#EVALUATION PORTION-------------------------------------------------------------
    #here you are supposed to calculate the evaluation measures indicated in the project proposal (accuracy, F-score etc)
result = clf.score(X_test, y_test)  #accuracy score
print(result)


#===============================================================================
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
