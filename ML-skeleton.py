import pandas as pd 
import numpy as np
import csv 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn import tree

try:
    # importing the csv file into a data structure known as pandas DataFrame
    df = pd.read_csv("test.csv", header=None)
    print(df)
except ValueError:
    print("Error... no csv file or file is curruppted")

# You might not need this next line if you do not care about losing information about flow_id etc.
#These are the names that will be assigned to the fields of the csv --> make sure to name these correctly
columns_list = ['srcIP', 'dstIP', 'srcPort', 'destPort', 'proto', 'totalPkts', 'srcPkts', 'destPkts', 'totalBytes', 'srcBytes', 'destBytes','time1', 'time2', 'label']
df.columns = columns_list

#These are the actual features that will be used to feed the machine learning model
features = ['totalPkts', 'srcPkts', 'destPkts', 'totalBytes', 'srcBytes', 'destBytes', 'time1', 'time2']

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
acc_scores = 0
for i in range(0, 10):
    
    #Splitting the dataset into training set and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

    #MACHINE LEARNING MODEL 1---------------------------------------------------
    #Decision Trees
    #Reference: https://scikit-learn.org/stable/modules/tree.html#tree
    
    clf = tree.DecisionTreeClassifier()            #initialize the Decision Tree
    clf.fit(X_train, y_train)                      #train the model with training sets
    y_pred_class = clf.predict(X_test)                    #test the model
    print('this is y_pred_class: ', y_pred_class)
    acc_scores += accuracy_score(y_test, y_pred_class)
    print(acc_scores)
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
print('')
print('')
print('total score: ', acc_scores)
results = (acc_scores/10)
print('avg score', results)
#===============================================================================
#EVALUATION PORTION-------------------------------------------------------------
    #here you are supposed to calculate the evaluation measures indicated in the project proposal (accuracy, F-score etc)
#result = clf.score(X_test, y_test)  #accuracy score
