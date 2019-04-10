import pandas as pd 
import numpy as np
import csv 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn import tree

# importing the csv file into a data structure known as pandas DataFrame
df = pd.read_csv("data.csv", header=None)

# You might not need this next line if you do not care about losing information about flow_id etc.
#These are the names that will be assigned to the fields of the csv --> make sure to name these correctly
columns_list = ['flow_id', 'IPsrc', 'IPdst', 'proto', 'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5', 'label']
df.columns = columns_list

#These are the actual features that will be used to feed the machine learning model
features = ['proto', 'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']

X = df[features]    #This are features, needed for training and testing sets
y = df['label']     #This are labels, needed for training and testing sets

#Labels
# 1 = Web browsing                      NEED 25 FLOWS FOR THIS CASE
# 2 = Video streaming (ie Youtube)      NEED 25 FLOWS FOR THIS CASE
# 3 = Vido conference (ie skype)        NEED 25 FLOWS FOR THIS CASE
# 4 = File download                     NEED 25 FLOWS FOR THIS CASE
#                                       FOR A TOTAL OF 100 SAMPLES (FLOWS)

#Will run the machine learning model 10 times (cross validation)
acc_scores = 0
for i in range(0, 10):
    
    #Splitting the dataset into training set and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

    #MACHINE LEARNING MODEL 1---------------------------------------------------
    #Decision Trees
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    #MACHINE LEARNING MODEL 2---------------------------------------------------
    # Neural network (MultiPerceptron Classifier)
    # clf = MLPClassifier()
    # clf.fit(X_train, y_train)

    #MACHINE LEARNING MODEL 3---------------------------------------------------
    #SVM's
    # clf = SVC(gamma='auto')     #SVC USE THIS
    # clf = LinearSVC()  #Linear SVC
    # clf.fit(X_train, y_train) 

#===============================================================================
#EVALUATION PORTION-------------------------------------------------------------
    #here you are supposed to calculate the evaluation measures indicated in the project proposal (accuracy, F-score etc)
    result = clf.score(X_test, y_test)  #accuracy score
