#import the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

###################################
#DATA EXPLORATION
###################################
dataset = pd.read_csv("data/creditcard.csv")

dataset.head()

dataset.shape
dataset.columns
dataset.info()

#statistical summary
dataset.describe()

#Dealing with missing values
dataset.isnull().values.any()

#false no null values in this data set

dataset.isnull().values.sum()

# Encoding categorical data

dataset.select_dtypes(include="object").columns

len(dataset.select_dtypes(include="object").columns)

# no the cathegorical values in this dataset

#Countplot
sns.countplot(dataset["Class"])
#non fraud transactions
(dataset.Class==0).sum()

# fraud transactions
(dataset.Class == 1).sum()

#Correlation matrix and heatmap

dataset_2 = dataset.drop(columns='Class')
#dropping the column
#specifify the target varaiable
#correclation of independendant variable with class

dataset_2.corrwith(dataset['Class']).plot.bar(
figsize=(16, 9), title='Correlated with Class', grid=True)

corr = dataset.corr()

plt.figure(figsize=(16, 9))
ax = sns.heatmap(corr, annot=True, linewidths=2)

#####################################
# Splitting the dataset to train and test set
####################################

dataset.head()

# matrix of features / independent variables
#define X
X = dataset.drop(columns='Class')

# target variable / dependent variable
#define y
y = dataset['Class']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

#import Class

X_train.shape

y_train.shape

X_test.shape

y_test.shape

#######################################

#Feature scaling, import library, creating instance of the class, applying feature scalling
#######################################

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


X_train
X_test

###################################
#BUILDING THE MODEL
###################################
# 1) Logistic regression
####################################

from sklearn.linear_model import LogisticRegression
classifier_lr = LogisticRegression(random_state=0)
classifier_lr.fit(X_train, y_train)
#import the library and train the model

y_pred = classifier_lr.predict(X_test)

#import metrics
from sklearn.metrics import confusion_matrix, accuracy_score

#define accuracy
acc = accuracy_score(y_test, y_pred)
print(acc*100)

#confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


#wrong predictions --46

###############################

#2)Random Forest
###############################
#import RandomforestClassifier, creating the instance of class, training the model

from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(random_state=0)
classifier_rf.fit(X_train, y_train)


y_pred = classifier_rf.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
acc = accuracy_score(y_test, y_pred)
print(acc*100)

cm = confusion_matrix(y_test, y_pred)
print(cm)

#################################

#3)XGBoost Classifier
################################

from xgboost import XGBClassifier 
classifier_xgb = XGBClassifier(random_state=0)
classifier_xgb.fit(X_train, y_train)


y_pred = classifier_xgb.predict(X_test)


from sklearn.metrics import confusion_matrix, accuracy_score
acc = accuracy_score(y_test, y_pred)
print(acc*100)

cm = confusion_matrix(y_test, y_pred)
print(cm)

#############################
#FINAL MODEL (RANDOM FOREST)
############################


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
acc = accuracy_score(y_test, y_pred)
print(acc*100)

cm = confusion_matrix(y_test, y_pred)
print(cm)


##########################
#Predicting a single observation
##########################

dataset.head()

dataset.shape

single_obs = [[0.0, -1.359807,-0.072781, 2.536347,1.378155,-0.338321,0.462388,0.239599,0.098698,0.363787,	0.090794,	-0.551600,	-0.617801,	-0.991390,	-0.311169,	1.468177,-0.470401,0.207971,0.025791,0.403993, 0.251412,-0.018307,	0.277838,	-0.110474,	0.066928,0.128539,-0.189115,0.133558,-0.021053,149.62]]


#applying feature scalling
classifier.predict(sc.transform(single_obs))