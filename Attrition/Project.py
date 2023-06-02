#Importing excel dataset into a DataFrame
import pandas as pd
df=pd.read_excel(r"C:\Users\Administrator\Desktop\alk\Attrition Rate-Dataset.xlsx")
output=pd.read_excel(r"C:\Users\Administrator\Desktop\alk\Attrition Rate-Dataset.xlsx")
print(df)

#Check if there is any null values
df.isnull().sum()

df.info()    #Check the data type of each column

df.value_counts()   #counting the values in each column

df.nunique()     #Finding how many unique values are there in each column

df.describe()   #describe the dataframe

print(df.groupby('Attrition')['EmployeeID'].count())

print(df.groupby('WorkLifeBalance')['EmployeeID'].count())

print(df.groupby('Designation')['EmployeeID'].count())

#Convert the categorical values into numerical values using label_encoder object
from sklearn.preprocessing import LabelEncoder
label_encoder =LabelEncoder()
 
df['Attrition']=label_encoder.fit_transform(df['Attrition'])     # Encode labels in column
df['Designation']= label_encoder.fit_transform(df['Designation'])
df['WorkLifeBalance']=label_encoder.fit_transform(df['WorkLifeBalance'])

#Drop the unwanted columns
df=df.drop(['EmployeeName','EmployeeID'],axis=1)

print(df)

### First Moment bussiness understanding.
## Mean dataframe
df.mean()

df.median()   #Median of dataframe

df.var()   #variance of dataframe

df.std()     ##Standard deviation of dataframe

## Skewness and kurtossis
df.skew()
df.kurt()

#For plotting import the libraries
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.figure(figsize=(10,5))
plt.bar(df['Designation'],df['TraininginHours'])
plt.xlabel("Designation")
plt.ylabel("TraininginHours")
plt.show()

plt.figure(figsize=(10,5))
plt.bar(df['Designation'],df['MonthlySalary'])
plt.xlabel("Designation")
plt.ylabel("MonthlySalary")
plt.show()

plt.figure(figsize=(10,5))
plt.bar(df['Designation'],df['PercentSalaryHike'])
plt.xlabel("Designation")
plt.ylabel("PercentSalaryHike")
plt.show()

plt.figure(figsize=(10,5))
plots=sns.barplot(x="Designation",y="MonthlySalary",data=df)
plt.show()

#Pie chart for Designation column
b=df['Designation'].value_counts().reset_index()
labels=b['index']
plt.pie(b['Designation'],labels=labels,autopct='%0.2f%%')
plt.show()

#Pie chart for TraininginHours column
b=df['TraininginHours'].value_counts().reset_index()
labels=b['index']
plt.pie(b['TraininginHours'],labels=labels,autopct='%0.2f%%')
plt.show()

#Pie chart for WorkLifeBalance column
b=df['WorkLifeBalance'].value_counts().reset_index()
labels=b['index']
plt.pie(b['WorkLifeBalance'],labels=labels,autopct='%0.2f%%')
plt.show()

"""Box plot using all columns"""

plt.boxplot(df.MonthlySalary)

plt.boxplot(df.Tenure)

plt.boxplot(df.PercentSalaryHike)

plt.boxplot(df.TraininginHours)

"""By using Boxplot for each column,PercentSalaryHike has no quartiles and MonthlySalary has most number of quartiles"""

plt.hist(df.Tenure)

plt.hist(df.Designation)

plt.hist(df.PercentSalaryHike)

plt.hist(df.TraininginHours)

plt.hist(df.WorkLifeBalance)

## Data Destribution
from scipy import stats
import pylab

stats.probplot(df.Tenure, dist="norm",plot=pylab)
plt.show()

stats.probplot(df.TraininginHours, dist="norm",plot=pylab)
plt.show()

stats.probplot(df.PercentSalaryHike, dist="norm",plot=pylab)
plt.show()

stats.probplot(df.MonthlySalary, dist="norm",plot=pylab)
plt.show()

## Heatmap for the attrition data
sns.heatmap(df.corr(), annot = True, fmt = '.0%')
plt.show()

#variance of each column
print("Variance of Designation",np.var(df['Designation']))
print("Variance of PercentSalaryHike",np.var(df['PercentSalaryHike']))
print("Variance of TraininginHours",np.var(df['TraininginHours']))
print("Variance of WorkLifeBalance",np.var(df['WorkLifeBalance']))
print("Variance of MonthlySalary",np.var(df['MonthlySalary'])) 	
print("Variance of Tenure",np.var(df['Tenure']))

"""Here MonthlySalary has highest variance;WorkLifeBalance has low variance"""

#range of each column
print("Range of Designation",np.ptp(df['Designation']))
print("Range of PercentSalaryHike",np.ptp(df['PercentSalaryHike']))
print("Range of TraininginHours",np.ptp(df['TraininginHours']))
print("Range of WorkLifeBalance",np.ptp(df['WorkLifeBalance']))
print("Range of MonthlySalary",np.ptp(df['MonthlySalary'])) 	
print("Range of Tenure",np.ptp(df['Tenure']))

"""Here MonthlySalary has highest range;WorkLifeBalance has low range"""

## Correlation values between the output and input variables
np.corrcoef(df.Attrition,df.MonthlySalary)

np.corrcoef(df.Attrition,df.Designation)

np.corrcoef(df.Attrition,df.PercentSalaryHike)

np.corrcoef(df.Attrition,df.Tenure)

np.corrcoef(df.Attrition,df.WorkLifeBalance)

np.corrcoef(df.Attrition,df.TraininginHours)

## Dividing the attrition data into input and output data
x=df.drop(['Attrition'],axis=1)
y=df['Attrition']

## Splitting thee data into train and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

#Using Decision Tree Algorithm
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
clf=DecisionTreeClassifier()
clf.fit(x_train, y_train)

y_pred_dt=clf.predict(x_test)     # Making Predictions with Our Model
# Measuring the accuracy of our model
print("Accuracy of the model using Decision Tree Classifier on test data:")
print(accuracy_score(y_test,y_pred_dt))

y_pred_dt1=clf.predict(x_train)
print("Accuracy of the model using Decision Tree Classifier on train data:")
print(accuracy_score(y_train,y_pred_dt1))

#Using RandomForestClassifier Algorithm
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

clf = RandomForestClassifier(n_estimators = 100)  
clf.fit(x_train, y_train)
  
# performing predictions on the test dataset
y_pred_rf=clf.predict(x_test)
print("Accuracy of the model using Random Forest Classifier on test data:")
print(metrics.accuracy_score(y_test, y_pred_rf))
# performing predictions on the train dataset
y_pred_rf1=clf.predict(x_train)
print("Accuracy of the model using Random Forest Classifier on train data:")
print(metrics.accuracy_score(y_train, y_pred_rf1))

from sklearn.model_selection import cross_val_score

# 10-Fold Cross validation
y_pred_rf=np.mean(cross_val_score(clf, x_test, y_test, cv=10))
y_pred_rf1=np.mean(cross_val_score(clf, x_train, y_train, cv=10))
print("For Test data:",y_pred_rf)
print("For Train data:",y_pred_rf1)

#Using XG Boost Algorithm
import xgboost as xgb
model=xgb.XGBClassifier()
model.fit(x_train, y_train)

# performing predictions on the test dataset
y_pred_xg=model.predict(x_test)
print("Accuracy of the model using XG Classifier on test dataset:")
print(accuracy_score(y_test, y_pred_xg))

# performing predictions on the train dataset
y_pred_xg1=model.predict(x_train)
print("Accuracy of the model using XG Classifier on train dataset:")
print(accuracy_score(y_train, y_pred_xg1))

#using Bagging Classifier Algorithm
from sklearn.ensemble import BaggingClassifier
model=BaggingClassifier(n_estimators=500)
model.fit(x_train, y_train)

# performing predictions on the test dataset
y_pred_bag=model.predict(x_test)
print("Accuracy of the model using Bagging on the test dataset:")
print(accuracy_score(y_test, y_pred_bag))

# performing predictions on the train dataset
y_pred_bag1=model.predict(x_train)
print("Accuracy of the model using Bagging on the train dataset:")
print(accuracy_score(y_train, y_pred_bag1))

#Using KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier().fit(x_train, y_train)   # fitting Training Set

# performing predictions on the test dataset
y_pred_knn=model.predict(x_test)
print('Accuracy of the model using KNeighbors Classifier on the test dataset:',accuracy_score(y_test,y_pred_knn))

# performing predictions on the train dataset
y_pred_knn1=model.predict(x_train)
print('Accuracy of the model using KNeighbors Classifier on the train dataset:',accuracy_score(y_train,y_pred_knn1))

#Using GaussianNB
from sklearn.naive_bayes import GaussianNB
model=GaussianNB().fit(x_train, y_train)

# performing predictions on the test dataset
y_pred_nb=model.predict(x_test)
print('Accuracy of the model using Naive Bayes Classifier on the test dataset:',accuracy_score(y_test,y_pred_nb))

# performing predictions on the train dataset
y_pred_nb1=model.predict(x_train)
print('Accuracy of the model using Naive Bayes Classifier on the train dataset:',accuracy_score(y_train,y_pred_nb1))

#Using Stacking Classifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier

estimators=[('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
              ('knn', KNeighborsClassifier(n_neighbors=10)),
              ('gbdt',GradientBoostingClassifier())]

clf=StackingClassifier(estimators=estimators,final_estimator=LogisticRegression(),cv=11)
clf.fit(x_train, y_train)

# performing predictions on the test dataset
y_pred=clf.predict(x_test)
print("Accuracy score using Stacking on the test dataset:",accuracy_score(y_test,y_pred))

# performing predictions on the train dataset
y_pred1=clf.predict(x_train)
print("Accuracy score using Stacking on the train dataset:",accuracy_score(y_train,y_pred1))

#Find the ROC-AUC score
from sklearn.metrics import roc_curve, roc_auc_score
rf = RandomForestClassifier(n_estimators=5, max_depth=2)      # Train a Random Forest classifier
rf.fit(x_train, y_train)
y_pred_prob = rf.predict_proba(x_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob, pos_label=1)
roc_auc = roc_auc_score(y_test, y_pred_prob)    # Compute the ROC AUC score
print("roc_auc score is:",roc_auc)

# Plot the ROC curve
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')       # roc curve for tpr = fpr 
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("Receiver Operating Characteristic")
plt.legend(loc="lower right")
plt.show()

# compute the confusion matrix
from sklearn.metrics import confusion_matrix

y_pred_rf=clf.predict(x_test)
cm = confusion_matrix(y_test,y_pred_rf)
 
#Plot the confusion matrix.
sns.heatmap(cm,
            annot=True,
            fmt='g')
plt.ylabel('Prediction')
plt.xlabel('Actual')
plt.title('Confusion Matrix')
plt.show()

y_pred_rf1=clf.predict(x_train)
cm = confusion_matrix(y_train,y_pred_rf1)
 
#Plot the confusion matrix.
sns.heatmap(cm,
            annot=True,
            fmt='g')
plt.ylabel('Prediction')
plt.xlabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Finding precision,recall and f1 scores on test data
from sklearn.metrics import precision_score, recall_score, f1_score
accuracy=accuracy_score(y_test, y_pred_rf)
print("Accuracy   :", accuracy)
precision=precision_score(y_test, y_pred_rf)
print("Precision :", precision)
recall=recall_score(y_test, y_pred_rf)
print("Recall    :", recall)
F1_score=f1_score(y_test, y_pred_rf)
print("F1-score  :", F1_score)

# Finding precision,recall and f1 scores on train data
accuracy=accuracy_score(y_train, y_pred1)
print("Accuracy   :", accuracy)
precision=precision_score(y_train, y_pred1)
print("Precision :", precision)
recall=recall_score(y_train, y_pred1)
print("Recall    :", recall)
F1_score=f1_score(y_train, y_pred1)
print("F1-score  :", F1_score)

## For Building model obeject ( UI ) #############

import pickle

x_train['pred']=y_pred_rf1
x_test['pred']=y_pred_rf

newTable =  pd.concat([x_train,x_test],axis=0)        
     
df4 = pd.merge(newTable, output[['EmployeeID','EmployeeName']], left_index=True, right_index=True)

with open('finalModel_randForest.pkl', 'wb') as f:
    pickle.dump(df4, f)