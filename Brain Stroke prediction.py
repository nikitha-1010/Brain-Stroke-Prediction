import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df.head(10)
df.isnull().sum()
# Data Cleaning
def clean(df1):
    col=['bmi']
    for c in col:
        df1[c].fillna(df1[c].median(),inplace=True)
    return df1
clean(df)
df.isnull().sum()
df.columns.values
df=df.drop('id',axis=1)
df['gender']=df['gender'].replace('Other','Female')
df.head()
#creating dummy variables for numeric-binary attributes
df[['hypertension','heart_disease','stroke']]=df[['hypertension','heart_disease','stroke']].astype(str)
df=pd.get_dummies(df,drop_first=True)
df.head()
from imblearn.over_sampling import RandomOverSampler

# Performing a minority oversampling
oversample = RandomOverSampler(sampling_strategy='minority')
x=df.drop(['stroke_1'],axis=1)
y=df['stroke_1']

# Obtaining the oversampled dataframes - testing and training
x_over, y_over = oversample.fit_resample(x, y)
from sklearn.preprocessing import StandardScaler
s=StandardScaler()
df[['bmi', 'avg_glucose_level', 'age']] = s.fit_transform(df[['bmi', 'avg_glucose_level', 'age']])
#splitting test-train data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_over,y_over,test_size=0.20,random_state=42)
x_train
print('x_train:', x_train.shape)
print('y_train:', y_train.shape)
print('x_test:', x_test.shape)
print('y_test:', y_test.shape)
# Decision Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import auc,roc_auc_score,roc_curve,precision_score,recall_score,f1_score
clf = DecisionTreeClassifier()
clf = clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn.metrics import auc,roc_auc_score,roc_curve,precision_score,recall_score,f1_score
# 2 neighbours because of the 2 classes
knn = KNeighborsClassifier(n_neighbors = 2)
knn.fit(x_train,y_train)
y_pred_knn = knn.predict(x_test)
y_pred_prob_knn = knn.predict_proba(x_test)[:, 1]
confusion_matrix(y_test, y_pred_knn)
print('Accuracy:',accuracy_score(y_test, y_pred_knn))
print('ROC AUC Score:', roc_auc_score(y_test, y_pred_prob_knn))
# Random forest Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
rf_clf = RandomForestClassifier(n_estimators = 100)
rf_clf.fit(x_train, y_train)
y_pred_rf = rf_clf.predict(x_test)
print('Accuracy:', accuracy_score(y_test, y_pred_rf))
age=21
avg_glucose_level=64.4
bmi=31
gender_Male=0
ever_married_Yes=0
work_type_Never_worked=0
work_type_Private=0
work_type_Self_employed=0
work_type_children=0
Residence_type_Urban=0
smoking_status_formerly_smoked=0
smoking_status_never_smoked=1
smoking_status_smokes=0
hypertension_1=0
heart_disease_1=0
input_features = [age,avg_glucose_level,bmi,gender_Male,hypertension_1,heart_disease_1,ever_married_Yes,work_type_Never_worked,work_type_Private,work_type_Self_employed,work_type_children	,Residence_type_Urban,	smoking_status_formerly_smoked,smoking_status_never_smoked,smoking_status_smokes]
features_value = [np.array(input_features)]
features_name = ['age','avg_glucose_level','bmi','gender_Male','hypertension_1','heart_disease_1','ever_married_Yes','work_type_Never_worked','work_type_Private','work_type_Self-employed','work_type_children','Residence_type_Urban','smoking_status_formerly smoked','smoking_status_never smoked','smoking_status_smokes']
df = pd.DataFrame(features_value, columns=features_name)
prediction = rf_clf.predict(df)[0]
if prediction==0:
    print('Congratulations! You are not Diagnosed with a stroke risk')
else:
    print('You are diagnosed with a stroke risk')
