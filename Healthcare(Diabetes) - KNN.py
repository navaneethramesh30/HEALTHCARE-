#!/usr/bin/env python
# coding: utf-8

# In[306]:


import sys
import numpy
import pandas
import matplotlib
import seaborn
import scipy
import sklearn


# In[307]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[308]:


data=pd.read_csv('health care diabetes.csv')
backup=pd.read_csv('health care diabetes.csv')


# In[309]:


data.info()


# In[310]:


data.head()


# In[372]:


data.describe()


# In[312]:


data.head()


# In[313]:


#some features are 0 in dataset indicating missing values so replacing them with Nan values
data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
data.info()
backup[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = backup[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
data.info()


# In[314]:


data['Glucose'].fillna( method ='ffill', inplace =True) 
data['BloodPressure'].fillna( method ='ffill', inplace =True) 
data['SkinThickness'].fillna( method ='ffill', inplace =True) 
data['Insulin'].fillna( method ='ffill', inplace =True) 
data['BMI'].fillna( method ='ffill', inplace =True) 
#df['Glucose'].fillna( method ='ffill', inplace =True) 


# In[315]:


data.head()


# In[316]:


data.info()
data.hist()
plt.show()


# In[317]:


(data['Pregnancies'].value_counts()) 
data['Pregnancies'].value_counts().plot()


# In[318]:


(data['Glucose'].value_counts())
data['Glucose'].value_counts().plot()


# In[319]:


(data['BloodPressure'].value_counts())
data['BloodPressure'].value_counts().plot()


# In[320]:


(data['SkinThickness'].value_counts())
data['SkinThickness'].value_counts().plot()


# In[321]:


(data['Insulin'].value_counts())
data['Insulin'].value_counts().plot()


# In[322]:


(data['BMI'].value_counts())
data['BMI'].value_counts().plot()


# In[323]:


(data['DiabetesPedigreeFunction'].value_counts())
data['DiabetesPedigreeFunction'].value_counts().plot()


# In[324]:


#print(df['Age'].value_counts())
data['Age'].value_counts().plot()


# In[325]:


print(data['Outcome'].value_counts())
data['Outcome'].value_counts().plot.bar()
#The number of non-diabetic is 500 the number of diabetic patients is 268


# In[326]:


data['Insulin'] = data['Insulin'].fillna(0)
data['Insulin'] = data['Insulin'].replace(0,100)


# In[ ]:





# In[327]:


sns.pairplot(data,hue='Outcome')


# In[328]:


data.head()


# In[329]:


cor=data.corr()
cor


# In[330]:


corr=data.corr()

sns.set(font_scale=1.15)
plt.figure(figsize=(14, 10))

sns.heatmap(corr, vmax=.8, linewidths=0.01,
            square=True,annot=True,cmap='YlGnBu',linecolor="black")
plt.title('Correlation between features');


# In[331]:


data.head()


# In[332]:


#relation between pregnancy and outcome , categorical vs numerical 
pd.crosstab(data['Pregnancies'],data['Outcome'])


# In[333]:


#graph between pregnancy and outcome categorical vs numerical dala
plt.bar(data['Outcome'],data['Pregnancies'],color='red',edgecolor='black')
plt.xlabel('Outcome', fontsize=16)
plt.ylabel('Pregnancies', fontsize=16)
plt.show()


# In[334]:


#Graph between glucose and outcome categorical vs numerical data
plt.bar(data['Outcome'],data['Glucose'],color='red',edgecolor='black')
plt.xlabel('Outcome', fontsize=16)
plt.ylabel('Glucose', fontsize=16)
plt.show()


# In[335]:


#graph between bmi and outcome, categorical vs numerical data
plt.bar(data['Outcome'],data['BMI'],color='red',edgecolor='black')
plt.xlabel('Outcome', fontsize=16)
plt.ylabel('BMI', fontsize=16)
plt.show()


# In[336]:


#analysis of pregnancies and age
sns.pointplot(data['Pregnancies'], data['Age'], hue=data['Outcome'])


# In[337]:


#analysis of insulin and skin thickness
sns.pointplot(data['Insulin'], data['SkinThickness'], hue=data['Outcome'])


# In[338]:


#analysis of BMI and Skin Thickness
sns.pointplot(data['BMI'], data['SkinThickness'], hue=data['Outcome'])


# In[339]:


#analysis of Insulin and Glucose
sns.pointplot(data['Insulin'], data['Glucose'], hue=data['Outcome'])


# In[340]:


df=data
temp=data


# In[341]:


#making groups for age
a=pd.Series([])
for i in df.index:
    if(df.loc[i:i,]['Age']<=24).bool():
        a=a.append(pd.Series(['<=24']))
    elif(df.loc[i:i,]['Age']<=30).bool():
        a=a.append(pd.Series(['25-30']))
    elif(df.loc[i:i,]['Age']<=40).bool():
        a=a.append(pd.Series(['31-40']))
    elif(df.loc[i:i,]['Age']<=55).bool():
        a=a.append(pd.Series(['41-55']))
    else:
        a=a.append(pd.Series(['>55']))
a.reset_index(drop=True,inplace=True)
df['Age']=a
df.head()

#Find the number of diabetic person in each age group

df1=df[temp['Outcome']==1].groupby('Age')[['Outcome']].count()
df1
df1.head()


# In[342]:


df2=df.groupby('Age')[['Outcome']].count()
df1['Diabetic %']=(df1['Outcome']/df2['Outcome'])*100
df1


# In[343]:


#graph for age
sns.barplot(df1.index,df1['Diabetic %'])


# In[344]:


data.head()


# In[354]:


data['Age'] = backup['Age']


# In[355]:


print("The dataset have nine attributes(parameters) in which there are eight independent variables (Pregnancies,Glucose,Blood Pressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age) and one dependent variable (Outcome).BMI and DiabetesPedigreeFunction are a float data type and other parameters are integer data type.The parameters do not contain any null values (missing values). However, this can not be true. As Insulin,SkinThickness,BloodPressure,BMI,Glucose have zero values.The Outcome parameter shows that there are 500 healthy people and 268 Diabetic people.It means that 65% people are diabetic and 34.9% people are healthy.The parameters Glucose, BloodPressure, BMI are normally distributed. Pregnancies,Insulin,Age,DiabetesPedigreeFunction are rightly skewed.The missing values '0' is replaced by the mean of the parameter to explore the dataset.BloodPressure,SkinThickness,Insulin,BMI have outliers.There are no convincing relationship between the parameters.Pregnancies and age have some kind of a linear line. BloodPressure and age have little relation. Most of the aged people have BloodPressure.Insulin and Glucose have some relation.Glucose, Age BMI and Pregnancies are the most Correlated features with the Outcome.Insulin and DiabetesPedigreeFunction have little correlation with the outcome. BloodPressure and SkinThickness have tiny correlation with the outcome.Age and Pregnancies,Insulin and Skin Thickness, BMI and Skin Thickness,Insulin and Glucose are little correlated.The midle aged women are most likely to be diabetic than the young women. As the percentage of diabetic women are 48% and 59% in the age group of 31-40 and 41-55.After Pregnancy people have more chance of diabeties.People with high Glucose level are more likely to have diabeties.People with high BloodPressure have more chance of diabeties.People with high Insulin level are more likely to have Diabetes.*/")


# In[356]:


data.head()


# In[357]:


data.describe()


# In[358]:


X = data.drop('Outcome',axis=1)
y = data['Outcome']


# In[359]:


#importing train_test_split
from sklearn.model_selection import train_test_split


# In[360]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=40)


# In[361]:


from sklearn.neighbors import KNeighborsClassifier

#Setup arrays to store training and test accuracies
neighbors = np.arange(1,9)
train_accuracy =np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))


# In[362]:


for i,k in enumerate(neighbors):
    #Setup a knn classifier with k neighbors
    knn = KNeighborsClassifier(n_neighbors=k)
    
    #Fit the model
    knn.fit(X_train, y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)
    
    #Compute accuracy on the test set
    test_accuracy[i] = knn.score(X_test, y_test) 
    
plt.title('k-NN Varying number of neighbors')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show() 


# In[363]:


for i in range(0,8):
    print("K = ",i+1," Train_accuracy :",train_accuracy[i])
    print("       ","Test_accuracy :",test_accuracy[i])


# In[ ]:





# ## Hence, We were at a conclusion that best k value is 7

# In[364]:


knn = KNeighborsClassifier(n_neighbors=7)


# In[365]:


knn.fit(X_train,y_train)


# In[366]:


from sklearn.metrics import confusion_matrix,accuracy_score


# In[367]:


y_pred = knn.predict(X_test)


# In[368]:


confusion_matrix(y_test,y_pred)


# In[369]:


print("Accuracy :",accuracy_score(y_test,y_pred))


# In[370]:


#import classification_report
from sklearn.metrics import classification_report


# In[371]:


print(classification_report(y_test,y_pred))


# In[ ]:




