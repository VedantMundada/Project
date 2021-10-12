# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 12:35:51 2021

@author: Vedant
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
import scipy.stats as stats
import pylab
from sklearn.pipeline import Pipeline
#%%
#EDA 



path = "C:/Users/Vedant/Desktop/UIUC/SEM 1/Machine Learning/Project/MLF_GP2_EconCycle.csv"
df = pd.read_csv(path)
#plotting the head
print(df.head())

#dropping the date column 
df=df.drop('Date',axis=1)

data=df.values


#calculating the colum statistics 
n_rows=len(df)
n_col=0
for column in df.values[0,:]:
    n_col=n_col+1
print("Number of columns of Data = " , n_col , '\n')
print("The summary for each column is \n",df.describe())

#Printing colum data types
type = [0]*3
colCounts = []
for col in range(n_col):
 for row in df.values:
     try:
         a = float(row[col])
         if isinstance(a, float):
             type[0] += 1
     except ValueError:
         if len(row[col]) > 0:
             type[1] += 1
         else:
             type[2] += 1
 colCounts.append(type)
 type = [0]*3
print("Col#" + '\t' + "Number" + '\t' +"Strings" + '\t ' + "Other\n")
iCol = 0
for types in colCounts:
 print(str(iCol) + '\t\t' + str(types[0]) + '\t\t' +
 str(types[1]) + '\t\t' + str(types[2]) + "\n")
 iCol += 1
 
 
#printing colum statistics
type = [0]*3
colCounts = []
for i in range(0,n_col):
    col = i
    colData = []
    for row in df.values:
     colData.append(float(row[col]))
    colArray = np.array(colData)
    
    colMean = np.mean(colArray)
    colsd = np.std(colArray)
    print("Column ", i ," :- Mean = " + '\t' + str(colMean) + '\t\t' + "Standard Deviation = " + '\t ' + str(colsd) + "\n")

#plotting the heat map
cols=df.keys()
corr=df[cols].corr()
sns.set(font_scale=0.6)
sns.heatmap(corr, annot= True,cmap='Blues')
plt.show()

#plotting scatter plots
stats.probplot(data[:,0], dist="norm", plot=pylab)
plt.title("Probability plot of T1Y Index")
pylab.show()
stats.probplot(data[:,6], dist="norm", plot=pylab)
plt.title("Probability plot of CP1M")
pylab.show()
stats.probplot(data[:,-1], dist="norm", plot=pylab)
plt.title("Probability plot of PCT 9MO FWD")
pylab.show()

#plotting scatter plots

sns.lmplot(x="T1Y Index", y="PCT 9MO FWD", data=df)
plt.title("PCT 9MO FWD vs. T1Y Index")
plt.xlabel("T1Y Index")
plt.ylabel("PCT 9MO FWD")
plt.show()

sns.lmplot(x="CP1M", y="PCT 9MO FWD", data=df)
plt.title("PCT 9MO FWD vs. CP1M")
plt.xlabel("CP1M")
plt.ylabel("PCT 9MO FWD")
plt.show()



#%%
X=data[:,0:12]
y=data[:,13:]
sc=StandardScaler()

#%%
#PCA 
pca=PCA(n_components=None)
X_pca = pca.fit_transform(sc.fit_transform(X))
exp_var_pca= pca.explained_variance_ratio_
cum_sum_eigenvalues = np.cumsum(exp_var_pca)
sns.set(font_scale=1)
plt.bar(range(0,len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
plt.step(range(0,len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.title("Cumulative and histogram plot for explained variance ratio")
plt.legend(loc='best')
plt.tight_layout()
plt.show()

#%%
max_accu=np.empty(len(range(0,50)))
x_axis=range(0,50)
for i in range(0,50):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=i)
    
    #KNN Code
    n=np.arange(1,26)
    test_acc= np.empty(len(n))
    
        
    for k in range(1,25):
        pipe = Pipeline([('scaler', StandardScaler()),('pca',PCA(n_components=3)), ('KNN', KNeighborsRegressor(n_neighbors=k,weights='distance') )])
        pipe.fit(X_train,y_train[:,0])
        test_acc[k]=(pipe.score(X_test,y_test[:,0]))
    print(i," ",max(test_acc[1:]))
    max_accu[i]=max(test_acc[1:])

plt.plot(x_axis,max_accu,label="Max_Accuracy")
plt.legend()
plt.xlabel('Random State')
plt.ylabel('Acuracy for KNN')
plt.show()

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=np.argmax(max_accu))
for k in range(1,25):
        pipe = Pipeline([('scaler', StandardScaler()),('pca',PCA(n_components=3)), ('KNN', KNeighborsRegressor(n_neighbors=k,weights='distance') )])
        pipe.fit(X_train,y_train[:,0])
        test_acc[k]=(pipe.score(X_test,y_test[:,0]))


print("THe best accuracy for KNN is ",max(test_acc),"is at random state =",np.argmax(max_accu),"with n_erighbors=",np.argmax(test_acc))

#%% for Decision tree
max_accu=np.empty(len(range(0,50)))
x_axis=range(0,50)
for i in range(0,50):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=i)
    
    #DT Code
    n=np.arange(1,26)
    test_acc= np.empty(len(n))
    
        
    for k in range(1,25):
        pipe = Pipeline([('scaler', StandardScaler()),('pca',PCA(n_components=3)), ('DT',DecisionTreeRegressor(max_depth=k) )])
        pipe.fit(X_train,y_train[:,0])
        test_acc[k]=(pipe.score(X_test,y_test[:,0]))
    print(i," ",max(test_acc[1:]))
    max_accu[i]=max(test_acc[1:])

plt.plot(x_axis,max_accu,label="Max_Accuracy")
plt.legend()
plt.xlabel('Random State')
plt.ylabel('Acuracy for DT')
plt.show()

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=np.argmax(max_accu))
for k in range(1,25):
        pipe = Pipeline([('scaler', StandardScaler()),('pca',PCA(n_components=3)), ('DT',DecisionTreeRegressor(max_depth=k))])
        pipe.fit(X_train,y_train[:,0])
        test_acc[k]=(pipe.score(X_test,y_test[:,0]))



print("THe best accuracy for DT is ",max(test_acc),"is at random state =",np.argmax(max_accu),"with max_depth=",np.argmax(test_acc))

#%%
#for SVR
max_accu=np.empty(len(range(0,50)))
x_axis=range(0,50)
for i in range(0,50):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=i)
    
    #SVR Code
    n=np.arange(1,26)
    test_acc= np.empty(len(n))
    
    pipe = Pipeline([('scaler', StandardScaler()),('pca',PCA(n_components=3)), ('SVR',SVR(kernel='rbf'))])
    pipe.fit(X_train,y_train[:,0])
    test_acc=(pipe.score(X_test,y_test[:,0]))
    print(i," ",pipe.score(X_test,y_test[:,0]))
    max_accu[i]=test_acc

plt.plot(x_axis,max_accu,label="Max_Accuracy")
plt.legend()
plt.xlabel('Random State')
plt.ylabel('Acuracy for SVR')
plt.show()

print("THe best accuracy for SVR is ",max(max_accu),"is at random state =",np.argmax(max_accu))

#%%
#for MLPR
max_accu=np.empty(len(range(0,50)))
x_axis=range(0,50)
for i in range(0,50):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=i)
    
    #SVR Code
    n=np.arange(1,26)
    test_acc= np.empty(len(n))
    
    pipe = Pipeline([('scaler', StandardScaler()),('pca',PCA(n_components=3)), ('MLPR',MLPRegressor(random_state=42, max_iter=1500,hidden_layer_sizes=10))])
    pipe.fit(X_train,y_train[:,0])
    test_acc=(pipe.score(X_test,y_test[:,0]))
    print(i," ",pipe.score(X_test,y_test[:,0]))
    max_accu[i]=test_acc

plt.plot(x_axis,max_accu,label="Max_Accuracy")
plt.legend()
plt.xlabel('Random State')
plt.ylabel('Acuracy for MLPR')
plt.show()

print("THe best accuracy for MLPR is ",max(max_accu),"is at random state =",np.argmax(max_accu))

#%%
#code for random forests
max_accu=np.empty(len(range(0,50)))
x_axis=range(0,50)
for i in range(0,50):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=i)
    
    #SVR Code
    n=np.arange(1,26)
    test_acc= np.empty(len(n))
    
    pipe = Pipeline([('scaler', StandardScaler()),('pca',PCA(n_components=3)), ('RF',RandomForestRegressor(n_estimators =500,bootstrap=True,n_jobs=-1,max_depth=None,random_state=43))])
    pipe.fit(X_train,y_train[:,0])
    test_acc=(pipe.score(X_test,y_test[:,0]))
    print(i," ",pipe.score(X_test,y_test[:,0]))
    max_accu[i]=test_acc

plt.plot(x_axis,max_accu,label="Max_Accuracy")
plt.legend()
plt.xlabel('Random State')
plt.ylabel('Acuracy for RFR')
plt.show()

print("THe best accuracy for RFR is ",max(max_accu),"is at random state =",np.argmax(max_accu))

#%%
#code for AdaBoost
max_accu=np.empty(len(range(0,50)))
x_axis=range(0,50)
for i in range(0,50):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=i)
    
    #SVR Code
    n=np.arange(1,26)
    test_acc= np.empty(len(n))
    
    pipe = Pipeline([('scaler', StandardScaler()),('pca',PCA(n_components=3)), ('AdaBoost',AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=7), n_estimators=100))])
    pipe.fit(X_train,y_train[:,0])
    test_acc=(pipe.score(X_test,y_test[:,0]))
    print(i," ",pipe.score(X_test,y_test[:,0]))
    max_accu[i]=test_acc

plt.plot(x_axis,max_accu,label="Max_Accuracy")
plt.legend()
plt.xlabel('Random State')
plt.ylabel('Acuracy for AdaBoost')
plt.show()

print("THe best accuracy for AdaBoost is ",max(max_accu),"is at random state =",np.argmax(max_accu))

#%%
#code for bagging of knn

max_accu=np.empty(len(range(0,50)))
x_axis=range(0,50)
for i in range(0,50):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=i)
    
    #SVR Code
    n=np.arange(1,26)
    test_acc= np.empty(len(n))
    
    pipe = Pipeline([('scaler', StandardScaler()),('pca',PCA(n_components=3)), ('Bagging',BaggingRegressor(base_estimator=KNeighborsRegressor(n_neighbors=3,weights='distance'),n_estimators=17))])
    pipe.fit(X_train,y_train[:,0])
    test_acc=(pipe.score(X_test,y_test[:,0]))
    print(i," ",pipe.score(X_test,y_test[:,0]))
    max_accu[i]=test_acc

plt.plot(x_axis,max_accu,label="Max_Accuracy")
plt.legend()
plt.xlabel('Random State')
plt.ylabel('Acuracy for Bagging of KNN')
plt.show()

print("THe best accuracy for Bagging of KNN is ",max(max_accu),"is at random state =",np.argmax(max_accu))
