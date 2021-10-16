# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 22:50:28 2021

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
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV

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
#KNN 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=43)
n=np.arange(0,25)
test_acc= np.empty(len(n))
neighbors=[]
yterms = ["PCT 3MO FWD","PCT 6MO FWD","PCT 9MO FWD"]
for i in range(0,3):
    for k in range(1,25):
        pipe = Pipeline([('scaler', StandardScaler()),('pca',PCA(n_components=3)), ('KNN', KNeighborsRegressor(n_neighbors=k,weights='distance') )])
        pipe.fit(X_train,y_train[:,i])
        test_acc[k]=(pipe.score(X_test,y_test[:,i]))
    neighbors.append(np.argmax(test_acc[1:])+1)
    print("The best accuracy for ",yterms[i]," is ",max(test_acc[1:]),"with n_neighbors=",np.argmax(test_acc[1:])+1)
    
    plt.plot(n,test_acc,label= yterms[i])
    plt.legend()
    plt.xlabel('Neighbors')
    plt.ylabel('Acuracy for KNN')
plt.title("Accuracy For KNN Regressor for")
plt.show()

for i in range(0,3):
    param_grid = {'KNN__n_neighbors': range(1,25)}
    gscv = GridSearchCV(estimator=pipe, param_grid=param_grid,cv=10)
    gscv.fit(X_train,y_train[:,i])
    print(gscv.best_params_)
    print(gscv.best_score_)
    
for i in range(0,3):    
    pipe = Pipeline([('scaler', StandardScaler()),('pca',PCA(n_components=3)), ('KNN', KNeighborsRegressor(n_neighbors=neighbors[i],weights='distance'))])
    pipe.fit(X_train,y_train[:,i])
    print("The accuracy for ",yterms[i]," is ",pipe.score(X_test,y_test[:,i]))
    print("The MSE for ",yterms[i],"is",MSE(y_test[:,i],pipe.predict(X_test)))


#%%
#DT 
test_acc= np.empty(len(n))
depth=[]
for i in range(0,3):
    for k in range(1,25):
        pipe = Pipeline([('scaler', StandardScaler()),('pca',PCA(n_components=3)), ('DT', DecisionTreeRegressor(max_depth=k,random_state=43) )])
        pipe.fit(X_train,y_train[:,i])
        test_acc[k]=(pipe.score(X_test,y_test[:,i]))
    depth.append(np.argmax(test_acc[1:])+1)
    print("The best accuracy for ",yterms[i]," is ",max(test_acc[1:]),"with max depth=",np.argmax(test_acc[1:])+1)
    
    plt.plot(n,test_acc,label= yterms[i])
    plt.legend()
    plt.xlabel('Max Depth')
    plt.ylabel('Acuracy for DT')
plt.title("Accuracy For DT Regressor for")
plt.show()

for i in range(0,3):
    param_grid = {'DT__max_depth': range(1,25)}
    gscv = GridSearchCV(estimator=pipe, param_grid=param_grid)
    gscv.fit(X_train,y_train[:,i])
    print(gscv.best_params_)
    
for i in range(0,3):    
    pipe = Pipeline([('scaler', StandardScaler()),('pca',PCA(n_components=3)), ('DT', DecisionTreeRegressor(max_depth=depth[i],random_state=43))])
    pipe.fit(X_train,y_train[:,i])
    print("The accuracy for ",yterms[i]," is ",pipe.score(X_test,y_test[:,i]))
    print("The MSE for ",yterms[i],"is",MSE(y_test[:,i],pipe.predict(X_test)))

#note DT has its own random state


#%%
#SVR
pipe = Pipeline([('scaler', StandardScaler()),('pca',PCA(n_components=3)), ('SVR',SVR(kernel='rbf',gamma=1,C=1))])
for i in range(0,3):
    param_grid = {'SVR__gamma': range(1,25),'SVR__C': range(1,25)}
    gscv = GridSearchCV(estimator=pipe, param_grid=param_grid)
    gscv.fit(X_train,y_train[:,i])
    print(gscv.best_params_)
    

for i in range(0,3):    
    pipe.fit(X_train,y_train[:,i])
    print("The accuracy for ",yterms[i]," is ",pipe.score(X_test,y_test[:,i]))
    print("The MSE for ",yterms[i],"is",MSE(y_test[:,i],pipe.predict(X_test)))


#%%
#code for MLPR
pipe = Pipeline([('scaler', StandardScaler()),('pca',PCA(n_components=3)), ('MLPR',MLPRegressor(random_state=42, max_iter=1500,hidden_layer_sizes=10))])

for i in range(0,3):
    param_grid = {'MLPR__max_iter': range(500,2500,200),'MLPR__hidden_layer_sizes': range(10,90,10)}
    gscv = GridSearchCV(estimator=pipe, param_grid=param_grid)
    gscv.fit(X_train,y_train[:,i])
    print(gscv.best_params_)
    
pipe = Pipeline([('scaler', StandardScaler()),('pca',PCA(n_components=3)), ('MLPR',MLPRegressor(random_state=42, max_iter=500,hidden_layer_sizes=80))])
    
for i in range(0,3):    
    pipe.fit(X_train,y_train[:,i])
    print("The accuracy for ",yterms[i]," is ",pipe.score(X_test,y_test[:,i]))
    print("The MSE for ",yterms[i],"is",MSE(y_test[:,i],pipe.predict(X_test)))

#%%
#code for RandomForests

pipe = Pipeline([('scaler', StandardScaler()),('pca',PCA(n_components=3)), ('RF',RandomForestRegressor(n_estimators =500,bootstrap=True,n_jobs=-1,max_depth=None,random_state=37))])
for i in range(0,3):
    param_grid = {'RF__n_estimators': range(100,1000,100)}
    gscv = GridSearchCV(estimator=pipe, param_grid=param_grid)
    gscv.fit(X_train,y_train[:,i])
    print(gscv.best_params_)
    
pipe = Pipeline([('scaler', StandardScaler()),('pca',PCA(n_components=3)), ('RF',RandomForestRegressor(n_estimators =200,bootstrap=True,n_jobs=-1,max_depth=None,random_state=43))])
pipe.fit(X_train,y_train[:,0])
print("The accuracy for ",yterms[0]," is ",pipe.score(X_test,y_test[:,0]))
print("The MSE for ",yterms[0],"is",MSE(y_test[:,0],pipe.predict(X_test)))

pipe = Pipeline([('scaler', StandardScaler()),('pca',PCA(n_components=3)), ('RF',RandomForestRegressor(n_estimators =500,bootstrap=True,n_jobs=-1,max_depth=None,random_state=43))])
pipe.fit(X_train,y_train[:,1])
print("The accuracy for ",yterms[1]," is ",pipe.score(X_test,y_test[:,1]))
print("The MSE for ",yterms[1],"is",MSE(y_test[:,1],pipe.predict(X_test)))

pipe = Pipeline([('scaler', StandardScaler()),('pca',PCA(n_components=3)), ('RF',RandomForestRegressor(n_estimators =700,bootstrap=True,n_jobs=-1,max_depth=None,random_state=43))])
pipe.fit(X_train,y_train[:,2])
print("The accuracy for ",yterms[2]," is ",pipe.score(X_test,y_test[:,2]))
print("The MSE for ",yterms[2],"is",MSE(y_test[:,1],pipe.predict(X_test)))

#%%
#AdaBoost
pipe = Pipeline([('scaler', StandardScaler()),('pca',PCA(n_components=3)), ('AdaBoost',AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=7), n_estimators=100))])

for i in range(0,3):
    param_grid = {'AdaBoost__n_estimators': range(100,1000,100)}
    gscv = GridSearchCV(estimator=pipe, param_grid=param_grid)
    gscv.fit(X_train,y_train[:,i])
    print(gscv.best_params_, "  ", gscv.best_score_)


#%%
#bagging KNN
pipe = Pipeline([('scaler', StandardScaler()),('pca',PCA(n_components=3)), ('Bagging',BaggingRegressor(base_estimator=KNeighborsRegressor(n_neighbors=3,weights='distance'),n_estimators=17))])
for i in range(0,3):
    param_grid = {'Bagging__n_estimators': range(1,70)}
    gscv = GridSearchCV(estimator=pipe, param_grid=param_grid)
    gscv.fit(X_train,y_train[:,i])
    print(gscv.best_params_, "  ", gscv.best_score_)



