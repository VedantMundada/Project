# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 09:18:57 2021

@author: Vedant
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
path = "C:/Users/Vedant/Desktop/UIUC/SEM 1/Machine Learning/Project/MLF_GP2_EconCycle.csv"
df = pd.read_csv(path)
print(df.head())
n_rows=len(df)
n_col=0
for column in df.values[0,:]:
    n_col=n_col+1
print("Number of columns of Data = " , n_col , '\n')
print("The summary for each column is \n",df.describe())


df=df.drop('Date',1)
print(df.head())

cols=df.keys()

corr=df[cols].corr()
sns.set(font_scale=0.6)
sns.heatmap(corr, annot= True,cmap="Blues")
plt.show()

data = df.values
sns.pairplot(df[cols[0:6]],height=2.5)
plt.tight_layout()
plt.show()
sns.pairplot(df[cols[6:9]],height=2.5)
plt.tight_layout()
plt.show()
sns.pairplot(df[cols[9:12]],height=2.5)
plt.tight_layout()
plt.show()
#%%
from sklearn.preprocessing import StandardScaler 
sc=StandardScaler()

X=data[:,1:12]
y=data[:,13:]

y3=y[:,0]
y3=sc.fit_transform(y3.reshape(-1,1))
y6=y[:,1]
y6=sc.fit_transform(y6.reshape(-1,1))
y9=y[:,-1]
y9=sc.fit_transform(y9.reshape(-1,1))



#%%
from sklearn.decomposition import PCA
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


pca=PCA(n_components=3)
X_set = pca.fit_transform(sc.fit_transform(X))
#%% 
#PCA on entire data set
pca=PCA(n_components=3)
X_pca = pca.fit_transform(sc.fit_transform(X))
exp_var_pca= pca.explained_variance_ratio_
cum_sum_eigenvalues = sum(exp_var_pca)
print("The explained variance ratio is ", cum_sum_eigenvalues)

#%%
#individual PCA for good corelated columns
pca=PCA(n_components=None)
X_pca = pca.fit_transform(sc.fit_transform(X[:,:6]))
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

pca=PCA(n_components=2)
X_pca1 = pca.fit_transform(sc.fit_transform(X[:,:6]))

pca=PCA(n_components=None)
X_pca = pca.fit_transform(sc.fit_transform(X[:,6:9]))
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

pca=PCA(n_components=2)
X_pca2 = pca.fit_transform(sc.fit_transform(X[:,6:9]))

pca=PCA(n_components=None)
X_pca = pca.fit_transform(sc.fit_transform(X[:,9:12]))
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

pca=PCA(n_components=1)
X_pca3 = pca.fit_transform(sc.fit_transform(X[:,9:12]))

#combining these individual PCA to get new X set
X_ipca = np.concatenate((X_pca1,X_pca2,X_pca3),axis=1)

#now we have two X sets X_ipca and X_set

#%%
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.linear_model import ElasticNet as elastic
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LogisticRegression
#%%

#splitting the X_set 
X_train3, X_test3, y3_train, y3_test = train_test_split(X_set, y3, test_size=0.2, random_state=43)
X_train6, X_test6, y6_train, y6_test = train_test_split(X_set, y6, test_size=0.2, random_state=43)
X_train9, X_test9, y9_train, y9_test = train_test_split(X_set, y9, test_size=0.2, random_state=43)

#%%
#fitting models to splits of X_set
knn = KNeighborsRegressor(n_neighbors=2)
knn.fit(X_train3,y3_train)
print("Score for 3 month via KNN = ", knn.score(X_test3,y3_test))

n=np.arange(1,26)
train_accuracy = np.empty(len(n))
test_accuracy = np.empty(len(n))
from sklearn import neighbors 
for k in range(1,25):
    knn = neighbors.KNeighborsRegressor(k ,weights='uniform')
    knn.fit(X_train3, y3_train)
    train_accuracy[k] = knn.score(X_train3,y3_train)
    test_accuracy[k] = knn.score(X_test3,y3_test)
plt.title("K-NN : Varying number of neighbors 3 months ")
plt.plot(n,test_accuracy,label="Test Accuracy")
plt.plot(n,train_accuracy,label="Train Accuracy")
plt.legend()
plt.xlabel('No of Neighbors')
plt.ylabel('Acuracy')
plt.show()

#so max acuracy at k = 2 for 3 months

n=np.arange(1,26)
train_accuracy = np.empty(len(n))
test_accuracy = np.empty(len(n))
from sklearn import neighbors 
for k in range(1,25):
    knn = neighbors.KNeighborsRegressor(k ,weights='uniform')
    knn.fit(X_train6, y6_train)
    train_accuracy[k] = knn.score(X_train6,y6_train)
    test_accuracy[k] = knn.score(X_test6,y6_test)
plt.title("K-NN : Varying number of neighbors 6 months ")
plt.plot(n,test_accuracy,label="Test Accuracy")
plt.plot(n,train_accuracy,label="Train Accuracy")
plt.legend()
plt.xlabel('No of Neighbors')
plt.ylabel('Acuracy')
plt.show()

#so max acuracy at k = 3 for 6 months
knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(X_train6,y6_train)
print("Score for 6 month via KNN = ", knn.score(X_test6,y6_test))


n=np.arange(1,26)
train_accuracy = np.empty(len(n))
test_accuracy = np.empty(len(n))
from sklearn import neighbors 
for k in range(1,25):
    knn = neighbors.KNeighborsRegressor(k ,weights='uniform')
    knn.fit(X_train9, y9_train)
    train_accuracy[k] = knn.score(X_train9,y9_train)
    test_accuracy[k] = knn.score(X_test9,y9_test)
plt.title("K-NN : Varying number of neighbors 9 months ")
plt.plot(n,test_accuracy,label="Test Accuracy")
plt.plot(n,train_accuracy,label="Train Accuracy")
plt.legend()
plt.xlabel('No of Neighbors')
plt.ylabel('Acuracy')
plt.show()

#so max acuracy at k = 3 for 9 months
knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(X_train9,y9_train)
print("Score for 9 month via KNN = ", knn.score(X_test9,y9_test))


#%%
#splitting the X_ipca and using it for regression
X_itrain3, X_itest3, y3_itrain, y3_itest = train_test_split(X_ipca, y3, test_size=0.2, random_state=43)
X_itrain6, X_itest6, y6_itrain, y6_itest = train_test_split(X_ipca, y6, test_size=0.2, random_state=43)
X_itrain9, X_itest9, y9_itrain, y9_itest = train_test_split(X_ipca, y9, test_size=0.2, random_state=43)

#%%
knn = KNeighborsRegressor(n_neighbors=2)
knn.fit(X_itrain3,y3_itrain)
print("Score for 3 month via KNN = ", knn.score(X_itest3,y3_itest))

knn = KNeighborsRegressor(n_neighbors=2)
knn.fit(X_itrain6,y6_itrain)
print("Score for 6 month via KNN = ", knn.score(X_itest6,y6_itest))

knn = KNeighborsRegressor(n_neighbors=2)
knn.fit(X_itrain9,y9_itrain)
print("Score for 9 month via KNN = ", knn.score(X_itest9,y9_itest))

#note max accuracy is at k=2 for all these

#%%
n=np.arange(1,25)
train_accuracy = []
test_accuracy = []

for k in range(1,25):
    dt = DecisionTreeRegressor(max_depth=k)
    dt.fit(X_train3, y3_train)
    train_accuracy.append( dt.score(X_train3,y3_train))
    test_accuracy.append(dt.score(X_test3,y3_test))
plt.title("Decision tree : Varying number of max depth 3 months ")
plt.plot(n,test_accuracy,label="Test Accuracy")
plt.plot(n,train_accuracy,label="Train Accuracy")
plt.legend()
plt.xlabel('Max Depth')
plt.ylabel('Acuracy')
plt.show()


dt = DecisionTreeRegressor(max_depth=5)
dt.fit(X_train3,y3_train)
print("Score for 3 month via DT = ", dt.score(X_test3,y3_test))



train_accuracy = []
test_accuracy = []
for k in range(1,25):
    dt = DecisionTreeRegressor(max_depth=k, random_state = 42)
    dt.fit(X_train6, y6_train)
    train_accuracy.append( dt.score(X_train6,y6_train))
    test_accuracy.append(dt.score(X_test6,y6_test))
plt.title("Decision tree : Varying number of max depth 6 months ")
plt.plot(n,test_accuracy,label="Test Accuracy")
plt.plot(n,train_accuracy,label="Train Accuracy")
plt.legend()
plt.xlabel('Max Depth')
plt.ylabel('Acuracy')
plt.show()

dt = DecisionTreeRegressor(max_depth=test_accuracy.index(max(test_accuracy))+1, random_state=42)
dt.fit(X_train6,y6_train)
print("Score for 6 month via DT = ", dt.score(X_test6,y6_test))


train_accuracy = []
test_accuracy = []
for k in range(1,25):
    dt = DecisionTreeRegressor(max_depth=k, random_state = 42)
    dt.fit(X_train9, y9_train)
    train_accuracy.append( dt.score(X_train9,y9_train))
    test_accuracy.append(dt.score(X_test9,y9_test))
plt.title("Decision tree : Varying number of max depth 9 months ")
plt.plot(n,test_accuracy,label="Test Accuracy")
plt.plot(n,train_accuracy,label="Train Accuracy")
plt.legend()
plt.xlabel('Max Depth')
plt.ylabel('Acuracy')
plt.show()

dt = DecisionTreeRegressor(max_depth=test_accuracy.index(max(test_accuracy))+1, random_state=42)
dt.fit(X_train9,y9_train)
print("Score for 9 month via DT = ", dt.score(X_test9,y9_test))

#%%
dt = DecisionTreeRegressor(max_depth=5, random_state = 42)
dt.fit(X_itrain3,y3_itrain)
print("Score for 3 month via DT = ", dt.score(X_itest3,y3_itest))

dt = DecisionTreeRegressor(max_depth=10, random_state = 42)
dt.fit(X_itrain6,y6_itrain)
print("Score for 6 month via DT = ", dt.score(X_itest6,y6_itest))

dt = DecisionTreeRegressor(max_depth=8, random_state = 42)
dt.fit(X_itrain9,y9_itrain)
print("Score for 9 month via DT = ", dt.score(X_itest9,y9_itest))

#note max accuracy is at k=2 for all these

train_accuracy = []
test_accuracy = []
for k in range(1,25):
    dt = DecisionTreeRegressor(max_depth=k, random_state = 42)
    dt.fit(X_itrain9, y9_itrain)
    train_accuracy.append( dt.score(X_itrain9,y9_itrain))
    test_accuracy.append(dt.score(X_itest9,y9_itest))
plt.title("Decision tree : Varying number of max depth 9 months ")
plt.plot(n,test_accuracy,label="Test Accuracy")
plt.plot(n,train_accuracy,label="Train Accuracy")
plt.legend()
plt.xlabel('Max Depth')
plt.ylabel('Acuracy')
plt.show()

#%%

from sklearn.svm import SVR
knls=['linear', 'poly', 'rbf']
for i in range(0,3):
    svr=SVR(kernel=knls[i])
    svr.fit(X_train3,y3_train)
    print("Accuracy for 3 months with kernel =  ",knls[i], " = ",svr.score(X_test3,y3_test))
    
    
    
#%%
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(X_itrain3,y3_itrain)
print("Score for 3 month via linear regression = ", lr.score(X_itest3,y3_itest))

lr.fit(X_itrain6,y6_itrain)
print("Score for 6 month via linear regression = ", lr.score(X_itest6,y6_itest))

lr.fit(X_itrain9,y9_itrain)
print("Score for 9 month via linear regression = ", lr.score(X_itest9,y9_itest))

#%%

from sklearn.linear_model import Ridge
ridge = Ridge(alpha= 0.03, normalize= True)
ridge.fit(X_itrain3,y3_itrain)
print("Score for 3 month via ridge regression = ", ridge.score(X_itest3,y3_itest))

ridge.fit(X_itrain6,y6_itrain)
print("Score for 6 month via ridge regression = ",ridge.score(X_itest6,y6_itest))

ridge.fit(X_itrain9,y9_itrain)
print("Score for 9 month via ridge regression = ", ridge.score(X_itest9,y9_itest))

from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.001,normalize=True)
lasso.fit(X_itrain3,y3_itrain)
print("Score for 3 month via lasso regression = ", lasso.score(X_itest3,y3_itest))

lasso.fit(X_itrain6,y6_itrain)
print("Score for 6 month via lasso regression = ",lasso.score(X_itest6,y6_itest))

lasso.fit(X_itrain9,y9_itrain)
print("Score for 9 month via lasso regression = ", lasso.score(X_itest9,y9_itest))
