
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 23:06:07 2021
@author: Vedant
"""
#%%
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


cols=['T1Y Index', 'T2Y Index', 'T3Y Index', 'T5Y Index', 'T7Y Index',
       'T10Y Index', 'CP1M', 'CP3M', 'CP6M', 'CP1M_T1Y', 'CP3M_T1Y',
       'CP6M_T1Y', 'USPHCI', 'PCT 3MO FWD', 'PCT 6MO FWD', 'PCT 9MO FWD']
# sns.pairplot(df[cols],height=2.5)
# plt.tight_layout()
# plt.show()


corr=df[cols].corr()
sns.set(font_scale=0.6)
sns.heatmap(corr, annot= True,cmap="Blues")
plt.show()

#%%

from sklearn.preprocessing import StandardScaler 
data = df.values
sc = StandardScaler()
data=sc.fit_transform(data)
X=data[:,1:12]
y=data[:,13:]
y1=y[:,0]

#%%
from sklearn.decomposition import PCA
pca=PCA(n_components=None)
X_pca = pca.fit_transform(X)
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
from sklearn.pipeline import Pipeline
pipe = Pipeline([('scaler', StandardScaler()),('pca',PCA(n_components=2))])
X1=pipe.fit_transform(X[:,:6])
X2=pipe.fit_transform(X[:,6:10])
X3=pipe.fit_transform(X[:,9:])
Xfin=np.concatenate((X1,X2,X3), axis=1)

#%%
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor as DT
from sklearn.neighbors import KNeighborsRegressor as knn
from sklearn.linear_model import ElasticNet as elastic
from sklearn.ensemble import BaggingRegressor
from sklearn.datasets import make_regression

#%%
X_train, X_test, y1_train, y1_test = train_test_split(X, y1, test_size=0.5, random_state=43)
pipe = Pipeline([('pca',PCA(n_components=2)), ('svr', SVR(gamma=10,kernel='linear') )])
pipe.fit(X_train,y1_train)
print("SVR (linear kernel) ", pipe.score(X_test,y1_test))
print("MSE for SVR (Linear Kernel)", MSE(pipe.predict(X_test),y1_test))


#%%
X_train2, X_test2, y1_train2, y1_test2 = train_test_split(X, y1, test_size=0.3, random_state=43)
pipe2 = Pipeline([('scaler', StandardScaler()), ('svr', SVR(gamma=10,kernel='rbf') )])
pipe2.fit(X_train2,y1_train2)
print("SVR (without PCA) (rbf Kernel)",pipe2.score(X_test2,y1_test2))
print("MSE (without PCA) for SVR (rbf Kernel)",MSE(pipe2.predict(X_test2),y1_test2))

#%%
X_train, X_test, y1_train, y1_test = train_test_split(X, y1, test_size=0.3)
pipe3 = Pipeline([('scaler', StandardScaler()),('pca',PCA(n_components=2)), ('svr', SVR(gamma=0.07,kernel='rbf') )])
pipe3.fit(X_train,y1_train)
print("SVR (rbf Kernel)",pipe3.score(X_test,y1_test))
print("MSE for SVR (rbf Kernel) ", MSE(pipe3.predict(X_test),y1_test))


#%%
pipe4 = Pipeline([('scaler', StandardScaler()),('pca',PCA(n_components=2)), ('dt', DT(max_depth=4) )])
pipe4.fit(X_train,y1_train)
print("Decision Tree Regressor ", pipe4.score(X_test,y1_test))
print("MSE for Decision Tree Regressor",MSE(pipe4.predict(X_test),y1_test))


#%%
X_train, X_test, y1_train, y1_test = train_test_split(X, y1, test_size=0.3)
pipe5 = Pipeline([('scaler', StandardScaler()),('pca',PCA(n_components=2)), ('KNN', knn(n_neighbors=9) )])
pipe5.fit(X_train,y1_train)
print("KNN Regressor",pipe5.score(X_test,y1_test))
print("MSE for KNN regressor",MSE(pipe5.predict(X_test),y1_test))

#%%
pipes=[pipe,pipe2,pipe3,pipe4,pipe5]
regressors = ["SVR (linear kernel)","SVR (without PCA) (rbf Kernel)","SVR (rbf Kernel)","Decision Tree Regressor ","KNN Regressor" ]
for i in range(0,5):
    regr = BaggingRegressor(base_estimator=pipes[i], n_estimators=100, random_state=0).fit(X_train, y1_train)
    print("Bagging Regressor for ",regressors[i],"  " , regr.score(X_test,y1_test))


#%%
from sklearn.linear_model import Lasso
X_train, X_test, y1_train, y1_test = train_test_split(X, y1, test_size=0.5, random_state=43)
sc.fit(X_train)
testacc_las=[]
trainacc_las=[]
coef_las=[]
y_axis=[]
X_zero=[]
for i in range (15,1000):
    X_zero.append(0)
for i in range (15,1000):
    y_axis.append(1/i)
for i in range(15,1000):
    lasso=Lasso(alpha = 1/i)
    lasso.fit(sc.transform(X_train),y1_train)
    coef_las.append(lasso.coef_)
    testacc_las.append(lasso.score(sc.transform(X_test),y1_test))
    trainacc_las.append(lasso.score(sc.transform(X_train),y1_train))
coef_las=np.array(coef_las)
print("the coefficients for the regression are ",coef_las)
plt.title(" Lasso Regression Model")
plt.plot(y_axis,coef_las)
plt.plot(y_axis,X_zero,color='black', lw=2)
plt.xscale('log')
plt.xlabel("Alpha Values")
plt.ylabel("Coefficients")
plt.legend(cols)
plt.show()


plt.title("Accuracy vs alpha scores for Lasso Regression Model")
plt.plot(y_axis,testacc_las,label="Test Accuracy")
plt.plot(y_axis,trainacc_las,label="Train Accuracy")
plt.legend()
plt.xlabel('Alpha Values')
plt.ylabel('Acuracy')
plt.show()

X_train, X_test, y1_train, y1_test = train_test_split(X, y1, test_size=0.5, random_state=43)
lasso=Lasso(alpha = 1/18)
#since 1/18 is max accuracy giving value for alpha
lasso.fit(sc.transform(X_train),y1_train)
y1_pred_train_las = lasso.predict(sc.transform(X_train))
y1_pred_las = lasso.predict(sc.transform(X_test))
plt.scatter(y1_pred_train_las,y1_pred_train_las-y1_train,c='steelblue',marker='o',edgecolor='white',label='Training Data')
plt.scatter(y1_pred_las,y1_pred_las-y1_test,c='limegreen',marker='s',edgecolor='white',label='Training Data')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0,xmin = -3,xmax = 2 , color='black', lw=2)
plt.xlim([-3,2])
plt.title("Residuals for the Lasso regression")
plt.show()


#%%
from sklearn.ensemble import VotingRegressor
vr_estimators =[("SVR (linear kernel)",pipe),("SVR (without PCA)(rbf Kernel)",pipe2), ("SVR (rbf Kernel)",pipe3),("Decision Tree Regressor ",pipe4),("KNN Regressor",pipe5)]
vr = VotingRegressor( estimators=vr_estimators)
vr.fit(X_train,y1_train)
print("Accuracy for voting regressor = ", vr.score(X_test,y1_test))

#%%
#function to plot decision bondaries
#work needed not yet done

def lin_regplot(X,y,model):
    plt.scatter(X,cols,c='steelblue',edgecolor='white',s=70)
    plt.plot(X,model.predict(X),color='black',lw=2)
    return None

lin_regplot(X_train, y1_train, pipe4)