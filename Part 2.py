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
# data=sc.fit_transform(data)
X=data[:,:12]
y=data[:,12:]
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
X_train, X_test, y1_train, y1_test = train_test_split(X, y1, test_size=0.3, random_state=43)
pipe = Pipeline([('scaler', StandardScaler()),('pca',PCA(n_components=2)), ('svr', SVR(gamma=10,kernel='linear') )])
pipe.fit(X_train,y1_train)
print(pipe.score(X_test,y1_test))
print(MSE(pipe.predict(X_test),y1_test))


#%%
X_train2, X_test2, y1_train2, y1_test2 = train_test_split(Xfin, y1, test_size=0.3, random_state=43)
pipe = Pipeline([('scaler', StandardScaler()), ('svr', SVR(gamma=10,kernel='linear') )])
pipe.fit(X_train2,y1_train2)
print(pipe.score(X_test2,y1_test2))
print(MSE(pipe.predict(X_test2),y1_test2))





