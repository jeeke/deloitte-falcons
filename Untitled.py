#!/usr/bin/env python
# coding: utf-8

# In[1]:


#libraries to be used
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer
get_ipython().run_line_magic('matplotlib', 'inline')
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import requests


# In[2]:


# data import 
url = "https://falcons-cyber.firebaseio.com/train.json"
r = requests.get(url)
v = r.json()


# In[3]:


trans=pd.DataFrame(v)
df=trans.T


# In[4]:


df.dropna(how='any',subset=['name'],axis=0,inplace=True)


# In[5]:


# Reseting index
df.reset_index(inplace=True,drop=True)


# In[6]:


fact=df[['timeSpentOnInternet','leisureTimeThingsToDo','monotonousAndBoring','unnecessaryTeamMeetings',
         'needShortBreak','taskCompleted','overTime','peopleAroundUsesInternet','internetUseEnjoyable']]
fact=fact.astype('int')


# In[7]:


# Create factor analysis object and perform factor analysis
fa = FactorAnalyzer(rotation=None)
fa.fit(fact)
# Check Eigenvalues
ev, v = fa.get_eigenvalues()
print(ev) #NEED THIS ON THE URL FOR FACTOR ANAlYSIS PLOT


# In[8]:


eigen=[]
for i in range(0,len(ev)):
    if ev[i]>=1:
        eigen.append(ev[i])


# In[9]:


# Create factor analysis object and perform factor analysis
fa1 = FactorAnalyzer(n_factors=len(eigen),rotation='Varimax',method='principal')
fa1.fit(fact)


# In[10]:


fa1.loadings_#NEED TO SEE THIS AS WELL


# In[11]:


factor_names=['Boredom','Work Completed','Work Environment','Habit']
factor=pd.DataFrame({'Factors':factor_names,'Eigen_Value':eigen})
factor.head()


# In[12]:


# Ploting Dominating Factors
plt.figure(figsize=(9,7))
sns.barplot(x='Factors',y='Eigen_Value',data=factor)
ax=plt.axes()
plt.title('Visulalization of Factors Dominating')
ax.set_facecolor('whitesmoke')
plt.ylabel('Eigen Value of the factors')
plt.savefig('Factor.png')


# In[13]:


d1 = {}
for i in range(0,len(factor)):
    d1[str(factor.iloc[i][0])] = factor.iloc[i][1].astype('str')


# In[14]:


res1 = {
    'list1':d1
}


# In[15]:


print(res1)


# In[16]:


clus=df[['timeSpentOnInternet','peopleAroundUsesInternet','internetUseEnjoyable','employeeId']]


# In[17]:


clus.set_index('employeeId',inplace=True)


# In[18]:


km = KMeans(n_clusters=2,random_state=0)
km.fit(clus)
km.predict(clus)
y=km.fit_predict(clus)


# In[19]:


a=pd.DataFrame({'cyberloaferType':km.labels_,'name':df['name'],'employeeId':df['employeeId']})


# In[20]:


a.set_index('employeeId',inplace=True)


# In[21]:


clus['cyberloaferType']=a['cyberloaferType'].copy()


# In[22]:


df.set_index('employeeId',inplace=True)


# In[23]:


df['cyberloaferType']=a['cyberloaferType'].copy()


# In[24]:


#Creating Dataframe for clustering
X=df[['timeSpentOnInternet','peopleAroundUsesInternet','internetUseEnjoyable','name']]
y=df['cyberloaferType'].map({0:'Low',1:'High'})


# In[25]:


X1=X[['timeSpentOnInternet','peopleAroundUsesInternet','internetUseEnjoyable']]


# In[639]:


x1=X.iloc[:,0:3]


# In[26]:


#Predicting throw K-NN
knn=KNeighborsClassifier()
knn.fit(X1,y)
#knn_pred=knn.predict(x1)


# In[641]:


out=pd.DataFrame({'cyberloaferType':knn_pred,'name':X['name']})


# In[642]:


out_1=out[['cyberloaferType']]


# In[643]:


sns.countplot(out['cyberloaferType'])


# In[644]:


out.reset_index(inplace=True)


# In[645]:


l = sum(out['cyberloaferType']=='Low')
h = sum(out['cyberloaferType']=='High')


# In[646]:


g=out.to_dict('records')


# In[647]:


res2 = {
    'list' : g,
    'graph' : {
        'Low' : l,
        'High' : h
    }
}
print(res2)


# In[648]:


out.set_index('employeeId',inplace=True)


# In[649]:


df4=df[['productionScore']]


# In[650]:


prod=pd.concat([out_1,df4],axis=1)


# In[651]:


prod.dropna(how='any',subset=['productionScore'],axis=0,inplace=True)


# In[652]:


prod['productionScore']=prod['productionScore'].astype(int)


# In[654]:


plt.figure(figsize=(6,6))
sns.barplot(x=prod['cyberloaferType'],y=prod['productionScore'])
plt.title('Productivity v/s Cyberloafing')
ax=plt.axes()
ax.set_facecolor('whitesmoke')
#plt.savefig('Prod.png')


# In[655]:


prod.reset_index(inplace=True)


# In[656]:


means = prod.groupby('cyberloaferType')['productionScore'].mean()



