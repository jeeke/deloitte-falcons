# Dependencies
from flask import Flask, request, jsonify
import traceback
import json
import requests

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
# from sklearn.externals import joblib
from sklearn.cluster import KMeans

# Your API definition
app = Flask(__name__)
global knn
knn = None
global model2FileData
model2FileData = None


@app.route('/', methods=['GET'])
def root():
    return jsonify({'message': "Welcome to Falcon\'s Web API"})


@app.route('/untrain', methods=['GET'])
def untrain():
    global knn, model2FileData
    knn = None
    model2FileData = None
    return jsonify({'message': "Model Untrained"})


@app.route('/train', methods=['GET'])
def train():
    url = "https://falcons-cyber.firebaseio.com/train.json"
    r = requests.get(url)
    if r:
        v = r.json()
        # print(v)
        trans = pd.DataFrame(v)
        df = trans.T
        df.reset_index(inplace=True, drop=True)
        df.head()

        clus = df[['timeSpentOnInternet',
                   'peopleAroundUsesInternet', 'internetUseEnjoyable','employeeId']]
        clus.set_index('employeeId',inplace=True)

        km = KMeans(n_clusters=2, random_state=1)
        km.fit(clus)
        y = km.fit_predict(clus)
        a = pd.DataFrame({'cyberloaferType': km.labels_,
                          'name': df['name'],
                          'employeeId':df['employeeId']})
        d = []
        for i in range(0, len(a)):
            dic = {'name': a.iloc[i][1],
                   'cyberloaferType': a.iloc[i][0].astype('str'),
                    'employeeId':a.iloc[i][2]}
            d.append(dic)
        global model2FileData
        model2FileData = d

        a = pd.DataFrame({'cyberloaferType': km.labels_,
                       'name': df['name'],
                      'employeeId':df['employeeId']})
        a.set_index('employeeId',inplace=True)

        clus['cyberloaferType'] = a['cyberloaferType'].copy()
        df.set_index('employeeId',inplace=True)
        df['cyberloaferType']=a['cyberloaferType'].copy()
       
        #Creating Dataframe for clustering
        X=df[['timeSpentOnInternet','peopleAroundUsesInternet','internetUseEnjoyable','name']]
        # y=df['cyberloaferType'].map({0:'Low',1:'High'})
        x1=X.iloc[:,0:3]
        global knn
        knn = KNeighborsClassifier()
        knn.fit(x1, y)
        return jsonify({'message': "Model Trained"})
    else:
        return jsonify({'message': "No data here to train"})


@app.route('/prediction', methods=['GET'])
def prediction():
    global knn
    x = knn
    global model2FileData
    if x == None or model2FileData == None:
        return jsonify({'message': 'Train the model first'})
    else:
        try:
            url = "https://falcons-cyber.firebaseio.com/predict.json"
            m = requests.get(url)
            n = m.json()
            if n:
                trans = pd.DataFrame(n)
                df1 = trans.T
                df1.dropna(how='any',subset=['employeeId'],axis=0,inplace=True)
                df1.set_index('employeeId', inplace=True)
                X_pred = df1[['timeSpentOnInternet',
                              'peopleAroundUsesInternet', 'internetUseEnjoyable', 'name']]
                X1 = X_pred[['timeSpentOnInternet',
                             'peopleAroundUsesInternet', 'internetUseEnjoyable']]
                knn_pred = knn.predict(X1)
                out = pd.DataFrame(
                    {'cyberloaferType': knn_pred, 'name': X_pred['name']})
                out.reset_index(inplace=True)
                g = out.to_dict('records')
                return json.dumps({'predict': g,
                                   'train': model2FileData})
            else:
                return jsonify({'train': model2FileData})
        except:
            return jsonify({'trace': traceback.format_exc()})


if __name__ == '__main__':
# On IBM Cloud Cloud Foundry, get the port number from the environment variable PORT
# When running this app on the local machine, default the port to 5000
    import os
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0',port=port)
