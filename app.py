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

@app.route('/', methods=['GET'])
def root():
    return jsonify({'message': "Welcome to Falcon\'s Web API"})

@app.route('/train', methods=['GET'])
def untrain():
    global knn
    knn = None
    return jsonify({'message': "Model Untrained"})

@app.route('/train', methods=['GET'])
def train():
    url = "https://falcons-cyber.firebaseio.com/train.json"
    r = requests.get(url)
    if r:
        v = r.json()
        # print(v)
        trans=pd.DataFrame(v)
        df=trans.T
        df.reset_index(inplace=True,drop=True)
        df.head()

        clus=df[['timeSpentOnInternet','peopleAroundUsesInternet','internetUseEnjoyable']]

        km = KMeans(n_clusters=2,random_state=1)
        km.fit(clus)
        y = km.fit_predict(clus)
        a=pd.DataFrame({'Cyberloafer Type':km.labels_,'Name of the Employee':df['name']})
        l = sum(a['Cyberloafer Type']==0)
        h = sum(a['Cyberloafer Type']==1)
        d = []
        for i in range(0,len(a)):
            dic = { 'name' : a.iloc[i][1],
                            'prediction' : a.iloc[i][0].astype('str') }
            d.append(dic)
        res = {
            'list' : d,
            'graph' : {
                'low' : l,
                'high' : h
            }
        }
        import json
        with open(r'model2_static.json', 'w') as f:
            json.dump(res, f)

        a=pd.DataFrame({'Cyberloafer Type':km.labels_,'Name of the Employee':df['name']})

        clus['Cyberloafer Type']=a['Cyberloafer Type']

        a.set_index('Name of the Employee',inplace=True)
        # a['Cyberloafer Type']=a['Cyberloafer Type'].map({0:'High Cyberloafer',1:'Low Cyberloafer'})
        # a.head()
        X=clus[['timeSpentOnInternet','peopleAroundUsesInternet','internetUseEnjoyable']]
        y=clus['Cyberloafer Type']
        global knn
        knn = KNeighborsClassifier()
        knn.fit(X,y)
        return jsonify({'message': "Model Trained"})
        # joblib.dump(knn, 'model3.pkl')
        # print("Model3 dumped!")
    else :
        return jsonify({'message': "No data here to train"})


@app.route('/prediction', methods=['GET'])
def prediction():
    # knn = joblib.load("model3.pkl")# Load "model.pkl"
    # print ('knn loaded')
    if knn:
        try:
            url = "https://falcons-cyber.firebaseio.com/predict.json"
            m = requests.get(url)
            if m:
                n = m.json()
                o=pd.DataFrame(n)
                p =o.T
                p.reset_index(inplace=True,drop=True)
                q=p[['timeSpentOnInternet','peopleAroundUsesInternet','internetUseEnjoyable','name']]
                q.set_index('name',inplace=True)
                knn_pred=knn.predict(q)
                out=pd.DataFrame({'Cyberloafer Type':knn_pred,'Name':q.index})
                out.set_index('Name',inplace=True)
                out.reset_index(inplace=True)
                print(out)
                d1 = []
                for i in range(0,len(out)):
                    dic = { 'name' : out.iloc[i][0],
                            'prediction' : out.iloc[i][1].astype('str') }
                    d1.append(dic) 
                return json.dumps(d1)
            else :
                return jsonify({'message': "No data here to predict"})
        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        return jsonify({'Train the model first'})


@app.route('/canalysis', methods=['GET'])
def canalysis():
        try:
            with open(r'model2_static.json') as f:
                return json.load(f)
        except:

            return jsonify({'trace': traceback.format_exc()})


if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True,host='0.0.0.0',port=port)
