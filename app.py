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

import db.database as DB
from cloudant.error import CloudantException
from cloudant.result import Result
from cloudant.document import Document

# Your API definition
app = Flask(__name__)
global knn
knn = None
global model2FileData
model2FileData = None

global train_db_name
train_db_name = "train-db"
global predict_db_name
predict_db_name = "predict-db"


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
    train_db = DB.client[train_db_name]
    v = Result(train_db.all_docs, include_docs=True)
    if v:
        w = {}
        for i in v:
            w[i['id']] = i['doc']
        trans = pd.DataFrame(w)
        df = trans.T
        df.reset_index(inplace=True, drop=True)
        df.head()
        # print(df)
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
            predict_db = DB.client[predict_db_name]
            n = Result(predict_db.all_docs, include_docs=True)
            q = {}
            for i in n:
                q[i['id']] = i['doc']
            if len(q) == 0:
                return jsonify({'train': model2FileData})
            else:
                trans = pd.DataFrame(q)
                print(trans)
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
        except:
            return jsonify({'trace': traceback.format_exc()})

@app.route('/submit-form', methods=['POST'])
def saveForm():
    try:
        global knn
        x = knn
        global model2FileData
        db_name = ""
        if x == None or model2FileData == None: db_name = "train-db"
        else: db_name = "predict-db"
        r = request.json
        r['_id'] = r['employeeId']
        my_database = DB.client[db_name]
        new_document = my_database.create_document(r)

        if new_document.exists():
            return jsonify({'message': "Form Saved Successfully"})
        else:
            return jsonify({'message': "Form Couldn't be saved"})
    except:
        return jsonify({'message': "Ensure all fields are filled"})

@app.route('/save-score', methods=['PUT'])
def saveScore():
    try:
        r = request.json
        emp_id = r['employeeId']
        score = r['performanceScore']
        try:
            train_db = DB.client[train_db_name]
            train_document = train_db[emp_id]
            train_document['performanceScore'] = score
            train_document.save()
            return jsonify({'message': "Score saved successfully"})
        except KeyError:
            try:
                predict_db = DB.client[predict_db_name]
                predict_document = predict_db[emp_id]
                predict_document['performanceScore'] = score
                predict_document.save()
                return jsonify({'message': "Score saved successfully"})
            except KeyError: 
                return jsonify({'message': "No form submitted with this employee id"})
    except:
        import sys
        print("Oops!", sys.exc_info()[0], "occured.")
        return jsonify({'message': "Bad request"})
    
@app.route('/form-count', methods=['GET'])
def formCount():
    train_db = DB.client[train_db_name]
    predict_db = DB.client[predict_db_name]
    count = train_db.doc_count() + predict_db.doc_count()
    return jsonify({'count': count})

@app.route('/batch-upload', methods=['POST'])
def upload():
    r = request.json
    train_db = DB.client[train_db_name]
    return jsonify({'status': train_db.bulk_docs(r)})

if __name__ == '__main__':
    app.run(host='0.0.0.0')
    # app.run()
