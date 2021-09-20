#!/usr/bin/env python
# coding: utf-8
import pandas as pd
from fbprophet import Prophet
import json
from flask import Flask, request, jsonify
app = Flask(__name__)


@app.route('/sendjson2/', methods=['POST','GET'])
def sendjson2():
    if request.method == 'POST':
        data = json.loads(request.get_data())
        df = pd.DataFrame()
        index=list()
        value=list()
        for i in data:
            index.append(i['ds'])
            value.append(i['y'])
        df['ds'] = index
        df['y'] = value
        #print(df)
        m = Prophet()
        m.fit(df)
        future = m.make_future_dataframe(periods=90)
        forecast = m.predict(future)
        temp=pd.DataFrame(forecast[['ds','yhat']])
        temp['ds'] = pd.to_datetime(temp['ds'])
        temp.index = temp['ds']
        temp=temp.resample('M').sum()
        date=temp['yhat'].index.date
        value=list(temp['yhat'])
        output={}
        for i,v in enumerate(date):
            output[str(v)[:-3]]=value[i]
                #output=json.dumps(output, sort_keys=True)
                #print(jsonify(output))
        return jsonify(output)
    else:
        print("get")
        return "see"


# A welcome message to test our server
@app.route('/')
def index():
    return "<h1>Welcome to our server !!</h1>"

if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)




