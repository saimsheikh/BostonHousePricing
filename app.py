import pickle
from flask import Flask,render_template,request,app,jsonify,url_for,session,render_template

import numpy as np
import pandas as pd


app=Flask(__name__)

#loading the model
regmodel=pickle.load(open("regmodel.pkl",'rb'))
scaler=pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data= request.get_json(force=True)['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output= regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_in=scaler.transform(np.array(data).reshape(1,-1))
    print(final_in)
    output=regmodel.predict(final_in)[0]
    return render_template("home.html",prediction_text="the house price prediction is {}".format(output))

    
if __name__=='__main__':
    app.run(debug=True)

