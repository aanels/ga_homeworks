from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import pickle

app = Flask(__name__)

with open('./data/models.pkl','rb') as fi:
    mds = pickle.load(fi)

def predict_for(drp_loc, drp_time):
    pdf = pd.DataFrame({'ds': [drp_time], 'cap' : 25.0})
    try:
        fc = mds[drp_loc].predict(pdf)
        #print (fc)
    except KeyError:
        return 'Location not trained for prediction'
    return 'Predicted trips: ' + str(int(round(fc.yhat[0],0)))

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/gettrips',methods=['POST','GET'])
def get_trips():
    if request.method=='POST':
        result=request.form

        dtm = datetime(int(result['yr']),int(result['mn']),int(result['dy']),int(result['hr']),int(result['min']))
        prediction = predict_for(result['dropoff'], dtm)

        return render_template('result.html',prediction=prediction)


if __name__ == '__main__':
	app.run()
