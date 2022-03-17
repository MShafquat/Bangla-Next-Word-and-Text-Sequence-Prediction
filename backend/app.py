from flask import Flask,request,jsonify
import numpy as np
import pickle

model = pickle.load(open('model.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello world"


if __name__ == '__main__':
    app.run(debug=True)