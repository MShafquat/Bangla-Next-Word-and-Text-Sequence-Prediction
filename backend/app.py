from http import cookies
from flask import Flask,request,jsonify
from flask import send_file

app = Flask(__name__)

@app.route('/predict/lstm', methods=['GET'])
def predict():
    return 'GET'

@app.route('/predict/gpt-2', methods=['GET'])
def predict():
    return jsonify(['ami', 'jani', 'ki', 'hdf'])
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)