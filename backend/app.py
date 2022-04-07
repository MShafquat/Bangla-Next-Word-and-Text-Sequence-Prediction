from base64 import encode
from http import cookies
from flask import Flask, request, jsonify
from flask import send_file

from ml_prediction.machine_learning import predict_lstm

app = Flask(__name__)

@app.route('/api/predict/lstm', methods=['GET'])
def predict_lstm_view():
    input = request.args.get('input')
    res = predict_lstm(str(input))
    return  jsonify(res)

@app.route('/api/predict/gpt2', methods=['GET'])
def predict_gpt2_view():
    return jsonify(['ami', 'jani', 'ki', 'hdf'])
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)