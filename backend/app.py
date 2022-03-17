from http import cookies
from flask import Flask,request,jsonify
from flask import send_file

app = Flask(__name__)

@app.route('/model')
def model():
    path = "mlmodels/bn_lstm.h5"
    return send_file(path, as_attachment=True)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return 'GET'
    elif request.method == 'POST':
        data = request.json
        return jsonify(data)
    else:
        return 'UNKNOWN'
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)