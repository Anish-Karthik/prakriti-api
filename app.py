from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)


regmodel=pickle.load(open('prakriti-classifier.pkl','rb'))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    data = np.array([list(data)])
    print(data)
    output=regmodel.predict(data)
    print(output[0])
    return jsonify(output[0])

if __name__ == '__main__':
  app.run(host='0.0.0.0', debug=True)