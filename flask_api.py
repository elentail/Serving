import numpy as np
import tensorflow as tf
from flask import Flask, request
from flask_restx import Resource, Api

app = Flask(__name__)
api = Api(app)

tf_model = tf.keras.models.load_model('tf_cnn_model/1')

@api.route('/predict')
class SimplePredict(Resource):

    def post(self):
        data = request.json.get('data')
        data = np.array(data, dtype=np.float32)
        print(data.shape)
        predict = tf_model.predict(data)

        return {'predict':np.argmax(predict, axis=1).tolist()}

        
if __name__ == '__main__':    
    #app.run(debug=True, host='0.0.0.0', port=80)
    app.run(host='0.0.0.0', port=80)
