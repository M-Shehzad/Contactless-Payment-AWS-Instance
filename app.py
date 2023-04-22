from flask import Flask, redirect, url_for, request, Response
import numpy as np
import json
import tensorflow as tf

app = Flask(__name__)

model = tf.keras.models.load_model('./model/0000001')

@app.route('/')
def index():
    return Response('Hello World!')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image = np.array(data['image'])

    prediction = model.predict(image)

    return json.dumps({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)