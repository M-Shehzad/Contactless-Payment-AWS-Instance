from flask import Flask, redirect, url_for, request, Response
import boto3
import numpy as np
import json
import tensorflow as tf
import time

app = Flask(__name__)

model = tf.keras.models.load_model('./model/0000001')

table = boto3.resource('dynamodb',region_name='ap-south-1').Table('palmPrints')

def putDynamoDB(data):
    try:
        timestamp = int(time.time())
        item = {'timestamp': timestamp} | data
        table.put_item(Item=item)
        return True
    except Exception as e:
        print(e)
        return False

@app.route('/')
def index():
    return Response('Hello World!')




@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image = np.array(data['image'])
    imageName = data['imageName']

    prediction = model.predict(image).tolist()

    DBstatus = putDynamoDB({'imageName': imageName, 'vector': [str(i) for i in prediction]})

    return json.dumps({'prediction': prediction, 'DBstatus': DBstatus})

@app.route('/getAllVectors', methods=['GET'])
def getAllVectors():
    response = table.scan()
    return json.dumps(response['Items'])

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8888)
