from flask import Flask, redirect, url_for, request, Response
import boto3
import numpy as np
import json
import tensorflow as tf
import time
from uuid import uuid4

app = Flask(__name__)

model = tf.keras.models.load_model('./model/0000001')

table = boto3.resource('dynamodb',region_name='ap-south-1').Table('palmPrints')

def putDynamoDB(data):
    try:
        timestamp = int(time.time())
        item = {**{'timestamp': timestamp}, **data}
        table.put_item(Item=item)
        return True
    except Exception as e:
        print(e)
        return False
    
def predictVector(image):
    return model.predict(image).tolist()

@app.route('/')
def index():
    return Response('Hello World!')


@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    image = np.array(data['image'])
    imageName = data['imageName']
    uniqueId = f"{int(time.time())}-{str(uuid4())}"

    prediction = predictVector(image)

    DBstatus = putDynamoDB({'imageName': imageName, 'vector': [str(i) for i in prediction[0]], 'uniqueId': uniqueId})
    # print({'imageName': imageName, 'vector': [str(i) for i in prediction[0]], 'uniqueId': uniqueId})

    return json.dumps({'prediction': prediction, 'DBstatus': DBstatus})

@app.route('/match', methods=['POST'])
def match():
    data = request.get_json()
    image = np.array(data['image'])
    prediction = predictVector(image)[0]

    response = table.scan(
        ProjectionExpression='vector'
    )
    allVectors = [list(map(float,item['vector'])) for item in response['Items']]

    difference = [np.linalg.norm(np.array(vector)-np.array(prediction)) for vector in allVectors]
    return json.dumps({'difference': difference})

@app.route('/getAllVectors', methods=['GET'])
def getAllVectors():
    response = table.scan()
    # print(response['Items'])
    data = {str(item['uniqueId']):(item['imageName'], item['vector']) for item in response['Items']}
    # print(data)
    return json.dumps(data)



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8888)
