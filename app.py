# some utilities
import os
import numpy as np
# from util import base64_to_pil
import re
import base64

import numpy as np
# import cv2
# from PIL import Image
from io import BytesIO

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect

#tensorflow

# Variables 
# Change them if you are using custom model or pretrained model with saved weigths
Model_json = ".json"
Model_weigths = ".h5"

# Import all necessary libaries
import os
import pathlib
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.image as img
import random
import tensorflow as tf


# Declare a flask app 
app = Flask(__name__)

def get_ImageClassifierModel():
    model = tf.keras.models.load_model('model.h5')

    return model    

def model_predict(img, model):
    
    labels = np.array(['DEFECT', 'HEALTHY'])
    
    # Import Image ansd Pre-process it
    # image = tf.io.read_file(custom_image_path)
    # print(img)
    image_tensor = tf.convert_to_tensor(img)
    image_tensor = tf.image.decode_image(image_tensor)
    image_tensor = tf.image.resize(image_tensor, size=[224,224])
    image_tensor = image_tensor/255
    image_tensor_reshaped = tf.expand_dims(image_tensor, axis=0)
    
    # Make a prediction
    pred = model.predict(image_tensor_reshaped)
    pred_class = labels[int(tf.round(pred))]
    
    # Plot the image and class label
    # plt.figure(figsize=(10,10))
    # plt.imshow(image_tensor)
    # plt.title(f'Prediction: {pred_class}')

    return pred_class


@app.route('/', methods=['GET'])
def index():
    '''
    Render the main page
    '''
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    '''
    predict function to predict the image
    Api hits this function when someone clicks submit.
    '''
    if request.method == 'POST':
        # Get the image from post request
        # img = base64_to_pil(request.json)
        img = re.sub('^data:image/.+;base64,', '', request.json)
        pil_image = BytesIO(base64.b64decode(img))
        img1 = pil_image.getbuffer().tobytes()

        # initialize model
        model = get_ImageClassifierModel()

        # Make prediction
        preds = model_predict(img1,model)
        
        # Serialize the result, you can add additional fields
        return jsonify(result=preds)
    return None


if __name__ == '__main__':
    # app.run(port=5002)
    PORT=8080
    app.run(debug=True, port=PORT)
    

