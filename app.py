from flask import Flask, request, render_template
from keras.models import load_model
from keras import backend as K
import tensorflow as tf
from PIL import Image
import keras as kr
import numpy as np
import base64

global model, graph

model = load_model('model/num_reader.h5')
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def imageProcess():

    # Setting image size to that of mnist images
    img_size = 28, 28

    # https://www.journaldev.com/23617/python-string-encode-decode
    encoded = request.values[('imgBase64')]
  
    with open('output.png', 'wb') as output:
        output.write(base64.b64decode(encoded[22:]))

    # https://stackoverflow.com/questions/9506841/using-python-pil-to-turn-a-rgb-image-into-a-pure-black-and-white-image
    img = Image.open('output.png').convert('L')
   
    # Resizing and reshaping the image
    # https://www.programcreek.com/python/example/223/Image.open
    img = img.resize(img_size, Image.ANTIALIAS)
    img = np.array(img, dtype=np.float32).reshape(1, 784)
    img /= 255

    setPrediction = model.predict(img)
    # print(setPrediction)
    getPrediction = np.array(setPrediction[0])
    #print(getPrediction)
    prediction = str(np.argmax(getPrediction))
    #print(prediction)

    print(prediction)
 
    return prediction
    
    

app.run(threaded=False)