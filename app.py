from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model,save_model
from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.feature import hog
from PIL import Image
from skimage import transform


app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0],2)

    return render_template('index.html',prediction_text="Stress will be {}".format(output))

if __name__ == '__main__':
    app.run(port=3000,debug=True)