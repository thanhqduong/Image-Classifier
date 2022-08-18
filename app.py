from __future__ import division, print_function

import os
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)

PATH = 'classify.h5'
model = load_model(PATH)



def model_predict(img_path, model):
    img = load_img(img_path, target_size=(128, 128))

    x = img_to_array(img)
    x = x/255
    x = np.expand_dims(x, axis=0)

    ret = []
    preds = model.predict(x)
    if preds[0][0] >= 0.5:
        ret = ["Cartoon", preds[0][0]]
    else:
        ret = ["Anime", 1 - preds[0][0]]

    return ret


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        preds = model_predict(file_path, model)

        return preds[0] + ' (' + str(round(preds[1] * 100, 2)) + '%)'
    return None


if __name__ == '__main__':
    app.run(debug=True)

