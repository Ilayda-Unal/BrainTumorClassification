import cv2
import os
from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# __name__ = "app"
app = Flask(__name__)

flag = {0: 'Glioma', 1: 'Meningioma', 2: 'No', 3: 'Pituitary'}

img_size = 256
model = load_model('BrainTrainingModel.h5')

model.make_predict_function()


def predict_label(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(gray, (img_size, img_size))
    i = image.img_to_array(resized) / 255.0
    i = i.reshape(1, img_size, img_size, 1)
    p = model.predict_classes(i)
    return flag[p[0]]


@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")


@app.route("/predict", methods=['GET', 'POST'])
def upload():

    if request.method == 'POST':
        img = request.files['file']
        img_path = "uploads/" + img.filename
        img.save(img_path)
        p = predict_label(img_path)
        return str(p)


if __name__ == "__main__":
    app.debug = True
    app.run(debug=True)
