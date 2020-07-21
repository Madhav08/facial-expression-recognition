from flask import Flask, render_template,request, Response
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from cv2 import cv2
import os
from werkzeug.utils import secure_filename
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from camera import VideoCamera


app = Flask(__name__) 

model=load_model("mymodel.h5")

emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}

@app.route('/', methods=['GET', 'POST'])  
def index(): 
	return render_template('index.html')

'''
@app.route('/predict',methods=['POST'])
def predict():
    im = cv2.imread("sad.jpg",0)
	im=np.array(im)
	im=cv2.resize(im, (48, 48))
	im=im.reshape(1,48,48,1)
	pred=model.predict(im)
	pred=pred.argmax()
	print(emotion_dict[pred])
'''
def model_predict(img_path, model):
    im = cv2.imread(img_path,0)
    im=np.array(im)
    im=cv2.resize(im, (48, 48))
    im=im.reshape(1,48,48,1)
    pred=model.predict(im)
    pred=pred.argmax()
    return emotion_dict[pred]


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files["file"]

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', f.filename)
        f.save(file_path)

        preds = model_predict(file_path, model)
        return preds
    return None
    	
@app.route('/video')
def video():
    return render_template('video.html') 

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':  
	app.run(debug=True) 
