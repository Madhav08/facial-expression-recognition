from cv2 import cv2
from tensorflow.keras.models import load_model
import numpy as np

font = cv2.FONT_HERSHEY_SIMPLEX
facec=cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
ds_factor=0.6

model=load_model("mymodel.h5")
emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        _, fr = self.video.read()
        gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray_fr, 1.3, 5)

        for (x, y, w, h) in faces:
            fc = gray_fr[y:y+h, x:x+w]

            roi = cv2.resize(fc, (48, 48))
            pred=model.predict(roi[np.newaxis, :, :, np.newaxis])
            pred=pred.argmax()
            cv2.putText(fr,  emotion_dict[pred], (x, y), font, 1, (255, 255, 0), 2)
            cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)

        _, jpeg = cv2.imencode('.jpg', fr)
        return jpeg.tobytes()
