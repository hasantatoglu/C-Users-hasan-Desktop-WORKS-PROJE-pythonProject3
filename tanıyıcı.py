import cv2
import numpy as np
import os
from time import sleep
from Client import sendMessage2Server as send2server


#!/usr/bin/env python3



recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
faceCascade = cv2.CascadeClassifier("Cascade/haarcascade_frontalface_default.xml")
font = cv2.FONT_HERSHEY_SIMPLEX
id = 0
names = ['Unknown', 'Brad', 'Angelina', 'Child ']

cam = cv2.VideoCapture(0)
cam.set(3, 640) # set Width
cam.set(4, 480) # set Height

minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)
while True:
    ret, img =cam.read()
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=1,minSize=(int(minW), int(minH)),)

    for (x, y, w, h) in faces:

        """color = img[y:y + h, x:x + w]"""
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])


        # confidence can be changed accordigly
        if (50 > confidence):

             #id = names[id]
             send2server("HIGH")
             confidence = "  {0}%".format(round(100 - confidence))
             print(names[id])

            # Network MSG send, or make connection via Multithread.
        else:

            send2server("LOW")
            confidence = "  {0}%".format(round(100 - confidence))
            id=0
            print(names[id])

        cv2.putText(img, str(names[id]), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)



    cv2.imshow('camera', img)
    if cv2.waitKey(5) & 0xff == ord('q'):
        break


cam.release()
cv2.destroyAllWindows()
