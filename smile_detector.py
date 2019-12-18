# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 16:32:15 2019

@author: Dell
"""

import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

def detect(gray,frame):
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    
    for (x, y, w, h) in faces:
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        # Detects eyes of different sizes in the input image 
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
        
        #To draw a rectangle in eyes 
        for (sx, sy, sw, sh) in smiles:
             cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
    return frame    

# Doing some Face Recognition with the webcam    
video_capture = cv2.VideoCapture(0)    
while True:
     _,frame = video_capture.read()
     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
     canvas = detect(gray, frame)
     
     #Display an image in a window 
     cv2.imshow('Video',canvas)
     if cv2.waitKey(1) & 0xFF == ord('q'):
         break


#turn off webcam
video_capture.release()
cv2.destroyAllWindows()