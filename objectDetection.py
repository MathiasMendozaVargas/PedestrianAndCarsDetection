################################################
#       OBJECT DETECTION WITH OPENCV
#              @MATHIAS_MENDOZA
################################################

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time

# Create our body and car classifier for pedestrians
body_classifier = cv.CascadeClassifier('Cascades/haarcascade_fullbody.xml')
car_classifier = cv.CascadeClassifier('Cascades/haarcascade_car.xml')


# Initiate video capture for video file
cap = cv.VideoCapture('Assets/streetVideo.mp4')

while cap.isOpened():
    ret, frame = cap.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Analyze frame with cascade
    bodies = body_classifier.detectMultiScale(gray, 1.1, 2)
    cars = car_classifier.detectMultiScale(gray, 1.1, 2)

    # Extract bounding boxes for any bodies identified
    for (x, y, w, h) in bodies:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

    for (x, y, w, h) in cars:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 180, 0), 2)
    
    cv.imshow('Pedestrian and Car Detection', frame)

    if cv.waitKey(1) == 13:
        break

cap.release()
cv.destroyAllWindows()

