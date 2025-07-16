from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("yolov8n.pt") # import AI model

cap = cv2.VideoCapture(0) # import camera

while True:
    ret, frame = cap.read() # read camera frame
    results = model(frame) # run AI model
    cv2.imshow("Frame", results[0].plot()) # show frame and results
    if cv2.waitKey(1) & 0xFF == ord("q"): # press q to quit
        break

cap.release() # release camera
cv2.destroyAllWindows() # close all windows
