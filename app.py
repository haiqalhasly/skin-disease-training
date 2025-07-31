from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("C:/Users/Administrator/Desktop/YOLO/model/skindisease_model.pt") # import AI model
#C:/Users/Administrator/Desktop/YOLO/model/skindisease_model.pt
#C:/Users/Administrator/Desktop/YOLO/model/acneModel.pt
#C:/Users/Administrator/Desktop/YOLO/model/maskModel.pt
#C:/Users/Administrator/Desktop/YOLO/model/fatModel.pt
#


cap = cv2.VideoCapture(0) # import camera

while True:
    ret, frame = cap.read() # read camera frame
    results = model(frame) # run AI model
    cv2.imshow("Frame", results[0].plot()) # show frame and results
    if cv2.waitKey(1) & 0xFF == ord("q"): # press q to quit
        break

cap.release() # release camera
cv2.destroyAllWindows() # close all windows
