import cv2 as cv
from PIL import Image
import torch
import numpy as np

from sort import Sort


yolo_v5n = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
font = cv.FONT_HERSHEY_SIMPLEX
mot_tracker = Sort()

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    results = yolo_v5n(frame)

    if results.pandas().xyxy[0].shape[0] == 0:
        dets = np.empty((0, 5))
        trackers = mot_tracker.update(dets)
    else:
        for idx in range(results.pandas().xyxy[0].shape[0]):

            res = results.pandas().xyxy[0].iloc[idx]
            x = int(res["xmin"])
            y = int(res["ymin"])
            w = int(res["xmax"]-res["xmin"])
            h = int(res["ymax"]-res["ymin"])
            #dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]

            tracked_objects = mot_tracker.update(np.array([x,y,w+x,h+y,res["confidence"]]))

            frame = cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv.putText(frame,res["name"] + " ("+str(res["confidence"]*100)[0:4] + "%)",(x+w+10,y+h),0,0.5,(0,255,0))

    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break
    

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()