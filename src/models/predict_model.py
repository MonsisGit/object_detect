import cv2 as cv
from PIL import Image
import torch
import numpy as np
from matplotlib import pyplot as plt

from sort import Sort

cmap = plt.get_cmap('tab20b')
colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
font = cv.FONT_HERSHEY_SIMPLEX

yolo_v5n = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)

mot_tracker = Sort()

points=[]

cap = cv.VideoCapture(3)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    if len(points)>200:
        points=[]
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

        #tracked_objects = mot_tracker.update(results.pandas().xyxy[0].values[:,0:5])
        tracked_objects = mot_tracker.update(results.pandas().xyxy[0].values[:,0:5][np.where(results.pandas().xyxy[0].name=="person")[0],0:5])

        classes=results.pandas().xyxy[0].values[np.where(results.pandas().xyxy[0].name=="person")[0],-1]
        for idx, track_out in enumerate(tracked_objects):
            x1, y1, x2, y2, obj_id = track_out
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            points.append([x1+(x2-x1)/2,y1+(y2-y1)/2]) 
            cv.polylines(frame, np.int32([points]), 2, (0,255,255))

            frame = cv.rectangle(frame,(x1,y1),(x1+x2,y1+y2),(0,255,0),2)
            color = colors[int(obj_id) % len(colors)]
            color = [i * 255 for i in color]
            cv.putText(frame,classes[idx] ,(x1+3,y1+25),0,0.5,(0,0,0))
            cv.putText(frame,"ObjID: " + str(obj_id)[0:5],(x1+3,y1+10),0,0.5,(0,0,0))
                
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break
    

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()