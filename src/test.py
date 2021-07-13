from yolov5 import YOLOv5
import cv2 
from yolov5.utils.plots import colors, plot_one_box
from yolov5.utils.general import xywh2xyxy
import numpy as np
import torch
import random
from collections import deque

model_path = "src/best.pt"
device = 'cpu'
cam = cv2.VideoCapture(0)

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

net = YOLOv5(model_path, device)

class PeopleTracker:

    def __init__(self, net, debug = True):
        self.debug = debug
        self.net = net
        self.frame = None
        self.debug_frame = None
        self.trackers = []
        self.detected_targets = []
        self.tracked_targets = []
        self.numOfTrackedTargets = 0

    def get_detections(self, frame):
        detections = self.net.predict(frame)
        detected = []
        for bbox in detections.pred:
            for *xyxy, cond, cls in bbox:
                if cls == 0 and cond > 0.7:
                    x,y,w,h = int(xyxy[0]),int(xyxy[1]),int(xyxy[2] - xyxy[0]),int(xyxy[3]-xyxy[1])
                    if self.debug:
                        label = f'{detections.names[int(cls)]} {cond:.2f}'
                        plot_one_box(xyxy, self.debug_frame, label=label, color=colors(int(cls),True), line_thickness=3)
                    detected.append((x,y,w,h))
        self.detected_targets = detected
        

    def init_tracking(self):
        for target in self.detected_targets:
            self.tracked_targets.append(TrackedTarget(random.randint(1, 101), target, self.frame))
        self.numOfTrackedTargets = len(self.detected_targets)
            
    def update_tracking(self):
        print(self.numOfTrackedTargets)
        for target in self.tracked_targets:
            if not target.update(self.frame, self.debug_frame):
                self.tracked_targets.remove(target)
                del(target)
        self.numOfTrackedTargets = len(self.tracked_targets)
                    

    def run(self, frame):
        self.frame = frame
        self.debug_frame = frame
        if self.numOfTrackedTargets == 0:
            self.get_detections(frame)
            self.init_tracking()
        self.update_tracking()


class TrackedTarget:

    def __init__(self, id, initBB, frame):
        self.id = id
        self.bbox = initBB
        self.tracker = cv2.TrackerKCF_create()
        self.reset(frame, initBB)

    def __del__(self):
        print("person " + str(self.id) + " is out of frame")

    def reset(self, frame, initBB):
        self.tracker.init(frame, initBB)

    def update(self, frame, debug_frame):
        success, box =self.tracker.update(frame)
        if success:
            (x, y, w, h) = [int(v) for v in box]
            self.bbox = box
            cv2.rectangle(debug_frame, (x, y), (x + w, y + h),
				(0, 255, 255), 2)
            cv2.putText(debug_frame, str(self.id), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
        return success


if __name__ == "__main__":

    tracker = PeopleTracker(net)
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        
        tracker.run(frame)
        cv2.imshow("debug", tracker.debug_frame)

        k = cv2.waitKey(1)
        if k%256 == 27:
            print("Escape hit, closing...")
            break
    cam.release()

    cv2.destroyAllWindows()