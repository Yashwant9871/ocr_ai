# ================================
# AI-Based Real-Time Bag Counter
# RTSP + Threaded + FastAPI
# ================================

# IMPORTS
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from fastapi import FastAPI
import uvicorn
import cv2
import numpy as np
import threading

import time
from ultralytics import YOLO            #YOLO model


# CONFIGURATION
MODEL_PATH = "D:\ObjectDetection\models\yolov8s_12_2_26.pt"                                            # Path to trained YOLO model
RTSP_URL = "rtsp://localhost:8554/mystream"  # MediaMTX stream


CONF_THRESHOLD = 0.4                                                                                # Detection confidence threshold
IOU_THRESHOLD = 0.3                                                                                 # IoU threshold for counting logic

# Tracking parameters
MAX_CENTROID_DISTANCE = 80                                                                          # Max allowed movement between frames
MAX_MISSING_FRAMES = 5     

SHOW_WINDOW = True  # Set False in production


# GLOBAL STATE
current_count = 0
current_frame = None
frame_lock = threading.Lock()                 # lock because threads + shared state = race conditions


# ================================
# TYPE DEFINITIONS
# ================================

# Bounding box format (x1, y1, x2, y2)
BBox = Tuple[int, int, int, int]

@dataclass
class Detection:
    """
    Stores detection information.
    """
    bbox: BBox
    confidence: float
    class_id: int


# ================================
# CENTROID TRACKER
# ================================

class CentroidTracker:
    """
    Simple centroid-based object tracker.
    Assigns unique IDs to detected bags and keeps tracking
    them across frames using Euclidean distance.
    """

    #Assigns a unique ID to every bag
    #Matches new detections to old ones using centroid distance
    #Removes objects that disappear too long

    def __init__(self, max_distance: int = 80, max_missing: int = 25) -> None:
        self.next_object_id = 0
        self.objects: "OrderedDict[int, Tuple[int, int]]" = OrderedDict()
        self.object_boxes: "OrderedDict[int, BBox]" = OrderedDict()
        self.missing_counts: "OrderedDict[int, int]" = OrderedDict()
        self.max_distance = max_distance
        self.max_missing = max_missing

    @staticmethod
    def _centroid(box: BBox) -> Tuple[int, int]:
        x1, y1, x2, y2 = box
        return (x1 + x2) // 2, (y1 + y2) // 2         # convert bounding box to centroid

    def register(self, box: BBox) -> None:
        centroid = self._centroid(box)
        self.objects[self.next_object_id] = centroid
        self.object_boxes[self.next_object_id] = box
        self.missing_counts[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id: int) -> None:
        del self.objects[object_id]
        del self.object_boxes[object_id]
        del self.missing_counts[object_id]

    def update(self, detections: List[BBox]) -> Dict[int, BBox]:
        """
        Updates tracked objects using new detections.
        Returns dictionary: {object_id: bounding_box}
        """

        if len(detections) == 0:
            for object_id in list(self.missing_counts.keys()):
                self.missing_counts[object_id] += 1
                if self.missing_counts[object_id] > self.max_missing:
                    self.deregister(object_id)
            return dict(self.object_boxes)

        input_centroids = np.array([self._centroid(box) for box in detections])

        if len(self.objects) == 0:
            for box in detections:
                self.register(box)
            return dict(self.object_boxes)

        object_ids = list(self.objects.keys())
        object_centroids = np.array(list(self.objects.values()))

        distance_matrix = np.linalg.norm(
            object_centroids[:, np.newaxis] - input_centroids[np.newaxis, :], axis=2
        )

        rows = distance_matrix.min(axis=1).argsort()
        cols = distance_matrix.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue

            if distance_matrix[row, col] > self.max_distance:
                continue

            object_id = object_ids[row]
            self.objects[object_id] = tuple(input_centroids[col])
            self.object_boxes[object_id] = detections[col]
            self.missing_counts[object_id] = 0

            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(distance_matrix.shape[0])) - used_rows
        unused_cols = set(range(distance_matrix.shape[1])) - used_cols

        for row in unused_rows:
            object_id = object_ids[row]
            self.missing_counts[object_id] += 1
            if self.missing_counts[object_id] > self.max_missing:
                self.deregister(object_id)

        for col in unused_cols:
            self.register(detections[col])

        return dict(self.object_boxes)


# ================================
# UTILITIES
# ================================
def compute_iou(box_a, box_b):
    """
    Computes Intersection over Union between two bounding boxes.
    Used to check overlap between bag and boiler opening.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)

    union = area_a + area_b - inter_area
    if union <= 0:
      return 0.0

    return inter_area /union        # if bag overlaps boiler opening (if >=0.3 then overlapping)


def center_inside(inner_box, outer_box):
    """
    Checks if center of bag lies inside boiler opening.
    """
    # to check if overlaps or center inside

    x1, y1, x2, y2 = inner_box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    ox1, oy1, ox2, oy2 = outer_box
    return ox1 <= cx <= ox2 and oy1 <= cy <= oy2


def resolve_class_ids(model: YOLO) -> Tuple[Optional[int], Optional[int]]:
    # label names from the trained model
    names = model.names

    if isinstance(names, dict):
        class_map = {idx: name.lower() for idx, name in names.items()}
    else:
        class_map = {idx: name.lower() for idx, name in enumerate(names)}

    boiler_id = None
    bag_id = None

    for idx, name in class_map.items():
        if name == "boiler_opening":
            boiler_id = idx
        if name == "bag":
            bag_id = idx

    return boiler_id, bag_id


def extract_detections(result, boiler_class_id, bag_class_id):
    # if multiple detection exist og boiler opening, ignore lower level confidence
    opening_candidates = []
    bag_boxes= []
    all_detections = []

    if result.boxes is None:
        return None, bag_boxes, all_detections

    for box in result.boxes:
        conf = float(box.conf[0])
        if conf < CONF_THRESHOLD:
            continue

        class_id = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        bbox = (x1, y1, x2, y2)

        all_detections.append(Detection(bbox, conf, class_id))

        if class_id == boiler_class_id:
            opening_candidates.append((bbox, conf))
        elif class_id == bag_class_id:
            bag_boxes.append(bbox)

    opening_box = max(opening_candidates, key=lambda x: x[1])[0] if opening_candidates else None
    
    return opening_box, bag_boxes, all_detections


def annotate_frame(
    frame: np.ndarray,
    opening_box: Optional[BBox],
    tracked_bags: Dict[int, BBox],
    counted_ids: set,
    total_count: int,
) -> np.ndarray:
    annotated = frame.copy()
    """
    Boiler Opening: Orange box
    Uncounted Bag: Blue box
    Counted Bag: Green box
    Total Bag Count: Top text
    """

    if opening_box is not None:
        x1, y1, x2, y2 = opening_box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 140, 255), 2)
        cv2.putText(
            annotated,"boiler_opening",(x1, max(y1 - 10, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 140, 255),2,)

    for bag_id, box in tracked_bags.items():
        x1, y1, x2, y2 = box
        is_counted = bag_id in counted_ids
        color = (0, 255, 0) if is_counted else (255, 0, 0)
        label = f"bag id={bag_id}" + (" counted" if is_counted else "")

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,label,(x1, max(y1 - 10, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2,)

    cv2.putText(
        annotated,
        f"Total Bags Counted: {total_count}",
        (20, 40),cv2.FONT_HERSHEY_SIMPLEX,1.0,
        (0, 255, 255),3,)

    return annotated



# ================================
# RTSP CAPTURE THREAD
# ================================

# RTSP capture thread function - continuously captures frames from RTSP stream and updates current_frame
# auto reconnects, avoid blocking, keeps latest frame only
def capture_frames(rtsp_url):
    global current_frame

    while True:
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        if not cap.isOpened():
            print("Failed to connect. Retrying in 5 seconds...")
            time.sleep(5)
            continue

        print("Connected to RTSP stream.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Stream lost. Reconnecting...")
                cap.release()
                break
            
            with frame_lock:
                current_frame = frame


# ================================
# PROCESSING THREAD
# ================================

def process_stream():
    """
    Main pipeline:
    - Load model
    - Read video
    - Detect objects
    - Track bags
    - Count when entering boiler opening
    - Save annotated video
    """
    global current_count, current_frame

    model = YOLO(MODEL_PATH)
    boiler_class_id, bag_class_id = resolve_class_ids(model)

    tracker = CentroidTracker(MAX_CENTROID_DISTANCE, MAX_MISSING_FRAMES)

    counted_ids = set()
    last_opening_box = None

    while True:

        with frame_lock:
            if current_frame is None:
                continue
            frame = current_frame.copy()
            current_frame = None   # <<< IMPORTANT - clear frame to avoid processing same frame multiple times

         # If no new frame came, skip
        if frame is None:
            continue

        
        results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)
        result = results[0]

        opening_box, bag_boxes, _ = extract_detections(
            result, boiler_class_id, bag_class_id,)

        if opening_box is not None:
            last_opening_box = opening_box

        tracked_bags = tracker.update(bag_boxes)

        if last_opening_box is not None:
            for bag_id, bag_box in tracked_bags.items():

                if bag_id in counted_ids:
                    continue

                overlaps = compute_iou(bag_box, last_opening_box) >= IOU_THRESHOLD
                center_in_opening = center_inside(bag_box, last_opening_box)

                if overlaps or center_in_opening:
                    counted_ids.add(bag_id)
                    with frame_lock:
                        current_count += 1

        if SHOW_WINDOW:
            annotated = annotate_frame(
                frame,
                last_opening_box,
                tracked_bags,
                counted_ids,
                current_count,
            )
            cv2.imshow("Live", annotated)
            cv2.waitKey(1)  

# print(f"Final total bag count: {current_count}")
  

# ================================
# FASTAPI
# ================================

app = FastAPI()

@app.get("/count")
def get_count():
    with frame_lock:
        return {"bag_count": current_count}
    

# ================================
# MAIN
# ================================

if __name__ == "__main__":

    # thread1 - RTSP capture (non-blocking, auto-reconnect, keeps latest frame only)
    # thread2 - main processing loop (runs YOLO, tracking, counting, annotation)
    # main thread - FastAPI server to serve count via API

    threading.Thread(target=capture_frames, args=(RTSP_URL,), daemon=True).start()
    threading.Thread(target=process_stream, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=8000)


