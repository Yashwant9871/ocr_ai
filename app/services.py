import time
from typing import Optional

import cv2
from ultralytics import YOLO

from app.config import (
    CONF_THRESHOLD,
    IOU_THRESHOLD,
    MAX_CENTROID_DISTANCE,
    MAX_MISSING_FRAMES,
    MODEL_PATH,
    SHOW_WINDOW,
)
from app.detection import center_inside, compute_iou, extract_detections, resolve_class_ids
from app.rendering import annotate_frame
from app.state import RuntimeState
from app.tracker import CentroidTracker
from app.types import BBox


def capture_frames(rtsp_url: str, state: RuntimeState) -> None:
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

            with state.lock:
                state.current_frame = frame


def process_stream(state: RuntimeState) -> None:
    model = YOLO(MODEL_PATH)
    boiler_class_id, bag_class_id = resolve_class_ids(model)

    tracker = CentroidTracker(MAX_CENTROID_DISTANCE, MAX_MISSING_FRAMES)

    counted_ids = set()
    last_opening_box: Optional[BBox] = None

    while True:
        frame = None
        with state.lock:
            if state.current_frame is not None:
                frame = state.current_frame.copy()
                state.current_frame = None

        if frame is None:
            continue

        results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)
        result = results[0]

        opening_box, bag_boxes, _ = extract_detections(
            result, boiler_class_id, bag_class_id
        )

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
                    with state.lock:
                        state.current_count += 1

        if SHOW_WINDOW:
            annotated = annotate_frame(
                frame,
                last_opening_box,
                tracked_bags,
                counted_ids,
                state.current_count,
            )
            cv2.imshow("Live", annotated)
            cv2.waitKey(1)
