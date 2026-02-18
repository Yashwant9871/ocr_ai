from typing import List, Optional, Tuple

from ultralytics import YOLO

from app.config import CONF_THRESHOLD
from app.types import BBox, Detection


def compute_iou(box_a: BBox, box_b: BBox) -> float:
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

    return inter_area / union


def center_inside(inner_box: BBox, outer_box: BBox) -> bool:
    x1, y1, x2, y2 = inner_box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    ox1, oy1, ox2, oy2 = outer_box
    return ox1 <= cx <= ox2 and oy1 <= cy <= oy2


def resolve_class_ids(model: YOLO) -> Tuple[Optional[int], Optional[int]]:
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


def extract_detections(
    result, boiler_class_id: Optional[int], bag_class_id: Optional[int]
) -> Tuple[Optional[BBox], List[BBox], List[Detection]]:
    opening_candidates = []
    bag_boxes: List[BBox] = []
    all_detections: List[Detection] = []

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

    opening_box = (
        max(opening_candidates, key=lambda candidate: candidate[1])[0]
        if opening_candidates
        else None
    )

    return opening_box, bag_boxes, all_detections
