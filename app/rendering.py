from typing import Dict, Optional, Set

import cv2
import numpy as np

from app.types import BBox


def annotate_frame(
    frame: np.ndarray,
    opening_box: Optional[BBox],
    tracked_bags: Dict[int, BBox],
    counted_ids: Set[int],
    total_count: int,
) -> np.ndarray:
    annotated = frame.copy()

    if opening_box is not None:
        x1, y1, x2, y2 = opening_box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 140, 255), 2)
        cv2.putText(
            annotated,
            "boiler_opening",
            (x1, max(y1 - 10, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 140, 255),
            2,
        )

    for bag_id, box in tracked_bags.items():
        x1, y1, x2, y2 = box
        is_counted = bag_id in counted_ids
        color = (0, 255, 0) if is_counted else (255, 0, 0)
        label = f"bag id={bag_id}" + (" counted" if is_counted else "")

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(y1 - 10, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    cv2.putText(
        annotated,
        f"Total Bags Counted: {total_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        3,
    )

    return annotated
