from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np

from app.types import BBox


class CentroidTracker:
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
        return (x1 + x2) // 2, (y1 + y2) // 2

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
