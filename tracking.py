import cv2

class MultiTracking:
    def __init__(self):
        self.multi_tracker = self._create_multi_tracker()
        self.tracked_boxes = []
        self.initialized = False

    def _create_multi_tracker(self):
        multiTracker = cv2.legacy.MultiTracker_create()
        return multiTracker

    def _create_single_tracker(self):
        return cv2.legacy.TrackerKCF_create()

    def start_tracker(self, frame, detections):
        self.multi_tracker = self._create_multi_tracker()
        self.tracked_boxes = []

        valid_count = 0
        for det in detections:
            x, y, w, h = det["bbox"]

            if x < 0 or y < 0 or w <= 0 or h <= 0:
                continue
            if x + w > frame.shape[1] or y + h > frame.shape[0]:
                continue

            tracker = self._create_single_tracker()
            self.multi_tracker.add(tracker, frame, (x, y, w, h))
            valid_count += 1

        self.initialized = valid_count > 0
        return self.initialized

    def update_tracker(self, frame):
        success, boxes = self.multi_tracker.update(frame)
        self.tracked_boxes = []

        for box in boxes:
            x, y, w, h = [int(v) for v in box]
            self.tracked_boxes.append((x, y, w, h))

        return success, self.tracked_boxes

    def draw_tracks(self, frame):
        for i, (x, y, w, h) in enumerate(self.tracked_boxes):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame,f"ID_{i}",(x, max(20, y - 10)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255, 0, 0), 2)
        return frame