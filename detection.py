import cv2
import numpy as np

class YOLOv3Detection:
    def __init__(self, model_config_path, model_weights_path, class_names=None, conf_threshold=0.2, nms_threshold=0.4, input_size=(416, 416)):
        self.net = cv2.dnn.readNetFromDarknet(model_config_path, model_weights_path)

        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers().flatten()]

        self.class_names = class_names if class_names is not None else ["potato"]
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.detections = []

    def _frame_to_blob(self, frame):
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, self.input_size, swapRB=True, crop=False)
        return blob


    def _clip_box(self, x, y, w, h, img_w, img_h):
        x = max(0, x)
        y = max(0, y)
        w = max(0, min(w, img_w - x))
        h = max(0, min(h, img_h - y))
        return x, y, w, h

    def detect(self, frame):
        img_h, img_w = frame.shape[:2]
        blob = self._frame_to_blob(frame)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        boxes, confidences, class_ids = [], [], []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])

                if confidence < self.conf_threshold:
                    continue

                center_x = int(detection[0] * img_w)
                center_y = int(detection[1] * img_h)
                width = int(detection[2] * img_w)
                height = int(detection[3] * img_h)

                x = int(center_x - width / 2)
                y = int(center_y - height / 2)
                x, y, width, height = self._clip_box(x, y, width, height, img_w, img_h)

                if width <= 0 or height <= 0:
                    continue

                boxes.append([x, y, width, height])
                confidences.append(confidence)
                class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)

        self.detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                label = (self.class_names[class_ids[i]]if class_ids[i] < len(self.class_names) else "unknown")
                self.detections.append(
                    {
                        "bbox": boxes[i],  # [x, y, w, h]
                        "confidence": confidences[i],
                        "class_id": class_ids[i],
                        "label": label,
                    }
                )

        return self.detections

    def draw_detections(self, frame, detections=None):
        if detections is None:
            detections = self.detections

        for det in detections:
            x, y, w, h = det["bbox"]
            conf = det["confidence"]
            label = det["label"]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {conf:.2f}", (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame

    def get_detections(self):
        return self.detections