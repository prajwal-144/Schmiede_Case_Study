import os
import cv2
from detection import YOLOv3Detection
from tracking import MultiTracking

def main():
    config_path = "config/yolov3_training.cfg"
    weights_path = "weights/yolov3_training.weights"
    video_path = "dataset/video.mp4"

    os.makedirs("outputs", exist_ok=True)
    detection_output_path = "outputs/detection_output.mp4"
    tracking_output_path = "outputs/tracking_output.mp4"

    detector = YOLOv3Detection(model_config_path=config_path, model_weights_path=weights_path, class_names=["potato"], conf_threshold=0.2, nms_threshold=0.4)
    tracker = MultiTracking()

    # Part 1: Object Detection
    video = cv2.VideoCapture(video_path)

    frame_w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    detection_writer = cv2.VideoWriter(detection_output_path, fourcc, fps, (frame_w, frame_h))

    frame_index = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break

        detections = detector.detect(frame)
        annotated = detector.draw_detections(frame.copy(), detections)
        detection_writer.write(annotated)
        cv2.imshow("Detection Output", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_index += 1

    video.release()
    detection_writer.release()
    cv2.destroyAllWindows()

    # Part 2: Object Tracking
    video = cv2.VideoCapture(video_path)

    tracking_writer = cv2.VideoWriter(tracking_output_path, fourcc, fps, (frame_w, frame_h))

    frame_index = 0
    detect_every = 50  # detect every n frames to reduce drift

    while True:
        ret, frame = video.read()
        if not ret:
            break

        annotated = frame.copy()

        if frame_index % detect_every == 0:
            detections = detector.detect(frame)
            tracker.start_tracker(frame, detections)
            annotated = detector.draw_detections(annotated, detections)

        else:
            success, tracked_boxes = tracker.update_tracker(frame)
            annotated = tracker.draw_tracks(annotated)

        tracking_writer.write(annotated)
        cv2.imshow("Tracking Output", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_index += 1

    video.release()
    tracking_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()