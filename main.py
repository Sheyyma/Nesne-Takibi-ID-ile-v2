import cv2
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort


def main():
    # ---------------- YOLOv5 ----------------
    print("YOLOv5 yükleniyor...")
    model = torch.hub.load(
        "ultralytics/yolov5",
        "yolov5s",
        pretrained=True
    )

    model.conf = 0.4        # güven eşiği
    model.classes = [0]    # sadece person

    # ---------------- TRACKER ----------------
    tracker = DeepSort(
        max_age=30,
        n_init=3,
        max_iou_distance=0.7
    )

    # ---------------- KAMERA ----------------
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Kamera açılamadı!")
        return

    print("Kamera açıldı. ESC ile çıkış.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO inference
        results = model(frame)
        detections = []

        for *box, conf, cls in results.xyxy[0]:
            x1, y1, x2, y2 = map(int, box)
            w = x2 - x1
            h = y2 - y1

            detections.append(
                ([x1, y1, w, h], conf.item(), "person")
            )

        # TRACKING (ID burada geliyor)
        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, r, b = map(int, track.to_ltrb())

            cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"ID: {track_id}",
                (l, t - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        cv2.imshow("YOLO Person Tracking", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
