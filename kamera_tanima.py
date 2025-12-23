import torch
import cv2

# YOLOv5 hazır modeli 
model = torch.hub.load(
    'ultralytics/yolov5',
    'yolov5s',
    pretrained=True
)

# Kamera açımı
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Kamera açılamadı")
    exit()

print("Kamera açıldı. Çıkmak için ESC.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("okunamadı")
        break

    results = model(frame)
    annotated_frame = results.render()[0]

    cv2.imshow("YOLOv5 Kamera Nesne Tanima", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
