import cv2

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

model = YOLO("yolo11s.pt")
names = model.names
cap = cv2.VideoCapture("path/to/video/file.mp4")

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
writer = cv2.VideoWriter("Ultralytics circle annotation.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

while True:
    ret, im0 = cap.read()
    if not ret:
        break

    annotator = Annotator(im0)
    results = model.predict(im0)
    boxes = results[0].boxes.xyxy.cpu()
    clss = results[0].boxes.cls.cpu().tolist()

    for box, cls in zip(boxes, clss):
        annotator.circle_label(box, label=names[int(cls)])

    writer.write(im0)
    cv2.imshow("Ultralytics circle annotation", im0)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

writer.release()
cap.release()
cv2.destroyAllWindows()
