
import cv2
import matplotlib.pyplot as plt
import numpy as np

model = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'ssd_mobilenet_v2_coco_2018_03_29.pbtxt')

with open('object_detection_classes_coco.txt', 'r') as f:
    class_names = f.read().strip().split('\n')

COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))

cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    image_height, image_width, _ = frame.shape
    model.setInput(cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False))
    output = model.forward()

    for detection in output[0, 0, :, :]:
        score = float(detection[2])
        if score > 0.3:
            class_id = int(detection[1])
            class_name = class_names[class_id - 1] if class_id - 1 < len(class_names) else "Unknown"
            color = COLORS[class_id % len(COLORS)]  

            left = int(detection[3] * image_width)
            top = int(detection[4] * image_height)
            right = int(detection[5] * image_width)
            bottom = int(detection[6] * image_height)

            cv2.rectangle(frame, (left, top), (right, bottom), color, thickness=3)
            cv2.putText(frame, f"{class_name}: {round(score * 100, 2)}%",
                        (left, max(30, top - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1)&0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()
