import cv2
import numpy as np
from sklearn.cluster import KMeans
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

def dominant_color(image, k=4):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    clt = KMeans(n_clusters=k)
    clt.fit(image)
    return clt.cluster_centers_

def compare_colors(color1, color2, threshold=40):
    color_diff = np.linalg.norm(color1 - color2, axis=1)
    return np.all(color_diff < threshold)

cap = cv2.VideoCapture(0)

unique_persons = []
person_ids = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    results = model(frame)[0]

    boxes = []
    confidences = []
    for result in results.boxes:
        if result.cls == 0 and result.conf > 0.5:
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            confidences.append(float(result.conf))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    torso_regions = []

    if len(indices) > 0:
        for i in indices:
            box = boxes[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            torso_y_start = y + int(h / 4)
            torso_y_end = y + int(3 * h / 4)
            torso_x_start = x
            torso_x_end = x + w
            torso_region = frame[torso_y_start:torso_y_end, torso_x_start:torso_x_end]
            torso_regions.append(torso_region)

    torso_colors = [dominant_color(torso) for torso in torso_regions]

    for i, colors in enumerate(torso_colors):
        is_unique = True
        for j, unique_colors in enumerate(unique_persons):
            if compare_colors(colors, unique_colors):
                is_unique = False
                break
        if is_unique:
            unique_persons.append(colors)
            person_ids.append(len(unique_persons))

    if len(indices) > 0:
        for idx, i in enumerate(indices):
            box = boxes[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            person_id = person_ids[idx]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Person {person_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

