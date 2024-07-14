import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from scipy.spatial import distance as dist

# Initialize MediaPipe Face Detection and Face Mesh
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=5, min_detection_confidence=0.5)

# Tracker type
tracker_type = "CSRT"

# Global face ID management
global_face_id = 0
multi_tracker1 = cv2.legacy.MultiTracker_create()
multi_tracker2 = cv2.legacy.MultiTracker_create()

# Queue to store landmarks for multiple faces
landmarks_queue = {i: deque(maxlen=10) for i in range(100)}  # Assuming a maximum of 100 faces

# Function to detect faces and draw bounding boxes
def detect_faces_and_draw_boxes(frame, face_id, trackers, multi_tracker):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)
    new_trackers = []
    ids_detected = []

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            cx, cy = x + w // 2, y + h // 2

            # Find if this face matches a previously detected face
            min_dist = float('inf')
            min_id = -1
            for tracker in trackers:
                tracker_id, (tx, ty, tw, th), (tcx, tcy) = tracker
                d = dist.euclidean((cx, cy), (tcx, tcy))
                if d < min_dist:
                    min_dist = d
                    min_id = tracker_id

            if min_dist < 50:  # Threshold to determine if it's the same face
                ids_detected.append(min_id)
                new_trackers.append((min_id, (x, y, w, h), (cx, cy)))
            else:
                ids_detected.append(face_id)
                new_trackers.append((face_id, (x, y, w, h), (cx, cy)))
                face_id += 1

            # Initialize tracker for the new face
            tracker = cv2.legacy.TrackerCSRT_create()
            multi_tracker.add(tracker, frame, (x, y, w, h))

            # Draw a green rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Put a unique ID for each face
            cv2.putText(frame, f'ID: {ids_detected[-1]}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame, face_id, new_trackers

# Function to detect landmarks and compare
def detect_landmarks_and_compare(frame, landmarks_queue):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    current_landmarks = {}

    if results.multi_face_landmarks:
        for i, face_landmarks in enumerate(results.multi_face_landmarks):
            landmarks = [(int(point.x * frame.shape[1]), int(point.y * frame.shape[0])) for point in face_landmarks.landmark]
            current_landmarks[i] = landmarks

            for (x, y) in landmarks:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

    similarity_ratios = {}
    if current_landmarks:
        for face_id, landmarks in current_landmarks.items():
            similarities = []
            if face_id in landmarks_queue:
                for stored_landmarks in landmarks_queue[face_id]:
                    similarity = np.mean([dist.euclidean(l1, l2) for l1, l2 in zip(stored_landmarks, landmarks)])
                    similarities.append(similarity)
                similarity_ratios[face_id] = np.mean(similarities)

    return frame, similarity_ratios

# Open two webcams
cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)
frames_with_faces = []
trackers1 = []
trackers2 = []

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if not ret1 or not ret2:
        break

    if len(trackers1) == 0:
        frame_with_boxes1, global_face_id, trackers1 = detect_faces_and_draw_boxes(frame1, global_face_id, trackers1, multi_tracker1)
    else:
        success1, boxes1 = multi_tracker1.update(frame1)
        for i, new_box in enumerate(boxes1):
            x, y, w, h = map(int, new_box)
            cx, cy = x + w // 2, y + h // 2
            trackers1[i] = (trackers1[i][0], (x, y, w, h), (cx, cy))
            # Draw a green rectangle around the face
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Put a unique ID for each face
            cv2.putText(frame1, f'ID: {trackers1[i][0]}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    if len(trackers2) == 0:
        frame_with_boxes2, global_face_id, trackers2 = detect_faces_and_draw_boxes(frame2, global_face_id, trackers2, multi_tracker2)
    else:
        success2, boxes2 = multi_tracker2.update(frame2)
        for i, new_box in enumerate(boxes2):
            x, y, w, h = map(int, new_box)
            cx, cy = x + w // 2, y + h // 2
            trackers2[i] = (trackers2[i][0], (x, y, w, h), (cx, cy))
            # Draw a green rectangle around the face
            cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Put a unique ID for each face
            cv2.putText(frame2, f'ID: {trackers2[i][0]}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Detect landmarks and compare
    frame_with_landmarks1, similarity_ratios1 = detect_landmarks_and_compare(frame1, landmarks_queue)
    frame_with_landmarks2, similarity_ratios2 = detect_landmarks_and_compare(frame2, landmarks_queue)
    
    # Display similarity ratios
    if similarity_ratios1:
        for i, similarity in similarity_ratios1.items():
            cv2.putText(frame_with_landmarks1, f'Similarity ID {i}: {similarity:.2f}', (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    if similarity_ratios2:
        for i, similarity in similarity_ratios2.items():
            cv2.putText(frame_with_landmarks2, f'Similarity ID {i}: {similarity:.2f}', (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    # Save the frame with detected faces to the list
    frames_with_faces.append((frame_with_landmarks1, frame_with_landmarks2))

    # Display the resulting frames
    cv2.imshow("Frame 1", frame_with_landmarks1)
    cv2.imshow("Frame 2", frame_with_landmarks2)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        # Save the current frame as a screenshot
        cv2.imwrite('screenshot1.png', frame_with_landmarks1)
        cv2.imwrite('screenshot2.png', frame_with_landmarks2)
        # Extract and store landmarks of the current frame
        results1 = face_mesh.process(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
        results2 = face_mesh.process(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
        if results1.multi_face_landmarks:
            for i, face_landmarks in enumerate(results1.multi_face_landmarks):
                landmarks = [(int(point.x * frame1.shape[1]), int(point.y * frame1.shape[0])) for point in face_landmarks.landmark]
                if i in landmarks_queue:
                    landmarks_queue[i].append(landmarks)
                else:
                    landmarks_queue[i] = deque([landmarks], maxlen=10)
        if results2.multi_face_landmarks:
            for i, face_landmarks in enumerate(results2.multi_face_landmarks):
                landmarks = [(int(point.x * frame2.shape[1]), int(point.y * frame2.shape[0])) for point in face_landmarks.landmark]
                if i in landmarks_queue:
                    landmarks_queue[i].append(landmarks)
                else:
                    landmarks_queue[i] = deque([landmarks], maxlen=10)

# Convert the list of frames to a NumPy array
frames_array = np.array(frames_with_faces)

# Release the capture and close windows
cap1.release()
cap2.release()
cv2.destroyAllWindows()

# Save the numpy array to a file
np.save('frames_with_faces.npy', frames_array)
