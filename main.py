
import cv2
import mediapipe as mp
import numpy as np
import time
import winsound
import csv
import os
import math
from datetime import datetime

# ==============================
# DRIVER DROWSINESS DETECTION
# ==============================

mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Mouth landmarks for yawning estimation
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 78
MOUTH_RIGHT = 308

# Settings
EAR_THRESHOLD = 0.20
CLOSED_SECONDS = 2.0
YAWN_THRESHOLD = 0.06
YAWN_SECONDS = 1.5

# State
eyes_closed_since = None
yawn_since = None
last_alert_time = 0
total_alerts = 0
alert_active = False
yawn_active = False

ALERT_FILE = "alert_history.csv"

# Create CSV history file
if not os.path.exists(ALERT_FILE):
    with open(ALERT_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Date", "Time", "Alert Type"])

def save_alert(alert_type):
    now = datetime.now()
    with open(ALERT_FILE, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M:%S"),
            alert_type
        ])

def calculate_ear(points):
    p1, p2, p3, p4, p5, p6 = points

    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)

    if horizontal == 0:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)

def get_landmark_points(face, indices, width, height):
    points = []

    for index in indices:
        landmark = face.landmark[index]
        points.append([
            landmark.x * width,
            landmark.y * height
        ])

    return np.array(points, dtype=np.float32)

def distance(point1, point2):
    return np.linalg.norm(
        np.array(point1) - np.array(point2)
    )

def get_point(face, index, width, height):
    landmark = face.landmark[index]
    return np.array([
        landmark.x * width,
        landmark.y * height
    ])

# Camera setup
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot access webcam. Check your camera.")
    raise SystemExit

try:
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while cap.isOpened():
            success, frame = cap.read()

            if not success:
                print("Cannot read webcam frame")
                break

            frame = cv2.flip(frame, 1)
            height, width, _ = frame.shape

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            status = "FACE NOT DETECTED"
            color = (0, 0, 255)
            ear = 0.0
            mouth_ratio = 0.0

            if results.multi_face_landmarks:
                face = results.multi_face_landmarks[0]

                # ==============================
                # 1. EYE TRACKING
                # ==============================

                left_eye = get_landmark_points(
                    face, LEFT_EYE, width, height
                )

                right_eye = get_landmark_points(
                    face, RIGHT_EYE, width, height
                )

                ear = (
                    calculate_ear(left_eye) +
                    calculate_ear(right_eye)
                ) / 2.0

                for points in [left_eye, right_eye]:
                    for x, y in points.astype(int):
                        cv2.circle(
                            frame, (x, y), 2,
                            (0, 255, 0), -1
                        )

                # ==============================
                # 2. YAWNING ESTIMATION
                # ==============================

                mouth_top = get_point(
                    face, MOUTH_TOP, width, height
                )
                mouth_bottom = get_point(
                    face, MOUTH_BOTTOM, width, height
                )
                mouth_left = get_point(
                    face, MOUTH_LEFT, width, height
                )
                mouth_right = get_point(
                    face, MOUTH_RIGHT, width, height
                )

                mouth_width = distance(
                    mouth_left, mouth_right
                )
                mouth_height = distance(
                    mouth_top, mouth_bottom
                )

                if mouth_width > 0:
                    mouth_ratio = mouth_height / mouth_width

                # ==============================
                # 3. HEAD POSE ESTIMATION
                # ==============================

                nose = get_point(face, 1, width, height)
                chin = get_point(face, 152, width, height)
                forehead = get_point(face, 10, width, height)

                face_height = distance(forehead, chin)

                # Simple vertical head movement estimate.
                # This is not a calibrated head-pose angle.
                nose_position = (
                    (nose[1] - forehead[1]) / face_height
                    if face_height > 0 else 0.5
                )

                head_status = "Head Position: Normal"

                if nose_position > 0.55:
                    head_status = "Head Tilted Down"

                # ==============================
                # 4. DROWSINESS TIMER
                # ==============================

                if ear < EAR_THRESHOLD:
                    if eyes_closed_since is None:
                        eyes_closed_since = time.time()

                    closed_duration = (
                        time.time() - eyes_closed_since
                    )

                    if closed_duration >= CLOSED_SECONDS:
                        status = "DROWSINESS ALERT!"
                        color = (0, 0, 255)

                        if not alert_active:
                            total_alerts += 1
                            save_alert("Eyes closed")
                            alert_active = True
                    else:
                        status = "Eyes Closed"
                        color = (0, 165, 255)

                else:
                    eyes_closed_since = None
                    alert_active = False
                    status = "NORMAL"
                    color = (0, 255, 0)

                # ==============================
                # 5. YAWNING TIMER
                # ==============================

                if mouth_ratio > YAWN_THRESHOLD:
                    if yawn_since is None:
                        yawn_since = time.time()

                    if time.time() - yawn_since >= YAWN_SECONDS:
                        status = "YAWNING DETECTED"
                        color = (0, 165, 255)

                        if not yawn_active:
                            total_alerts += 1
                            save_alert("Possible yawn")
                            yawn_active = True
                else:
                    yawn_since = None
                    yawn_active = False

                # ==============================
                # 6. DISPLAY VALUES
                # ==============================

                cv2.putText(
                    frame, f"EAR: {ear:.2f}",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2
                )

                cv2.putText(
                    frame, f"Mouth Ratio: {mouth_ratio:.2f}",
                    (30, 75), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 0), 2
                )

                cv2.putText(
                    frame, head_status,
                    (30, 110), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2
                )

            else:
                eyes_closed_since = None
                alert_active = False
                yawn_since = None
                yawn_active = False

            # ==============================
            # 7. ALERT SOUND
            # ==============================

            if status in [
                "DROWSINESS ALERT!",
                "YAWNING DETECTED"
            ]:
                if time.time() - last_alert_time >= 2:
                    try:
                        winsound.Beep(1000, 500)
                    except RuntimeError:
                        print("Warning alert!")

                    last_alert_time = time.time()

            # ==============================
            # 8. STATUS AND COUNTER
            # ==============================

            cv2.putText(
                frame, f"STATUS: {status}",
                (30, 150), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, color, 2
            )

            cv2.putText(
                frame, f"TOTAL ALERTS: {total_alerts}",
                (30, 185), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 0), 2
            )

            cv2.putText(
                frame, "Press Q to Quit",
                (30, height - 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2
            )

            cv2.imshow(
                "Advanced Driver Drowsiness Detection",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    cap.release()
    cv2.destroyAllWindows()

print(f"Session finished. Total alerts: {total_alerts}")
print(f"Alert history saved in: {ALERT_FILE}")
