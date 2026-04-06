import cv2
import dlib
import numpy as np
import pygame
from scipy.spatial import distance as dist
import time
import influxdb_client
from influxdb_client.client.write_api import ASYNCHRONOUS
from datetime import datetime

# Initialize pygame mixer
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound("audio/audio.mp3")
is_playing = False # To prevent the sound from overlapping itself

# 1. PATHS AND THRESHOLDS
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
EYE_AR_THRESH = 0.27   # Threshold for closed eyes (EAR)
MOUTH_AR_THRESH = 0.6  # Threshold for yawning (MAR)
BUFFER_FRAMES = 7      # Wait for 7 consecutive frames to trigger DROWSY

# Landmark indices for the 68-point model
L_EYE = list(range(36, 42))
R_EYE = list(range(42, 48))
MOUTH = [48, 54, 51, 57] # Left corner, Right corner, Top lip, Bottom lip

def get_ear(eye_points):
    # Vertical distances
    v1 = dist.euclidean(eye_points[1], eye_points[5])
    v2 = dist.euclidean(eye_points[2], eye_points[4])
    # Horizontal distance
    h = dist.euclidean(eye_points[0], eye_points[3])
    return (v1 + v2) / (2.0 * h)

def get_mar(mouth_points):
    # Horizontal distance (corners)
    h = dist.euclidean(mouth_points[0], mouth_points[1])
    # Vertical distance (center of lips)
    v = dist.euclidean(mouth_points[2], mouth_points[3])
    return v / h

# --- 2. INFLUXDB SETUP ---
INFLUX_URL = "http://192.168.1.X:8086" # REPLACE 'X' WITH YOUR LAPTOP IP
INFLUX_TOKEN = "YOUR_INFLUX_TOKEN"     # REPLACE WITH YOUR ACTUAL TOKEN
INFLUX_ORG = "Your_Org"                # REPLACE WITH YOUR ORG NAME
INFLUX_BUCKET = "DriverSafety"         # REPLACE WITH YOUR BUCKET NAME

print("[INFO] Connecting to InfluxDB Telemetry...")
client = influxdb_client.InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
write_api = client.write_api(write_options=ASYNCHRONOUS) # Critical for maintaining FPS
delete_api = client.delete_api()
# -------------------------

# 3. INITIALIZE DLIB
print("[INFO] Initializing Dlib Detectors...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

cap = cv2.VideoCapture(0)
drowsy_counter = 0

# Variables for FPS calculation
prev_frame_time = 0
new_frame_time = 0

while True:
    ret, frame = cap.read()
    if not ret: break

    # Calculate live FPS
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time > 0 else 0
    prev_frame_time = new_frame_time

    # BLACK AND WHITE CONVERSION
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    rects = detector(gray, 0)

    for rect in rects:
        # Get Landmarks
        shape = predictor(gray, rect)
        coords = np.zeros((68, 2), dtype="int")
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)

        # Calculate Aspect Ratios
        leftEAR = get_ear(coords[L_EYE])
        rightEAR = get_ear(coords[R_EYE])
        ear = (leftEAR + rightEAR) / 2.0
        mar = get_mar(coords[MOUTH])

        # BUFFER LOGIC
        if ear < EYE_AR_THRESH or mar > MOUTH_AR_THRESH:
            drowsy_counter += 1
        else:
            drowsy_counter = 0

        # Determine Status
        if drowsy_counter >= BUFFER_FRAMES:
            status = "DROWSY"
            color = (0, 0, 255) # Red for alert
            is_drowsy_int = 1   # For Grafana tracking
            if not is_playing:
                alarm_sound.play(loops=-1) 
                is_playing = True
        else:
            status = "NON-DROWSY"
            color = (0, 255, 0) # Green for safe
            is_drowsy_int = 0   # For Grafana tracking
            if is_playing:
                alarm_sound.stop()
                is_playing = False

        # --- 4. WRITE TELEMETRY TO INFLUXDB ---
        p = influxdb_client.Point("drowsiness_metrics") \
            .tag("driver_id", "Pranav") \
            .field("EAR", float(ear)) \
            .field("MAR", float(mar)) \
            .field("fps", float(fps)) \
            .field("is_drowsy", is_drowsy_int)

        write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=p)
        # --------------------------------------

        # DRAWING ON SCREEN
        cv2.putText(frame, f"STATUS: {status}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        cv2.putText(frame, f"EAR: {ear:.2f} | MAR: {mar:.2f} | FPS: {int(fps)}", (20, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Draw the points on the mouth and eyes
        for (x, y) in coords[36:68]:
            cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)

    cv2.imshow("Jetson Orin Nano - Geometric DDD", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

# --- 5. CLEANUP AND DATA DELETION ---
print("[INFO] Quitting and wiping local telemetry data from bucket...")

# Grab current time to tell InfluxDB up to when it should delete
end_time = datetime.utcnow()

# Delete all data in the 'drowsiness_metrics' measurement from 1970 to now
delete_api.delete(
    start="1970-01-01T00:00:00Z",
    stop=end_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
    predicate='_measurement="drowsiness_metrics"',
    bucket=INFLUX_BUCKET,
    org=INFLUX_ORG
)

cap.release()
pygame.mixer.quit()
client.close()
cv2.destroyAllWindows()
print("[INFO] Data wiped successfully. Systems offline.")
