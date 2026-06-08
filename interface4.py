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
is_playing = False

# 1. PATHS AND THRESHOLDS
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
EYE_AR_THRESH = 0.26
MOUTH_AR_THRESH = 0.68
BUFFER_FRAMES = 7

# Landmark indices
L_EYE = list(range(36, 42))
R_EYE = list(range(42, 48))
MOUTH = [48, 54, 51, 57]

# Roll calibration globals
ROLL_CALIBRATED = False
ROLL_OFFSET = 0.0
ROLL_SAMPLES = []

def get_ear(eye_points):
    v1 = dist.euclidean(eye_points[1], eye_points[5])
    v2 = dist.euclidean(eye_points[2], eye_points[4])
    h = dist.euclidean(eye_points[0], eye_points[3])
    return (v1 + v2) / (2.0 * h)

def get_mar(mouth_points):
    h = dist.euclidean(mouth_points[0], mouth_points[1])
    v = dist.euclidean(mouth_points[2], mouth_points[3])
    return v / h

# Sunglasses detection
def detect_sunglasses(frame, coords):
    left_eye = coords[36:42]
    right_eye = coords[42:48]

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [left_eye], 255)
    cv2.fillPoly(mask, [right_eye], 255)

    eye_region = cv2.bitwise_and(frame, frame, mask=mask)
    gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)

    valid_pixels = gray[gray > 0]
    if len(valid_pixels) == 0:
        return False

    brightness = np.mean(valid_pixels)
    return brightness < 40

# Head pose estimation
def get_head_pose(coords, frame):
    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0)
    ])

    image_points = np.array([
        coords[30],
        coords[8],
        coords[36],
        coords[45],
        coords[48],
        coords[54]
    ], dtype="double")

    h, w = frame.shape[:2]
    focal_length = w
    center = (w / 2, h / 2)

    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))

    success, rvec, tvec = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs
    )

    rmat, _ = cv2.Rodrigues(rvec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

    pitch = float(angles[0] * 180)
    yaw = float(angles[1] * 180)
    roll = float(angles[2] * 180)

    return pitch, yaw, roll

# InfluxDB setup
INFLUX_URL = "http://192.168.1.X:8086"
INFLUX_TOKEN = "YOUR_INFLUX_TOKEN"
INFLUX_ORG = "Your_Org"
INFLUX_BUCKET = "DriverSafety"

print("[INFO] Connecting to InfluxDB Telemetry...")
client = influxdb_client.InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
write_api = client.write_api(write_options=ASYNCHRONOUS)
delete_api = client.delete_api()

# Initialize Dlib
print("[INFO] Initializing Dlib Detectors...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev_frame_time = 0
ear = 0.0
mar = 0.0
fps = 0.0
status = "NON-DROWSY"
drowsy_counter = 0

frame_count = 0
cached_rects = []
PROCESS_EVERY_N_FRAMES = 3

# Graph buffers
GRAPH_W = 260
GRAPH_H = 50
ear_hist = np.zeros(GRAPH_W, dtype=np.float32)
mar_hist = np.zeros(GRAPH_W, dtype=np.float32)
fps_hist = np.zeros(GRAPH_W, dtype=np.float32)

def draw_graph(history, max_val, color):
    graph = np.zeros((GRAPH_H, GRAPH_W, 3), dtype=np.uint8)
    scaled = (history / max_val) * (GRAPH_H - 2)
    scaled = np.clip(scaled, 0, GRAPH_H - 2).astype(np.int32)
    for x in range(1, GRAPH_W):
        y1 = GRAPH_H - 1 - scaled[x - 1]
        y2 = GRAPH_H - 1 - scaled[x]
        cv2.line(graph, (x - 1, y1), (x, y2), color, 1)
    return graph

while True:
    ret, frame = cap.read()
    if not ret:
        break

    new_frame_time = time.time()
    if prev_frame_time > 0:
        fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if frame_count % PROCESS_EVERY_N_FRAMES == 0:
        rects = detector(gray, 0)
        cached_rects = rects if len(rects) > 0 else []
    else:
        rects = cached_rects
    frame_count += 1

    is_drowsy_int = 0
    wearing_sunglasses = False
    slouching = False

    for rect in rects:
        shape = predictor(gray, rect)
        coords = np.zeros((68, 2), dtype="int")
        for i in range(68):
            coords[i] = (shape.part(i).x, shape.part(i).y)

        leftEAR = get_ear(coords[L_EYE])
        rightEAR = get_ear(coords[R_EYE])
        ear = (leftEAR + rightEAR) / 2.0
        mar = get_mar(coords[MOUTH])

        wearing_sunglasses = detect_sunglasses(frame, coords)
        pitch, yaw, roll = get_head_pose(coords, frame)

        # Roll calibration
        if not ROLL_CALIBRATED:
            ROLL_SAMPLES.append(roll)
            if len(ROLL_SAMPLES) >= 30:
                ROLL_OFFSET = np.mean(ROLL_SAMPLES)
                ROLL_CALIBRATED = True
                print(f"[CALIBRATION] Roll offset set to: {ROLL_OFFSET:.2f}")

        corrected_roll = roll - ROLL_OFFSET

        # Slouch detection
        slouching = pitch < 30 or abs(corrected_roll) < 25

        if wearing_sunglasses:
            if slouching:
                status = "DROWSY"
                is_drowsy_int = 1
                if not is_playing:
                    alarm_sound.play(loops=-1)
                    is_playing = True
            else:
                status = "NON-DROWSY"
                is_drowsy_int = 0
                if is_playing:
                    alarm_sound.stop()
                    is_playing = False
        else:
            if ear < EYE_AR_THRESH or mar > MOUTH_AR_THRESH:
                drowsy_counter += 1
            else:
                drowsy_counter = 0

            if drowsy_counter >= BUFFER_FRAMES or slouching:
                status = "DROWSY"
                is_drowsy_int = 1
                if not is_playing:
                    alarm_sound.play(loops=-1)
                    is_playing = True
            else:
                status = "NON-DROWSY"
                is_drowsy_int = 0
                if is_playing:
                    alarm_sound.stop()
                    is_playing = False

        p = influxdb_client.Point("drowsiness_metrics") \
            .tag("driver_id", "Pranav") \
            .field("EAR", float(ear)) \
            .field("MAR", float(mar)) \
            .field("fps", float(fps)) \
            .field("is_drowsy", is_drowsy_int) \
            .field("pitch", float(pitch)) \
            .field("yaw", float(yaw)) \
            .field("roll", float(roll)) \
            .field("sunglasses", int(wearing_sunglasses)) \
            .field("slouching", int(slouching))
        write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=p)

        for (pt_x, pt_y) in coords[36:68]:
            cv2.circle(frame, (pt_x, pt_y), 2, (0, 255, 255), -1)

    ear_hist[:-1] = ear_hist[1:]
    ear_hist[-1] = ear
    mar_hist[:-1] = mar_hist[1:]
    mar_hist[-1] = mar
    fps_hist[:-1] = fps_hist[1:]
    fps_hist[-1] = fps

    h_frame, w_frame, _ = frame.shape
    sidebar = np.zeros((h_frame, 320, 3), dtype=np.uint8)

    cv2.putText(sidebar, "EDGE-AI TELEMETRY", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.line(sidebar, (20, 50), (300, 50), (255, 255, 255), 1)

    sidebar[60:60+GRAPH_H, 20:20+GRAPH_W] = draw_graph(ear_hist, 0.40, (0, 255, 255))
    sidebar[120:120+GRAPH_H, 20:20+GRAPH_W] = draw_graph(mar_hist, 1.0, (255, 0, 255))
    sidebar[180:180+GRAPH_H, 20:20+GRAPH_W] = draw_graph(fps_hist, 60, (0, 255, 0))

    cv2.putText(sidebar, "EAR", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(sidebar, "MAR", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(sidebar, "FPS", (20, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    fps_color = (0, 255, 0) if fps > 25 else (0, 0, 255)
    cv2.putText(sidebar, f"SYSTEM FPS: {int(fps)}", (20, 260),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)

    cv2.putText(sidebar, f"EAR (Target > {EYE_AR_THRESH}): {ear:.2f}", (20, 280),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    ear_width = min(200, int((ear / 0.40) * 200))
    ear_color = (0, 255, 0) if ear > EYE_AR_THRESH else (0, 0, 255)
    cv2.rectangle(sidebar, (20, 295), (220, 315), (100, 100, 100), 2)
    cv2.rectangle(sidebar, (20, 295), (20 + ear_width, 315), ear_color, -1)

    cv2.putText(sidebar, f"MAR (Target < {MOUTH_AR_THRESH}): {mar:.2f}", (20, 330),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    mar_width = min(200, int((mar / 1.0) * 200))
    mar_color = (0, 255, 0) if mar < MOUTH_AR_THRESH else (0, 0, 255)
    cv2.rectangle(sidebar, (20, 345), (220, 365), (100, 100, 100), 2)
    cv2.rectangle(sidebar, (20, 345), (20 + mar_width, 365), mar_color, -1)

    status_bg = (0, 255, 0) if status == "NON-DROWSY" else (0, 0, 255)
    cv2.rectangle(sidebar, (20, 380), (300, 460), status_bg, -1)
    text_x = 70 if status == "NON-DROWSY" else 105
    cv2.putText(sidebar, status, (text_x, 430),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3)

    cv2.putText(sidebar, f"SUNGLASSES: {'YES' if wearing_sunglasses else 'NO'}", (20, 480),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(sidebar, f"SLOUCHING: {'YES' if slouching else 'NO'}", (20, 500),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    final_hud = np.hstack((frame, sidebar))
    cv2.imshow("Jetson Orin Nano - Interface 4 (Sunglasses + Slouch)", final_hud)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("[INFO] Quitting and wiping local telemetry data...")
try:
    delete_api.delete(
        start="1970-01-01T00:00:00Z",
        stop=datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        predicate='_measurement="drowsiness_metrics"',
        bucket=INFLUX_BUCKET,
        org=INFLUX_ORG
    )
except:
    pass

cap.release()
pygame.mixer.quit()
client.close()
cv2.destroyAllWindows()
print("[INFO] Systems offline.")
