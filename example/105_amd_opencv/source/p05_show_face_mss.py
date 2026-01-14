import cv2 as cv
import mediapipe as mp
import mss
import numpy as np
import time
import argparse

# https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task

# Command-line arguments
parser = argparse.ArgumentParser(description="Scale the output window and configure algorithm parameters")
parser.add_argument("-s", "--scale", type=int, choices=[1, 2, 3, 4], default=1, help="Scale factor for the output window")
parser.add_argument("-n", "--num-faces", type=int, default=1, help="Maximum number of faces to detect")
parser.add_argument("-dc", "--detection-confidence", type=float, default=0.15, help="Minimum detection confidence")
parser.add_argument("-pc", "--presence-confidence", type=float, default=0.15, help="Minimum presence confidence")
parser.add_argument("-tc", "--tracking-confidence", type=float, default=0.15, help="Minimum tracking confidence")
parser.add_argument("-cl", "--clahe-clip-limit", type=float, default=15.0, help="CLAHE clip limit for contrast enhancement")
args = parser.parse_args()
scale = args.scale
num_faces = args.num_faces
detection_confidence = args.detection_confidence
presence_confidence = args.presence_confidence
tracking_confidence = args.tracking_confidence
clahe_clip_limit = args.clahe_clip_limit

# New MediaPipe Tasks API
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Store latest result from async callback
latest_result = None
latest_timestamp = 0

def result_callback(result, output_image, timestamp_ms):
    global latest_result, latest_timestamp
    latest_result = result
    latest_timestamp = timestamp_ms

def draw_landmarks(image, face_landmarks_list):
    """Draw face landmarks on the image."""
    if not face_landmarks_list:
        return image

    h, w = image.shape[:2]

    for face_landmarks in face_landmarks_list:
        # Draw each landmark as a small circle
        for landmark in face_landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv.circle(image, (x, y), 1, (0, 255, 0), -1)

        # Lips outer
        lip_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 61]
        for i in range(len(lip_indices) - 1):
            pt1 = face_landmarks[lip_indices[i]]
            pt2 = face_landmarks[lip_indices[i + 1]]
            cv.line(image,
                   (int(pt1.x * w), int(pt1.y * h)),
                   (int(pt2.x * w), int(pt2.y * h)),
                   (0, 255, 0), 1)

        # Left eye
        left_eye = [33, 7, 163, 144, 145, 153, 154, 155, 133, 33]
        for i in range(len(left_eye) - 1):
            pt1 = face_landmarks[left_eye[i]]
            pt2 = face_landmarks[left_eye[i + 1]]
            cv.line(image,
                   (int(pt1.x * w), int(pt1.y * h)),
                   (int(pt2.x * w), int(pt2.y * h)),
                   (0, 255, 0), 1)

        # Right eye
        right_eye = [362, 382, 381, 380, 374, 373, 390, 249, 263, 362]
        for i in range(len(right_eye) - 1):
            pt1 = face_landmarks[right_eye[i]]
            pt2 = face_landmarks[right_eye[i + 1]]
            cv.line(image,
                   (int(pt1.x * w), int(pt1.y * h)),
                   (int(pt2.x * w), int(pt2.y * h)),
                   (0, 255, 0), 1)

    return image

def apply_clahe(image, clip_limit):
    """Apply CLAHE to improve contrast."""
    clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=(32, 18))
    lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
    lab_planes = cv.split(lab)
    lclahe = clahe.apply(lab_planes[0])
    lab = cv.merge([lclahe, lab_planes[1], lab_planes[2]])
    return cv.cvtColor(lab, cv.COLOR_LAB2BGR)

# Create FaceLandmarker with live stream mode
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='face_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_faces=num_faces,
    min_face_detection_confidence=detection_confidence,
    min_face_presence_confidence=presence_confidence,
    min_tracking_confidence=tracking_confidence,
    result_callback=result_callback
)

sct = mss.mss()
monitor = sct.monitors[1]
start_time = time.time()
loop_time = time.time()

with FaceLandmarker.create_from_options(options) as landmarker:
    while True:
        # Capture screenshot
        screenshot = sct.grab(monitor)
        frame = np.array(screenshot)

        # Convert BGRA to BGR
        frame = cv.cvtColor(frame, cv.COLOR_BGRA2BGR)

        # Apply CLAHE for contrast enhancement
        frame = apply_clahe(frame, clahe_clip_limit)

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Calculate timestamp in milliseconds
        timestamp_ms = int((time.time() - start_time) * 1000)

        # Run async detection
        landmarker.detect_async(mp_image, timestamp_ms)

        # Draw latest results on frame
        if latest_result and latest_result.face_landmarks:
            frame = draw_landmarks(frame, latest_result.face_landmarks)

        # Apply scaling
        frame = cv.resize(frame, None, fx=scale, fy=scale)

        cv.imshow('Face Landmarks (Screen)', frame)

        # FPS calculation
        delta = time.time() - loop_time
        target_period = (1 / 60.0) - 1e-4
        if delta < target_period:
            time.sleep(target_period - delta)
        fps = 1 / delta
        loop_time = time.time()

        if timestamp_ms % 2000 == 0:
            print(f"{time.time() - start_time:.2f} nil fps={fps:.2f}")

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

cv.destroyAllWindows()
