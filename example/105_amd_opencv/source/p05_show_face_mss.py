import cv2 as cv
import mediapipe as mp
import mss
import numpy as np
import time

# https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task

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

# Create FaceLandmarker with live stream mode
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='face_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    result_callback=result_callback
)

sct = mss.mss()
monitor = sct.monitors[1]
start_time = time.time()

with FaceLandmarker.create_from_options(options) as landmarker:
    while True:
        # Capture screenshot
        screenshot = sct.grab(monitor)
        frame = np.array(screenshot)

        # Convert BGRA to BGR
        frame = cv.cvtColor(frame, cv.COLOR_BGRA2BGR)

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

        # Resize for display
        display = cv.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
        cv.imshow('Face Landmarks (Screen)', display)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

cv.destroyAllWindows()

