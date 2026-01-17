import cv2 as cv
import mediapipe as mp
import mss
import numpy as np
import time
import argparse

# Download a pose_landmarker_heavy.task model from MediaPipe and place it next to this script.

# Command-line arguments
parser = argparse.ArgumentParser(description="Scale the output window and configure algorithm parameters")
parser.add_argument("-s", "--scale", type=int, choices=[1, 2, 3, 4], default=1, help="Scale factor for the output window")
parser.add_argument("-n", "--num-poses", type=int, default=1, help="Maximum number of poses to detect")
parser.add_argument("-dc", "--detection-confidence", type=float, default=0.5, help="Minimum detection confidence")
parser.add_argument("-pc", "--presence-confidence", type=float, default=0.5, help="Minimum presence confidence")
parser.add_argument("-tc", "--tracking-confidence", type=float, default=0.5, help="Minimum tracking confidence")
parser.add_argument("-cl", "--clahe-clip-limit", type=float, default=15.0, help="CLAHE clip limit for contrast enhancement")
parser.add_argument("--no-clahe", action="store_true", help="Disable CLAHE contrast enhancement")
parser.add_argument("--wire-grid", action="store_true", help="Draw pose wire grid connections")
parser.add_argument("--segmentation-mask", action="store_true", help="Overlay segmentation mask")
parser.add_argument("-rx", "--roi-x", type=int, default=20, help="ROI top-left X coordinate (default: 0)")
parser.add_argument("-ry", "--roi-y", type=int, default=100, help="ROI top-left Y coordinate (default: 0)")
parser.add_argument("-rw", "--roi-width", type=int, default=1200, help="ROI width (default: full screen width)")
parser.add_argument("-rh", "--roi-height", type=int, default=512, help="ROI height (default: full screen height)")
args = parser.parse_args()
scale = args.scale
num_poses = args.num_poses
detection_confidence = args.detection_confidence
presence_confidence = args.presence_confidence
tracking_confidence = args.tracking_confidence
clahe_clip_limit = args.clahe_clip_limit
use_clahe = not args.no_clahe
use_wire_grid = args.wire_grid
use_segmentation_mask = args.segmentation_mask

# ROI parameters
sct_temp = mss.mss()
full_monitor = sct_temp.monitors[1]
sct_temp.close()

roi_x = args.roi_x if args.roi_x is not None else 0
roi_y = args.roi_y if args.roi_y is not None else 0
roi_width = args.roi_width if args.roi_width is not None else full_monitor["width"]
roi_height = args.roi_height if args.roi_height is not None else full_monitor["height"]

# Clamp ROI to screen boundaries
roi_x = max(0, min(roi_x, full_monitor["width"] - 1))
roi_y = max(0, min(roi_y, full_monitor["height"] - 1))
roi_width = min(roi_width, full_monitor["width"] - roi_x)
roi_height = min(roi_height, full_monitor["height"] - roi_y)

print(f"ROI: x={roi_x}, y={roi_y}, width={roi_width}, height={roi_height}")

# New MediaPipe Tasks API
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31),
    (24, 26), (26, 28), (28, 30), (28, 32)
]

# Store latest result from async callback
latest_result = None
latest_timestamp = 0


def result_callback(result, output_image, timestamp_ms):
    global latest_result, latest_timestamp
    latest_result = result
    latest_timestamp = timestamp_ms


def draw_pose_landmarks(image, pose_landmarks_list, draw_wire):
    """Draw pose landmarks on the image."""
    if not pose_landmarks_list:
        return image

    h, w = image.shape[:2]

    for pose_landmarks in pose_landmarks_list:
        for landmark in pose_landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            if not (np.isfinite(x) and np.isfinite(y) and 0 <= x < w and 0 <= y < h):
                continue
            cv.circle(image, (x, y), 2, (0, 255, 0), -1)

        if not draw_wire:
            continue

        for connection in POSE_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx >= len(pose_landmarks) or end_idx >= len(pose_landmarks):
                continue
            pt1 = pose_landmarks[start_idx]
            pt2 = pose_landmarks[end_idx]
            x1, y1 = int(pt1.x * w), int(pt1.y * h)
            x2, y2 = int(pt2.x * w), int(pt2.y * h)
            if (np.isfinite(x1) and np.isfinite(y1) and np.isfinite(x2) and np.isfinite(y2) and
                    0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h):
                cv.line(image, (x1, y1), (x2, y2), (0, 255, 0), 1)

    return image


def apply_segmentation_mask(image, mask_image, color=(0, 255, 0), alpha=0.6):
    mask = mask_image.numpy_view()
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    mask = np.clip(mask, 0.0, 1.0)

    h, w = image.shape[:2]
    if mask.shape[0] != h or mask.shape[1] != w:
        mask = cv.resize(mask, (w, h), interpolation=cv.INTER_LINEAR)

    frame_f = image.astype(np.float32)
    overlay = np.zeros_like(frame_f)
    overlay[:, :] = color
    mask_f = (mask * alpha)[:, :, None]
    blended = frame_f * (1.0 - mask_f) + overlay * mask_f
    return blended.astype(np.uint8)


def apply_clahe(image, clip_limit):
    """Apply CLAHE to improve contrast."""
    clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=(32, 18))
    lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
    lab_planes = cv.split(lab)
    lclahe = clahe.apply(lab_planes[0])
    lab = cv.merge([lclahe, lab_planes[1], lab_planes[2]])
    return cv.cvtColor(lab, cv.COLOR_LAB2BGR)


# Create PoseLandmarker with live stream mode
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='pose_landmarker_heavy.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_poses=num_poses,
    min_pose_detection_confidence=detection_confidence,
    min_pose_presence_confidence=presence_confidence,
    min_tracking_confidence=tracking_confidence,
    output_segmentation_masks=use_segmentation_mask,
    result_callback=result_callback
)

sct = mss.mss()
monitor = sct.monitors[1]

# Configure monitor with ROI
monitor = {
    "top": monitor["top"] + roi_y,
    "left": monitor["left"] + roi_x,
    "width": roi_width,
    "height": roi_height
}

start_time = time.time()
loop_time = time.time()

with PoseLandmarker.create_from_options(options) as landmarker:
    while True:
        # Capture screenshot
        screenshot = sct.grab(monitor)
        frame = np.array(screenshot)

        # Convert BGRA to BGR
        frame = cv.cvtColor(frame, cv.COLOR_BGRA2BGR)

        # Apply CLAHE for contrast enhancement
        if use_clahe:
            frame = apply_clahe(frame, clahe_clip_limit)

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Calculate timestamp in milliseconds
        timestamp_ms = int((time.time() - start_time) * 1000)

        # Run async detection
        landmarker.detect_async(mp_image, timestamp_ms)

        # Apply segmentation mask if enabled
        if use_segmentation_mask and latest_result and latest_result.segmentation_masks:
            frame = apply_segmentation_mask(frame, latest_result.segmentation_masks[0])

        # Draw latest results on frame
        if latest_result and latest_result.pose_landmarks:
            frame = draw_pose_landmarks(frame, latest_result.pose_landmarks, use_wire_grid)

        # Apply scaling
        frame = cv.resize(frame, None, fx=scale, fy=scale)

        cv.imshow('Pose Landmarks (Screen)', frame)

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
