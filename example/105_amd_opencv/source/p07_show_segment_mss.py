import argparse
import time

import cv2 as cv
import mediapipe as mp
import mss
import numpy as np

# Download a compatible image_segmenter .tflite model and place it next to this script.

parser = argparse.ArgumentParser(description="Screen capture image segmentation with MediaPipe Tasks")
parser.add_argument("-s", "--scale", type=int, choices=[1, 2, 3, 4], default=1, help="Scale factor for the output window")
parser.add_argument("-m", "--model", default="deeplab_v3.tflite", help="Path to the segmentation model (.tflite)")
parser.add_argument("--category-index", action="store_true", help="Show category index visualization and legend")
parser.add_argument("-cl", "--clahe-clip-limit", type=float, default=15.0, help="CLAHE clip limit for contrast enhancement")
parser.add_argument("--no-clahe", action="store_true", help="Disable CLAHE contrast enhancement")
parser.add_argument("-rx", "--roi-x", type=int, default=20, help="ROI top-left X coordinate (default: 0)")
parser.add_argument("-ry", "--roi-y", type=int, default=100, help="ROI top-left Y coordinate (default: 0)")
parser.add_argument("-rw", "--roi-width", type=int, default=1200, help="ROI width (default: full screen width)")
parser.add_argument("-rh", "--roi-height", type=int, default=512, help="ROI height (default: full screen height)")
args = parser.parse_args()

scale = args.scale
model_path = args.model
use_category_index = args.category_index
clahe_clip_limit = args.clahe_clip_limit
use_clahe = not args.no_clahe

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
print(f"Model: {model_path}")

BaseOptions = mp.tasks.BaseOptions
ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
VisionRunningMode = mp.tasks.vision.RunningMode

latest_result = None
latest_timestamp = 0


def result_callback(result, output_image, timestamp_ms):
    global latest_result, latest_timestamp
    latest_result = result
    latest_timestamp = timestamp_ms


def apply_clahe(image, clip_limit):
    """Apply CLAHE to improve contrast."""
    clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=(32, 18))
    lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
    lab_planes = cv.split(lab)
    lclahe = clahe.apply(lab_planes[0])
    lab = cv.merge([lclahe, lab_planes[1], lab_planes[2]])
    return cv.cvtColor(lab, cv.COLOR_LAB2BGR)


def build_palette(num_colors=256):
    rng = np.random.default_rng(7)
    palette = rng.integers(0, 255, size=(num_colors, 3), dtype=np.uint8)
    palette[0] = np.array([0, 0, 0], dtype=np.uint8)
    return palette


def to_category_indices(category_mask):
    mask = category_mask
    if mask.ndim == 3 and mask.shape[2] == 1:
        mask = mask[:, :, 0]
    if not np.issubdtype(mask.dtype, np.integer):
        max_val = float(np.max(mask)) if mask.size else 0.0
        if max_val <= 1.0:
            mask = mask * 255.0
        mask = np.rint(mask)
    return mask.astype(np.int32)


def colorize_category_mask(category_mask, palette):
    mask = to_category_indices(category_mask)
    mask = np.clip(mask, 0, len(palette) - 1)
    return palette[mask]


def draw_legend(image, palette, max_labels=12):
    h, w = image.shape[:2]
    padding = 6
    box_size = 14
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1

    for idx in range(min(max_labels, len(palette))):
        x = padding
        y = padding + idx * (box_size + 6) + box_size
        if y + padding > h:
            break
        color = tuple(int(c) for c in palette[idx])
        cv.rectangle(image, (x, y - box_size), (x + box_size, y), color, -1)
        cv.putText(image, str(idx), (x + box_size + 6, y - 2), font, font_scale, (255, 255, 255), thickness, cv.LINE_AA)


def overlay_segmentation(frame, category_mask, palette, alpha=0.6):
    colored_mask = colorize_category_mask(category_mask, palette)
    if colored_mask.ndim == 2:
        colored_mask = np.repeat(colored_mask[:, :, None], 3, axis=2)
    colored_mask = cv.resize(colored_mask, (frame.shape[1], frame.shape[0]), interpolation=cv.INTER_NEAREST)
    frame_f = frame.astype(np.float32)
    mask_f = colored_mask.astype(np.float32)
    blended = frame_f * (1.0 - alpha) + mask_f * alpha
    return blended.astype(np.uint8)


palette = build_palette()

options = ImageSegmenterOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    output_category_mask=True,
    result_callback=result_callback
)

sct = mss.mss()
monitor = sct.monitors[1]

monitor = {
    "top": monitor["top"] + roi_y,
    "left": monitor["left"] + roi_x,
    "width": roi_width,
    "height": roi_height,
}

start_time = time.time()
loop_time = time.time()

with ImageSegmenter.create_from_options(options) as segmenter:
    while True:
        screenshot = sct.grab(monitor)
        frame = np.array(screenshot)

        frame = cv.cvtColor(frame, cv.COLOR_BGRA2BGR)

        if use_clahe:
            frame = apply_clahe(frame, clahe_clip_limit)

        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        timestamp_ms = int((time.time() - start_time) * 1000)
        segmenter.segment_async(mp_image, timestamp_ms)

        if latest_result and latest_result.category_mask is not None:
            category_mask = latest_result.category_mask.numpy_view()
            frame = overlay_segmentation(frame, category_mask, palette)
            if use_category_index:
                draw_legend(frame, palette)
                mask_vis = colorize_category_mask(category_mask, palette)
                mask_vis = cv.resize(mask_vis, (frame.shape[1], frame.shape[0]), interpolation=cv.INTER_NEAREST)
                cv.imshow("Category Index", mask_vis)

        frame = cv.resize(frame, None, fx=scale, fy=scale)
        cv.imshow("Image Segmenter (Screen)", frame)

        delta = time.time() - loop_time
        target_period = (1 / 60.0) - 1e-4
        if delta < target_period:
            time.sleep(target_period - delta)
        fps = 1 / delta
        loop_time = time.time()

        if timestamp_ms % 2000 == 0:
            print(f"{time.time() - start_time:.2f} nil fps={fps:.2f}")

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

cv.destroyAllWindows()
