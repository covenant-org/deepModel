import cv2
import numpy as np
import time
import torch
from transformers import AutoModelForDepthEstimation, AutoImageProcessor
from PIL import Image
import math
import os

# --- User Configuration ---
VIDEO_PATH = 'sample.mov' # Your video file
DISPLAY_WIDTH = 1280 # Max width for the display window

# --- Performance Tuning ---
PROCESS_HEIGHT = 360 # Lower = faster depth. Higher = potentially better depth.
TARGET_FPS = 2.0     # Target processing FPS for depth and halo updates
DEPTH_OVERLAY_ALPHA = 0.6 # <<< Increased alpha for more dominant depth map

# --- Obstacle Halo Configuration ---
# Proximity Thresholds (Sorted: Closest first)
CRITICAL_THRESHOLD = 1.5 # meters <<< Start of Red zone
ALERT_THRESHOLD = 3.0    # meters <<< Start of Orange zone
WARN_THRESHOLD = 5.0     # meters <<< Start of Yellow zone
# Halo Geometry & Appearance
HALO_NUM_SEGMENTS = 360   # Keep high number for smooth gradient appearance
HALO_X_RADIUS_FACTOR = 0.40 # Horizontal radius relative to frame width
HALO_Y_RADIUS_FACTOR = 0.30 # Vertical radius relative to frame height (elliptical)
HALO_THICKNESS = 15       # Halo thickness
# Base Colors (Simple BGR Tuples for OpenCV)
BGR_INACTIVE = np.array([100, 100, 100]) # Gray base (use numpy array for interpolation)
BGR_WARN     = np.array([0, 255, 255])   # Yellow (BGR)
BGR_ALERT    = np.array([0, 165, 255])   # Orange (BGR)
BGR_CRITICAL = np.array([0, 0, 255])     # Red (BGR)

# --- Helper Functions for Color Interpolation ---
def interpolate_color(color1, color2, factor):
    """Linearly interpolates between two BGR colors."""
    factor = np.clip(factor, 0.0, 1.0)
    result = color1 * (1 - factor) + color2 * factor
    # Ensure result is standard python ints tuple for OpenCV
    return tuple(int(c) for c in result) # Explicitly cast to int

def get_halo_color_for_depth(depth, inactive_clr, warn_clr, alert_clr, critical_clr, warn_thresh, alert_thresh, critical_thresh):
    """Calculates the BGR color based on depth thresholds and interpolation."""
    if depth < critical_thresh:
        return tuple(int(c) for c in critical_clr) # Explicitly cast to int tuple
    elif depth < alert_thresh:
        factor = (depth - critical_thresh) / (alert_thresh - critical_thresh)
        return interpolate_color(critical_clr, alert_clr, factor)
    elif depth < warn_thresh:
        factor = (depth - alert_thresh) / (warn_thresh - alert_thresh)
        return interpolate_color(alert_clr, warn_clr, factor)
    else:
        return tuple(int(c) for c in inactive_clr) # Explicitly cast to int tuple

# --- 1. Initialize Depth Estimation Model ---
print("Loading depth estimation model..."); MODEL_NAME = "LiheYoung/depth-anything-small-hf"
DEVICE = torch.device("cpu"); print(f"Using device: {DEVICE}")
try:
    image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    depth_model = AutoModelForDepthEstimation.from_pretrained(MODEL_NAME).to(DEVICE)
    print("Depth model loaded successfully.")
except Exception as e: print(f"Error loading depth model: {e}"); exit()

# --- 2. Initialize Video Capture ---
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened(): print(f"Error: Could not open video file: {VIDEO_PATH}"); exit()
actual_frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Actual video resolution: {actual_frame_width}x{actual_frame_height}")

# --- 3. Halo Drawing Function (Ensure Types) ---
def draw_obstacle_halo(frame, segment_colors, num_segments, radius_x, radius_y, thickness):
    """Draws the obstacle halo using a list of calculated segment colors."""
    h, w = frame.shape[:2]
    # Ensure center and axes are standard Python integers
    center_x = int(w // 2); center_y = int(h // 2)
    axes = (int(radius_x), int(radius_y))
    segment_angle = 360.0 / num_segments
    start_angle_offset = -90.0 # Use float for angles

    for i in range(num_segments):
        # Ensure color is a standard tuple of 3 Python integers
        color = tuple(int(c) for c in segment_colors[i])
        # Ensure angles are floats
        segment_start_angle = float(start_angle_offset + i * segment_angle)
        segment_end_angle = float(segment_start_angle + segment_angle + 0.5) # Add overlap
        try:
            cv2.ellipse(frame,
                        (center_x, center_y),
                        axes,
                        0.0, # Angle of ellipse rotation (float)
                        segment_start_angle,
                        segment_end_angle,
                        color, # Explicitly confirmed tuple of ints
                        int(thickness)) # Ensure thickness is int
        except Exception as e:
            print(f"Error drawing halo segment {i} (Color: {color}, Start: {segment_start_angle}, End: {segment_end_angle}): {e}")


# --- 4. Main Processing Loop ---
frame_count = 0; total_process_time = 0; processed_frame_counter = 0
TARGET_PROCESS_INTERVAL = 1.0 / TARGET_FPS; last_process_time = 0.0
last_halo_segment_colors = [tuple(int(c) for c in BGR_INACTIVE)] * HALO_NUM_SEGMENTS # Initialize with int tuples
last_depth_map_color = None

while True: # Main loop starts here
    ret, frame = cap.read()
    if not ret: print("End of video."); break
    frame_count += 1
    current_time = time.time()

    # --- Resize original frame for display ---
    aspect_ratio = actual_frame_width / actual_frame_height
    process_width = int(PROCESS_HEIGHT * aspect_ratio)
    frame_display = cv2.resize(frame, (process_width, PROCESS_HEIGHT))

    # --- Timed Processing Block (Depth + Halo Status) ---
    process_this_frame = (current_time - last_process_time >= TARGET_PROCESS_INTERVAL)

    if process_this_frame:
        last_process_time = current_time; processed_frame_counter += 1
        start_time_process = time.time()
        current_depth_map = None
        try:
            image_pil = Image.fromarray(cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB))
            inputs = image_processor(images=image_pil, return_tensors="pt").to(DEVICE)
            with torch.no_grad(): outputs = depth_model(**inputs)
            prediction = torch.nn.functional.interpolate(outputs.predicted_depth.unsqueeze(1), size=(PROCESS_HEIGHT, process_width), mode="bicubic", align_corners=False).squeeze()
            current_depth_map = prediction.cpu().numpy()
            depth_normalized = (current_depth_map - current_depth_map.min()) / (current_depth_map.max() - current_depth_map.min() + 1e-6)
            depth_display_color = cv2.applyColorMap((depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
            last_depth_map_color = depth_display_color
        except Exception as e: print(f"Frame {frame_count}: Error depth: {e}")

        # --- Halo Obstacle Check and Color Calculation ---
        temp_halo_colors = [tuple(int(c) for c in BGR_INACTIVE)] * HALO_NUM_SEGMENTS # Default for this run
        if current_depth_map is not None:
            depth_h, depth_w = current_depth_map.shape
            segment_width_px = max(1, depth_w // HALO_NUM_SEGMENTS)
            roi_y_center = depth_h // 2; roi_y_half_height = int(depth_h * 0.25)
            roi_y_start = max(0, roi_y_center - roi_y_half_height); roi_y_end = min(depth_h, roi_y_center + roi_y_half_height)
            for i in range(HALO_NUM_SEGMENTS):
                roi_x_start = int(i * depth_w / HALO_NUM_SEGMENTS)
                roi_x_end = int((i + 1) * depth_w / HALO_NUM_SEGMENTS)
                roi_x_end = max(roi_x_start + 1, roi_x_end)
                if roi_y_start < roi_y_end and roi_x_start < roi_x_end:
                     segment_depth_roi = current_depth_map[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
                     if segment_depth_roi.size > 0:
                         valid_depths = segment_depth_roi[segment_depth_roi > 0.1]
                         if valid_depths.size > 0:
                             min_depth = np.min(valid_depths)
                             temp_halo_colors[i] = get_halo_color_for_depth(
                                 min_depth, BGR_INACTIVE, BGR_WARN, BGR_ALERT, BGR_CRITICAL,
                                 WARN_THRESHOLD, ALERT_THRESHOLD, CRITICAL_THRESHOLD
                             )
            last_halo_segment_colors = temp_halo_colors

        process_time = (time.time() - start_time_process) * 1000
        total_process_time += process_time

    # --- Frame Display Logic (Always runs) ---
    display_output = frame_display.copy()
    if last_depth_map_color is not None: # Overlay depth map
        if last_depth_map_color.shape == display_output.shape:
            display_output = cv2.addWeighted(last_depth_map_color, DEPTH_OVERLAY_ALPHA, display_output, 1.0 - DEPTH_OVERLAY_ALPHA, 0.0)

    # Draw Halo using the list of calculated segment colors
    halo_radius_x_px = int(display_output.shape[1] * HALO_X_RADIUS_FACTOR)
    halo_radius_y_px = int(display_output.shape[0] * HALO_Y_RADIUS_FACTOR)
    # Call drawing function (which now ensures types)
    draw_obstacle_halo(display_output, last_halo_segment_colors, HALO_NUM_SEGMENTS,
                       halo_radius_x_px, halo_radius_y_px, HALO_THICKNESS)

    # Add Info Text
    avg_process_time_ms = total_process_time / processed_frame_counter if processed_frame_counter > 0 else 0
    effective_fps = 1000 / avg_process_time_ms if avg_process_time_ms > 0 else TARGET_FPS
    info_text = f"Frame: {frame_count} Proc FPS: {effective_fps:.1f} (Target: {TARGET_FPS:.1f})"
    cv2.putText(display_output, info_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Final Scaling for Display Window Size
    final_display_frame = display_output
    if final_display_frame.shape[1] > DISPLAY_WIDTH:
         scale_factor = DISPLAY_WIDTH / final_display_frame.shape[1]; display_height = int(final_display_frame.shape[0] * scale_factor)
         final_display_frame = cv2.resize(final_display_frame, (DISPLAY_WIDTH, display_height))
    cv2.imshow('Gradient Depth Halo Overlay', final_display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

# --- 5. Cleanup & Final Stats ---
cap.release(); cv2.destroyAllWindows()
avg_fps_final = 1000 / (total_process_time / processed_frame_counter) if processed_frame_counter > 0 else 0
print(f"\nProcessing finished. Processed {processed_frame_counter}/{frame_count} frames.")
print(f"Average Processing FPS: {avg_fps_final:.2f} (Target was {TARGET_FPS:.1f})")