import cv2
import numpy as np
import time
import torch
from transformers import AutoModelForDepthEstimation, AutoImageProcessor
from PIL import Image
import matplotlib.pyplot as plt # Keep for final plot (optional)
import os
import math # Needed for halo geometry

# --- User Configuration ---
VIDEO_PATH = 'sample.mov' # Your video file
DISPLAY_WIDTH = 1280 # Max width for the combined display window

# --- Approximate Calibration for DJI Avata ---
DJI_AVATA_NATIVE_WIDTH = 1920
DJI_AVATA_NATIVE_HEIGHT = 1080
GUESSED_FX = 550.0; GUESSED_FY = 550.0
GUESSED_CX = DJI_AVATA_NATIVE_WIDTH / 2; GUESSED_CY = DJI_AVATA_NATIVE_HEIGHT / 2
K_base = np.array([[GUESSED_FX, 0, GUESSED_CX], [0, GUESSED_FY, GUESSED_CY], [0, 0, 1]])
print(f"Using GUESSED base K matrix (for {DJI_AVATA_NATIVE_WIDTH}x{DJI_AVATA_NATIVE_HEIGHT}):\n{K_base}")
print("!!! WARNING: This K matrix is a GUESS. Accurate results require proper calibration !!!")

# --- Performance Tuning ---
PROCESS_HEIGHT = 360 # Lower = faster
FRAME_SKIP = 2       # Process 1 out of every N frames
ORB_FEATURES = 1500

# --- Plotting Configuration ---
PLOT_CANVAS_HEIGHT = 480; PLOT_CANVAS_WIDTH = 640
PLOT_MARGIN = 0.1; PLOT_INITIAL_SCALE = 20.0

# --- Obstacle Halo Configuration ---
OBSTACLE_THRESHOLD = 2.0 # meters <<< Objects closer than this trigger red halo
HALO_NUM_SEGMENTS = 5   # Number of segments (e.g., Left, Center-Left, Center, Center-Right, Right)
HALO_RADIUS_FACTOR = 0.3 # Radius relative to frame height
HALO_CENTER_Y_FACTOR = 0.9 # Vertical position (0.5=center, 0.9=lower)
HALO_THICKNESS = 10
HALO_TOTAL_ANGLE = 160 # Total angular span of the halo (degrees)
HALO_COLOR_INACTIVE = (100, 100, 100) # Gray
HALO_COLOR_ACTIVE = (0, 0, 255) # Red

# --- 1. Initialize Depth Estimation Model ---
print("Loading depth estimation model..."); MODEL_NAME = "LiheYoung/depth-anything-small-hf"
DEVICE = torch.device("cpu"); print(f"Using device: {DEVICE}")
try:
    image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    depth_model = AutoModelForDepthEstimation.from_pretrained(MODEL_NAME).to(DEVICE)
    print("Depth model loaded successfully.")
except Exception as e: print(f"Error loading depth model: {e}"); exit()

# --- 2. Initialize VO Components ---
orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened(): print(f"Error: Could not open video file: {VIDEO_PATH}"); exit()
actual_frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Actual video resolution: {actual_frame_width}x{actual_frame_height}")

# --- 3. Initialize State Variables ---
prev_frame_gray = None; prev_keypoints = None; prev_descriptors = None
previous_depth_map = None; last_processed_frame_count = -FRAME_SKIP
trajectory_points = [np.array([0, 0, 0], dtype=np.float64)]
current_R = np.identity(3); current_t = np.zeros((3, 1))

# --- 4. Plotting & Halo Functions ---
plot_canvas = np.ones((PLOT_CANVAS_HEIGHT, PLOT_CANVAS_WIDTH, 3), dtype=np.uint8) * 255

def calculate_adaptive_scale_and_origin(points_array, canvas_width, canvas_height, margin=0.1, initial_scale=20.0):
    # (Function remains the same as before)
    if points_array.shape[0] < 2: return initial_scale, canvas_width // 2, canvas_height // 2
    min_x, max_x = points_array[:, 0].min(), points_array[:, 0].max(); min_z, max_z = points_array[:, 2].min(), points_array[:, 2].max()
    world_range_x = max_x - min_x + (max_x - min_x) * margin * 2; world_range_z = max_z - min_z + (max_z - min_z) * margin * 2
    epsilon = 1e-6; world_range_x = max(world_range_x, epsilon); world_range_z = max(world_range_z, epsilon)
    scale_x = canvas_width / world_range_x; scale_z = canvas_height / world_range_z
    new_scale = min(scale_x, scale_z)
    center_x = (min_x + max_x) / 2; center_z = (min_z + max_z) / 2
    canvas_center_u = canvas_width / 2; canvas_center_v = canvas_height / 2
    new_origin_u = int(canvas_center_u - center_x * new_scale); new_origin_v = int(canvas_center_v + center_z * new_scale)
    # Prevent excessive zoom in at start
    if new_scale > initial_scale * 5: new_scale = initial_scale * 5
    return new_scale, new_origin_u, new_origin_v

def draw_trajectory(canvas, points_array, origin_u, origin_v, scale):
    # (Function remains the same as before)
    canvas.fill(255); h, w = canvas.shape[:2]
    world_origin_u, world_origin_v = int(origin_u), int(origin_v)
    if 0 <= world_origin_u < w and 0 <= world_origin_v < h:
        cv2.circle(canvas, (world_origin_u, world_origin_v), 5, (0, 200, 0), -1)
        # cv2.putText(canvas, "(0,0)", (world_origin_u + 5, world_origin_v - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 128, 0), 1)
    if points_array.shape[0] < 2: return
    for i in range(points_array.shape[0] - 1):
        p1, p2 = points_array[i], points_array[i+1]
        u1, v1 = int(origin_u + p1[0] * scale), int(origin_v - p1[2] * scale)
        u2, v2 = int(origin_u + p2[0] * scale), int(origin_v - p2[2] * scale)
        if max(u1, u2) >= 0 and min(u1, u2) < w and max(v1, v2) >= 0 and min(v1, v2) < h:
             cv2.line(canvas, (u1, v1), (u2, v2), (255, 0, 0), 2)
    curr_p = points_array[-1]; curr_u, curr_v = int(origin_u + curr_p[0] * scale), int(origin_v - curr_p[2] * scale)
    if 0 <= curr_u < w and 0 <= curr_v < h: cv2.circle(canvas, (curr_u, curr_v), 4, (0, 0, 255), -1)

def draw_obstacle_halo(frame, activation_status, num_segments, total_angle, radius, center_y_factor, thickness, color_active, color_inactive):
    """Draws the obstacle halo segments on the frame."""
    h, w = frame.shape[:2]
    center_x = w // 2
    center_y = int(h * center_y_factor)
    segment_angle = total_angle / num_segments
    # Start angle needs to place the arc correctly (0 is right, 90 is down in OpenCV ellipse)
    # For a bottom arc centered horizontally, start angle is 90 - total_angle/2
    start_angle_offset = 90 - total_angle / 2

    for i in range(num_segments):
        segment_start_angle = start_angle_offset + i * segment_angle
        segment_end_angle = segment_start_angle + segment_angle
        color = color_active if activation_status[i] else color_inactive
        axes = (radius, radius) # Circle

        try:
            # Use cv2.ellipse to draw thick arcs
            cv2.ellipse(frame, (center_x, center_y), axes, 0, segment_start_angle, segment_end_angle, color, thickness)
        except Exception as e:
            print(f"Error drawing halo segment {i}: {e}") # Catch potential drawing errors


# --- 5. Main Processing Loop ---
frame_count = 0; total_process_time = 0; processed_frame_counter = 0
plot_scale = PLOT_INITIAL_SCALE; plot_origin_u = PLOT_CANVAS_WIDTH // 2; plot_origin_v = PLOT_CANVAS_HEIGHT // 2

while True: # Main loop starts here
    ret, frame = cap.read()
    if not ret: print("End of video."); break
    frame_count += 1
    if frame_count % FRAME_SKIP != 0: continue

    start_time_frame = time.time(); processed_frame_counter += 1

    # --- Resize frame ---
    aspect_ratio = actual_frame_width / actual_frame_height
    process_width = int(PROCESS_HEIGHT * aspect_ratio)
    frame_resized = cv2.resize(frame, (process_width, PROCESS_HEIGHT))
    frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

    # --- Calculate Scaled K ---
    scale_x = process_width / DJI_AVATA_NATIVE_WIDTH; scale_y = PROCESS_HEIGHT / DJI_AVATA_NATIVE_HEIGHT
    K_resized = np.array([[GUESSED_FX * scale_x, 0, GUESSED_CX * scale_x],[0, GUESSED_FY * scale_y, GUESSED_CY * scale_y],[0, 0, 1]])

    # --- Depth Estimation ---
    start_time_depth = time.time(); current_depth_map = None
    try:
        image_pil = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
        inputs = image_processor(images=image_pil, return_tensors="pt").to(DEVICE)
        with torch.no_grad(): outputs = depth_model(**inputs)
        prediction = torch.nn.functional.interpolate(outputs.predicted_depth.unsqueeze(1), size=(PROCESS_HEIGHT, process_width), mode="bicubic", align_corners=False).squeeze()
        current_depth_map = prediction.cpu().numpy()
        depth_normalized = (current_depth_map - current_depth_map.min()) / (current_depth_map.max() - current_depth_map.min() + 1e-6)
        depth_display = (depth_normalized * 255).astype(np.uint8); depth_display_color = cv2.applyColorMap(depth_display, cv2.COLORMAP_INFERNO)
    except Exception as e: depth_display_color = np.zeros((PROCESS_HEIGHT, process_width, 3), dtype=np.uint8)
    depth_time = (time.time() - start_time_depth) * 1000

    # --- Feature Detection ---
    keypoints, descriptors = orb.detectAndCompute(frame_gray, None)

    # --- Prepare frame_display early for drawing halo ---
    frame_display = cv2.drawKeypoints(frame_resized, keypoints if keypoints is not None else [], None, color=(0, 255, 0), flags=0)

    # --- Obstacle Halo Logic ---
    halo_activation = [False] * HALO_NUM_SEGMENTS
    if current_depth_map is not None:
        depth_h, depth_w = current_depth_map.shape
        segment_width_px = depth_w // HALO_NUM_SEGMENTS
        # Define vertical ROI for depth check (e.g., bottom half, excluding very bottom edge maybe)
        roi_y_start = depth_h // 2
        roi_y_end = depth_h - int(depth_h * 0.05) # Exclude bottom 5%

        for i in range(HALO_NUM_SEGMENTS):
            # Define horizontal ROI for this segment
            roi_x_start = i * segment_width_px
            roi_x_end = (i + 1) * segment_width_px
            # Ensure ROI is valid
            if roi_y_start < roi_y_end and roi_x_start < roi_x_end:
                 segment_depth_roi = current_depth_map[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
                 # Check if ROI is not empty
                 if segment_depth_roi.size > 0:
                     # Filter out non-positive values which might be invalid background
                     valid_depths_in_roi = segment_depth_roi[segment_depth_roi > 0.1]
                     if valid_depths_in_roi.size > 0:
                         min_depth_in_segment = np.min(valid_depths_in_roi)
                         if min_depth_in_segment < OBSTACLE_THRESHOLD:
                             halo_activation[i] = True
                     # else: print(f"No valid depths in ROI {i}") # Debugging
                 # else: print(f"Empty ROI {i}") # Debugging


        # Draw the halo based on activation status
        halo_radius_px = int(frame_display.shape[0] * HALO_RADIUS_FACTOR)
        draw_obstacle_halo(frame_display, halo_activation, HALO_NUM_SEGMENTS, HALO_TOTAL_ANGLE,
                           halo_radius_px, HALO_CENTER_Y_FACTOR, HALO_THICKNESS,
                           HALO_COLOR_ACTIVE, HALO_COLOR_INACTIVE)

    # --- Feature Matching and Pose Estimation ---
    estimated_scale = 1.0; num_good_matches = 0; pose_updated = False
    if descriptors is not None and current_depth_map is not None and \
       prev_frame_gray is not None and prev_descriptors is not None and previous_depth_map is not None:
        # (Matching, Essential Matrix, Recover Pose, Scale Estimation logic remains the same)
        matches = bf.knnMatch(descriptors, prev_descriptors, k=2); good_matches = []
        try:
            if len(matches) > 0 and len(matches[0]) == 2:
                for m, n in matches:
                    if m.distance < 0.75 * n.distance: good_matches.append(m)
            num_good_matches = len(good_matches)
        except IndexError: pass
        if num_good_matches > 10:
            current_pts_img=np.float32([keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
            prev_pts_img=np.float32([prev_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
            E, mask_e = cv2.findEssentialMat(current_pts_img, prev_pts_img, K_resized, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            if E is not None and E.shape==(3,3):
                retval, R_rel, t_rel, mask_rp = cv2.recoverPose(E, current_pts_img, prev_pts_img, K_resized, mask=mask_e)
                num_recoverpose_inliers = np.sum(mask_rp) if mask_rp is not None else 0
                if num_recoverpose_inliers > 5:
                    inlier_indices_goodmatches=[i for i, flag in enumerate(mask_rp) if flag[0]==1]
                    inlier_indices_prev_kpts=[good_matches[i].trainIdx for i in inlier_indices_goodmatches]
                    valid_depths=[];
                    if len(inlier_indices_prev_kpts)>0:
                        prev_inlier_pts_coords=np.int32([prev_keypoints[idx].pt for idx in inlier_indices_prev_kpts])
                        for u,v in prev_inlier_pts_coords:
                           if 0<=v<previous_depth_map.shape[0] and 0<=u<previous_depth_map.shape[1]:
                               depth=previous_depth_map[v,u];
                               if depth > 0.1 and depth < 100: valid_depths.append(depth)
                    if len(valid_depths)>5: estimated_scale = np.median(valid_depths)
                    if t_rel is not None:
                        if t_rel.shape == (1,3): t_rel=t_rel.T
                        if t_rel.shape == (3,): t_rel=t_rel.reshape(3,1)
                        if t_rel.shape == (3,1):
                            adjusted_scale = estimated_scale
                            current_t = current_t + current_R @ t_rel * adjusted_scale
                            current_R = R_rel @ current_R
                            trajectory_points.append(current_t.flatten()); pose_updated = True

    # --- Update Previous Frame Data ---
    if descriptors is not None: # Only update if we had features this frame
        prev_frame_gray = frame_gray.copy(); prev_keypoints = keypoints; prev_descriptors = descriptors
        if current_depth_map is not None: previous_depth_map = current_depth_map.copy()
        last_processed_frame_count = frame_count

    # --- Calculate Adaptive Scale and Draw Trajectory ---
    if len(trajectory_points) > 1 :
        traj_array = np.array(trajectory_points)
        plot_scale, plot_origin_u, plot_origin_v = calculate_adaptive_scale_and_origin(traj_array, PLOT_CANVAS_WIDTH, PLOT_CANVAS_HEIGHT, PLOT_MARGIN, PLOT_INITIAL_SCALE)
        draw_trajectory(plot_canvas, traj_array, plot_origin_u, plot_origin_v, plot_scale)
    else: draw_trajectory(plot_canvas, np.array(trajectory_points), plot_origin_u, plot_origin_v, plot_scale)

    # --- Display Combined Results ---
    frame_time = (time.time() - start_time_frame) * 1000; total_process_time += frame_time
    avg_process_time = total_process_time / processed_frame_counter
    effective_fps = 1000 / avg_process_time if avg_process_time > 0 else 0
    info_text = f"F:{frame_count}({processed_frame_counter}) FPS:{effective_fps:.1f}"
    info_text2 = f"M:{num_good_matches} Scl:{estimated_scale:.2f} DptT:{depth_time:.0f}"
    # Draw info text on frame_display (which already has halo and keypoints)
    cv2.putText(frame_display, info_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame_display, info_text2, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    # Combine frame_display with depth map
    depth_display_color_resized = cv2.resize(depth_display_color, (frame_display.shape[1], frame_display.shape[0]))
    combined_top = np.hstack((frame_display, depth_display_color_resized))
    if combined_top.shape[1] > DISPLAY_WIDTH:
         scale_factor = DISPLAY_WIDTH / combined_top.shape[1]; display_height = int(combined_top.shape[0] * scale_factor)
         combined_top_display = cv2.resize(combined_top, (DISPLAY_WIDTH, display_height))
    else: combined_top_display = combined_top
    cv2.imshow('Metric VO, Depth & Halo', combined_top_display)
    cv2.imshow('Aerial Trajectory (X-Z)', plot_canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

# --- 6. Cleanup & Final Stats ---
cap.release(); cv2.destroyAllWindows()
avg_fps_final = 1000/(total_process_time/processed_frame_counter) if processed_frame_counter > 0 else 0
print(f"\nProcessing finished. Processed {processed_frame_counter}/{frame_count} frames. Avg FPS: {avg_fps_final:.2f}")
if 'current_t' in locals() and isinstance(current_t, np.ndarray): print(f"Final position (guess): {current_t.flatten()}")
else: print("Final position N/A.")

# --- Optional: Plot Final Trajectory with Matplotlib ---
try:
    print("Plotting final trajectory..."); trajectory = np.array(trajectory_points)
    if trajectory.shape[0] > 1:
        fig=plt.figure(figsize=(8,8)); ax=fig.add_subplot(111); ax.plot(trajectory[:,0], trajectory[:,2], marker='.', markersize=2, linestyle='-', label='Est. Traj (X-Z)')
        ax.set_xlabel('X [m](guess)'); ax.set_ylabel('Z [m](guess)'); ax.set_title('Final Estimated Trajectory'); ax.grid(True); ax.legend(); ax.axis('equal'); plt.show()
    else: print("Not enough points to plot.")
except ImportError: print("\nMatplotlib not found.")
except Exception as e: print(f"\nError plotting: {e}")