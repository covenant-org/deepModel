import cv2
import numpy as np
import time
import torch
from transformers import AutoModelForDepthEstimation, AutoImageProcessor
from PIL import Image
import matplotlib.pyplot as plt # Keep for final plot (optional)
import os

# --- User Configuration ---
VIDEO_PATH = 'sample.mov' # Your video file
DISPLAY_WIDTH = 1280 # Max width for the combined display window

# --- Approximate Calibration for DJI Avata ---
DJI_AVATA_NATIVE_WIDTH = 1920
DJI_AVATA_NATIVE_HEIGHT = 1080
GUESSED_FX = 550.0
GUESSED_FY = 550.0
GUESSED_CX = DJI_AVATA_NATIVE_WIDTH / 2
GUESSED_CY = DJI_AVATA_NATIVE_HEIGHT / 2
K_base = np.array([[GUESSED_FX, 0, GUESSED_CX],
                   [0, GUESSED_FY, GUESSED_CY],
                   [0, 0, 1]])
print(f"Using GUESSED base K matrix (for {DJI_AVATA_NATIVE_WIDTH}x{DJI_AVATA_NATIVE_HEIGHT}):\n{K_base}")
print("!!! WARNING: This K matrix is a GUESS. Accurate results require proper calibration !!!")

# --- Performance Tuning ---
PROCESS_HEIGHT = 360 # <<< Lower this (e.g., 240) for more speed, less accuracy
FRAME_SKIP = 3       # <<< Process 1 out of every N frames (e.g., 2, 3, 5). Higher = faster but less smooth tracking.
ORB_FEATURES = 1500  # <<< Reduced ORB features

# --- 1. Initialize Depth Estimation Model ---
print("Loading depth estimation model...")
MODEL_NAME = "LiheYoung/depth-anything-small-hf" # Keep the small one
cache_dir = os.path.expanduser('~/.cache/huggingface/hub')
model_dir = os.path.join(cache_dir, f'models--{MODEL_NAME.replace("/", "--")}')
if not os.path.exists(model_dir):
     print(f"Model not found in cache, will download...")

DEVICE = torch.device("cpu") # Stick to CPU
print(f"Using device: {DEVICE}")
try:
    image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    depth_model = AutoModelForDepthEstimation.from_pretrained(MODEL_NAME).to(DEVICE)
    print("Depth model loaded successfully.")
except Exception as e:
    print(f"Error loading depth model: {e}")
    exit()

# --- 2. Initialize VO Components ---
orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Could not open video file: {VIDEO_PATH}")
    exit()

# Get video properties (use these for scaling K later)
actual_frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Actual video resolution: {actual_frame_width}x{actual_frame_height}")

# --- 3. Initialize State Variables ---
prev_frame_gray = None
prev_keypoints = None
prev_descriptors = None
previous_depth_map = None
last_processed_frame_count = -FRAME_SKIP # Ensure first frame is processed

# Trajectory storage
trajectory_points = [np.array([0, 0, 0], dtype=np.float64)] # Start at origin [X, Y, Z]
current_R = np.identity(3)
current_t = np.zeros((3, 1))

# --- 4. Initialize Aerial View Plot Canvas ---
plot_canvas_height = 480
plot_canvas_width = 640
plot_canvas = np.ones((plot_canvas_height, plot_canvas_width, 3), dtype=np.uint8) * 255 # White canvas
plot_origin_u = int(plot_canvas_width / 2)
plot_origin_v = int(plot_canvas_height * 0.8)
plot_scale = 20 # Pixels per meter (MAY need adjustment based on depth results)

def draw_trajectory(canvas, points, origin_u, origin_v, scale):
    """Draws the trajectory on the canvas (X-Z plane, top-down)."""
    canvas.fill(255) # Clear canvas
    cv2.line(canvas, (origin_u, 0), (origin_u, canvas.shape[0]), (200, 200, 200), 1) # Z-axis (vertical)
    cv2.line(canvas, (0, origin_v), (canvas.shape[1], origin_v), (200, 200, 200), 1) # X-axis (horizontal)
    cv2.putText(canvas, "X", (canvas.shape[1] - 20, origin_v - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    cv2.putText(canvas, "Z", (origin_u + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    if len(points) < 2: return

    for i in range(len(points) - 1):
        p1, p2 = points[i], points[i+1]
        u1, v1 = int(origin_u + p1[0] * scale), int(origin_v - p1[2] * scale)
        u2, v2 = int(origin_u + p2[0] * scale), int(origin_v - p2[2] * scale)
        h, w = canvas.shape[:2]
        if 0 <= u1 < w and 0 <= v1 < h and 0 <= u2 < w and 0 <= v2 < h:
             cv2.line(canvas, (u1, v1), (u2, v2), (255, 0, 0), 2)
    curr_p = points[-1]
    curr_u, curr_v = int(origin_u + curr_p[0] * scale), int(origin_v - curr_p[2] * scale)
    if 0 <= curr_u < w and 0 <= curr_v < h:
        cv2.circle(canvas, (curr_u, curr_v), 4, (0, 0, 255), -1)

# --- 5. Main Processing Loop ---
frame_count = 0
total_process_time = 0
processed_frame_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video reached or error reading frame.")
        break

    frame_count += 1

    # --- Frame Skipping ---
    if frame_count % FRAME_SKIP != 0:
        continue # Skip this frame

    start_time_frame = time.time()
    processed_frame_counter += 1

    # --- Resize frame for processing ---
    aspect_ratio = actual_frame_width / actual_frame_height
    process_width = int(PROCESS_HEIGHT * aspect_ratio)
    frame_resized = cv2.resize(frame, (process_width, PROCESS_HEIGHT))
    frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

    # Calculate the K matrix scaled for the PROCESS resolution
    scale_x = process_width / DJI_AVATA_NATIVE_WIDTH
    # ***** CORRECTED TYPO HERE *****
    scale_y = PROCESS_HEIGHT / DJI_AVATA_NATIVE_HEIGHT
    # ***** /CORRECTED TYPO HERE *****
    K_resized = np.array([[GUESSED_FX * scale_x, 0, GUESSED_CX * scale_x],
                          [0, GUESSED_FY * scale_y, GUESSED_CY * scale_y],
                          [0, 0, 1]])

    # --- Depth Estimation ---
    start_time_depth = time.time()
    try:
        image_pil = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
        inputs = image_processor(images=image_pil, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = depth_model(**inputs)
            predicted_depth = outputs.predicted_depth
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1), size=(PROCESS_HEIGHT, process_width),
            mode="bicubic", align_corners=False,
        ).squeeze()
        current_depth_map = prediction.cpu().numpy()
        # Add epsilon to prevent division by zero if max == min
        depth_normalized = (current_depth_map - current_depth_map.min()) / (current_depth_map.max() - current_depth_map.min() + 1e-6)
        depth_display = (depth_normalized * 255).astype(np.uint8)
        depth_display_color = cv2.applyColorMap(depth_display, cv2.COLORMAP_INFERNO)
    except Exception as e:
        print(f"Frame {frame_count}: Error during depth estimation: {e}")
        current_depth_map = None # Signal failure
        depth_display_color = np.zeros((PROCESS_HEIGHT, process_width, 3), dtype=np.uint8)
    depth_time = (time.time() - start_time_depth) * 1000

    # --- Feature Detection ---
    keypoints, descriptors = orb.detectAndCompute(frame_gray, None)

    if descriptors is None or current_depth_map is None: # Also skip if depth failed
        if descriptors is None: print(f"Frame {frame_count}: Warning: No descriptors found")
        # Keep previous data ONLY if descriptors were found this frame (for next iteration)
        if descriptors is not None:
            # Decide if keeping old data is best, maybe clearing is better if depth fails often
            prev_frame_gray = frame_gray.copy() # Update frame anyway
            prev_keypoints = keypoints # Update features
            prev_descriptors = descriptors
            # Don't update previous_depth_map if current one failed
        continue # Skip processing for this frame

    frame_display = cv2.drawKeypoints(frame_resized, keypoints, None, color=(0, 255, 0), flags=0)

    # --- Feature Matching and Pose Estimation ---
    estimated_scale = 1.0 # Default fallback
    num_good_matches = 0
    pose_updated = False # Flag to check if pose was updated

    if prev_frame_gray is not None and prev_descriptors is not None and previous_depth_map is not None:
        matches = bf.knnMatch(descriptors, prev_descriptors, k=2)
        good_matches = []
        try:
            if len(matches) > 0 and len(matches[0]) == 2: # Basic check for k=2 results
                for m, n in matches:
                    if m.distance < 0.75 * n.distance: good_matches.append(m)
            num_good_matches = len(good_matches)
        except IndexError: pass # Ignore if match format is weird

        if num_good_matches > 10: # Need a reasonable number of good matches
            current_pts_img = np.float32([keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            prev_pts_img = np.float32([prev_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Use the K_resized calculated earlier
            E, mask_e = cv2.findEssentialMat(current_pts_img, prev_pts_img, K_resized, method=cv2.RANSAC, prob=0.999, threshold=1.0)

            if E is not None and E.shape == (3, 3):
                # Pass mask_e to recoverPose for better inlier check
                retval, R_rel, t_rel, mask_rp = cv2.recoverPose(E, current_pts_img, prev_pts_img, K_resized, mask=mask_e)

                # Check if retval (inliers) is a reasonable fraction of good_matches AFTER Essential Matrix
                # Use mask_rp to count inliers for recoverPose step
                num_recoverpose_inliers = np.sum(mask_rp) if mask_rp is not None else 0

                # Maybe adjust this threshold, 0.5 might be too high if E has few inliers
                if num_recoverpose_inliers > 5 : # Require at least 5 inliers for recoverPose
                    # --- Estimate Scale from Depth ---
                    # Get indices relative to `good_matches` list
                    inlier_indices_goodmatches = [i for i, inlier_flag in enumerate(mask_rp) if inlier_flag[0] == 1]
                    # Get corresponding indices in the *previous* frame's keypoint list
                    inlier_indices_prev_kpts = [good_matches[i].trainIdx for i in inlier_indices_goodmatches]

                    valid_depths = []
                    if len(inlier_indices_prev_kpts) > 0:
                        prev_inlier_pts_coords = np.int32([prev_keypoints[idx].pt for idx in inlier_indices_prev_kpts]) # Get (u,v) coords

                        for u, v in prev_inlier_pts_coords:
                           # Check bounds carefully
                           if 0 <= v < previous_depth_map.shape[0] and 0 <= u < previous_depth_map.shape[1]:
                               depth = previous_depth_map[v, u]
                               # Add sanity checks for depth values (e.g., ignore zero or very large values)
                               if depth > 0.1 and depth < 100: # Adjust range based on expected scene scale
                                   valid_depths.append(depth)

                    if len(valid_depths) > 5: # Need a few depth points for robustness
                         estimated_scale = np.median(valid_depths) # Median is more robust to outliers
                    #else: print(f"Frame {frame_count}: Warning: Not enough valid depths ({len(valid_depths)}) for scale.")

                    # --- Update Full Trajectory (Metric Scale) ---
                    if t_rel is not None: # Ensure t_rel is valid from recoverPose
                        # Ensure t_rel has the correct shape (3, 1)
                        if t_rel.shape == (1, 3): t_rel = t_rel.T
                        if t_rel.shape == (3,): t_rel = t_rel.reshape(3, 1)

                        if t_rel.shape == (3, 1):
                            # Scale update by time between *processed* frames
                            # delta_frames = frame_count - last_processed_frame_count # Not reliable if FPS varies
                            # Simple scale adjustment - NEEDS refinement if using actual time / FPS
                            adjusted_scale = estimated_scale # Using median depth directly

                            current_t = current_t + current_R @ t_rel * adjusted_scale
                            current_R = R_rel @ current_R # Update rotation matrix
                            trajectory_points.append(current_t.flatten()) # Store as 1D array
                            pose_updated = True # Mark that pose was updated
                        # else: print(f"Frame {frame_count}: Warning: t_rel shape issue {t_rel.shape} after reshape")
                    # else: print(f"Frame {frame_count}: Warning: t_rel is None after recoverPose")

                # else: print(f"Frame {frame_count}: Warning: recoverPose failed or low inliers ({num_recoverpose_inliers})")
            # else: print(f"Frame {frame_count}: Warning: Essential Matrix estimation failed or invalid shape")
        # else: print(f"Frame {frame_count}: Warning: Not enough good matches ({num_good_matches}) for pose estimation.")


    # --- Update Previous Frame Data only if pose was updated or features are good ---
    # Keep previous data to allow matching on the *next* processed frame
    prev_frame_gray = frame_gray.copy()
    prev_keypoints = keypoints
    prev_descriptors = descriptors
    # Update previous depth only if the current one was valid
    if current_depth_map is not None:
        previous_depth_map = current_depth_map.copy()
    # Store the frame number we just processed
    last_processed_frame_count = frame_count

    # --- Draw Trajectory on Aerial Plot ---
    draw_trajectory(plot_canvas, trajectory_points, plot_origin_u, plot_origin_v, plot_scale)

    # --- Display Combined Results ---
    frame_time = (time.time() - start_time_frame) * 1000
    total_process_time += frame_time
    avg_process_time = total_process_time / processed_frame_counter
    effective_fps = 1000 / avg_process_time if avg_process_time > 0 else 0

    info_text = f"Frame: {frame_count} (Proc: {processed_frame_counter}) Eff FPS: {effective_fps:.1f}"
    info_text2 = f"Matches: {num_good_matches} Scale: {estimated_scale:.2f} DepthT: {depth_time:.0f}ms"
    cv2.putText(frame_display, info_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame_display, info_text2, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Resize depth display to match frame display height for stacking
    depth_display_color_resized = cv2.resize(depth_display_color, (frame_display.shape[1], frame_display.shape[0]))

    # Combine main view and depth view side-by-side
    combined_top = np.hstack((frame_display, depth_display_color_resized))

    # Scale down combined top view if too large for screen
    if combined_top.shape[1] > DISPLAY_WIDTH:
         scale_factor = DISPLAY_WIDTH / combined_top.shape[1]
         display_height = int(combined_top.shape[0] * scale_factor)
         combined_top_display = cv2.resize(combined_top, (DISPLAY_WIDTH, display_height))
    else:
         combined_top_display = combined_top

    cv2.imshow('Metric Monocular VO & Depth', combined_top_display)
    cv2.imshow('Aerial Trajectory (X-Z)', plot_canvas)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'): # Use waitKey(1) for max speed
        break

# --- 6. Cleanup & Final Stats ---
cap.release()
cv2.destroyAllWindows()

avg_fps_final = 1000 / (total_process_time / processed_frame_counter) if processed_frame_counter > 0 else 0
print(f"\nProcessing finished.")
print(f"Processed {processed_frame_counter} frames out of {frame_count}.")
print(f"Average Effective FPS: {avg_fps_final:.2f}")
# Ensure current_t exists and is numpy array before printing
if 'current_t' in locals() and isinstance(current_t, np.ndarray):
    print(f"Final estimated position (metric scale guess): {current_t.flatten()}")
else:
     print("Final estimated position not available (processing might have stopped early).")


# --- Optional: Plot Final Trajectory with Matplotlib ---
try:
    print("Plotting final trajectory with Matplotlib...")
    trajectory = np.array(trajectory_points)
    if trajectory.shape[0] > 1: # Need at least 2 points to plot
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        ax.plot(trajectory[:, 0], trajectory[:, 2], marker='.', markersize=2, linestyle='-', label='Estimated Trajectory (X-Z plane)')
        ax.set_xlabel('X [m] (guessed scale)')
        ax.set_ylabel('Z [m] (guessed scale)')
        ax.set_title('Final Estimated Camera Trajectory (Top-Down)')
        ax.grid(True)
        ax.legend()
        ax.axis('equal')
        plt.show()
    else:
        print("Not enough trajectory points generated to plot.")
except ImportError:
    print("\nMatplotlib not found. Skipping final trajectory plot.")
except Exception as e:
    print(f"\nError plotting final trajectory: {e}")