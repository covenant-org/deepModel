import cv2
import numpy as np
import time
import torch # Added
from transformers import AutoModelForDepthEstimation, AutoImageProcessor # Added
from PIL import Image # Added
import matplotlib.pyplot as plt # Keep for final plot (optional)
import os # Added for checking model cache

# --- User Configuration ---
VIDEO_PATH = 'sample.mov' # Your video file
# Camera Intrinsics (PLACEHOLDERS - REPLACE FOR ACCURACY!)
fx = 718.8560; fy = 718.8560; cx = 607.1928; cy = 185.2157
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
# Output Display Size (optional, makes windows manageable)
DISPLAY_WIDTH = 1280 # Adjust as needed

# --- 1. Initialize Depth Estimation Model ---
print("Loading depth estimation model...")
# Choose model: LiheYoung/depth-anything-small-hf is faster, base-hf is moderate
MODEL_NAME = "LiheYoung/depth-anything-small-hf"
# Check if model is cached to avoid re-downloading messages if possible
cache_dir = os.path.expanduser('~/.cache/huggingface/hub')
model_dir = os.path.join(cache_dir, f'models--{MODEL_NAME.replace("/", "--")}')

if not os.path.exists(model_dir):
     print(f"Model not found in cache, will download (may take time)...")

# Check for MPS (Metal Performance Shaders) on Mac for potential speedup
# if torch.backends.mps.is_available():
#     DEVICE = torch.device("mps")
# else:
#     DEVICE = torch.device("cpu")
DEVICE = torch.device("cpu") # Stick to CPU for now for simplicity

print(f"Using device: {DEVICE}")

try:
    image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    depth_model = AutoModelForDepthEstimation.from_pretrained(MODEL_NAME).to(DEVICE)
    print("Depth model loaded successfully.")
except Exception as e:
    print(f"Error loading depth model: {e}")
    print("Please ensure 'torch' and 'transformers' are installed correctly.")
    exit()

# --- 2. Initialize VO Components ---
orb = cv2.ORB_create(nfeatures=3000) # Increased features slightly
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Could not open video file: {VIDEO_PATH}")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video resolution: {frame_width}x{frame_height}")

# --- 3. Initialize State Variables ---
prev_frame_gray = None
prev_keypoints = None
prev_descriptors = None
previous_depth_map = None # To store the previous depth map

# Trajectory storage
trajectory_points = [np.array([0, 0, 0], dtype=np.float64)] # Start at origin [X, Y, Z]
current_R = np.identity(3)
current_t = np.zeros((3, 1))

# --- 4. Initialize Aerial View Plot Canvas ---
plot_canvas_height = 480 # Height of the plot window
plot_canvas_width = 640 # Width of the plot window
plot_canvas = np.ones((plot_canvas_height, plot_canvas_width, 3), dtype=np.uint8) * 255 # White canvas
plot_origin_u = int(plot_canvas_width / 2) # Center x
plot_origin_v = int(plot_canvas_height * 0.8) # Start lower down y
plot_scale = 30 # Pixels per meter (adjust as needed)

def draw_trajectory(canvas, points, origin_u, origin_v, scale):
    """Draws the trajectory on the canvas (X-Z plane, top-down)."""
    canvas.fill(255) # Clear canvas
    # Draw axes markers (optional)
    cv2.line(canvas, (origin_u, 0), (origin_u, canvas.shape[0]), (200, 200, 200), 1) # Z-axis (vertical)
    cv2.line(canvas, (0, origin_v), (canvas.shape[1], origin_v), (200, 200, 200), 1) # X-axis (horizontal)
    cv2.putText(canvas, "X", (canvas.shape[1] - 20, origin_v - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    cv2.putText(canvas, "Z", (origin_u + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)


    if len(points) < 2:
        return

    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i+1]

        # Project X (p[0]) and Z (p[2]) to canvas coordinates (u, v)
        # Assuming standard camera coordinates (+Z forward) maps to +v downwards on canvas
        u1 = int(origin_u + p1[0] * scale)
        v1 = int(origin_v - p1[2] * scale) # Subtract Z because +Z is forward, but +v is down
        u2 = int(origin_u + p2[0] * scale)
        v2 = int(origin_v - p2[2] * scale)

        # Basic bounds check
        h, w = canvas.shape[:2]
        if 0 <= u1 < w and 0 <= v1 < h and 0 <= u2 < w and 0 <= v2 < h:
             cv2.line(canvas, (u1, v1), (u2, v2), (255, 0, 0), 2) # Blue line

    # Draw current position marker
    curr_p = points[-1]
    curr_u = int(origin_u + curr_p[0] * scale)
    curr_v = int(origin_v - curr_p[2] * scale)
    if 0 <= curr_u < w and 0 <= curr_v < h:
        cv2.circle(canvas, (curr_u, curr_v), 4, (0, 0, 255), -1) # Red circle

# --- 5. Main Processing Loop ---
frame_count = 0
processing_time_list = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video reached or error reading frame.")
        break

    start_time_frame = time.time()
    frame_count += 1

    # Resize frame for faster processing (optional, but recommended for depth)
    aspect_ratio = frame.shape[1] / frame.shape[0]
    process_height = 480 # Process at this height
    process_width = int(process_height * aspect_ratio)
    frame_resized = cv2.resize(frame, (process_width, process_height))
    frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

    # --- Depth Estimation ---
    start_time_depth = time.time()
    try:
        # Prepare image for model (convert BGR to RGB PIL Image)
        image_pil = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
        inputs = image_processor(images=image_pil, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = depth_model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Interpolate to original frame size (or processed size) and convert to numpy
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(process_height, process_width),
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        current_depth_map = prediction.cpu().numpy() # Store metric depth (relative)

        # Normalize depth map for visualization (0-255)
        depth_normalized = (current_depth_map - current_depth_map.min()) / (current_depth_map.max() - current_depth_map.min())
        depth_display = (depth_normalized * 255).astype(np.uint8)
        depth_display_color = cv2.applyColorMap(depth_display, cv2.COLORMAP_INFERNO)

    except Exception as e:
        print(f"Error during depth estimation: {e}")
        # Use a blank map if error occurs, or skip frame?
        current_depth_map = np.zeros((process_height, process_width), dtype=np.float32)
        depth_display_color = np.zeros((process_height, process_width, 3), dtype=np.uint8)

    depth_time = (time.time() - start_time_depth) * 1000

    # --- Feature Detection ---
    keypoints, descriptors = orb.detectAndCompute(frame_gray, None)

    if descriptors is None:
        print(f"Warning: No descriptors found in frame {frame_count}")
        # Keep previous frame data for the next iteration, including depth
        prev_frame_gray = frame_gray.copy()
        prev_keypoints = keypoints
        prev_descriptors = descriptors
        previous_depth_map = current_depth_map.copy() if current_depth_map is not None else None
        continue

    frame_display = cv2.drawKeypoints(frame_resized, keypoints, None, color=(0, 255, 0), flags=0)

    # --- Feature Matching and Pose Estimation (if not the first frame) ---
    estimated_scale = 1.0 # Default scale if estimation fails
    num_good_matches = 0

    if prev_frame_gray is not None and prev_descriptors is not None and previous_depth_map is not None:
        matches = bf.knnMatch(descriptors, prev_descriptors, k=2)
        good_matches = []
        try:
            if len(matches) > 0 and len(matches[0]) == 2:
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
            num_good_matches = len(good_matches)
        except IndexError:
            print(f"Warning: Issue with knnMatch output structure in frame {frame_count}")
            good_matches = []


        #print(f"Frame {frame_count}: Found {len(keypoints)} kpts, {num_good_matches} good matches.")

        if num_good_matches > 10: # Increased threshold slightly
            current_pts_img = np.float32([keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            prev_pts_img = np.float32([prev_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Adjust K matrix for resized frame
            scale_x = process_width / frame_width
            scale_y = process_height / frame_height
            K_resized = np.array([[fx * scale_x, 0, cx * scale_x],
                                  [0, fy * scale_y, cy * scale_y],
                                  [0, 0, 1]])

            E, mask_e = cv2.findEssentialMat(current_pts_img, prev_pts_img, K_resized, method=cv2.RANSAC, prob=0.999, threshold=1.0)

            if E is not None and E.shape == (3, 3):
                retval, R_rel, t_rel, mask_rp = cv2.recoverPose(E, current_pts_img, prev_pts_img, K_resized, mask=mask_e)

                if retval > 0.5 * num_good_matches : # Check if enough inliers support the pose
                    # --- Estimate Scale from Depth ---
                    inlier_indices_prev = [m.trainIdx for i, m in enumerate(good_matches) if mask_rp[i, 0] == 1] # Get indices of previous keypoints that are inliers
                    valid_depths = []
                    if len(inlier_indices_prev) > 0:
                        prev_inlier_pts = np.int32([prev_keypoints[idx].pt for idx in inlier_indices_prev]) # Get (u,v) coords

                        for u, v in prev_inlier_pts:
                           if 0 <= v < previous_depth_map.shape[0] and 0 <= u < previous_depth_map.shape[1]:
                               depth = previous_depth_map[v, u]
                               # Add sanity checks for depth values (e.g., ignore zero or very large values)
                               if depth > 0.1 and depth < 100: # Adjust range based on expected scene scale
                                   valid_depths.append(depth)

                    if len(valid_depths) > 5: # Need a few depth points for robustness
                         # estimated_scale = np.mean(valid_depths)
                         estimated_scale = np.median(valid_depths) # Median is more robust to outliers
                         #print(f"  Estimated scale: {estimated_scale:.3f} from {len(valid_depths)} points")
                    else:
                        #print(f"Warning: Not enough valid depths ({len(valid_depths)}) for scale estimation. Using default scale 1.0.")
                        estimated_scale = 1.0 # Fallback

                    # --- Update Full Trajectory (Metric Scale) ---
                    # Ensure t_rel has the correct shape (3, 1)
                    if t_rel.shape == (3,):
                        t_rel = t_rel.reshape(3, 1)
                    elif t_rel.shape == (1, 3):
                        t_rel = t_rel.T # Transpose if it's (1, 3)

                    if t_rel.shape == (3, 1):
                        current_t = current_t + current_R @ t_rel * estimated_scale
                        current_R = R_rel @ current_R
                        trajectory_points.append(current_t.flatten())
                    else:
                        print(f"Warning: t_rel has unexpected shape {t_rel.shape} after recoverPose. Skipping update.")


                else:
                     print(f"Warning: recoverPose failed or low inliers ({retval}/{num_good_matches}) in frame {frame_count}")
                     estimated_scale = 1.0 # Reset scale if pose fails
            else:
                 print(f"Warning: Essential Matrix estimation failed or invalid shape in frame {frame_count}")
                 estimated_scale = 1.0 # Reset scale
        else:
             #print(f"Warning: Not enough good matches ({num_good_matches}) for pose estimation.")
             estimated_scale = 1.0 # Reset scale


    # --- Update Previous Frame Data ---
    prev_frame_gray = frame_gray.copy()
    prev_keypoints = keypoints
    prev_descriptors = descriptors
    previous_depth_map = current_depth_map.copy() if current_depth_map is not None else None


    # --- Draw Trajectory on Aerial Plot ---
    draw_trajectory(plot_canvas, trajectory_points, plot_origin_u, plot_origin_v, plot_scale)

    # --- Display Combined Results ---
    frame_time = (time.time() - start_time_frame) * 1000
    processing_time_list.append(frame_time)

    # Prepare text info
    info_text = f"Frame: {frame_count} FPS: {1000/frame_time:.1f} Matches: {num_good_matches}"
    info_text2 = f"Scale: {estimated_scale:.2f} Depth Time: {depth_time:.0f} ms"
    cv2.putText(frame_display, info_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame_display, info_text2, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Resize depth display to match frame display height for stacking
    depth_display_color_resized = cv2.resize(depth_display_color, (frame_display.shape[1], frame_display.shape[0]))

    # Combine main view and depth view side-by-side
    combined_top = np.hstack((frame_display, depth_display_color_resized))

    # Resize plot canvas to match combined top width if desired, or display separately
    # To stack below:
    # plot_canvas_resized = cv2.resize(plot_canvas, (combined_top.shape[1], plot_canvas_height))
    # combined_all = np.vstack((combined_top, plot_canvas_resized))
    # cv2.imshow('Metric Monocular VO', combined_all)

    # Display separately for simplicity
    # Scale down combined top view if too large
    if combined_top.shape[1] > DISPLAY_WIDTH:
         scale_factor = DISPLAY_WIDTH / combined_top.shape[1]
         display_height = int(combined_top.shape[0] * scale_factor)
         combined_top_display = cv2.resize(combined_top, (DISPLAY_WIDTH, display_height))
    else:
         combined_top_display = combined_top

    cv2.imshow('Visual Odometry & Depth', combined_top_display)
    cv2.imshow('Aerial Trajectory (X-Z)', plot_canvas)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'): # Use waitKey(1) for max speed
        break

# --- 6. Cleanup & Final Stats ---
cap.release()
cv2.destroyAllWindows()

if processing_time_list:
    avg_fps = 1000 / np.mean(processing_time_list)
    print(f"\nProcessing finished.")
    print(f"Average FPS: {avg_fps:.2f}")
else:
    print("\nProcessing finished (no frames processed).")

print(f"Final estimated position (metric scale): {current_t.flatten()}")

# --- Optional: Plot Final Trajectory with Matplotlib ---
# (Keep the existing Matplotlib code here if you still want the static final plot)
try:
    print("Plotting final trajectory with Matplotlib...")
    trajectory = np.array(trajectory_points)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.plot(trajectory[:, 0], trajectory[:, 2], marker='o', markersize=2, linestyle='-', label='Estimated Trajectory (X-Z plane)')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Z [m]')
    ax.set_title('Final Estimated Camera Trajectory (Top-Down)')
    ax.grid(True)
    ax.legend()
    ax.axis('equal')
    plt.show()
except ImportError:
    print("\nMatplotlib not found. Skipping final trajectory plot.")
except Exception as e:
    print(f"\nError plotting final trajectory: {e}")