import cv2
import numpy as np
import time # To see processing time

# --- 1. Configuration & Placeholders ---

# IMPORTANT: Replace with your actual camera calibration data!
# These are generic placeholders and WILL NOT give accurate results.
# You need focal lengths (fx, fy) and principal point (cx, cy)
fx = 718.8560 # Example value
fy = 718.8560 # Example value
cx = 607.1928 # Example value
cy = 185.2157 # Example value
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])
# Assuming no lens distortion for simplicity, otherwise add distortion coefficients
D = np.zeros(5) # Placeholder for distortion coefficients [k1, k2, p1, p2, k3]

# Path to your video file
video_path = 'sample.mov' # <<< CHANGE THIS TO YOUR VIDEO FILE

# Feature Detector/Descriptor (ORB)
orb = cv2.ORB_create(nfeatures=2000) # Increase features for potentially better matching

# Feature Matcher (Brute-Force with Hamming distance, suitable for ORB)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False) # crossCheck=False for ratio test

# --- 2. Initialization ---
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video file: {video_path}")
    exit()

prev_frame_gray = None
prev_keypoints = None
prev_descriptors = None

# Variables to store the full trajectory
trajectory_points = [np.array([0, 0, 0], dtype=np.float64)] # Start at origin
current_R = np.identity(3) # Current cumulative rotation
current_t = np.zeros((3, 1)) # Current cumulative translation (relative scale)

# --- 3. Main Processing Loop ---
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video reached or error reading frame.")
        break

    start_time = time.time()
    frame_count += 1
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --- Feature Detection and Description ---
    keypoints, descriptors = orb.detectAndCompute(frame_gray, None)

    if descriptors is None: # Handle cases where no features are found
        print(f"Warning: No descriptors found in frame {frame_count}")
        # Keep previous frame data for the next iteration
        continue

    # Visualization: Draw keypoints on the current frame
    frame_display = cv2.drawKeypoints(frame, keypoints, None, color=(0, 255, 0), flags=0)

    # --- Feature Matching and Pose Estimation (if not the first frame) ---
    if prev_frame_gray is not None and prev_descriptors is not None:
        # Match descriptors between current and previous frame
        matches = bf.knnMatch(descriptors, prev_descriptors, k=2) # Find 2 nearest neighbors

        # Apply Lowe's Ratio Test to filter good matches
        good_matches = []
        if len(matches) > 0 and len(matches[0]) == 2: # Ensure knnMatch returned pairs
            for m, n in matches:
                if m.distance < 0.75 * n.distance: # Adjust ratio threshold as needed
                    good_matches.append(m)
        else:
             print(f"Warning: Not enough neighbors found for ratio test in frame {frame_count}")


        print(f"Frame {frame_count}: Found {len(keypoints)} keypoints, {len(good_matches)} good matches.")

        # Need enough matches to estimate pose (at least 5 for Essential Matrix with RANSAC)
        if len(good_matches) > 8:
            # Get coordinates of matched keypoints
            current_pts = np.float32([keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            prev_pts = np.float32([prev_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # --- Estimate Essential Matrix ---
            # K is the camera intrinsic matrix
            E, mask_e = cv2.findEssentialMat(current_pts, prev_pts, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

            if E is not None and E.shape == (3, 3):
                # --- Recover Relative Pose (Rotation R_rel, Translation t_rel) ---
                # t_rel is normalized (unit vector) due to scale ambiguity
                retval, R_rel, t_rel, mask_rp = cv2.recoverPose(E, current_pts, prev_pts, K, mask=mask_e)

                if retval > 0.5 * len(good_matches): # Check if enough points support the pose
                    # --- Update Full Trajectory ---
                    # IMPORTANT: Scale Ambiguity! t_rel is only a direction.
                    # We add the translation scaled by the *previous* rotation state.
                    # We don't know the true scale, so the trajectory length is arbitrary.
                    scale_factor = 1.0 # Arbitrary scale factor for visualization
                    current_t = current_t + current_R @ t_rel * scale_factor
                    current_R = R_rel @ current_R # Update rotation matrix

                    # Store the new position
                    trajectory_points.append(current_t.flatten()) # Store as 1D array

                else:
                    print(f"Warning: recoverPose failed or low inliers ({retval}/{len(good_matches)}) in frame {frame_count}")
            else:
                 print(f"Warning: Essential Matrix estimation failed or invalid shape in frame {frame_count}")


            # Visualization: Draw good matches
            # frame_display = cv2.drawMatches(frame, keypoints, prev_frame_color, prev_keypoints, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        else:
             print(f"Warning: Not enough good matches ({len(good_matches)}) for pose estimation in frame {frame_count}.")


    # --- Update Previous Frame Data ---
    prev_frame_gray = frame_gray.copy()
    # prev_frame_color = frame.copy() # Keep color version if drawing matches between frames
    prev_keypoints = keypoints
    prev_descriptors = descriptors

    # --- Display Results ---
    processing_time = (time.time() - start_time) * 1000 # ms
    cv2.putText(frame_display, f"Frame: {frame_count} Matches: {len(good_matches) if 'good_matches' in locals() else 0} Time: {processing_time:.1f} ms",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Very basic trajectory visualization on frame (draw last N points)
    traj_vis_len = 100 # Number of trajectory points to show
    for i in range(max(0, len(trajectory_points) - traj_vis_len), len(trajectory_points) - 1):
        # Project 3D points to 2D (simple x, z -> u, v - needs improvement)
        # This is a crude visualization - a proper 3D plot is better
        p1 = trajectory_points[i]
        p2 = trajectory_points[i+1]
        # Offset to see it on screen
        offset_x, offset_y = int(frame_display.shape[1] * 0.5), int(frame_display.shape[0] * 0.8)
        # Scale for visibility (adjust as needed)
        vis_scale = 5
        try:
            cv2.line(frame_display,
                     (int(p1[0]*vis_scale + offset_x), int(p1[2]*vis_scale + offset_y)), # Using X and Z for a top-down view sketch
                     (int(p2[0]*vis_scale + offset_x), int(p2[2]*vis_scale + offset_y)),
                     (255, 0, 0), 2)
        except OverflowError:
            print("Warning: Overflow error during trajectory drawing - values too large?")


    cv2.imshow('Monocular Odometry Demo', frame_display)

    # Exit on 'q' key press
    if cv2.waitKey(10) & 0xFF == ord('q'): # waitKey(1) for max speed, waitKey(30) ~30fps
        break

# --- 4. Cleanup ---
cap.release()
cv2.destroyAllWindows()

# --- 5. Optional: Plot Trajectory (using Matplotlib) ---
try:
    import matplotlib.pyplot as plt
    print("Plotting trajectory...")
    trajectory = np.array(trajectory_points)
    fig = plt.figure()
    ax = fig.add_subplot(111) # Or projection='3d' for 3D plot
    # Plotting X vs Z (top-down view)
    ax.plot(trajectory[:, 0], trajectory[:, 2], marker='o', markersize=2, linestyle='-', label='Estimated Trajectory (X-Z plane)')
    # Plotting X vs Y (side view)
    # ax.plot(trajectory[:, 0], trajectory[:, 1], marker='.', linestyle='-', label='Estimated Trajectory (X-Y plane)')
    ax.set_xlabel('X [m] (arbitrary scale)')
    ax.set_ylabel('Z [m] (arbitrary scale)') # Change label if plotting Y
    ax.set_title('Estimated Camera Trajectory')
    ax.grid(True)
    ax.legend()
    ax.axis('equal') # Important for aspect ratio
    plt.show()
except ImportError:
    print("\nMatplotlib not found. Skipping trajectory plot.")
    print("To install: pip install matplotlib")

print("Processing finished.")
print(f"Final estimated position (arbitrary scale): {current_t.flatten()}")
