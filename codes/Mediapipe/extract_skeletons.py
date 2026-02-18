import cv2
import mediapipe as mp
import numpy as np
import os
from tqdm import tqdm

# --- Configuration ---
INPUT_FOLDER = '../kinetics_subset/videos'
OUTPUT_FOLDER = './mediapipe_skeletons'
MODEL_COMPLEXITY = 1

def extract_skeletons():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=MODEL_COMPLEXITY,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        # Find all .mp4 files in the input folder
        try:
            video_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith('.mp4')]
            if not video_files:
                print(f"Error: No .mp4 files found in the '{INPUT_FOLDER}' directory.")
                return
        except FileNotFoundError:
            print(f"Error: The directory '{INPUT_FOLDER}' was not found.")
            return

        print(f"Found {len(video_files)} videos to process.")
        
        for video_file in tqdm(video_files, desc="Processing Videos"):
            video_path = os.path.join(INPUT_FOLDER, video_file)
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Warning: Could not open video file {video_file}. Skipping.")
                continue
            
            # List to store the skeleton data for each frame of the current video
            video_landmarks = []

            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                # Convert the BGR image to RGB before processing
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                image_rgb.flags.writeable = False
                results = pose.process(image_rgb)
                
                if results.pose_world_landmarks:
                    # Use pose_world_landmarks for 3D coordinates in meters.
                    # These are camera-independent.
                    landmarks = results.pose_world_landmarks.landmark
                    
                    # Create a (33, 4) numpy array for the current frame
                    # Each row is a landmark: [x, y, z, visibility]
                    frame_landmarks = np.array(
                        [[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks],
                        dtype=np.float32
                    )
                    video_landmarks.append(frame_landmarks)
                else:
                    # If no pose is detected, append a placeholder of zeros
                    # This maintains the sequence length equal to the frame count
                    video_landmarks.append(np.zeros((33, 4), dtype=np.float32))

            cap.release()

            if video_landmarks:
                # Stack the list of (33, 4) arrays into a single (num_frames, 33, 4) array
                sequence_array = np.stack(video_landmarks, axis=0)

                base_name = os.path.splitext(video_file)[0]
                output_path = os.path.join(OUTPUT_FOLDER, f"{base_name}.npy")
                
                # Save the numpy array
                np.save(output_path, sequence_array)
        
        print("\nProcessing complete.")
        print(f"Skeletons saved to the '{OUTPUT_FOLDER}' directory.")

if __name__ == '__main__':
    extract_skeletons()