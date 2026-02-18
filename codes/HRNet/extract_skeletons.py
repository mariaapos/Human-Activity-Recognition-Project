import cv2
import os
import numpy as np
from tqdm import tqdm

from HRNET import HRNET, ModelType

def process_videos(video_dir, output_dir, model_path, model_type, conf_thres):
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the HRNet model once
    print("Initializing HRNet model...")
    try:
        hrnet = HRNET(path=model_path, model_type=model_type, conf_thres=conf_thres)
    except Exception as e:
        print(f"Error initializing model: {e}")
        return
    print("Model initialized successfully.")

    # Get a list of all .mp4 files in the video directory
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    
    if not video_files:
        print(f"No .mp4 videos found in '{video_dir}'.")
        return

    print(f"Found {len(video_files)} videos to process.")

    # Loop through each video file
    for video_name in tqdm(video_files, desc="Processing Videos"):
        video_path = os.path.join(video_dir, video_name)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Could not open video file {video_name}. Skipping.")
            continue

        all_frame_skeletons = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Loop through each frame of the video
        for _ in range(frame_count):
            success, frame = cap.read()
            if not success:
                break

            _, skeleton = hrnet(frame)

            # The 'skeleton' is a numpy array of shape (num_keypoints, 2).
            # If no keypoints are found above the confidence threshold,
            # the HRNet.py script already populates them with np.nan.
            # So, we just append the result directly.
            all_frame_skeletons.append(skeleton)

        cap.release()
        
        if not all_frame_skeletons:
            print(f"Warning: No frames were processed for video {video_name}. Skipping save.")
            continue

        # Convert the list of skeleton arrays into a single NumPy array
        # The shape will be (num_frames, num_keypoints, 2)
        skeletons_array = np.array(all_frame_skeletons)

        output_filename = os.path.splitext(video_name)[0] + '.npy'
        output_path = os.path.join(output_dir, output_filename)
        
        # Save the array
        np.save(output_path, skeletons_array)

    print("\nProcessing complete.")
    print(f"Skeleton data saved to '{output_dir}'.")

if __name__ == '__main__':
    VIDEO_INPUT_DIR = "../kinetics_subset/videos"
    SKELETON_OUTPUT_DIR = "./hrnet_skeletons"
    MODEL_PATH = "models/hrnet_coco_w48_384x288.onnx"
    
    if not os.path.isdir(VIDEO_INPUT_DIR):
        print(f"Error: Input directory '{VIDEO_INPUT_DIR}' not found.")
        print("Please create it and place your .mp4 files inside.")
    else:
        process_videos(
            video_dir=VIDEO_INPUT_DIR,
            output_dir=SKELETON_OUTPUT_DIR,
            model_path=MODEL_PATH,
            model_type=ModelType.COCO,
            conf_thres=0.6
        )