import os
import glob
import argparse
import numpy as np
import torch
from tqdm import tqdm
from pixielib.pixie import PIXIE
from pixielib.datasets.body_datasets import TestData
from pixielib.utils import util
from pixielib.utils.config import cfg as pixie_cfg
from pixielib.utils.tensor_cropper import transform_points
import shutil

# Fixed SMPL-X joint count output by PIXIE
N_JOINTS = 55

def parse_args():
    parser = argparse.ArgumentParser(description="Extract skeletons from Kinetics videos using PIXIE")
    parser.add_argument('--video_root', type=str, default='../kinetics_subset/videos',
                        help='Directory containing .mp4 clips')
    parser.add_argument('--output_dir', type=str, default='./pixie_skeletons',
                        help='Directory to save .npy skeleton files')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Computation device (e.g. cuda:0 or cpu)')
    parser.add_argument('--iscrop', action='store_true',
                        help='Apply cropping before PIXIE')
    parser.add_argument('--reproject', action='store_true',
                        help='Reproject joints into original image space using transform_points')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug printing')
    return parser.parse_args()


def extract_joints_from_video(video_path, pixie_model, device, iscrop, reproject, debug=False):
    # Initialize TestData loader on the video
    testdata = TestData(video_path, iscrop=iscrop, body_detector='rcnn')
    if len(testdata) == 0:
        if debug:
            print(f"[DEBUG] No frames/person detected in {video_path}")
        return np.zeros((0, N_JOINTS, 3), dtype=np.float32)

    all_joints = []
    for idx, batch in enumerate(tqdm(testdata, desc=os.path.basename(video_path), leave=False)):
        util.move_dict_to_device(batch, device)
        batch['image']    = batch['image'].unsqueeze(0)
        batch['image_hd'] = batch['image_hd'].unsqueeze(0)
        data = {'body': batch}

        with torch.no_grad():
            # Run PIXIE encode & decode with thresholding and local fitting
            param_dict = pixie_model.encode(
                data,
                threthold=True,
                keep_local=True,
                copy_and_paste=False
            )
            codedict = param_dict['body']
            opdict   = pixie_model.decode(codedict, param_type='body')

        if debug and idx < 3:
            print(f"[DEBUG] Frame {idx} opdict keys: {list(opdict.keys())}")

        joints_tensor = opdict.get('smplx_kpt3d', None)
        if joints_tensor is None:
            joints_tensor = opdict.get('joints', None)

        if joints_tensor is None:
            if debug and idx < 1:
                print(f"[DEBUG] No 'smplx_kpt3d' or 'joints' for frame {idx}")
            joints_np = np.zeros((N_JOINTS, 3), dtype=np.float32)
        else:
            joints_np = joints_tensor.squeeze(0).cpu().numpy()

        if reproject and 'tform' in batch and joints_np.size:
            # Using transform_points to map back
            points = torch.tensor(joints_np, device=device).unsqueeze(0)  # [1,J,3]
            tform  = batch['tform'].unsqueeze(0)                         # [1,3,3]
            reprojected = transform_points(points, tform).squeeze(0).cpu().numpy()
            joints_np = reprojected

        all_joints.append(joints_np)

    return np.stack(all_joints, axis=0)


def main():
    args = parse_args()
    args.video_root = os.path.abspath(args.video_root)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    pixie_model = PIXIE(config=pixie_cfg, device=device)

    os.makedirs(args.output_dir, exist_ok=True)
    videos = sorted(glob.glob(os.path.join(args.video_root, '*.mp4')))
    print(f"Found {len(videos)} videos in {args.video_root}")

    # Create a temporary directory inside the output folder for video copies
    temp_video_dir = os.path.join(args.output_dir, 'temp_videos_to_process')
    
    temp_video_dir = os.path.abspath(temp_video_dir)

    os.makedirs(temp_video_dir, exist_ok=True)

    try:
        for vp in tqdm(videos, desc='Videos'):
            vid_id = os.path.splitext(os.path.basename(vp))[0]
            out_path = os.path.join(args.output_dir, f"{vid_id}.npy")
            if os.path.exists(out_path):
                continue

            temp_video_path = os.path.join(temp_video_dir, os.path.basename(vp))
            shutil.copy(vp, temp_video_path)

            # Process the temporary video copy
            joints_arr = extract_joints_from_video(
                temp_video_path,
                pixie_model,
                device,
                args.iscrop,
                args.reproject,
                debug=args.debug
            )
            
            print(f"{vid_id}: extracted {joints_arr.shape[0]} frames, coord sum {joints_arr.sum():.5f}")
            np.save(out_path, joints_arr)
            
            # Clean up the temporary files for this video
            temp_frame_dir = os.path.splitext(temp_video_path)[0]
            if os.path.exists(temp_frame_dir):
                shutil.rmtree(temp_frame_dir)
            os.remove(temp_video_path)

    finally:
        # Clean up the main temporary directory
        if os.path.exists(temp_video_dir):
            print(f"Cleaning up temporary directory: {temp_video_dir}")
            shutil.rmtree(temp_video_dir)

    print("Done. Skeletons saved to", args.output_dir)

if __name__ == '__main__':
    main()